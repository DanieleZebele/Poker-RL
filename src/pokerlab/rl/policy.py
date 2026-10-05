"""The actor-critic network. The only module in pokerlab that imports torch.

One shared trunk, one policy head, one value head, with the street arriving as
part of the input vector rather than selecting between per-street networks --
see `rl/features.py` for why. Should diagnostics ever show a single head
starving one street, per-street heads can be bolted onto this same trunk
without retraining it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pokerlab.players.rl_agent import PolicyDecision, PolicyFn
from pokerlab.rl.action_space import ACTION_DIM
from pokerlab.rl.features import OBS_DIM

# Not -inf: with only one legal action, -inf makes log_softmax return -inf for
# the masked entries and turns Categorical.entropy() into NaN. A large finite
# fill still underflows to exactly zero probability in float32.
MASK_FILL = -1e9


# The defaults of every shape knob. A shape is four integers and `config.toml`'s
# `[network]` section sets them for the fleet; a checkpoint records the four it was
# built with, so `build_model_from_checkpoint` can rebuild it whatever the file says
# today. The head defaults describe the network as it was before heads had hidden
# layers: no extra layer, just the output `Linear`.
DEFAULT_HIDDEN = 512
DEFAULT_NUM_LAYERS = 3
DEFAULT_HEAD_HIDDEN = 256
DEFAULT_HEAD_LAYERS = 0

SHAPE_KEYS = ("hidden", "num_layers", "head_hidden", "head_layers")


def _block(in_features: int, out_features: int) -> list[nn.Module]:
    return [nn.Linear(in_features, out_features), nn.LayerNorm(out_features), nn.ReLU()]


class PokerActorCritic(nn.Module):
    """A shared trunk and two heads, each a small MLP of its own.

    - **Trunk**: `num_layers` blocks of `Linear -> LayerNorm -> ReLU`, `hidden` wide.
    - **Each head**: `head_layers` such blocks, `head_hidden` wide, and then the output
      `Linear` (11 logits for the policy, 1 value for the critic). With
      `head_layers = 0` a head is the output `Linear` alone, which is what the network
      was before heads had hidden layers.

    The two heads have the same shape but separate weights, so the critic's gradient
    reaches the trunk and not the policy's private layers, and the other way round:
    giving the heads room is what lets them specialise on a trunk both depend on.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = DEFAULT_HIDDEN,
        num_layers: int = DEFAULT_NUM_LAYERS,
        head_hidden: int = DEFAULT_HEAD_HIDDEN,
        head_layers: int = DEFAULT_HEAD_LAYERS,
    ) -> None:
        super().__init__()
        if hidden < 1 or num_layers < 1 or head_hidden < 1 or head_layers < 0:
            raise ValueError(
                "a network needs hidden >= 1, num_layers >= 1, head_hidden >= 1 and "
                f"head_layers >= 0, got {hidden}/{num_layers}/{head_hidden}/{head_layers}"
            )
        # Recorded in checkpoints so an archived model can be rebuilt without
        # the caller having to remember what shape it was trained at.
        self.hidden = hidden
        self.num_layers = num_layers
        self.head_hidden = head_hidden
        self.head_layers = head_layers
        layers: list[nn.Module] = []
        in_features = obs_dim
        for _ in range(num_layers):
            layers.extend(_block(in_features, hidden))
            in_features = hidden
        self.trunk = nn.Sequential(*layers)
        self.policy_head = self._head(hidden, action_dim)
        self.value_head = self._head(hidden, 1)

    def _head(self, in_features: int, out_features: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for _ in range(self.head_layers):
            layers.extend(_block(in_features, self.head_hidden))
            in_features = self.head_hidden
        layers.append(nn.Linear(in_features, out_features))
        return nn.Sequential(*layers)

    @property
    def shape(self) -> dict[str, int]:
        return {key: getattr(self, key) for key in SHAPE_KEYS}

    def forward(self, features: Tensor, legal_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Masked logits `(B, ACTION_DIM)` and state values `(B,)`."""
        hidden = self.trunk(features)
        logits = self.policy_head(hidden).masked_fill(~legal_mask, MASK_FILL)
        return logits, self.value_head(hidden).squeeze(-1)


def make_policy_fn(
    model: PokerActorCritic, *, device: str | torch.device = "cpu", greedy: bool = False
) -> PolicyFn:
    """Adapt a network into the callable `RLAgentPlayer` expects.

    This is the whole bridge between torch and the engine: everything on the
    engine side stays plain Python lists.
    """
    model.to(device)

    def policy_fn(features: list[float], mask: list[bool]) -> PolicyDecision:
        with torch.no_grad():
            feature_batch = torch.tensor([features], dtype=torch.float32, device=device)
            mask_batch = torch.tensor([mask], dtype=torch.bool, device=device)
            logits, value = model(feature_batch, mask_batch)
            distribution = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(dim=-1) if greedy else distribution.sample()
            return PolicyDecision(
                action_index=int(action.item()),
                log_prob=float(distribution.log_prob(action).item()),
                value=float(value.item()),
            )

    return policy_fn
