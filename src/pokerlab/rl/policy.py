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


class PokerActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = 512,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        # Recorded in checkpoints so an archived model can be rebuilt without
        # the caller having to remember what shape it was trained at.
        self.hidden = hidden
        self.num_layers = num_layers
        layers: list[nn.Module] = []
        in_features = obs_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_features, hidden), nn.LayerNorm(hidden), nn.ReLU()])
            in_features = hidden
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

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
