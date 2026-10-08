"""The actor-critic network. The only module in pokerlab that imports torch.

A policy and a critic that each read the cards through the equity network
(`rl/equity_net.py`): the policy through its frozen per-player encoder, the critic through
the equity of every player. The street arrives as part of the input vector rather than
selecting between per-street networks -- see `rl/features.py` for why.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from pokerlab.players.rl_agent import DealView, PolicyDecision, PolicyFn
from pokerlab.rl.action_space import ACTION_DIM
from pokerlab.rl.equity_net import CARDS, EquityEncoder, EquityNet
from pokerlab.rl.features import CARDS_DIM, MAX_SEATS, OBS_DIM
from pokerlab.rl.rollout import CriticFns

# Not -inf: with only one legal action, -inf makes log_softmax return -inf for
# the masked entries and turns Categorical.entropy() into NaN. A large finite
# fill still underflows to exactly zero probability in float32.
MASK_FILL = -1e9


# The defaults of every shape knob. A shape is four integers and `config.toml`'s
# `[network]` section sets them for the fleet; a checkpoint records the four it was
# built with, so `build_model_from_checkpoint` can rebuild it whatever the file says
# today.
DEFAULT_HIDDEN = 512
DEFAULT_NUM_LAYERS = 3
DEFAULT_HEAD_HIDDEN = 256
DEFAULT_HEAD_LAYERS = 0

SHAPE_KEYS = ("hidden", "num_layers", "head_hidden", "head_layers")

# Where the card planes sit in the observation: the first `CARDS_DIM` features, six planes
# of 52 (hole, flop, turn, river, whole board, hole + board). The network reads two of them
# through the equity encoder and drops all six.
PLANE = 52
HOLE_PLANE = 0
BOARD_PLANE = 4
# What the critic is given in place of the cards: one equity per seat, in
# the order of the observation's seats (slot 0 is the deciding player), 0 for a seat that
# has folded.
EQUITY_SLOTS = MAX_SEATS


def _block(in_features: int, out_features: int) -> list[nn.Module]:
    return [nn.Linear(in_features, out_features), nn.LayerNorm(out_features), nn.ReLU()]


class PokerActorCritic(nn.Module):
    """A policy and a critic, each a trunk and a head, reading the cards through the equity network.

    - **Trunk**: `num_layers` blocks of `Linear -> LayerNorm -> ReLU`, `hidden` wide.
    - **Each head**: `head_layers` such blocks, `head_hidden` wide, and then the output
      `Linear` (11 logits for the policy, 1 value for the critic). With
      `head_layers = 0` a head is the output `Linear` alone.

    `equity` is the shape of an equity network's per-player encoder (see `rl/equity_net.py`)
    and is required: there is no network without it.

    - The policy reads the observation with its card planes replaced by the encoder's
      output on the model's own cards and the board. It sees nothing but legal information.
    - The critic is a trunk and a head of its own, of the same shape, reading the observation
      with the card planes replaced by `critic_extra`: the equity of every player still in the
      hand, which needs everyone's cards and so exists only while training. It is not
      computed when `critic_extra` is not given (a model that is only playing), and the
      value is then 0.

    The encoder's weights are frozen and ride in the model's state, so a checkpoint is
    self-contained.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = DEFAULT_HIDDEN,
        num_layers: int = DEFAULT_NUM_LAYERS,
        head_hidden: int = DEFAULT_HEAD_HIDDEN,
        head_layers: int = DEFAULT_HEAD_LAYERS,
        *,
        equity: dict[str, int],
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
        self.equity = dict(equity)
        rest = obs_dim - CARDS_DIM
        self.equity_encoder = EquityEncoder(**self.equity)
        self.trunk = self._trunk(rest + self.equity["latent"], hidden, num_layers)
        self.policy_head = self._head(hidden, action_dim)
        self.value_trunk = self._trunk(rest + EQUITY_SLOTS, hidden, num_layers)
        self.value_head = self._head(hidden, 1)

    @staticmethod
    def _trunk(in_features: int, hidden: int, num_layers: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.extend(_block(in_features, hidden))
            in_features = hidden
        return nn.Sequential(*layers)

    def _head(self, in_features: int, out_features: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for _ in range(self.head_layers):
            layers.extend(_block(in_features, self.head_hidden))
            in_features = self.head_hidden
        layers.append(nn.Linear(in_features, out_features))
        return nn.Sequential(*layers)

    def policy_parameters(self) -> list[nn.Parameter]:
        """The policy's own trainable weights: its trunk and head (the encoder is frozen)."""
        return [*self.trunk.parameters(), *self.policy_head.parameters()]

    def critic_parameters(self) -> list[nn.Parameter]:
        """The critic's own weights: its trunk and head. With `policy_parameters`, every
        trainable weight of the model, each exactly once."""
        return [*self.value_trunk.parameters(), *self.value_head.parameters()]

    def critic_value(self, features: Tensor, critic_extra: Tensor) -> Tensor:
        """The state values `(B,)`, without the policy's half of the work."""
        hidden = self.value_trunk(torch.cat([features[:, CARDS_DIM:], critic_extra], dim=-1))
        return self.value_head(hidden).squeeze(-1)

    @property
    def shape(self) -> dict[str, int]:
        return {key: getattr(self, key) for key in SHAPE_KEYS}

    def forward(
        self, features: Tensor, legal_mask: Tensor, critic_extra: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Masked logits `(B, ACTION_DIM)` and state values `(B,)`.

        `critic_extra` `(B, EQUITY_SLOTS)` is what the critic reads; without it (a model that
        is only playing) the values are zero."""
        rest = features[:, CARDS_DIM:]
        latent = self.equity_encoder(
            features[:, HOLE_PLANE * PLANE : (HOLE_PLANE + 1) * PLANE],
            features[:, BOARD_PLANE * PLANE : (BOARD_PLANE + 1) * PLANE],
        )
        logits = self.policy_head(self.trunk(torch.cat([rest, latent], dim=-1)))
        logits = logits.masked_fill(~legal_mask, MASK_FILL)
        if critic_extra is None:
            return logits, features.new_zeros(features.shape[0])
        return logits, self.critic_value(features, critic_extra)


def make_policy_fn(
    model: PokerActorCritic, *, device: str | torch.device = "cpu", greedy: bool = False
) -> PolicyFn:
    """Adapt a network into the callable `RLAgentPlayer` expects.

    This is the whole bridge between torch and the engine: everything on the
    engine side stays plain Python lists.
    """
    model.to(device)

    def policy_fn(
        features: list[float], mask: list[bool], bias: list[float] | None = None, temperature: float = 1.0
    ) -> PolicyDecision:
        """`bias` and `temperature` are an opponent's style (`rl/styles.py`): added to the
        logits, then divided. A masked bin stays masked (its logit is -1e9). The learner
        never passes them, so what PPO is given is its own policy's probabilities."""
        with torch.no_grad():
            feature_batch = torch.tensor([features], dtype=torch.float32, device=device)
            mask_batch = torch.tensor([mask], dtype=torch.bool, device=device)
            logits, value = model(feature_batch, mask_batch)
            if bias is not None:
                logits = (logits + torch.tensor([bias], dtype=logits.dtype, device=device)) / temperature
            distribution = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(dim=-1) if greedy else distribution.sample()
            return PolicyDecision(
                action_index=int(action.item()),
                log_prob=float(distribution.log_prob(action).item()),
                value=float(value.item()),
            )

    def distribution(
        features: list[float], mask: list[bool], bias: list[float] | None = None, temperature: float = 1.0
    ) -> list[float]:
        """The probability of each bin for this decision -- what `policy_fn` samples from,
        without sampling (and without touching the random stream the game depends on)."""
        with torch.no_grad():
            logits, _value = model(
                torch.tensor([features], dtype=torch.float32, device=device),
                torch.tensor([mask], dtype=torch.bool, device=device),
            )
            if bias is not None:
                logits = (logits + torch.tensor([bias], dtype=logits.dtype, device=device)) / temperature
            return torch.softmax(logits, dim=-1)[0].tolist()

    policy_fn.distribution = distribution  # type: ignore[attr-defined]
    return policy_fn


def make_critic_fns(
    model: PokerActorCritic, equity_net: EquityNet, *, device: str | torch.device = "cpu"
) -> CriticFns:
    """The two batched callables a collector needs to train a model.

    `equity_net` is the full, frozen equity network: it reads every player's cards, which
    only the table knows, and its output is what the critic is given in place of the card
    planes. The closures read the live model, like `make_policy_fn`'s."""
    equity_net.to(device)

    def equity(views: Sequence[DealView]) -> list[list[float]]:
        count = len(views)
        holes = torch.zeros(count, EQUITY_SLOTS, CARDS)
        board = torch.zeros(count, CARDS)
        mask = torch.zeros(count, EQUITY_SLOTS, dtype=torch.bool)
        for row, view in enumerate(views):
            for slot, cards in enumerate(view.holes):
                if cards is not None:
                    holes[row, slot, list(cards)] = 1.0
                    mask[row, slot] = True
            board[row, list(view.board)] = 1.0
        with torch.no_grad():
            shares = equity_net(holes.to(device), board.to(device), mask.to(device))
        return shares.cpu().tolist()

    def value(features: Sequence[list[float]], extras: Sequence[list[float]]) -> list[float]:
        with torch.no_grad():
            values = model.critic_value(
                torch.tensor(features, dtype=torch.float32, device=device),
                torch.tensor(extras, dtype=torch.float32, device=device),
            )
        return values.cpu().tolist()

    return CriticFns(equity=equity, value=value)
