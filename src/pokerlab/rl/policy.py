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
from pokerlab.rl.features import CARDS_DIM, FEATURE_VERSION, MAX_SEATS, observation_dim
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
        obs_dim: int | None = None,
        action_dim: int = ACTION_DIM,
        hidden: int = DEFAULT_HIDDEN,
        num_layers: int = DEFAULT_NUM_LAYERS,
        head_hidden: int = DEFAULT_HEAD_HIDDEN,
        head_layers: int = DEFAULT_HEAD_LAYERS,
        *,
        equity: dict[str, int],
        feature_version: int = FEATURE_VERSION,
    ) -> None:
        super().__init__()
        # The encoding the model reads (`rl/features.py`): its input has that many numbers, and
        # whoever feeds it encodes at that version. `obs_dim` is for a caller who wants to say
        # it; it can only agree with the version.
        expected = observation_dim(feature_version)
        if obs_dim is not None and obs_dim != expected:
            raise ValueError(
                f"feature version {feature_version} has {expected} features, not obs_dim={obs_dim}"
            )
        obs_dim = expected
        self.feature_version = feature_version
        self.obs_dim = obs_dim
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
    # What `RLAgentPlayer` encodes with when it is not told otherwise: the model's own version.
    policy_fn.feature_version = model.feature_version  # type: ignore[attr-defined]
    return policy_fn


def make_search_policy_fn(
    model: PokerActorCritic, *, device: str | torch.device = "cpu", cache_size: int = 200_000
) -> PolicyFn:
    """The policy of `make_policy_fn`, for the rollouts of a search (`rl/search.py`), faster.

    The probabilities are the same (the same layers, in the same order: pinned by a test). What
    changes is the cost of asking:

    - the card encoder's output depends on the hole cards and the board alone, and a search asks
      about the same cards on the same board thousands of times, so it is remembered by those
      cards (up to `cache_size` of them, then started afresh);
    - the action is drawn from the softmax directly: no `Categorical`, no log-probability and no
      value, which a search does not keep (the decision carries 0 for both).

    The draw comes from torch's stream, like `make_policy_fn`'s, so `torch_seeded` still governs
    it. Not for training: PPO needs the log-probability this leaves out."""
    model.to(device)
    latents: dict[tuple[tuple[int, ...], tuple[int, ...]], Tensor] = {}
    hole_slice = slice(HOLE_PLANE * PLANE, (HOLE_PLANE + 1) * PLANE)
    board_slice = slice(BOARD_PLANE * PLANE, (BOARD_PLANE + 1) * PLANE)

    def cards_of(features: list[float]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        hole = tuple(index for index, value in enumerate(features[hole_slice]) if value)
        board = tuple(index for index, value in enumerate(features[board_slice]) if value)
        return hole, board

    def latents_of(keys: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> Tensor:
        """The encoder's output for each `(hole, board)`, those not remembered in one call."""
        missing = list(dict.fromkeys(key for key in keys if key not in latents))
        if missing:
            if len(latents) + len(missing) > cache_size:
                latents.clear()
                missing = list(dict.fromkeys(keys))
            holes = torch.zeros(len(missing), PLANE, device=device)
            boards = torch.zeros(len(missing), PLANE, device=device)
            for row, (hole, board) in enumerate(missing):
                holes[row, list(hole)] = 1.0
                boards[row, list(board)] = 1.0
            with torch.no_grad():
                encoded = model.equity_encoder(holes, boards)
            for row, key in enumerate(missing):
                latents[key] = encoded[row : row + 1]
        return torch.cat([latents[key] for key in keys])

    def latent_of(features: list[float]) -> Tensor:
        return latents_of([cards_of(features)])

    def logits_of(
        features: list[float], mask: list[bool], bias: list[float] | None, temperature: float
    ) -> Tensor:
        with torch.no_grad():
            rest = torch.tensor([features[CARDS_DIM:]], dtype=torch.float32, device=device)
            logits = model.policy_head(model.trunk(torch.cat([rest, latent_of(features)], dim=-1)))
            logits = logits.masked_fill(~torch.tensor([mask], dtype=torch.bool, device=device), MASK_FILL)
            if bias is not None:
                logits = (logits + torch.tensor([bias], dtype=logits.dtype, device=device)) / temperature
            return logits

    def policy_fn(
        features: list[float], mask: list[bool], bias: list[float] | None = None, temperature: float = 1.0
    ) -> PolicyDecision:
        probabilities = torch.softmax(logits_of(features, mask, bias, temperature), dim=-1)
        return PolicyDecision(action_index=int(torch.multinomial(probabilities, 1).item()))

    def distribution(
        features: list[float], mask: list[bool], bias: list[float] | None = None, temperature: float = 1.0
    ) -> list[float]:
        return torch.softmax(logits_of(features, mask, bias, temperature), dim=-1)[0].tolist()

    def batch_distribution(features: list[list[float]], masks: list[list[bool]]) -> list[list[float]]:
        """`distribution` for many decisions in one forward pass (what `rl/search.py`'s
        `BatchGate` evaluates): the same probabilities, row by row."""
        with torch.no_grad():
            latent = latents_of([cards_of(row) for row in features])
            rest = torch.tensor([row[CARDS_DIM:] for row in features], dtype=torch.float32, device=device)
            logits = model.policy_head(model.trunk(torch.cat([rest, latent], dim=-1)))
            logits = logits.masked_fill(~torch.tensor(masks, dtype=torch.bool, device=device), MASK_FILL)
            return torch.softmax(logits, dim=-1).tolist()

    policy_fn.distribution = distribution  # type: ignore[attr-defined]
    policy_fn.batch_distribution = batch_distribution  # type: ignore[attr-defined]
    policy_fn.feature_version = model.feature_version  # type: ignore[attr-defined]
    return policy_fn


def make_leaf_value(
    model: PokerActorCritic, equity_net: EquityNet, *, value_unit: float, device: str | torch.device = "cpu"
):
    """What a decision is worth to the player deciding, in big blinds, by the model's critic: the
    `leaf_value` of `rl/search.py`.

    The critic reads the decision's features and every player's equity (from `equity_net`, the
    full network the critic was trained with), so it needs every player's cards: a search's world
    knows them. It estimates the player's result for the whole hand in the training reward's
    unit; `value_unit` turns that into big blinds (1 / the reward scale: with the default scale,
    big blind over the deepest stack, it is that stack in big blinds)."""
    from pokerlab.players.rl_agent import deal_view

    critic = make_critic_fns(model, equity_net, device=device)

    def leaf_value(features: list[float], observation, holes) -> float:
        extra = critic.equity([deal_view(observation, holes)])[0]
        return critic.value([features], [extra])[0] * value_unit

    return leaf_value


def deal_tensors(views: Sequence[DealView]) -> tuple[Tensor, Tensor, Tensor]:
    """The equity network's inputs for a batch of deals: every seat's hole cards `(n, EQUITY_SLOTS,
    CARDS)`, the board `(n, CARDS)` and which seats hold cards `(n, EQUITY_SLOTS)`.

    The ones are written with one indexed assignment per tensor from flat lists of
    (row, seat, card) indices. Assigning them one view and one seat at a time was ~7 s of pure
    Python on an iteration's 66,000 decisions, and the same on a machine with a GPU."""
    count = len(views)
    holes = torch.zeros(count, EQUITY_SLOTS, CARDS)
    board = torch.zeros(count, CARDS)
    mask = torch.zeros(count, EQUITY_SLOTS, dtype=torch.bool)
    hole_rows: list[int] = []
    hole_seats: list[int] = []
    hole_cards: list[int] = []
    seat_rows: list[int] = []
    seat_seats: list[int] = []
    board_rows: list[int] = []
    board_cards: list[int] = []
    for row, view in enumerate(views):
        for seat, cards in enumerate(view.holes):
            if cards is not None:
                hole_rows.extend([row] * len(cards))
                hole_seats.extend([seat] * len(cards))
                hole_cards.extend(cards)
                seat_rows.append(row)
                seat_seats.append(seat)
        board_rows.extend([row] * len(view.board))
        board_cards.extend(view.board)
    if hole_cards:
        holes[hole_rows, hole_seats, hole_cards] = 1.0
        mask[seat_rows, seat_seats] = True
    if board_cards:
        board[board_rows, board_cards] = 1.0
    return holes, board, mask


def make_critic_fns(
    model: PokerActorCritic, equity_net: EquityNet, *, device: str | torch.device = "cpu"
) -> CriticFns:
    """The two batched callables a collector needs to train a model.

    `equity_net` is the full, frozen equity network: it reads every player's cards, which
    only the table knows, and its output is what the critic is given in place of the card
    planes. The closures read the live model, like `make_policy_fn`'s."""
    equity_net.to(device)

    def equity(views: Sequence[DealView]) -> list[list[float]]:
        holes, board, mask = deal_tensors(views)
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
