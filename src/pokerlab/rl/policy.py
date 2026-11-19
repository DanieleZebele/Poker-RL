"""The actor-critic network. The only module in pokerlab that imports torch.

A policy and a critic that each read the cards through the equity network
(`rl/equity_net.py`): the policy through its frozen per-player encoder, the critic through
the equity of every player. The street arrives as part of the input vector rather than
selecting between per-street networks -- see `rl/features.py` for why.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from pokerlab.players.rl_agent import DealView, PolicyDecision, PolicyFn
from pokerlab.rl.action_space import ACTION_DIM, RAISE_SIZINGS
from pokerlab.rl.action_space import raise_sizing as process_raise_sizing
from pokerlab.rl.belief import CONFIG_KEYS as BELIEF_KEYS
from pokerlab.rl.belief import BeliefModel, BeliefReader
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

# The critic as a classifier (HL-Gauss, "Stop Regressing", Farebrother et al. 2024): its head
# gives `bins` logits over a fixed support, the value is the softmax's mean over the bins'
# centres and the loss the cross-entropy against a Gaussian `CRITIC_SIGMA_RATIO` bins wide
# centred on the target. The bins are uniform in symlog(v) = sign(v) log(1 + |v| / unit), with
# `unit` one big blind in the value's scale (the reward scale): fine near 0, where a short
# stack's hands live, coarse in the tails. `low`/`high` are the support's ends in the value's
# own unit. A model's critic is a dict of those four (`PokerActorCritic.critic`), or None for
# the scalar head trained on the squared error. Measured offline (`studies/critic_loss`): 101
# symlog bins explain more of the held-out return than the squared error in every group, worth
# ~1.6x the data.
CRITIC_KEYS = ("bins", "low", "high", "unit")
CRITIC_SIGMA_RATIO = 0.75
# How the classifier's output layer is started (`reset_critic_output`), recorded in every
# checkpoint: 2 is the start that learns. A classifier checkpoint without it was saved by a run
# whose critic began at 1, which barely learns, and `ppo.load_checkpoint` starts that layer afresh.
CRITIC_OUTPUT_INIT = 2


def symlog(value: Tensor, unit: float) -> Tensor:
    return torch.sign(value) * torch.log1p(value.abs() / unit)


def critic_edges(critic: dict[str, Any]) -> Tensor:
    """The `bins + 1` edges of the bins, in symlog space."""
    unit = float(critic["unit"])
    low = math.copysign(math.log1p(abs(critic["low"]) / unit), critic["low"])
    high = math.copysign(math.log1p(abs(critic["high"]) / unit), critic["high"])
    return torch.linspace(low, high, int(critic["bins"]) + 1, dtype=torch.float64).float()


def critic_centres(critic: dict[str, Any]) -> Tensor:
    """Each bin's centre as a value (back from symlog): what the predicted distribution is
    averaged over."""
    edges = critic_edges(critic).double()
    middle = (edges[1:] + edges[:-1]) / 2
    return (torch.sign(middle) * torch.expm1(middle.abs()) * float(critic["unit"])).float()


def hl_gauss_targets(targets: Tensor, edges: Tensor) -> Tensor:
    """The probability of each bin under a Gaussian on `targets` (already in symlog space)
    `CRITIC_SIGMA_RATIO` bins wide, renormalised to the support (a target outside it is
    clipped to its end): `(B,)` -> `(B, bins)`."""
    sigma = CRITIC_SIGMA_RATIO * float(edges[1] - edges[0])
    clipped = targets.clamp(float(edges[0]), float(edges[-1]))
    cdf = 0.5 * (1 + torch.erf((edges[None, :] - clipped[:, None]) / (math.sqrt(2) * sigma)))
    return (cdf[:, 1:] - cdf[:, :-1]) / (cdf[:, -1:] - cdf[:, :1])


def normalise_critic(critic: dict[str, Any] | None) -> dict[str, Any] | None:
    """A critic spec as the model keeps and a checkpoint records it, refusing one that cannot
    be built."""
    if critic is None:
        return None
    missing = [key for key in CRITIC_KEYS if key not in critic]
    if missing:
        raise ValueError(f"a classifier critic needs {CRITIC_KEYS}, missing {missing}")
    spec = {"bins": int(critic["bins"]), "low": float(critic["low"]),
            "high": float(critic["high"]), "unit": float(critic["unit"])}
    if spec["bins"] < 2 or not spec["low"] < 0 < spec["high"] or spec["unit"] <= 0:
        raise ValueError(f"a classifier critic needs bins >= 2, low < 0 < high and unit > 0, got {spec}")
    return spec


def _block(in_features: int, out_features: int) -> list[nn.Module]:
    return [nn.Linear(in_features, out_features), nn.LayerNorm(out_features), nn.ReLU()]


class ResidualBlock(nn.Module):
    """`x + Linear(ReLU(Linear(LayerNorm(x))))`, `width` wide throughout (SimBa's block, with
    no widening inside)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.inner = nn.Linear(width, width)
        self.outer = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.outer(torch.relu(self.inner(self.norm(x))))


class PokerActorCritic(nn.Module):
    """A policy and a critic, each a trunk and a head, reading the cards through the equity network.

    - **Trunk**: `num_layers` blocks of `Linear -> LayerNorm -> ReLU`, `hidden` wide.
    - **Each head**: `head_layers` such blocks, `head_hidden` wide, and then the output
      `Linear` (11 logits for the policy; for the critic 1 value, or `critic["bins"]` logits
      when it is a classifier, see `CRITIC_KEYS`). With
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
        critic: dict[str, Any] | None = None,
        belief: dict[str, Any] | None = None,
        residual: int = 0,
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
        # 1: each trunk is a `Linear` into `hidden`, `num_layers` residual blocks and a final
        # LayerNorm (SimBa, Lee et al. 2025), instead of `num_layers` plain blocks. A study
        # setting: the fleet's models are all 0, and a checkpoint without the key is 0.
        if residual not in (0, 1):
            raise ValueError(f"residual is 0 or 1, not {residual!r}")
        self.residual = int(residual)
        self.equity = dict(equity)
        rest = obs_dim - CARDS_DIM
        self.equity_encoder = EquityEncoder(**self.equity)
        self.trunk = self._trunk(rest + self.equity["latent"], hidden, num_layers, self.residual)
        self.policy_head = self._head(hidden, action_dim)
        self.value_trunk = self._trunk(rest + EQUITY_SLOTS, hidden, num_layers, self.residual)
        self.critic = normalise_critic(critic)
        self.value_head = self._head(hidden, self.critic["bins"] if self.critic else 1)
        if self.critic is not None:
            # Not persistent: rebuilt from `critic`, which the checkpoint records, so the
            # state of a scalar-critic model and of a classifier differ only in the head.
            self.register_buffer("critic_edges", critic_edges(self.critic), persistent=False)
            self.register_buffer("critic_centres", critic_centres(self.critic), persistent=False)
            self.reset_critic_output()
        # Feature version 5 reads a belief about every opponent: the belief network (`rl/belief.py`),
        # frozen, rides in the model's own state like the equity encoder. `belief` is its shape and
        # the fingerprint of the file it came from; the weights are loaded by whoever builds the
        # model (`train.py` from the file, `build_model_from_checkpoint` from the checkpoint).
        if (feature_version == 5) != (belief is not None):
            raise ValueError("a network of feature version 5 needs a belief network, and only it has one")
        self.belief = None if belief is None else {**{k: int(belief[k]) for k in BELIEF_KEYS},
                                                   "fingerprint": str(belief["fingerprint"])}
        self.belief_net = (
            None if belief is None
            else BeliefModel(self.equity["latent"], **{k: self.belief[k] for k in BELIEF_KEYS})
        )

    def reset_critic_output(self) -> None:
        """Start the classifier's output layer afresh: the weights as PyTorch initialises a
        `Linear`, the biases `-|z|` for a bin whose centre is `z` in symlog space -- a broad bump
        at 0, so the value starts at about +1 big blind (a near-uniform distribution would start
        it at about +1 deepest stack) and every logit is within reach (the far tail at -6.6).

        Measured (`CRITIC_OUTPUT_INIT` 1 was the other way): weights 0 and biases the log of the
        narrow HL-Gauss target of 0, floored at 1e-6, put almost every logit at -13.8, and with
        Adam's steps the critic explained 0.006 of the return after 400 steps against 0.24 from
        this start; in PPO it explained nothing after 10 iterations. The scalar head starts near
        0 by itself."""
        if self.critic is None:
            return
        output = self.value_head[-1]
        output.reset_parameters()
        centres = (self.critic_edges[1:] + self.critic_edges[:-1]) / 2
        with torch.no_grad():
            output.bias.copy_(-centres.abs())

    @staticmethod
    def _trunk(in_features: int, hidden: int, num_layers: int, residual: int = 0) -> nn.Sequential:
        if residual:
            return nn.Sequential(nn.Linear(in_features, hidden),
                                 *(ResidualBlock(hidden) for _ in range(num_layers)), nn.LayerNorm(hidden))
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

    def belief_reader(self, device: str | torch.device = "cpu") -> BeliefReader | None:
        """What computes this model's beliefs at a decision (None below version 5)."""
        if self.belief_net is None:
            return None
        return BeliefReader(self.belief_net, self.equity_encoder, self.belief["fingerprint"], device)

    def policy_parameters(self) -> list[nn.Parameter]:
        """The policy's own trainable weights: its trunk and head (the encoder is frozen)."""
        return [*self.trunk.parameters(), *self.policy_head.parameters()]

    def critic_parameters(self) -> list[nn.Parameter]:
        """The critic's own weights: its trunk and head. With `policy_parameters`, every
        trainable weight of the model, each exactly once."""
        return [*self.value_trunk.parameters(), *self.value_head.parameters()]

    def _critic_output(self, features: Tensor, critic_extra: Tensor) -> Tensor:
        """The head's output: `(B, 1)` for the scalar critic, `(B, bins)` logits for a classifier."""
        return self.value_head(
            self.value_trunk(torch.cat([features[:, CARDS_DIM:], critic_extra], dim=-1))
        )

    def _value_of(self, output: Tensor) -> Tensor:
        if self.critic is None:
            return output.squeeze(-1)
        return torch.softmax(output, dim=-1) @ self.critic_centres

    def critic_value(self, features: Tensor, critic_extra: Tensor) -> Tensor:
        """The state values `(B,)`, without the policy's half of the work."""
        return self._value_of(self._critic_output(features, critic_extra))

    def critic_terms(
        self, features: Tensor, critic_extra: Tensor, returns: Tensor
    ) -> tuple[Tensor, Tensor]:
        """The values `(B,)` and each decision's critic loss `(B,)` against `returns`: the
        squared error for the scalar critic, the cross-entropy against the HL-Gauss target
        for a classifier."""
        output = self._critic_output(features, critic_extra)
        values = self._value_of(output)
        if self.critic is None:
            return values, (values - returns) ** 2
        target = hl_gauss_targets(symlog(returns, self.critic["unit"]), self.critic_edges)
        return values, -(target * torch.log_softmax(output, dim=-1)).sum(dim=-1)

    def policy_logits(self, features: Tensor, legal_mask: Tensor) -> Tensor:
        """The masked logits `(B, ACTION_DIM)`, without the critic."""
        latent = self.equity_encoder(
            features[:, HOLE_PLANE * PLANE : (HOLE_PLANE + 1) * PLANE],
            features[:, BOARD_PLANE * PLANE : (BOARD_PLANE + 1) * PLANE],
        )
        logits = self.policy_head(self.trunk(torch.cat([features[:, CARDS_DIM:], latent], dim=-1)))
        return logits.masked_fill(~legal_mask, MASK_FILL)

    @property
    def shape(self) -> dict[str, int]:
        return {**{key: getattr(self, key) for key in SHAPE_KEYS}, "residual": self.residual}

    def forward(
        self, features: Tensor, legal_mask: Tensor, critic_extra: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Masked logits `(B, ACTION_DIM)` and state values `(B,)`.

        `critic_extra` `(B, EQUITY_SLOTS)` is what the critic reads; without it (a model that
        is only playing) the values are zero."""
        logits = self.policy_logits(features, legal_mask)
        if critic_extra is None:
            return logits, features.new_zeros(features.shape[0])
        return logits, self.critic_value(features, critic_extra)


# How many card inputs a policy function remembers the encoder's output for (`make_policy_fn`):
# the preflop has 169 of them once the suits are in canonical order, the later streets many more.
ENCODER_CACHE_SIZE = 20_000


def canonical_cards(hole: Sequence[int], board: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """`equity_net.canonical_suits` for one player, in plain Python: the indices (`suit * 13 +
    rank`) of the hole cards and of the board once the suits are put in canonical order. The
    same order, ties included (pinned by a test); a few microseconds instead of a dozen small
    tensor operations."""
    key = [0, 0, 0, 0]
    for card in board:
        suit, rank = divmod(card, 13)
        key[suit] += 10**12 + (1 << rank) * 10**5
    for card in hole:
        suit, rank = divmod(card, 13)
        key[suit] += 10**9 + (1 << rank)
    order = sorted(range(4), key=lambda suit: -key[suit])  # stable: ties keep their order
    place = [0, 0, 0, 0]
    for new, old in enumerate(order):
        place[old] = new
    return (
        tuple(sorted(place[c // 13] * 13 + c % 13 for c in hole)),
        tuple(sorted(place[c // 13] * 13 + c % 13 for c in board)),
    )


def _plane_cards(features: Sequence[float], plane: int) -> list[int]:
    start = plane * PLANE
    return [index for index, value in enumerate(features[start : start + PLANE]) if value]


def make_policy_fn(
    model: PokerActorCritic,
    *,
    device: str | torch.device = "cpu",
    greedy: bool = False,
    raise_sizing: str | None = None,
) -> PolicyFn:
    """Adapt a network into the callable `RLAgentPlayer` expects.

    This is the whole bridge between torch and the engine: everything on the
    engine side stays plain Python lists.

    The probabilities are `model.policy_logits`'s and the draw is `Categorical`'s, step for
    step (pinned by tests), but a decision is asked for at a fraction of the cost, since at
    batch 1 the network's arithmetic is a minority of it: the suits are put in canonical
    order in Python (`canonical_cards`), the card encoder's output is remembered by those
    cards (the encoder is frozen; `ENCODER_CACHE_SIZE` of them, then started afresh), and
    the action is drawn from the normalised logits without building a distribution.

    `raise_sizing` (default: the process's, `action_space.set_raise_sizing`) is how a drawn raise is sized when
    the model plays: `bin`, the bin's own level, or `weighted`, the mean of the open raise bins'
    levels weighted by their probabilities (`action_space.weighted_raise_to`). Which bin is drawn,
    and its log-probability, do not depend on it.
    """
    model.to(device)
    sizing = process_raise_sizing() if raise_sizing is None else raise_sizing
    if sizing not in RAISE_SIZINGS:
        raise ValueError(f"raise sizing {sizing!r} is not one of {RAISE_SIZINGS}")
    encoder = model.equity_encoder
    latents: dict[tuple[tuple[int, ...], tuple[int, ...]], Tensor] = {}

    def logits_of(features: list[float], mask: list[bool]) -> Tensor:
        cards = canonical_cards(_plane_cards(features, HOLE_PLANE), _plane_cards(features, BOARD_PLANE))
        latent = latents.get(cards)
        if latent is None:
            if len(latents) >= ENCODER_CACHE_SIZE:
                latents.clear()
            planes = [0.0] * (2 * PLANE)
            for card in cards[0]:
                planes[card] = 1.0
            for card in cards[1]:
                planes[PLANE + card] = 1.0
            latent = functional.layer_norm(
                encoder.phi(torch.tensor([planes], dtype=torch.float32, device=device)), (encoder.latent,)
            )
            latents[cards] = latent
        inputs = torch.cat(
            [torch.tensor([features[CARDS_DIM:]], dtype=torch.float32, device=device), latent], dim=-1
        )
        logits = model.policy_head(model.trunk(inputs))
        return logits.masked_fill(~torch.tensor([mask], dtype=torch.bool, device=device), MASK_FILL)

    def styled(logits: Tensor, bias: list[float] | None, temperature: float) -> Tensor:
        if bias is None:
            return logits
        return (logits + torch.tensor([bias], dtype=logits.dtype, device=device)) / temperature

    def policy_fn(
        features: list[float], mask: list[bool], bias: list[float] | None = None, temperature: float = 1.0
    ) -> PolicyDecision:
        """`bias` and `temperature` are an opponent's style (`rl/styles.py`): added to the
        logits, then divided. A masked bin stays masked (its logit is -1e9). The learner
        never passes them, so what PPO is given is its own policy's probabilities."""
        with torch.inference_mode():
            logits = styled(logits_of(features, mask), bias, temperature)
            # What `Categorical(logits=...)` does: normalise, softmax, one draw with replacement.
            normalised = logits - logits.logsumexp(dim=-1, keepdim=True)
            if greedy:
                action = int(logits.argmax(dim=-1))
            else:
                action = int(torch.multinomial(torch.softmax(normalised, dim=-1), 1, True))
            return PolicyDecision(
                action_index=action,
                log_prob=float(normalised[0, action]),
                value=0.0,
                probabilities=normalised.exp()[0].tolist() if sizing == "weighted" else None,
            )

    def distribution(
        features: list[float], mask: list[bool], bias: list[float] | None = None, temperature: float = 1.0
    ) -> list[float]:
        """The probability of each bin for this decision -- what `policy_fn` samples from,
        without sampling (and without touching the random stream the game depends on)."""
        with torch.inference_mode():
            return torch.softmax(styled(logits_of(features, mask), bias, temperature), dim=-1)[0].tolist()

    policy_fn.distribution = distribution  # type: ignore[attr-defined]
    # What `RLAgentPlayer` encodes with when it is not told otherwise: the model's own version.
    policy_fn.feature_version = model.feature_version  # type: ignore[attr-defined]
    # Version 5: what computes the beliefs `RLAgentPlayer` puts in the features.
    policy_fn.beliefs = model.belief_reader(device)  # type: ignore[attr-defined]
    # How `RLAgentPlayer` sizes a drawn raise.
    policy_fn.raise_sizing = sizing  # type: ignore[attr-defined]
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
    policy_fn.beliefs = model.belief_reader(device)  # type: ignore[attr-defined]
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
