"""The belief network: every opponent's range of hole cards, from what the table saw.

Feature version 5 (`rl/features.py`) gives the policy, for every opponent still in the hand, a
vector describing the cards the model believes that opponent holds. This module computes it:

    context  h = MLP([GRU(the hand's tokens, the target's own marked) | target's statics | board
                      | memory of the target])
    combos   psi(c) = MLP([phi(c, board) | class embedding of c])        for all 1326 combos
    range    p(c) = softmax(<A h, psi(c)> / sqrt(rank)),  0 on the combos the observer sees

`phi` is the frozen per-player encoder of the equity network, the 32 numbers the policy already
reads for its own cards. The vector the policy is given (`BELIEF_FEATURES`) is the range's mean in
that space and how wide the range is (its entropy over the largest possible, `log 1326`): for a
range that is one hand, exactly `phi` of that hand and 0 -- which is what lets training show the
policy, for a share of its decisions, the opponents' true cards in the same place (`oracle`).

The **memory** of a player is a state per player and table: a GRU over the player's earlier hands
at this table (all their tokens, its cards only when it showed them at a showdown, its result),
one step per hand (`PlayerMemory`); before a player's first hand it is a learned start. The table
keeps the hands (`belief_inputs.MemoryTracker`), each network its own states.

The network is trained apart, on self-play where every card is known (`studies/belief`), and
frozen: a file (`[network] belief_model` in `config.toml`) is copied into the model when it is
built, so a published model carries its belief network as it carries its equity encoder, and
records the file's fingerprint (`load_belief_file`). It never sees another player's cards: it reads
the observation, and the memory reads only the cards a showdown revealed.

Like `rl/policy.py`, this imports torch, and is imported only from there and from training code.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from pokerlab.rl.belief_inputs import (
    COMBO_CARDS,
    NUM_COMBOS,
    STATIC_DIM,
    TOKEN_DIM,
    MemoryHand,
    decision_question,
    near_combos,
)
from pokerlab.rl.equity_net import EquityEncoder
from pokerlab.rl.features import BELIEF_LATENT

MASK_FILL = -1e9
CONFIG_KEYS = ("hidden", "gru", "rank", "state", "embed")
LOG_COMBOS = math.log(NUM_COMBOS)
# How many boards a reader remembers the combos' features for (each ~0.5 MB): a hand's board is
# asked about at every decision and for every opponent on its street.
BOARD_CACHE_SIZE = 64


def _block(inputs: int, width: int) -> list[nn.Module]:
    return [nn.Linear(inputs, width), nn.LayerNorm(width), nn.ReLU()]


def combo_classes() -> Tensor:
    """The starting-hand class (0..168) of every combo: pairs, suited, offsuit."""
    classes = []
    for a, b in COMBO_CARDS:
        ra, rb, sa, sb = a % 13, b % 13, a // 13, b // 13
        hi, lo = max(ra, rb), min(ra, rb)
        classes.append(hi * 13 + lo if (sa == sb or hi == lo) else lo * 13 + hi)
    return torch.tensor(classes)


def card_planes(cards: Tensor) -> Tensor:
    """`(B, k)` card indices, -1 for none, as `(B, 52)` in {0, 1}."""
    planes = torch.zeros(cards.shape[0], 53, device=cards.device)
    planes.scatter_(1, torch.where(cards < 0, 52, cards).long(), 1.0)
    return planes[:, :52]


class BeliefNet(nn.Module):
    """The range of one target, for a batch of (observer, target) questions."""

    def __init__(self, latent: int, hidden: int, gru: int, rank: int, state: int) -> None:
        super().__init__()
        self.token_in = nn.Linear(TOKEN_DIM + 1, gru)
        self.gru = nn.GRU(gru, gru, batch_first=True)
        self.context = nn.Sequential(*_block(gru + STATIC_DIM + 52 + state, hidden), *_block(hidden, hidden))
        self.query = nn.Linear(hidden, rank)
        self.class_embedding = nn.Embedding(169, 16)
        self.psi = nn.Sequential(*_block(latent + 16, hidden), nn.Linear(hidden, rank))
        self.rank = rank
        self.register_buffer("classes", combo_classes(), persistent=False)

    def queries(self, tokens: Tensor, lengths: Tensor, static: Tensor, board: Tensor, memory: Tensor) -> Tensor:
        """`(B, rank)`: `tokens` `(B, L, TOKEN_DIM + 1)` (last column: the target's own), `lengths`
        `(B,)`, `static` `(B, STATIC_DIM)`, `board` `(B, 52)` planes, `memory` `(B, state)`."""
        out, _ = self.gru(torch.relu(self.token_in(tokens)))
        summary = out[torch.arange(tokens.shape[0], device=tokens.device), lengths - 1]
        return self.query_of(summary, static, board, memory)

    def query_of(self, summary: Tensor, static: Tensor, board: Tensor, memory: Tensor) -> Tensor:
        """The query from the GRU's output after the last token, `(B, gru)`."""
        return self.query(self.context(torch.cat([summary, static, board, memory], dim=-1)))

    def combo_features(self, phi: Tensor) -> Tensor:
        """`phi` `(..., 1326, latent)` -> `(..., 1326, rank)`."""
        classes = self.class_embedding(self.classes).expand(*phi.shape[:-1], -1)
        return self.psi(torch.cat([phi, classes], dim=-1))


class PlayerMemory(nn.Module):
    """One vector per (hand, player), then a GRU over a player's hands in the order they came."""

    def __init__(self, latent: int, gru: int, embed: int, state: int) -> None:
        super().__init__()
        self.token_in = nn.Linear(TOKEN_DIM + 1, gru)
        self.hand_gru = nn.GRU(gru, gru, batch_first=True)
        self.hand_out = nn.Sequential(*_block(gru + 1 + latent + 1, embed))
        self.player_gru = nn.GRU(embed, state, batch_first=True)
        self.start = nn.Parameter(torch.zeros(state))

    def hand_vectors(self, tokens: Tensor, lengths: Tensor, shown: Tensor, phi: Tensor, result: Tensor) -> Tensor:
        """`tokens` `(N, L, TOKEN_DIM + 1)`, `shown` `(N,)`, `phi` `(N, latent)` of the cards (read
        where shown), `result` `(N,)` in big blinds -> `(N, embed)`."""
        out, _ = self.hand_gru(torch.relu(self.token_in(tokens)))
        summary = out[torch.arange(len(lengths), device=out.device), lengths - 1]
        scaled = torch.sign(result) * torch.log1p(result.abs()) / math.log1p(100.0)
        return self.hand_out(torch.cat([summary, shown.unsqueeze(1), phi * shown.unsqueeze(1),
                                        scaled.unsqueeze(1)], dim=1))


class BeliefModel(nn.Module):
    """The two networks of a belief file, frozen. The equity encoder they read is the model's
    own (`PokerActorCritic.equity_encoder`), passed in at every call, never stored twice."""

    def __init__(self, latent: int, hidden: int, gru: int, rank: int, state: int, embed: int) -> None:
        super().__init__()
        if latent != BELIEF_LATENT:
            raise ValueError(f"feature version 5 holds a range of {BELIEF_LATENT} numbers, the encoder has {latent}")
        self.config = {"hidden": hidden, "gru": gru, "rank": rank, "state": state, "embed": embed}
        self.belief = BeliefNet(latent, hidden, gru, rank, state)
        self.memory = PlayerMemory(latent, gru, embed, state)
        self.requires_grad_(False)

    def train(self, mode: bool = True) -> BeliefModel:
        return super().train(False)


def file_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_belief_file(path: str | Path) -> dict[str, Any]:
    """A belief file (`studies/belief/belief_history.py --export`): `config` (`CONFIG_KEYS`),
    `state` (`belief.*`, `memory.*`), the equity file it was trained on, and its fingerprint."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if "config" not in saved or "state" not in saved:
        raise ValueError(f"{path} is not a belief network file (no config/state)")
    missing = [key for key in CONFIG_KEYS if key not in saved["config"]]
    if missing:
        raise ValueError(f"{path}: its config lacks {missing}")
    return {**saved, "fingerprint": file_fingerprint(path)}


def belief_spec(saved: dict[str, Any]) -> dict[str, Any]:
    """What a model records of its belief network: the shape and the file's fingerprint."""
    return {**{key: int(saved["config"][key]) for key in CONFIG_KEYS}, "fingerprint": saved["fingerprint"]}


class BeliefReader:
    """Every live opponent's belief vector at a decision: `reader(observation, big_blind, oracle)`
    -> `{seat: BELIEF_FEATURES numbers}`.

    `oracle`, in training only, is `(share, holes, mode, near)`: every opponent's true cards (seat ->
    two card indices), how much of them to give and how (`belief_inputs.ORACLE_MODES`): `mix`, every
    vector `(1 - share) * range + share * true hand` (the true hand being `phi` of its cards and a
    width of 0); `swap`, with probability `share` an opponent is read as its true hand, else as its
    range; `near`, with probability `share` its range is kept to the combos nearest the true hand in
    `phi` and renormalised (a range that knows roughly what it holds), as many as `near` says for
    this street (`belief_inputs.near_combos`). The draw is
    made once per opponent and hand (python's `random`, reseeded per forked copy). Off by default;
    the policy then reads only what it may know.

    The memory states are brought up to date through the observation's `MemoryView` (none: every
    player reads as never seen). The combos' features depend only on the board and are remembered
    for `BOARD_CACHE_SIZE` boards."""

    def __init__(self, model: BeliefModel, encoder: EquityEncoder, key: str, device: str | torch.device = "cpu") -> None:
        self.model, self.encoder, self.key, self.device = model, encoder, key, torch.device(device)
        self._boards: dict[tuple[int, ...], tuple[Tensor, Tensor]] = {}
        self._combo_planes = torch.zeros(NUM_COMBOS, 52, device=self.device)
        cards = torch.tensor(COMBO_CARDS, device=self.device)
        self._combo_planes[torch.arange(NUM_COMBOS), cards[:, 0]] = 1.0
        self._combo_planes[torch.arange(NUM_COMBOS), cards[:, 1]] = 1.0
        self._combo_cards = cards
        # Which combos hold each card, `(53, 1326)` (row 52: no card): the combos an observer rules out.
        self._holds = torch.zeros(53, NUM_COMBOS, dtype=torch.bool, device=self.device)
        self._holds[cards[:, 0], torch.arange(NUM_COMBOS)] = True
        self._holds[cards[:, 1], torch.arange(NUM_COMBOS)] = True
        # The GRU's outputs over the hand being played, per target: a target's run over the tokens
        # depends on nothing but the hand and which tokens are its own, whoever asks, so each
        # decision runs only the tokens that came since (`_summaries`). Keyed by the hand's first
        # record, held so its identity cannot be reused.
        self._hand: tuple[Any, dict[int, Tensor]] | None = None
        # The oracle's draws in the hand being played (`swap`, `near`): seat -> told or not.
        self._told: tuple[Any, dict[int, bool]] | None = None

    def _combos(self, board: tuple[int, ...]) -> tuple[Tensor, Tensor]:
        found = self._boards.get(board)
        if found is None:
            if len(self._boards) >= BOARD_CACHE_SIZE:
                self._boards.pop(next(iter(self._boards)))
            planes = card_planes(torch.tensor([list(board) or [-1]], device=self.device))
            phi = self.encoder(self._combo_planes, planes.expand(NUM_COMBOS, -1))
            # transposed and scaled once, for `queries @ scores`
            scores = (self.model.belief.combo_features(phi).T / math.sqrt(self.model.belief.rank)).contiguous()
            found = (phi, scores)
            self._boards[board] = found
        return found

    def update_memory(self, hands: Sequence[MemoryHand], states: dict[str, Any]) -> None:
        """Read `hands` (in order) into `states` (player id -> the GRU's hidden state)."""
        memory = self.model.memory
        zero = torch.zeros_like(memory.start)
        with torch.inference_mode():
            for hand in hands:
                players = hand.players
                count = len(players)
                seats = torch.tensor([p.seat for p in players], device=self.device).unsqueeze(1)
                own = (torch.tensor(hand.actors, device=self.device).unsqueeze(0) == seats).float()
                tokens = torch.cat([torch.tensor(hand.tokens, device=self.device).unsqueeze(0).expand(count, -1, -1),
                                    own.unsqueeze(-1)], dim=-1)
                shown = torch.tensor([float(p.shown) for p in players], device=self.device)
                phi = torch.zeros(count, self.encoder.latent, device=self.device)
                rows = [row for row, p in enumerate(players) if p.shown]
                if rows:
                    board = card_planes(torch.tensor([list(hand.board)], device=self.device))
                    cards = torch.tensor([list(players[row].cards) for row in rows], device=self.device)
                    phi[rows] = self.encoder(card_planes(cards), board.expand(len(rows), -1))
                result = torch.tensor([p.result_bb for p in players], device=self.device)
                lengths = torch.full((count,), len(hand.tokens), device=self.device)
                vectors = memory.hand_vectors(tokens, lengths, shown, phi, result)
                hidden = torch.stack([states.get(p.player_id, zero) for p in players]).unsqueeze(0)
                _, hidden = memory.player_gru(vectors.unsqueeze(1), hidden)
                for row, player in enumerate(players):
                    states[player.player_id] = hidden[0, row]

    def _summaries(self, observation, targets: list[int], read_tokens: list[list[float]],
                   actors: list[int]) -> Tensor:
        """The GRU's output after the hand's last token, for each target `(n, gru)`."""
        first = observation.action_history[0]
        if self._hand is None or self._hand[0] is not first:
            self._hand = (first, {})
        outputs = self._hand[1]
        length = len(read_tokens)
        start = min(outputs[seat].shape[0] if seat in outputs else 0 for seat in targets)
        belief = self.model.belief
        if start < length:
            new = torch.tensor(read_tokens[start:], device=self.device)
            seats = torch.tensor(targets, device=self.device).unsqueeze(1)
            own = (torch.tensor(actors[start:], device=self.device).unsqueeze(0) == seats).float()
            tokens = torch.cat([new.unsqueeze(0).expand(len(targets), -1, -1), own.unsqueeze(-1)], dim=-1)
            hidden = torch.stack([
                outputs[seat][start - 1] if start > 0 else torch.zeros(belief.gru.hidden_size, device=self.device)
                for seat in targets
            ]).unsqueeze(0)
            out, _ = belief.gru(torch.relu(belief.token_in(tokens)), hidden)
            for row, seat in enumerate(targets):
                outputs[seat] = torch.cat([outputs[seat][:start], out[row]]) if start else out[row]
        return torch.stack([outputs[seat][length - 1] for seat in targets])

    def _drawn(self, observation, targets: list[int], share: float) -> list[bool]:
        """Whether each target is given its true cards in this hand (drawn once per hand)."""
        first = observation.action_history[0]
        if self._told is None or self._told[0] is not first:
            self._told = (first, {})
        told = self._told[1]
        for seat in targets:
            if seat not in told:
                told[seat] = random.random() < share
        return [told[seat] for seat in targets]

    def __call__(self, observation, big_blind: int,
                 oracle: tuple[float, dict[int, tuple[int, int]], str, tuple[int, ...]] | None = None,
                 ) -> dict[int, list[float]]:
        asked = decision_question(observation, big_blind)
        if asked is None:
            return {}
        targets = asked["targets"]
        board = asked["board"]
        memory = self.model.memory
        view = getattr(observation, "seat_memory", None)
        states = view.states(self.key, self.update_memory) if view is not None else {}
        with torch.inference_mode():
            summary = self._summaries(observation, targets, asked["tokens"], asked["actors"])
            rows = [states.get(seat) for seat in targets]
            board_planes = card_planes(torch.tensor([list(board) or [-1]], device=self.device))
            queries = self.model.belief.query_of(
                summary,
                torch.tensor(asked["static"], device=self.device),
                board_planes.expand(len(targets), -1),
                torch.stack([memory.start if row is None else row for row in rows]),
            )
            phi, scores = self._combos(board)
            logits = queries @ scores
            blocked = self._holds[list(asked["known"])].any(dim=0)
            logits = logits.masked_fill(blocked, MASK_FILL)
            log_p = torch.log_softmax(logits, dim=-1)
            p = log_p.exp()
            mean = p @ phi
            width = -(p * log_p).sum(dim=-1) / LOG_COMBOS
            if oracle is not None and oracle[0] > 0:
                share, holes, mode, near = oracle
                true = self.encoder(card_planes(torch.tensor([list(holes[seat]) for seat in targets],
                                                             device=self.device)),
                                    board_planes.expand(len(targets), -1))
                if mode == "mix":
                    mean = (1 - share) * mean + share * true
                    width = (1 - share) * width
                else:
                    for row, told in enumerate(self._drawn(observation, targets, share)):
                        if not told:
                            continue
                        if mode == "swap":
                            mean[row], width[row] = true[row], 0.0
                            continue
                        # near: the combos closest to the true hand, weighed by the range
                        distance = ((phi - true[row]) ** 2).sum(dim=1).masked_fill(blocked, math.inf)
                        count = max(1, min(near_combos(near, len(board)), int((~blocked).sum())))
                        nearest = torch.topk(distance, count, largest=False).indices
                        kept = torch.log_softmax(logits[row, nearest], dim=-1)
                        mean[row] = kept.exp() @ phi[nearest]
                        width[row] = -(kept.exp() * kept).sum() / LOG_COMBOS
            vectors = torch.cat([mean, width.unsqueeze(1)], dim=1).tolist()
        return dict(zip(targets, vectors, strict=True))
