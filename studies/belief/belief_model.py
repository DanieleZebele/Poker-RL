"""The belief network: the range of hole cards a player holds, from what everyone saw.

    context  h = MLP([GRU(tokens, each marked "the target's own" or not) | target's statics | board])
    combos   psi(c) = MLP([phi(c, board) | class embedding of c])   for all 1326 combos
    range    logit(c) = <A h, psi(c)>,  -inf on the combos the observer can rule out

`phi` is the frozen per-player encoder of the equity network (`rl/equity_net.py`), the 32 numbers
the policy already reads for its own cards. A combo is scored in that space, so a combo the encoder
cannot tell from another (two suits equivalent on this board) gets the same score -- which is exact,
by suit symmetry -- and a range is a distribution over points of the space the policy reads: the
vector the policy is to be given is the range's mean in it (`range_embedding`), and the same vector
for the player's true cards is `phi` of them (the oracle).

The class embedding (169 starting-hand classes) lets the scores tell apart preflop hands the
encoder may have placed close together.
"""

from __future__ import annotations

import math

import torch
from belief_records import BOARD_AT, COMBO_CARDS, HUD_DIM, NUM_COMBOS, POSITIONS, STREETS, TOKEN_DIM
from torch import Tensor, nn

from pokerlab.rl.equity_net import (
    EquityEncoder,
    encoder_config,
    load_encoder_weights,
    load_equity_checkpoint,
)

STATIC_DIM = len(POSITIONS) + 1 + 1 + HUD_DIM + len(STREETS) + 1
MASK_FILL = -1e9


def combo_classes() -> Tensor:
    """The starting-hand class (0..168) of every combo: pairs, suited, offsuit."""
    classes = []
    for a, b in COMBO_CARDS:
        ra, rb, sa, sb = a % 13, b % 13, a // 13, b // 13
        hi, lo = max(ra, rb), min(ra, rb)
        classes.append(hi * 13 + lo if (sa == sb or hi == lo) else lo * 13 + hi)
    return torch.tensor(classes)


def combo_planes() -> Tensor:
    """Every combo as 52 numbers in {0, 1}: the `hole` input of the encoder."""
    planes = torch.zeros(NUM_COMBOS, 52)
    cards = torch.tensor(COMBO_CARDS)
    planes[torch.arange(NUM_COMBOS), cards[:, 0]] = 1.0
    planes[torch.arange(NUM_COMBOS), cards[:, 1]] = 1.0
    return planes


def board_planes(board: Tensor) -> Tensor:
    """`(B, 5)` card indices, -1 for none, as `(B, 52)` in {0, 1}."""
    planes = torch.zeros(board.shape[0], 53, device=board.device)
    planes.scatter_(1, torch.where(board < 0, 52, board).long(), 1.0)
    return planes[:, :52]


def _block(inputs: int, width: int) -> list[nn.Module]:
    return [nn.Linear(inputs, width), nn.LayerNorm(width), nn.ReLU()]


HAND_ENCODERS = ("gru", "transformer")
MEMORIES = ("gru", "attention")
HEADS = ("dot", "mixture", "mlp")
MAX_POSITIONS = 256  # tokens of a hand, hands of a player in a session (both under 200 in the data)


def _transformer(width: int, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=2 * width, dropout=0.0,
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class TokenTransformer(nn.Module):
    """A hand's tokens read by attention instead of a GRU: the summary is the output at the last
    token, which attends to every token of the hand (padding masked)."""

    def __init__(self, width: int, layers: int = 2) -> None:
        super().__init__()
        self.position = nn.Embedding(MAX_POSITIONS, width)
        self.body = _transformer(width, layers)
        self.norm = nn.LayerNorm(width)

    def forward(self, embedded: Tensor, lengths: Tensor) -> Tensor:
        steps = torch.arange(embedded.shape[1], device=embedded.device)
        x = embedded + self.position(steps.clamp_max(MAX_POSITIONS - 1)).unsqueeze(0)
        padding = steps.unsqueeze(0) >= lengths.unsqueeze(1)
        out = self.norm(self.body(x, src_key_padding_mask=padding))
        return out[torch.arange(len(lengths), device=out.device), lengths - 1]


def summarise(module: nn.Module, embedded: Tensor, lengths: Tensor) -> Tensor:
    """The summary of each sequence: a GRU's output at its last token, or a `TokenTransformer`'s."""
    if isinstance(module, nn.GRU):
        out, _ = module(embedded)
        return out[torch.arange(len(lengths), device=out.device), lengths - 1]
    return module(embedded, lengths)


class BeliefNet(nn.Module):
    def __init__(self, encoder: EquityEncoder, hidden: int = 256, gru: int = 128, rank: int = 64,
                 use_tokens: bool = True, use_hud: bool = True, extra: int = 0, condition: bool = False,
                 hand_encoder: str = "gru", head: str = "dot", mixture: int = 4) -> None:
        super().__init__()
        if hand_encoder not in HAND_ENCODERS or head not in HEADS:
            raise ValueError(f"hand encoder {hand_encoder!r}, head {head!r}")
        self.encoder = encoder  # frozen
        self.use_tokens, self.use_hud, self.condition = use_tokens, use_hud, condition
        self.head, self.mixture = head, mixture if head == "mixture" else 1
        self.config = {"hidden": hidden, "gru": gru, "rank": rank, "use_tokens": use_tokens, "use_hud": use_hud,
                       "extra": extra, "condition": condition, "hand_encoder": hand_encoder, "head": head,
                       "mixture": mixture}
        self.token_in = nn.Linear(TOKEN_DIM + 1, gru)
        self.token_width = gru
        # the attribute names of the GRU version are kept, so the saved networks still load
        if hand_encoder == "gru":
            self.gru = nn.GRU(gru, gru, batch_first=True)
        else:
            self.hand_tf = TokenTransformer(gru)
        # `extra`: numbers about the target from elsewhere (the memory of its past hands)
        # `condition`: one more static number, 1 when the target's cards are known because it showed
        # them at a showdown (real hands), 0 for the range with no condition (what play asks)
        self.context = nn.Sequential(*_block(gru + STATIC_DIM + int(condition) + 52 + extra, hidden),
                                     *_block(hidden, hidden))
        # head "dot": logit(c) = <q(h), psi(c)>; "mixture": the log of a sum of `mixture` such
        # exponentials (a range made of several parts, e.g. strong hands and bluffs); "mlp": the dot
        # product plus a small MLP on psi(c) + q'(h) (any shape, at the cost of B x 1326 x rank)
        self.query = nn.Linear(hidden, rank * self.mixture)
        self.class_embedding = nn.Embedding(169, 16)
        latent = encoder.latent
        self.psi = nn.Sequential(*_block(latent + 16, hidden), nn.Linear(hidden, rank * self.mixture))
        if head == "mlp":
            self.mlp_query = nn.Linear(hidden, rank)
            self.mlp = nn.Sequential(nn.Linear(rank, rank), nn.ReLU(), nn.Linear(rank, 1))
        self.rank = rank
        self.register_buffer("classes", combo_classes(), persistent=False)
        self.register_buffer("planes", combo_planes(), persistent=False)
        self.register_buffer("combo_cards", torch.tensor(COMBO_CARDS), persistent=False)

    def combo_phi(self, boards: Tensor) -> Tensor:
        """`phi(c, board)` of every combo on each of `boards` `(U, 52)`: `(U, 1326, latent)`.
        A combo that shares a card with the board is encoded anyway (and masked later)."""
        count = boards.shape[0]
        hole = self.planes.unsqueeze(0).expand(count, -1, -1).reshape(-1, 52)
        board = boards.unsqueeze(1).expand(-1, NUM_COMBOS, -1).reshape(-1, 52)
        return self.encoder(hole, board).view(count, NUM_COMBOS, -1)

    def combo_features(self, phi: Tensor) -> Tensor:
        classes = self.class_embedding(self.classes).unsqueeze(0).expand(phi.shape[0], -1, -1)
        return self.psi(torch.cat([phi, classes], dim=-1))

    def forward(self, tokens: Tensor, lengths: Tensor, static: Tensor, board: Tensor,
                known: Tensor, extra: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Logits over the 1326 combos `(B, 1326)`, the combos' `phi` per unique board and the
        index of each sample's board in it.

        `tokens` `(B, L, TOKEN_DIM + 1)` (the last column: the target's own), `lengths` `(B,)`,
        `static` `(B, STATIC_DIM + 1)` (the last column: the condition, read only if `condition`),
        `board` `(B, 5)` the visible cards (-1 none), `known`
        `(B, 7)` the cards the observer sees (board and its own, -1 none)."""
        batch = tokens.shape[0]
        if self.use_tokens:
            reader = self.gru if hasattr(self, "gru") else self.hand_tf
            summary = summarise(reader, torch.relu(self.token_in(tokens)), lengths)
        else:
            summary = torch.zeros(batch, self.token_width, device=tokens.device)
        if not self.use_hud:
            static = static.clone()
            start = len(POSITIONS) + 2
            static[:, start : start + HUD_DIM] = 0.0
        if not self.condition:
            static = static[:, :STATIC_DIM]
        planes = board_planes(board)
        parts = [summary, static, planes] + ([extra] if extra is not None else [])
        h = self.context(torch.cat(parts, dim=-1))
        query = self.query(h)
        unique, inverse = torch.unique(planes, dim=0, return_inverse=True)
        phi = self.combo_phi(unique)
        features = self.combo_features(phi)[inverse]  # (B, 1326, rank * mixture)
        if self.mixture > 1:
            parts = torch.einsum("bmr,bcmr->bcm", query.view(batch, self.mixture, self.rank),
                                 features.view(batch, NUM_COMBOS, self.mixture, self.rank))
            logits = torch.logsumexp(parts / math.sqrt(self.rank), dim=-1)
        else:
            logits = torch.einsum("br,bcr->bc", query, features) / math.sqrt(self.rank)
        if self.head == "mlp":
            logits = logits + self.mlp(torch.relu(features + self.mlp_query(h).unsqueeze(1))).squeeze(-1)
        blocked = torch.zeros(batch, 53, dtype=torch.bool, device=tokens.device)
        blocked.scatter_(1, torch.where(known < 0, 52, known).long(), True)
        blocked = blocked[:, :52]
        mask = blocked[:, self.combo_cards[:, 0]] | blocked[:, self.combo_cards[:, 1]]
        return logits.masked_fill(mask, MASK_FILL), phi, inverse


class PlayerMemory(nn.Module):
    """What the table has seen a player do, hand after hand: a state per player and hand.

    Each past hand of the player becomes a vector: a GRU over all of that hand's tokens (the
    player's own marked), the player's cards if it showed them (`phi` on the final board: the
    only private cards an observer ever sees), and its result. A second GRU runs over those
    vectors in the order the hands were played; the state *before* hand k is what is known when
    hand k starts (a learned state before the first: a player never seen)."""

    def __init__(self, encoder: EquityEncoder, gru: int = 128, embed: int = 64, state: int = 128,
                 hand_encoder: str = "gru", memory: str = "gru") -> None:
        super().__init__()
        if hand_encoder not in HAND_ENCODERS or memory not in MEMORIES:
            raise ValueError(f"hand encoder {hand_encoder!r}, memory {memory!r}")
        self.encoder = encoder  # frozen, shared with the belief network
        self.token_in = nn.Linear(TOKEN_DIM + 1, gru)
        if hand_encoder == "gru":
            self.hand_gru = nn.GRU(gru, gru, batch_first=True)
        else:
            self.hand_tf = TokenTransformer(gru)
        self.hand_out = nn.Sequential(*_block(gru + 1 + encoder.latent + 1, embed))
        # memory "attention": a causal transformer over the player's hands, so the state before
        # hand k can look back at any one of them (a showdown, a hand like this one) directly
        if memory == "gru":
            self.player_gru = nn.GRU(embed, state, batch_first=True)
        else:
            self.player_in = nn.Linear(embed, state)
            self.player_position = nn.Embedding(MAX_POSITIONS, state)
            self.player_tf = _transformer(state, 2)
            self.player_norm = nn.LayerNorm(state)
        self.start = nn.Parameter(torch.zeros(state))
        self.state = state

    def hand_vectors(self, tokens: Tensor, lengths: Tensor, shown: Tensor, cards: Tensor, board: Tensor,
                     result: Tensor) -> Tensor:
        """One vector per (hand, player): `tokens` `(N, L, TOKEN_DIM + 1)`, `shown` `(N,)`,
        `cards` `(N, 2)` its hole cards (read only where shown), `board` `(N, 5)` the final board,
        `result` `(N,)` in big blinds."""
        reader = self.hand_gru if hasattr(self, "hand_gru") else self.hand_tf
        summary = summarise(reader, torch.relu(self.token_in(tokens)), lengths)
        hole = torch.zeros(len(cards), 52, device=cards.device)
        hole.scatter_(1, cards.clamp_min(0).long(), 1.0)
        phi = self.encoder(hole, board_planes(board)) * shown.unsqueeze(1)
        scaled = torch.sign(result) * torch.log1p(result.abs()) / math.log1p(100.0)
        return self.hand_out(torch.cat([summary, shown.unsqueeze(1), phi, scaled.unsqueeze(1)], dim=1))

    def states_before(self, vectors: Tensor) -> Tensor:
        """`(P, K, embed)` the vectors of P players' K hands in order -> `(P, K, state)` the state
        before each hand."""
        if hasattr(self, "player_gru"):
            out, _ = self.player_gru(vectors)
            first = self.start.expand(out.shape[0], 1, -1)
            return torch.cat([first, out[:, :-1]], dim=1)
        # position k reads the start and hands 0..k-1 (padding comes after a player's last hand,
        # so the causal mask keeps it out of every state that matters)
        count = vectors.shape[1]
        sequence = torch.cat([self.start.expand(vectors.shape[0], 1, -1), self.player_in(vectors[:, :-1])], dim=1)
        steps = torch.arange(count, device=vectors.device)
        sequence = sequence + self.player_position(steps.clamp_max(MAX_POSITIONS - 1)).unsqueeze(0)
        causal = torch.triu(torch.ones(count, count, dtype=torch.bool, device=vectors.device), diagonal=1)
        return self.player_norm(self.player_tf(sequence, mask=causal))


def range_embedding(logits: Tensor, phi: Tensor, inverse: Tensor) -> tuple[Tensor, Tensor]:
    """The range's mean and standard deviation in the encoder's space, `(B, latent)` each."""
    probabilities = torch.softmax(logits, dim=-1)
    mean = torch.einsum("bc,bcl->bl", probabilities, phi[inverse])
    second = torch.einsum("bc,bcl->bl", probabilities, phi[inverse] ** 2)
    return mean, (second - mean**2).clamp_min(0).sqrt()


def build_encoder(equity_model: str) -> EquityEncoder:
    saved = load_equity_checkpoint(equity_model)
    encoder = EquityEncoder(**encoder_config(saved))
    load_encoder_weights(encoder, saved)
    return encoder


def visible_board(board: Tensor, street: Tensor) -> Tensor:
    """The board cards dealt by each sample's street, -1 for the rest."""
    counts = torch.tensor([BOARD_AT[s] for s in STREETS], device=board.device)[street.long()]
    keep = torch.arange(5, device=board.device).unsqueeze(0) < counts.unsqueeze(1)
    return torch.where(keep, board, torch.full_like(board, -1))



def load_belief(path: str, device: str = "cpu") -> BeliefNet:
    """A `BeliefNet` saved by `belief_train.py`, on its equity encoder."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    net = BeliefNet(build_encoder(saved["equity_model"]), **saved["config"])
    missing, unexpected = net.load_state_dict(saved["state"], strict=False)
    if unexpected or any(not key.startswith("encoder.") for key in missing):
        raise ValueError(f"{path}: missing {missing}, unexpected {unexpected}")
    return net.to(device).eval()
