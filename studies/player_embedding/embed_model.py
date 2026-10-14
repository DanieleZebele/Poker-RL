"""The player embedder and what trains it.

    hands 0..t-1 of one player  ->  HandEncoder -> GRU  ->  head  ->  e_t  (before hand t)

- `HandEncoder` reads one hand: its decisions (`DECISION_DIM` numbers each, through an MLP and a
  masked mean and max over the decisions) and the showdown it was seen to take (`SHOW_DIM`), and
  gives `hidden` numbers.
- `PlayerEmbedder` runs a **sequence model** over the hands (`SEQUENCE_MODELS`: a GRU, one step per
  hand, from a learned initial state; the plain running mean of the hands' encodings, with no order
  and no memory to learn; or a causal transformer, each hand attending to the ones before it), and
  a head turns the state *before* each hand into an embedding of `embedding` numbers, next to
  three counters (hands seen, showdowns seen, mucks seen). The three are there to be compared:
  if the mean does as well as the GRU, the GRU is not earning its keep. **The embedding for hand t uses hands
  0..t-1 only** -- the hand in progress is read by the policy from the actions in front of it --
  which `embeddings` guarantees by shifting the states by one and `test_..._no_leak` pins.
- `Predictor` is the teacher that makes the embedding mean something: given the embedding, the
  public spot of a decision *and the player's own cards* (private, known in the training data), it
  predicts the action; and, given the embedding, the spot and the action, the strength of the
  cards. Having the cards as input is the point: the embedding is then rewarded only for what the
  cards do not explain, which is the player's style. At use only the embedding is given to a
  policy; the predictor is thrown away.

Small on purpose (tens of thousands of parameters): a step per hand per player is all it costs at a
table.
"""

from __future__ import annotations

import torch
from embed_records import ACTION_DIM, DECISION_DIM, NUM_ACTIONS, SHOW_DIM, SHOW_MUCKED, SHOW_SHOWED, STATE_DIM
from torch import Tensor, nn


def show_features(show: Tensor, won: Tensor, shown: Tensor) -> Tensor:
    """`(B, T)` showdown arrays -> `(B, T, SHOW_DIM)`: one-hot of the state, won, strength shown."""
    onehot = torch.nn.functional.one_hot(show.long(), 3).float()
    return torch.cat([onehot, won.float()[..., None], shown.float()[..., None]], dim=-1)


class HandEncoder(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.decision = nn.Sequential(
            nn.Linear(DECISION_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.hand = nn.Sequential(nn.Linear(2 * hidden + SHOW_DIM, hidden), nn.ReLU())

    def forward(self, x: Tensor, mask: Tensor, show: Tensor, won: Tensor, shown: Tensor) -> Tensor:
        """`x` (B, T, A, DECISION_DIM), `mask` (B, T, A) bool -> (B, T, hidden)."""
        encoded = self.decision(x.float())
        keep = mask[..., None]
        count = keep.sum(dim=2).clamp(min=1)
        mean = (encoded * keep).sum(dim=2) / count
        maximum = encoded.masked_fill(~keep, -1e4).max(dim=2).values
        empty = ~mask.any(dim=2, keepdim=True)  # a hand with no decision: nothing to pool
        maximum = maximum.masked_fill(empty, 0.0)
        return self.hand(torch.cat([mean, maximum, show_features(show, won, shown)], dim=-1))


SEQUENCE_MODELS = ("gru", "mean", "transformer")
MAX_HANDS = 512  # the longest sequence the transformer's positions cover


class PlayerEmbedder(nn.Module):
    def __init__(self, hidden: int = 64, embedding: int = 16, sequence_model: str = "gru") -> None:
        super().__init__()
        if sequence_model not in SEQUENCE_MODELS:
            raise ValueError(f"sequence_model is one of {SEQUENCE_MODELS}, not {sequence_model!r}")
        self.hidden, self.embedding, self.sequence_model = hidden, embedding, sequence_model
        self.encoder = HandEncoder(hidden)
        if sequence_model == "gru":
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
        elif sequence_model == "transformer":
            self.positions = nn.Embedding(MAX_HANDS, hidden)
            layer = nn.TransformerEncoderLayer(
                hidden, nhead=4, dim_feedforward=2 * hidden, dropout=0.0, batch_first=True
            )
            self.attention = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.initial = nn.Parameter(torch.zeros(hidden))
        self.head = nn.Sequential(nn.Linear(hidden + 3, 64), nn.ReLU(), nn.Linear(64, embedding))

    def embeddings(self, x: Tensor, mask: Tensor, show: Tensor, won: Tensor, shown: Tensor) -> Tensor:
        """`(B, T, embedding)`: entry t is the embedding *before* hand t."""
        batch, hands = show.shape
        encoded = self.encoder(x, mask, show, won, shown)
        states = self._states(encoded)
        before = torch.cat([self.initial.expand(batch, 1, -1), states[:, :-1]], dim=1)
        seen = torch.arange(hands, device=x.device, dtype=torch.float32).expand(batch, hands)
        showed = (show == SHOW_SHOWED).float().cumsum(dim=1) - (show == SHOW_SHOWED).float()
        mucked = (show == SHOW_MUCKED).float().cumsum(dim=1) - (show == SHOW_MUCKED).float()
        counters = torch.stack([torch.log1p(seen), torch.log1p(showed), torch.log1p(mucked)], dim=-1)
        return self.head(torch.cat([before, counters], dim=-1))


    def _states(self, encoded: Tensor) -> Tensor:
        """`(B, T, hidden)`: entry t summarises hands 0..t, whichever the sequence model."""
        batch, hands, _ = encoded.shape
        if self.sequence_model == "gru":
            states, _last = self.gru(encoded, self.initial.expand(1, batch, -1).contiguous())
            return states
        if self.sequence_model == "mean":
            count = torch.arange(1, hands + 1, device=encoded.device, dtype=encoded.dtype)
            return encoded.cumsum(dim=1) / count[None, :, None]
        if hands > MAX_HANDS:
            raise ValueError(f"the transformer covers {MAX_HANDS} hands, got {hands}")
        positioned = encoded + self.positions(torch.arange(hands, device=encoded.device))[None]
        causal = torch.triu(torch.ones(hands, hands, dtype=torch.bool, device=encoded.device), diagonal=1)
        return self.attention(positioned, mask=causal)


class Predictor(nn.Module):
    def __init__(self, embedding: int = 16, hidden: int = 128) -> None:
        super().__init__()
        self.action = nn.Sequential(
            nn.Linear(embedding + STATE_DIM + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, NUM_ACTIONS),
        )
        self.strength = nn.Sequential(
            nn.Linear(embedding + STATE_DIM + ACTION_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def action_logits(self, embedding: Tensor, state: Tensor, strength: Tensor) -> Tensor:
        return self.action(torch.cat([embedding, state.float(), strength.float()], dim=-1))

    def strength_guess(self, embedding: Tensor, state: Tensor, action: Tensor) -> Tensor:
        return torch.sigmoid(self.strength(torch.cat([embedding, state.float(), action.float()], dim=-1))).squeeze(-1)
