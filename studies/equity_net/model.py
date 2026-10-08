"""The equity network: a permutation-invariant network over the players of a hand.

    the four suits are put in a canonical order (always, see `canonical_suits`)
    each player: [hole cards (52) | visible board (52)]
        -> encoder phi, shared by every player          (one vector per player)

and then, depending on `arch`:

    sets:   sum over the players -> rho -> [own vector | the table] -> head
    attn:   self-attention layers across the players (padded seats masked) -> head

    softmax over the players of the hand                 (shares that sum to 1)

Swapping two players swaps their outputs and changes nothing else: nothing in it depends
on the order they are listed in. Players are padded to `MAX_PLAYERS` with a mask, which the
sum, the attention and the softmax all respect.

The equity of a deal does not change when the four suits are renamed, but the input does. So
the suits are always put in a canonical order first (by what they hold), and the network does
not have to learn 24 copies of everything. One order for the whole deal, not one per player: a
flush of one player blocks another's.

Options (the defaults are the original network):

- `arch="attn"`: the players look at each other directly. With a sum over the players, two
  nearly equal hands (two full houses of the same trips) are hard to tell apart.
- `blocks`: residual blocks in phi instead of its two plain layers; recognising pairs,
  flushes, straights and draws is a nonlinear computation that wants depth.
"""

from __future__ import annotations

from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional

MAX_PLAYERS = 9
CARDS = 52
MASK_FILL = -1e9


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(pairwise(sizes)):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class _Block(nn.Module):
    """Pre-norm residual block: x + down(gelu(up(norm(x))))."""

    def __init__(self, width: int, inner: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, inner)
        self.down = nn.Linear(inner, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down(functional.gelu(self.up(self.norm(x))))


def canonical_suits(holes: torch.Tensor, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The same deal with its four suits in a canonical order: the suit with the most board
    cards first, then the one held by most players' cards, then by which ranks are on the
    board and which are held (ties are left as they are: an arbitrary but fixed order)."""
    batch, players, _ = holes.shape
    by_suit = holes.view(batch, players, 4, 13)
    on_board = board.view(batch, 4, 13)
    weights = 2 ** torch.arange(13, device=holes.device)  # a rank mask is below 8192
    board_i, hole_i = on_board.long(), by_suit.long()
    held = hole_i.amax(dim=1)  # (batch, 4, 13): some player holds this rank in this suit
    key = (
        board_i.sum(-1) * 10**12
        + hole_i.sum(dim=(1, 3)) * 10**9
        + (board_i * weights).sum(-1) * 10**5
        + (held * weights).sum(-1)
    )
    order = torch.sort(key, dim=-1, descending=True, stable=True).indices  # (batch, 4)
    by_suit = by_suit.gather(2, order[:, None, :, None].expand(-1, players, -1, 13))
    on_board = on_board.gather(1, order[:, :, None].expand(-1, -1, 13))
    return by_suit.reshape(batch, players, CARDS), on_board.reshape(batch, CARDS)


class EquityNet(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        latent: int = 128,
        arch: str = "sets",
        blocks: int = 0,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if arch not in ("sets", "attn"):
            raise ValueError(f"unknown arch {arch!r}")
        self.hidden, self.latent, self.arch = hidden, latent, arch
        self.blocks, self.layers, self.heads = blocks, layers, heads
        if blocks:
            self.phi = nn.Sequential(nn.Linear(2 * CARDS, latent), *[_Block(latent, hidden) for _ in range(blocks)])
        else:
            self.phi = _mlp([2 * CARDS, hidden, latent])
        if arch == "sets":
            self.rho = _mlp([latent, hidden, latent])
            self.head = _mlp([2 * latent, hidden, 1])
        else:
            layer = nn.TransformerEncoderLayer(
                latent, heads, dim_feedforward=2 * latent, dropout=0.0, activation="gelu",
                batch_first=True, norm_first=True,
            )
            # nested tensors would drop the padded seats from the output's shape
            self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(latent)
            self.head = _mlp([latent, hidden, 1])

    def config(self) -> dict:
        """What `EquityNet(**config)` needs to rebuild this network."""
        return {"hidden": self.hidden, "latent": self.latent, "arch": self.arch, "blocks": self.blocks,
                "layers": self.layers, "heads": self.heads}

    @classmethod
    def from_checkpoint(cls, saved: dict) -> EquityNet:
        # the first checkpoint was written before the options existed: it has only these two
        if "config" in saved:
            return cls(**saved["config"])
        return cls(saved["hidden"], saved["latent"])

    def forward(self, holes: torch.Tensor, board: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`holes` (B, 9, 52) and `mask` (B, 9) bool, `board` (B, 52): the shares (B, 9),
        zero for padded seats, summing to 1 over the real ones."""
        holes, board = canonical_suits(holes, board)
        players = holes.shape[1]
        per_player = torch.cat([holes, board.unsqueeze(1).expand(-1, players, -1)], dim=-1)
        if self.arch == "sets":
            encoded = torch.relu(self.phi(per_player))
            table = (encoded * mask.unsqueeze(-1)).sum(dim=1)
            table = torch.relu(self.rho(table))
            joint = torch.cat([encoded, table.unsqueeze(1).expand(-1, players, -1)], dim=-1)
        else:
            joint = self.norm(self.encoder(self.phi(per_player), src_key_padding_mask=~mask))
        scores = self.head(joint).squeeze(-1).masked_fill(~mask, MASK_FILL)
        return torch.softmax(scores, dim=-1)

    def log_shares(self, holes: torch.Tensor, board: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.log(self.forward(holes, board, mask).clamp_min(1e-9))
