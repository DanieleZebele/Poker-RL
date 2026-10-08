"""The equity network of `studies/equity_net`, as the training code uses it.

It estimates, for every player still in a hand, the share of the pot they will win, given
everyone's cards and the board so far. Two things in training read it:

- **The critic** gets its output (`EquityNet`) in place of the card planes. It needs the
  hole cards of every player, which no player may see, so it exists only while training:
  a published model never calls it.
- **The policy** gets only the per-player encoder (`EquityEncoder`, the network's `phi`),
  applied to the model's own cards and the board, in place of the card planes. That is
  legal information, and the encoder's weights travel inside the policy's checkpoint.

The classes mirror `studies/equity_net/model.py` layer for layer, so a checkpoint written
by that study loads here unchanged. `src/` cannot import from `studies/`, hence the copy.
Cards are indexed 0..51 as `suit * 13 + rank - 2` (clubs, diamonds, hearts, spades), the
index `rl/features.py` uses for its card planes.

Like `rl/policy.py`, this module imports torch: it is only ever imported from there, from
`rl/train.py` and from `rl/ppo.py`, never by anything the core install loads.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
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

    def forward(self, x: Tensor) -> Tensor:
        return x + self.down(functional.gelu(self.up(self.norm(x))))


def _phi(hidden: int, latent: int, blocks: int) -> nn.Sequential:
    """The per-player encoder: [hole cards | board] (104 numbers) -> a vector of `latent`."""
    if blocks:
        return nn.Sequential(nn.Linear(2 * CARDS, latent), *[_Block(latent, hidden) for _ in range(blocks)])
    return _mlp([2 * CARDS, hidden, latent])


def canonical_suits(holes: Tensor, board: Tensor) -> tuple[Tensor, Tensor]:
    """The same deal with its four suits in a canonical order.

    The equity of a deal does not change when the suits are renamed, but the input does.
    The suit with the most board cards goes first, then the one held by most cards, then
    by which ranks are on the board and which are held; ties keep their order.
    `holes` is `(batch, players, 52)` and `board` `(batch, 52)`."""
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
    """Shares of the pot for every player of a hand: `forward(holes, board, mask)`.

    `holes` is `(B, 9, 52)`, `board` `(B, 52)` and `mask` `(B, 9)` bool (which seats are
    in the hand); the result is `(B, 9)`, zero for the seats not in the hand and summing
    to 1 over the others. Permuting the players permutes the result and nothing else."""

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
            raise ValueError(f"unknown equity network arch {arch!r}")
        self.hidden, self.latent, self.arch = hidden, latent, arch
        self.blocks, self.layers, self.heads = blocks, layers, heads
        self.phi = _phi(hidden, latent, blocks)
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

    def config(self) -> dict[str, Any]:
        return {"hidden": self.hidden, "latent": self.latent, "arch": self.arch, "blocks": self.blocks,
                "layers": self.layers, "heads": self.heads}

    def forward(self, holes: Tensor, board: Tensor, mask: Tensor) -> Tensor:
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


class EquityEncoder(nn.Module):
    """The equity network's per-player encoder, alone, as a feature extractor for one player.

    `forward(hole, board)` takes the model's own two cards and the visible board, each as
    52 numbers in {0, 1} (the `hole` and `board` planes of the observation), and returns
    `latent` numbers. It never sees another player's cards.

    Two choices worth knowing. The suits are put in canonical order from this player's own
    cards and the board alone (the network was trained with an order decided by the whole
    deal, which a player cannot know), and the result is normalised without learned
    parameters so its scale does not depend on how the equity network happened to be
    trained. The weights are frozen: the equity network is trained elsewhere, and this
    is only a fixed map of cards to features."""

    def __init__(self, hidden: int, latent: int, blocks: int) -> None:
        super().__init__()
        self.hidden, self.latent, self.blocks = hidden, latent, blocks
        self.phi = _phi(hidden, latent, blocks)
        self.requires_grad_(False)

    def config(self) -> dict[str, int]:
        return {"hidden": self.hidden, "latent": self.latent, "blocks": self.blocks}

    def train(self, mode: bool = True) -> EquityEncoder:
        # nothing in it behaves differently in training, and it must stay as loaded
        return super().train(False)

    def forward(self, hole: Tensor, board: Tensor) -> Tensor:
        own, shared = canonical_suits(hole.unsqueeze(1), board)
        with torch.no_grad():
            latent = self.phi(torch.cat([own.squeeze(1), shared], dim=-1))
        return functional.layer_norm(latent, (self.latent,))


def load_equity_checkpoint(path: str | Path) -> dict[str, Any]:
    """A checkpoint written by `studies/equity_net` (a `config` and a `state`)."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if "config" not in saved or "state" not in saved:
        raise ValueError(f"{path} is not an equity network checkpoint (no config/state)")
    return saved


def equity_net_from_checkpoint(saved: dict[str, Any]) -> EquityNet:
    """The full network, frozen and in inference mode, for the critic's input."""
    net = EquityNet(**saved["config"])
    net.load_state_dict(saved["state"])
    return net.requires_grad_(False).eval()


def encoder_config(saved: dict[str, Any]) -> dict[str, int]:
    """The shape of the per-player encoder of a loaded equity checkpoint."""
    config = saved["config"]
    return {"hidden": config["hidden"], "latent": config["latent"], "blocks": config["blocks"]}


def load_encoder_weights(encoder: EquityEncoder, saved: dict[str, Any]) -> None:
    """Copy the `phi` weights of an equity checkpoint into `encoder`."""
    phi = {key[len("phi."):]: value for key, value in saved["state"].items() if key.startswith("phi.")}
    encoder.phi.load_state_dict(phi)
