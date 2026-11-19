"""The belief network trained on every source at once: our models with styles, and real hands.

    PYTHONPATH=src:studies/belief python studies/belief/belief_unified.py \\
        --data checkpoints/studies/belief/data-v2 dataset_real_hands/unified \\
        --weights selfplay=0.45 handhq=0.35 acpc=0.2 --steps 60000 --out checkpoints/studies/belief/unified/run

The `memory` arm of `belief_history.py` (a GRU over the hand, a memory of the target's past hands,
no opponent statistics), with three things the real hands need:

- **A player is who it is by name** (`player`, `seq_pos` from `belief_data.pack_sessions`): the
  memory runs over each player's own hands in a session, whoever sat where.
- **A condition input**: a real online hand tells a player's cards only if they were shown at a
  showdown, which picks the hands that went that far. Those samples are read with the condition 1
  ("the cards are known because they were shown"); every sample whose cards were dealt known (our
  models, the ACPC) reads 0 (no condition: the range play asks for), and a share
  (`--condition-share`) of them is read again with 1 when the target did reach a showdown, so the
  network learns what the condition means where the truth is known. Play asks with 0.
- **Sources mixed by weight** (`--weights group=w`; a source is in the group its name starts
  with): sessions are drawn per step from each group in proportion. `--test-only` sources (Pluribus,
  the televised WSOP table: humans, every card known) are never trained on: they are the honest test
  of how the network reads real players.

The report: by source group, by street, by how the cards were known, by hands of the target seen.
Saves `belief.pt` as `belief_history.py` does (arm `memory`, `condition` in its config), which
`belief_export.py` turns into the production file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from belief_data import concatenate, load
from belief_model import HAND_ENCODERS, HEADS, MEMORIES, BeliefNet, PlayerMemory, build_encoder
from belief_records import STREETS
from belief_train import HELD_OUT_EVERY, Data, format_report
from torch.nn import functional

SEEN = (("visto 0", 0, 0), ("visto 1-9", 1, 9), ("visto 10-49", 10, 49), ("visto 50+", 50, 10**6))


def held_out(label: str) -> bool:
    return zlib.crc32(label.encode()) % HELD_OUT_EVERY == 0


class UnifiedData(Data):
    def __init__(self, arrays: dict[str, np.ndarray], device: str, groups: list[str], test_only: list[str]) -> None:
        super().__init__(arrays, device)
        session = arrays["session"]
        starts = np.flatnonzero(np.r_[True, session[1:] != session[:-1]])
        self.session_start = starts
        self.session_count = np.diff(np.r_[starts, len(session)])
        self.session_id = session[starts]
        self.sample_start = np.searchsorted(arrays["s_hand"], np.arange(len(session) + 1))
        self.offset_cpu = arrays["token_offset"]
        self.player_cpu = arrays["player"].astype(np.int64)
        self.seq_cpu = arrays["seq_pos"].astype(np.int64)
        self.player = torch.from_numpy(self.player_cpu).to(device)
        self.seq = torch.from_numpy(self.seq_cpu).to(device)
        self.shown = torch.from_numpy(arrays["shown"]).to(device)
        self.result_bb = torch.from_numpy(arrays["result_bb"].astype(np.float32)).to(device)
        self.mode = torch.from_numpy(arrays["s_mode"].astype(np.int64)).to(device)
        self.shows = torch.from_numpy(arrays["s_shows"]).to(device)
        sources = arrays["sources"].tolist()
        self.source_names = sources
        hand_source = arrays["source"].astype(np.int64)
        self.session_source = hand_source[starts]
        group_of_source = np.array([next((i for i, g in enumerate(groups) if name.startswith(g)), -1)
                                    for name in sources])
        self.session_group = group_of_source[self.session_source]
        test_source = np.array([any(name.startswith(t) for t in test_only) for name in sources])
        test = (self.session_id % HELD_OUT_EVERY == 0) | test_source[self.session_source]
        target_model = arrays["seat_model"][arrays["s_hand"], arrays["s_target"]]
        unseen_models = np.array([held_out(m) for m in arrays["models"].tolist()] + [False])
        self.unseen = torch.from_numpy(unseen_models[target_model]).to(device)
        self.sample_source = torch.from_numpy(hand_source[arrays["s_hand"]]).to(device)
        self.train_by_group = [np.flatnonzero(~test & (self.session_group == g)) for g in range(len(groups))]
        self.test_sessions = np.flatnonzero(test)

    def memory_inputs(self, sessions: np.ndarray) -> tuple[dict, int, int]:
        """Every (hand, player) of `sessions`, and where its vector goes in `(S * players, hands)`."""
        hands = np.concatenate([np.arange(self.session_start[s], self.session_start[s] + self.session_count[s])
                                for s in sessions])
        local = np.concatenate([np.full(self.session_count[s], i) for i, s in enumerate(sessions)])
        players = self.player_cpu[hands]
        width = int(players.max()) + 1
        hand_index, seat = np.nonzero(players >= 0)
        hand = hands[hand_index]
        # one row per player present, not per possible player id: an online session can name
        # hundreds of players, and a grid of every id was most of the GPU's memory
        keys = local[hand_index] * width + players[hand_index, seat]
        present, rows = np.unique(keys, return_inverse=True)
        row_of = np.full(len(sessions) * width, -1, dtype=np.int64)
        row_of[present] = np.arange(len(present))
        cols = self.seq_cpu[hand, seat]
        lengths = self.offset_cpu[hand + 1] - self.offset_cpu[hand]
        device = self.device
        hand_t, seat_t = torch.from_numpy(hand).to(device), torch.from_numpy(seat).to(device)
        length = int(lengths.max())
        steps = torch.arange(length, device=device)
        lengths_t = torch.from_numpy(lengths).to(device)
        where = (self.offset[hand_t].unsqueeze(1) + steps).clamp_max(len(self.tokens) - 1)
        valid = (steps.unsqueeze(0) < lengths_t.unsqueeze(1)).float()
        own = (self.token_actor[where] == seat_t.unsqueeze(1)).float() * valid
        tokens = torch.cat([self.tokens[where].float() * valid.unsqueeze(-1), own.unsqueeze(-1)], dim=-1)
        inputs = {
            "tokens": tokens, "lengths": lengths_t, "shown": self.shown[hand_t, seat_t].float(),
            "cards": self.holes[hand_t, seat_t], "board": self.board[hand_t],
            "result": self.result_bb[hand_t, seat_t],
            "rows": torch.from_numpy(rows).to(device), "cols": torch.from_numpy(cols).to(device),
            "row_of": torch.from_numpy(row_of).to(device), "players": len(present),
        }
        return inputs, width, int(cols.max()) + 1

    def samples(self, sessions: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        index, local = [], []
        for i, s in enumerate(sessions):
            first = self.sample_start[self.session_start[s]]
            last = self.sample_start[self.session_start[s] + self.session_count[s]]
            index.append(np.arange(first, last))
            local.append(np.full(last - first, i))
        return (torch.from_numpy(np.concatenate(index)).to(self.device),
                torch.from_numpy(np.concatenate(local)).to(self.device))


class Model(torch.nn.Module):
    def __init__(self, encoder, hidden: int, gru: int, rank: int, state: int, condition: bool = True,
                 hand_encoder: str = "gru", memory: str = "gru", head: str = "dot", mixture: int = 4) -> None:
        super().__init__()
        self.memory = PlayerMemory(encoder, gru=gru, state=state, hand_encoder=hand_encoder, memory=memory)
        self.belief = BeliefNet(encoder, hidden=hidden, gru=gru, rank=rank, use_hud=False, extra=state,
                                condition=condition, hand_encoder=hand_encoder, head=head, mixture=mixture)

    def forward(self, data: UnifiedData, sessions, index, local, condition, augment: bool):
        batch = data.batch(index, augment=augment, condition=condition)
        inputs, width, hands = data.memory_inputs(sessions)
        vectors = self.memory.hand_vectors(inputs["tokens"], inputs["lengths"], inputs["shown"], inputs["cards"],
                                           inputs["board"], inputs["result"])
        grid = vectors.new_zeros(inputs["players"], hands, vectors.shape[1])
        grid[inputs["rows"], inputs["cols"]] = vectors
        states = self.memory.states_before(grid)
        hand, target = data.s_hand[index], data.s_target[index]
        extra = states[inputs["row_of"][local * width + data.player[hand, target]], data.seq[hand, target]]
        logits, _phi, _inverse = self.belief(batch["tokens"], batch["lengths"], batch["static"], batch["board"],
                                             batch["known"], extra)
        return logits, batch


def architecture(config: dict) -> dict:
    """The architecture of a run's network; a run saved before the choices existed is all GRU and dot."""
    return {"hand_encoder": config.get("hand_encoder", "gru"), "memory": config.get("memory", "gru"),
            "head": config.get("head", "dot"), "mixture": int(config.get("mixture", 4))}


def draw(data: UnifiedData, rng: np.random.Generator, weights: np.ndarray, count: int) -> np.ndarray:
    groups = rng.choice(len(weights), size=count, p=weights)
    return np.array([rng.choice(data.train_by_group[g]) for g in groups])


@torch.no_grad()
def evaluate(model: Model, data: UnifiedData, groups: list[str], per_step: int, keep_every: int) -> dict:
    model.eval()
    keys = ("loss", "uniform", "entropy", "rank", "street", "acted", "seen", "unseen", "source", "mode", "reread")
    rows: dict[str, list] = {k: [] for k in keys}
    for first in range(0, len(data.test_sessions), per_step):
        sessions = data.test_sessions[first : first + per_step]
        index, local = data.samples(sessions)
        index, local = index[::keep_every], local[::keep_every]
        if len(index) == 0:
            continue
        for start in range(0, len(index), 4096):
            part, part_local = index[start : start + 4096], local[start : start + 4096]
            # as known: shown -> 1, dealt -> 0 (no condition); and every sample dealt known whose
            # target showed read again with 1, so every source has a "showdown, read as shown" row
            again = (data.mode[part] == 0) & data.shows[part]
            reread = torch.cat([torch.zeros(len(part), device=part.device), torch.ones(int(again.sum()), device=part.device)])
            part, part_local = torch.cat([part, part[again]]), torch.cat([part_local, part_local[again]])
            condition = torch.maximum(data.mode[part].float(), reread)
            logits, batch = model(data, sessions, part, part_local, condition, augment=False)
            log_p = torch.log_softmax(logits, dim=-1)
            loss = -log_p.gather(1, batch["target"].unsqueeze(1)).squeeze(1)
            allowed = logits > -1e8
            count = allowed.sum(dim=1).float()
            p = log_p.exp()
            rows["loss"].append(loss)
            rows["uniform"].append(torch.log(count))
            rows["entropy"].append(-(torch.where(allowed, p * log_p, torch.zeros_like(p))).sum(dim=1))
            rows["rank"].append(((log_p > (-loss).unsqueeze(1)) & allowed).sum(dim=1).float() / count)
            rows["street"].append(batch["street"])
            rows["acted"].append(batch["acted"])
            hand, target = data.s_hand[part], data.s_target[part]
            rows["seen"].append(data.seq[hand, target])
            rows["unseen"].append(data.unseen[part])
            rows["source"].append(data.sample_source[part])
            rows["mode"].append(data.mode[part])
            rows["reread"].append(reread)
    model.train()
    cat = {key: torch.cat(value).float().cpu() for key, value in rows.items()}

    def summary(mask) -> dict:
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        loss, uniform = float(cat["loss"][mask].mean()), float(cat["uniform"][mask].mean())
        return {"n": n, "loss": loss, "uniform": uniform, "gain_bits": (uniform - loss) / math.log(2),
                "range": float(cat["entropy"][mask].exp().mean()), "rank": float(cat["rank"][mask].mean())}

    acted = cat["acted"] > 0
    out = {}
    for name in data.source_names:
        index = data.source_names.index(name)
        mine = (cat["source"] == index) & acted
        base = mine & (cat["reread"] == 0)
        if int(base.sum()) == 0:
            continue
        report = {"agito": summary(base),
                  "showdown, letto come mostrato": summary(mine & ((cat["reread"] > 0) | (cat["mode"] == 1)))}
        for i, street in enumerate(STREETS):
            report[street.value] = summary(base & (cat["street"] == i))
        report["carte date"] = summary(base & (cat["mode"] == 0))
        report["carte mostrate"] = summary(base & (cat["mode"] == 1))
        report["mai visto"] = summary(base & (cat["unseen"] > 0))
        for label, low, high in SEEN:
            report[label] = summary(base & (cat["seen"] >= low) & (cat["seen"] <= high))
        out[name] = report
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", default=["selfplay=0.45", "handhq=0.35", "acpc=0.2"])
    parser.add_argument("--test-only", nargs="+", default=["pluribus", "wsop"])
    parser.add_argument("--no-condition", action="store_true",
                        help="the control: no condition input, a shown hand read as if dealt known")
    parser.add_argument("--evaluate-run", type=Path, default=None,
                        help="only evaluate the belief.pt of this run (belief_history.py's or this one's)")
    parser.add_argument("--condition-share", type=float, default=0.2,
                        help="share of a batch re-read with the condition 1 from the samples dealt known")
    parser.add_argument("--equity-model", default="equity_evaluator/distill-lat32_h256_h1.pt")
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--bf16", action="store_true", help="train in bfloat16 mixed precision (CUDA)")
    parser.add_argument("--keep-evaluations", action="store_true",
                        help="also keep a copy of the network at every evaluation (belief-<step>.pt)")
    parser.add_argument("--stop-after", type=int, default=0,
                        help="end (with an evaluation) after this many steps of a schedule of --steps: arms "
                        "compared at equal steps and learning rate")
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gru", type=int, default=128)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--state", type=int, default=128)
    parser.add_argument("--hand-encoder", choices=HAND_ENCODERS, default="gru",
                        help="what reads a hand's tokens (the current hand and each past hand of the memory)")
    parser.add_argument("--memory", choices=MEMORIES, default="gru", help="what reads a player's past hands")
    parser.add_argument("--head", choices=HEADS, default="dot", help="how a combo is scored against the context")
    parser.add_argument("--mixture", type=int, default=4, help="parts of a range with --head mixture")
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--eval-keep-every", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)
    groups = [w.split("=")[0] for w in args.weights]
    weights = np.array([float(w.split("=")[1]) for w in args.weights])

    started = time.time()
    arrays = concatenate([load(path) for path in args.data])
    data = UnifiedData(arrays, args.device, groups, args.test_only)
    del arrays
    present = [len(t) > 0 for t in data.train_by_group]
    weights = np.where(present, weights, 0.0)
    weights = weights / weights.sum()
    print({g: (len(t), float(w)) for g, t, w in zip(groups, data.train_by_group, weights, strict=True)},
          f"training sessions per group; {len(data.test_sessions)} test sessions; "
          f"{len(data.s_hand)} samples ({time.time() - started:.0f}s to load)", flush=True)
    if args.evaluate_run is not None:
        saved = torch.load(args.evaluate_run / "belief.pt", map_location="cpu", weights_only=False)
        c = saved["config"]
        model = Model(build_encoder(c["equity_model"]), int(c["hidden"]), int(c["gru"]), int(c["rank"]),
                      int(c["state"]), condition=bool(c.get("condition", False)), **architecture(c))
        missing, unexpected = model.load_state_dict(saved["state"], strict=False)
        assert not unexpected and all("encoder" in key for key in missing), (missing, unexpected)
        report = evaluate(model.to(args.device), data, groups, args.sessions, args.eval_keep_every)
        for source, rows in report.items():
            print(format_report(f"{args.evaluate_run.name} {source}", rows), flush=True)
        (args.out / "evaluation.json").write_text(json.dumps({"run": str(args.evaluate_run), "report": report}, indent=1))
        return 0
    model = Model(build_encoder(args.equity_model), args.hidden, args.gru, args.rank, args.state,
                  condition=not args.no_condition, **architecture(vars(args))).to(args.device)
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters, "
          f"{architecture(vars(args))}", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.03)
    history, running = [], 0.0
    for step in range(1, args.steps + 1):
        sessions = draw(data, rng, weights, args.sessions)
        index, local = data.samples(sessions)
        keep = ~data.unseen[index]
        index, local = index[keep], local[keep]
        if len(index) == 0:
            continue
        pick = torch.randperm(len(index), device=args.device)[: args.batch_size]
        index, local = index[pick], local[pick]
        condition = data.mode[index].float()
        extra = torch.nonzero((data.mode[index] == 0) & data.shows[index]).squeeze(1)
        share = 0.0 if args.no_condition else args.condition_share
        extra = extra[torch.randperm(len(extra), device=args.device)[: int(share * len(index))]]
        if len(extra):
            index = torch.cat([index, index[extra]])
            local = torch.cat([local, local[extra]])
            condition = torch.cat([condition, torch.ones(len(extra), device=args.device)])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            logits, batch = model(data, sessions, index, local, condition, augment=True)
        loss = functional.cross_entropy(logits.float(), batch["target"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        schedule.step()
        running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 500 == 0:
            print(f"step {step:6d}  loss {running:.4f}  {time.time() - started:.0f}s", flush=True)
        last = step == args.steps or step == args.stop_after
        if step % args.eval_every == 0 or last:
            report = evaluate(model, data, groups, args.sessions, args.eval_keep_every)
            history.append({"step": step, "report": report})
            for source, rows in report.items():
                print(format_report(f"step {step} {source}", rows), flush=True)
            torch.save({"arm": "memory", "config": vars(args) | {"out": str(args.out), "data": [str(d) for d in args.data],
                                                                   "condition": not args.no_condition},
                        "state": {k: v for k, v in model.state_dict().items() if ".encoder." not in k and not k.startswith("encoder")}},
                       args.out / "belief.pt")
            if args.keep_evaluations:
                (args.out / f"belief-{step:07d}.pt").write_bytes((args.out / "belief.pt").read_bytes())
            (args.out / "result.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                              "history": history}, indent=1))
        if step == args.stop_after:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
