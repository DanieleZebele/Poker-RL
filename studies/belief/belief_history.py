"""The belief network with a memory of the target's past hands, against the opponent statistics.

    PYTHONPATH=src:studies/belief python studies/belief/belief_history.py --data checkpoints/studies/belief/data \\
        --arm memory --out checkpoints/studies/belief/history/memory

Training goes by whole sessions: a step takes `--sessions` sessions, encodes every hand of every
player in them (`belief_model.PlayerMemory`), runs the memory over each player's hands in order and
trains the range on `--batch-size` of the samples of those sessions, each reading the state its
target's memory had at the start of that hand. Arms (`--arm`), all through this same pipeline so
they are compared on equal terms:

- `hud`: the 20 statistics of `engine/stats.py`, no memory (what the models read today);
- `memory`: the memory, no statistics;
- `both`: the two;
- `none`: neither (only the current hand).

The report (`belief_train.format_report`) adds groups by how many hands of the target the table has
already seen in the session (`visto 0`, `1-9`, `10-49`, `50+`): where a memory can only help with
time. Test sessions and held-out models as in `belief_train.py`; here both are read on the test
sessions (held-out models are never a target in training, in any session).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from belief_data import load
from belief_model import BeliefNet, PlayerMemory, build_encoder
from belief_records import MAX_SEATS, STREETS
from belief_train import HELD_OUT_EVERY, Data, format_report, held_out_model
from torch.nn import functional

ARMS = {"hud": (True, False), "memory": (False, True), "both": (True, True), "none": (False, False)}
SEEN_GROUPS = (("visto 0", 0, 0), ("visto 1-9", 1, 9), ("visto 10-49", 10, 49), ("visto 50+", 50, 10**6))


class SessionData(Data):
    def __init__(self, arrays: dict[str, np.ndarray], device: str) -> None:
        super().__init__(arrays, device)
        session = arrays["session"]
        starts = np.flatnonzero(np.r_[True, session[1:] != session[:-1]])
        self.session_start = starts
        self.session_count = np.diff(np.r_[starts, len(session)])
        self.session_id = session[starts]
        self.session_size = (arrays["position"][starts] >= 0).sum(axis=1)
        self.sample_start = np.searchsorted(arrays["s_hand"], np.arange(len(session) + 1))
        self.offset_cpu = arrays["token_offset"]
        self.hand_number = torch.from_numpy(arrays["hand_number"].astype(np.int64)).to(device)
        self.shown = torch.from_numpy(arrays["shown"]).to(device)
        self.result_bb = torch.from_numpy(arrays["result_bb"].astype(np.float32)).to(device)
        target_model = arrays["seat_model"][arrays["s_hand"], arrays["s_target"]]
        unseen_models = np.array([held_out_model(m) for m in arrays["models"].tolist()] + [False])
        self.unseen = torch.from_numpy(unseen_models[target_model]).to(device)
        self.styled = torch.from_numpy(arrays["styled"][arrays["s_hand"], arrays["s_target"]]).to(device)
        test = self.session_id % HELD_OUT_EVERY == 0
        self.train_sessions = np.flatnonzero(~test)
        self.test_sessions = np.flatnonzero(test)

    def memory_inputs(self, sessions: np.ndarray) -> tuple[dict, int]:
        """Every (hand, player) of `sessions`, with where its vector goes in `(S * 9, K)`."""
        hands, seats, rows, cols = [], [], [], []
        for local, s in enumerate(sessions):
            start, count, size = self.session_start[s], self.session_count[s], self.session_size[s]
            for seat in range(size):
                hands.append(np.arange(start, start + count))
                seats.append(np.full(count, seat))
                rows.append(np.full(count, local * MAX_SEATS + seat))
                cols.append(np.arange(count))
        hand = np.concatenate(hands)
        lengths = self.offset_cpu[hand + 1] - self.offset_cpu[hand]
        device = self.device
        hand_t, seat_t = torch.from_numpy(hand).to(device), torch.from_numpy(np.concatenate(seats)).to(device)
        length = int(lengths.max())
        steps = torch.arange(length, device=device)
        lengths_t = torch.from_numpy(lengths).to(device)
        where = (self.offset[hand_t].unsqueeze(1) + steps).clamp_max(len(self.tokens) - 1)
        valid = (steps.unsqueeze(0) < lengths_t.unsqueeze(1)).float()
        own = (self.token_actor[where] == seat_t.unsqueeze(1)).float() * valid
        tokens = torch.cat([self.tokens[where] * valid.unsqueeze(-1), own.unsqueeze(-1)], dim=-1)
        inputs = {
            "tokens": tokens, "lengths": lengths_t, "shown": self.shown[hand_t, seat_t].float(),
            "cards": self.holes[hand_t, seat_t], "board": self.board[hand_t],
            "result": self.result_bb[hand_t, seat_t],
            "rows": torch.from_numpy(np.concatenate(rows)).to(device),
            "cols": torch.from_numpy(np.concatenate(cols)).to(device),
        }
        return inputs, int(self.session_count[sessions].max())

    def samples(self, sessions: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Every sample of `sessions` and the local index of its session."""
        index, local = [], []
        for i, s in enumerate(sessions):
            first = self.sample_start[self.session_start[s]]
            last = self.sample_start[self.session_start[s] + self.session_count[s]]
            index.append(np.arange(first, last))
            local.append(np.full(last - first, i))
        return (torch.from_numpy(np.concatenate(index)).to(self.device),
                torch.from_numpy(np.concatenate(local)).to(self.device))


class Model(torch.nn.Module):
    def __init__(self, encoder, arm: str, hidden: int, gru: int, rank: int, state: int) -> None:
        super().__init__()
        use_hud, use_memory = ARMS[arm]
        self.memory = PlayerMemory(encoder, gru=gru, state=state) if use_memory else None
        self.belief = BeliefNet(encoder, hidden=hidden, gru=gru, rank=rank, use_hud=use_hud,
                                extra=state if use_memory else 0)

    def forward(self, data: SessionData, sessions: np.ndarray, index: torch.Tensor, local: torch.Tensor,
                augment: bool) -> tuple[torch.Tensor, dict]:
        batch = data.batch(index, augment=augment)
        extra = None
        if self.memory is not None:
            inputs, hands = data.memory_inputs(sessions)
            vectors = self.memory.hand_vectors(inputs["tokens"], inputs["lengths"], inputs["shown"],
                                               inputs["cards"], inputs["board"], inputs["result"])
            grid = vectors.new_zeros(len(sessions) * MAX_SEATS, hands, vectors.shape[1])
            grid[inputs["rows"], inputs["cols"]] = vectors
            states = self.memory.states_before(grid)
            rows = local * MAX_SEATS + data.s_target[index]
            extra = states[rows, data.hand_number[data.s_hand[index]]]
        logits, _phi, _inverse = self.belief(batch["tokens"], batch["lengths"], batch["static"], batch["board"],
                                             batch["known"], extra)
        return logits, batch


@torch.no_grad()
def evaluate(model: Model, data: SessionData, sessions_per_step: int, keep_every: int) -> tuple[dict, dict]:
    model.eval()
    rows: dict[str, list] = {k: [] for k in ("loss", "uniform", "entropy", "rank", "street", "live", "acted",
                                            "seen", "unseen", "styled")}
    for first in range(0, len(data.test_sessions), sessions_per_step):
        sessions = data.test_sessions[first : first + sessions_per_step]
        index, local = data.samples(sessions)
        index, local = index[::keep_every], local[::keep_every]
        for start in range(0, len(index), 4096):
            part, part_local = index[start : start + 4096], local[start : start + 4096]
            logits, batch = model(data, sessions, part, part_local, augment=False)
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
            rows["live"].append(batch["live"])
            rows["acted"].append(batch["acted"])
            rows["seen"].append(data.hand_number[data.s_hand[part]])
            rows["unseen"].append(data.unseen[part])
            rows["styled"].append(data.styled[part])
    model.train()
    cat = {key: torch.cat(value).float().cpu() for key, value in rows.items()}

    def summary(mask) -> dict:
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        loss, uniform = float(cat["loss"][mask].mean()), float(cat["uniform"][mask].mean())
        return {"n": n, "loss": loss, "uniform": uniform, "gain_bits": (uniform - loss) / math.log(2),
                "range": float(cat["entropy"][mask].exp().mean()), "rank": float(cat["rank"][mask].mean())}

    def report(base) -> dict:
        out = {"all": summary(base)}
        for i, street in enumerate(STREETS):
            out[street.value] = summary(base & (cat["street"] == i))
        out["ha agito"] = summary(base & (cat["acted"] > 0))
        out["non ancora"] = summary(base & (cat["acted"] == 0))
        for name, low, high in (("2 in", 2, 2), ("3-4 in", 3, 4), ("5+ in", 5, 9)):
            out[name] = summary(base & (cat["live"] >= low) & (cat["live"] <= high))
        for name, styled in (("stile agito", 1), ("senza stile agito", 0)):
            out[name] = summary(base & (cat["styled"] == styled) & (cat["acted"] > 0))
        for name, low, high in SEEN_GROUPS:
            out[name] = summary(base & (cat["seen"] >= low) & (cat["seen"] <= high))
            out[name + " agito"] = summary(base & (cat["seen"] >= low) & (cat["seen"] <= high) & (cat["acted"] > 0))
        return out

    return report(cat["unseen"] == 0), report(cat["unseen"] > 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("checkpoints/studies/belief/data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), default="memory")
    parser.add_argument("--equity-model", default="equity_evaluator/distill-lat32_h256_h1.pt")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--sessions", type=int, default=16, help="sessions per step")
    parser.add_argument("--batch-size", type=int, default=4096, help="samples per step")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gru", type=int, default=128)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--state", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--eval-keep-every", type=int, default=4, help="evaluate one test sample in this many")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluate-run", type=Path, default=None,
                        help="only evaluate the belief.pt of this --out directory on --data's test sessions")
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    data = SessionData(load(args.data), args.device)
    print(f"{len(data.train_sessions)} training sessions, {len(data.test_sessions)} test sessions "
          f"({time.time() - started:.0f}s to load)", flush=True)
    if args.evaluate_run is not None:
        saved = torch.load(args.evaluate_run / "belief.pt", map_location="cpu", weights_only=False)
        c = saved["config"]
        model = Model(build_encoder(c["equity_model"]), saved["arm"], int(c["hidden"]), int(c["gru"]),
                      int(c["rank"]), int(c["state"]))
        missing, unexpected = model.load_state_dict(saved["state"], strict=False)
        assert not unexpected and all("encoder" in key for key in missing), (missing, unexpected)
        seen, unseen = evaluate(model.to(args.device), data, args.sessions, args.eval_keep_every)
        print(format_report(f"{args.evaluate_run} test", seen), flush=True)
        print(format_report("modelli mai visti", unseen), flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "evaluation.json").write_text(json.dumps({"run": str(args.evaluate_run), "seen": seen,
                                                              "unseen": unseen}, indent=1))
        return 0
    model = Model(build_encoder(args.equity_model), args.arm, args.hidden, args.gru, args.rank,
                  args.state).to(args.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    print(f"arm {args.arm}: {sum(p.numel() for p in trainable)} trainable parameters", flush=True)

    history, running = [], 0.0
    for step in range(1, args.steps + 1):
        sessions = rng.choice(data.train_sessions, size=args.sessions, replace=False)
        index, local = data.samples(sessions)
        keep = ~data.unseen[index]
        index, local = index[keep], local[keep]
        pick = torch.randperm(len(index), device=args.device)[: args.batch_size]
        logits, batch = model(data, sessions, index[pick], local[pick], augment=True)
        loss = functional.cross_entropy(logits, batch["target"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        schedule.step()
        running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 200 == 0:
            print(f"step {step:6d}  loss {running:.4f}  {time.time() - started:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            seen, unseen = evaluate(model, data, args.sessions, args.eval_keep_every)
            history.append({"step": step, "seen": seen, "unseen": unseen})
            print(format_report(f"step {step} test", seen), flush=True)
    print(format_report("modelli mai visti", history[-1]["unseen"]), flush=True)
    (args.out / "result.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                      "history": history}, indent=1))
    torch.save({"arm": args.arm, "config": vars(args) | {"out": str(args.out), "data": str(args.data)},
                "state": {k: v for k, v in model.state_dict().items() if ".encoder." not in k and not k.startswith("encoder")}},
               args.out / "belief.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
