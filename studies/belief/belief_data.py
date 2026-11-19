"""The hands of `belief_generate.py` as arrays: what the belief network trains on.

    PYTHONPATH=src:studies/belief python studies/belief/belief_data.py --data checkpoints/studies/belief/data --jobs 40

One `arrays-NNNN.npz` next to each `hands-NNNN.jsonl.gz`. Per hand (H): its tokens, flat, with
`token_offset` (H + 1) saying where each hand's start; the board (`board`, H x 5, -1 where no card
was dealt); every seat's hole cards (`holes`, H x 9 x 2, -1 for no seat), position (`position`,
index in `POSITIONS`, -1), starting stack in big blinds and statistics at the start of the hand
(`hud`, H x 9 x 20); the session (`session`) and every seat's model (`seat_model`, an index into
the shard's `models`) and whether it was styled; the hand's number in its session
(`hand_number`), every seat's result in big blinds (`result_bb`, what anyone at the table sees its
stack do) and whether it showed its cards (`shown`: still in at a showdown, which the engine
reveals -- the only private cards a player at the table ever sees). Per sample (S): the hand, the token it stands
before (`prefix`), the target and the observer, the street and how many were still in.

The featurisation is `belief_records.py`; this only packs it. Numpy, no torch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from belief_generate import read_shard
from belief_records import (
    ACTION_CLASSES,
    HUD_DIM,
    MAX_SEATS,
    POSITIONS,
    STREETS,
    TOKEN_DIM,
    card_index,
    hand_tokens,
    hud_before_hands,
    positions,
    samples_of,
)

from pokerlab.engine.history import _hand_history_from_dict

FOLD_COLUMN = len(STREETS) + len(POSITIONS) + ACTION_CLASSES.index("fold")


MAX_PLAYERS = 4000  # distinct players a session may hold (a real table's chunk sees players come and go)


def pack_sessions(sessions: list[dict], *, hud: bool = False) -> dict[str, np.ndarray]:
    """Sessions (`belief_generate.py`'s, every card known; or `phh_convert.py`'s, real hands, some
    cards unknown) as arrays. A player is who it is by name (`seat_names`) within a session
    (`player`, and `seq_pos`, the place of the hand among that player's hands, what the memory
    reads). A sample asks about a target whose cards are known, with how they came to be known
    (`s_mode`: 0 dealt -- whatever the player did, an unbiased label -- or 1 shown at a showdown)
    and whether the target reached a showdown (`s_shows`)."""
    models: list[str] = []
    model_index: dict[str, int] = {}
    sources: list[str] = []
    source_index: dict[str, int] = {}
    tokens, actors, offsets = [], [], [0]
    board, holes, position, stack, hud_rows_all, session_of, seat_model, styled = [], [], [], [], [], [], [], []
    player_of, seq_pos, source_of = [], [], []
    s_hand, s_prefix, s_target, s_observer, s_street, s_live, s_mode, s_shows = [], [], [], [], [], [], [], []
    hand_number, shown, result = [], [], []

    def index_of(table: dict, names: list, label: str) -> int:
        if label not in table:
            table[label] = len(names)
            names.append(label)
        return table[label]

    for session in sessions:
        labels = session.get("models")
        source = index_of(source_index, sources, session.get("source", "selfplay"))
        hands = [_hand_history_from_dict(h) for h in session["hands"]]
        known_by = [h.get("known_by") for h in session["hands"]]
        huds = hud_before_hands(hands) if hud else None
        players: dict[str, int] = {}
        played: dict[int, int] = {}
        for number, hand in enumerate(hands):
            h = len(board)
            read = hand_tokens(hand)
            still_in = set(hand.starting_stacks)
            for actor, token in zip(read.actors, read.tokens, strict=True):
                if token[FOLD_COLUMN] == 1.0:
                    still_in.discard(actor)
            showdown = len(still_in) >= 2
            hand_number.append(number)
            source_of.append(source)
            tokens.extend(read.tokens)
            actors.extend(read.actors)
            offsets.append(len(tokens))
            cards = [card_index(c) for c in hand.community_cards]
            board.append(cards + [-1] * (5 - len(cards)))
            names = positions(hand)
            row = {k: [] for k in ("hole", "pos", "stack", "model", "styled", "shown", "result", "player", "seq")}
            hud_rows = np.zeros((MAX_SEATS, HUD_DIM), dtype=np.float16)
            modes: dict[int, int] = {}
            for seat in range(MAX_SEATS):
                if seat in hand.starting_stacks:
                    pair = hand.hole_cards.get(seat)
                    if pair is not None:
                        how = (known_by[number] or {}).get(str(seat), "dealt")
                        modes[seat] = 0 if how == "dealt" else 1
                    name = hand.seat_names.get(seat, f"seat{seat}")
                    player = players.setdefault(name, len(players))
                    if player >= MAX_PLAYERS:
                        raise ValueError(f"a session of more than {MAX_PLAYERS} players")
                    row["hole"].append([card_index(pair[0]), card_index(pair[1])] if pair else [-1, -1])
                    row["pos"].append(POSITIONS.index(names[seat]))
                    row["stack"].append(hand.starting_stacks[seat] / hand.big_blind)
                    row["model"].append(index_of(model_index, models, labels[seat] if labels else name))
                    row["styled"].append(bool(session["styled"][seat]) if "styled" in session else False)
                    row["shown"].append(showdown and seat in still_in and pair is not None)
                    row["result"].append((hand.final_stacks[seat] - hand.starting_stacks[seat]) / hand.big_blind)
                    row["player"].append(player)
                    row["seq"].append(played.get(player, 0))
                    played[player] = played.get(player, 0) + 1
                    if huds is not None and seat in huds[number]:
                        hud_rows[seat] = huds[number][seat]
                else:
                    for key, empty in (("hole", [-1, -1]), ("pos", -1), ("stack", 0.0), ("model", -1),
                                       ("styled", False), ("shown", False), ("result", 0.0), ("player", -1),
                                       ("seq", -1)):
                        row[key].append(empty)
            holes.append(row["hole"])
            position.append(row["pos"])
            stack.append(row["stack"])
            hud_rows_all.append(hud_rows)
            session_of.append(session["session"])
            seat_model.append(row["model"])
            styled.append(row["styled"])
            shown.append(row["shown"])
            result.append(row["result"])
            player_of.append(row["player"])
            seq_pos.append(row["seq"])
            for sample in samples_of(read):
                if sample.target not in modes:
                    continue  # nothing known of its cards: nothing to learn
                s_hand.append(h)
                s_prefix.append(sample.prefix)
                s_target.append(sample.target)
                s_observer.append(sample.observer)
                s_street.append(sample.street)
                s_live.append(sample.live)
                s_mode.append(modes[sample.target])
                s_shows.append(showdown and sample.target in still_in)
    packed = {
        "tokens": np.asarray(tokens, dtype=np.float16).reshape(-1, TOKEN_DIM),
        "token_actor": np.asarray(actors, dtype=np.int8),
        "token_offset": np.asarray(offsets, dtype=np.int64),
        "board": np.asarray(board, dtype=np.int8).reshape(-1, 5),
        "holes": np.asarray(holes, dtype=np.int8).reshape(-1, MAX_SEATS, 2),
        "position": np.asarray(position, dtype=np.int8).reshape(-1, MAX_SEATS),
        "stack_bb": np.asarray(stack, dtype=np.float16).reshape(-1, MAX_SEATS),
        "session": np.asarray(session_of, dtype=np.int64),
        "seat_model": np.asarray(seat_model, dtype=np.int32).reshape(-1, MAX_SEATS),
        "styled": np.asarray(styled, dtype=bool).reshape(-1, MAX_SEATS),
        "models": np.asarray(models),
        "sources": np.asarray(sources),
        "source": np.asarray(source_of, dtype=np.int16),
        "player": np.asarray(player_of, dtype=np.int16).reshape(-1, MAX_SEATS),
        "seq_pos": np.asarray(seq_pos, dtype=np.int16).reshape(-1, MAX_SEATS),
        "hand_number": np.asarray(hand_number, dtype=np.int16),
        "shown": np.asarray(shown, dtype=bool).reshape(-1, MAX_SEATS),
        "result_bb": np.asarray(result, dtype=np.float16).reshape(-1, MAX_SEATS),
        "s_hand": np.asarray(s_hand, dtype=np.int32),
        "s_prefix": np.asarray(s_prefix, dtype=np.int16),
        "s_target": np.asarray(s_target, dtype=np.int8),
        "s_observer": np.asarray(s_observer, dtype=np.int8),
        "s_street": np.asarray(s_street, dtype=np.int8),
        "s_live": np.asarray(s_live, dtype=np.int8),
        "s_mode": np.asarray(s_mode, dtype=np.int8),
        "s_shows": np.asarray(s_shows, dtype=bool),
    }
    if hud:
        packed["hud"] = np.stack(hud_rows_all) if hud_rows_all else np.zeros((0, MAX_SEATS, HUD_DIM), dtype=np.float16)
    return packed


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Several packed shards as one: indices into hands, tokens and models are shifted."""
    models: list[str] = []
    index: dict[str, int] = {}
    sources: list[str] = []
    source_index: dict[str, int] = {}
    shared = set.intersection(*(set(part) for part in parts))
    parts = [{key: part[key] for key in shared} for part in parts]
    out: dict[str, list] = {key: [] for key in parts[0] if key not in ("models", "sources")}
    hands = tokens = 0
    for part in parts:
        remap = []
        for label in part["models"].tolist():
            if label not in index:
                index[label] = len(models)
                models.append(label)
            remap.append(index[label])
        remap_array = np.asarray(remap + [-1], dtype=np.int32)  # -1 stays -1
        source_remap = np.zeros(max(len(part.get("sources", [])), 1), dtype=np.int16)
        for i, label in enumerate(part.get("sources", np.asarray(["selfplay"])).tolist()):
            if label not in source_index:
                source_index[label] = len(sources)
                sources.append(label)
            source_remap[i] = source_index[label]
        for key, value in part.items():
            if key in ("models", "sources"):
                continue
            if key == "token_offset":
                value = value[1:] + tokens if out[key] else value + tokens
            elif key == "s_hand":
                value = value + hands
            elif key == "seat_model":
                value = remap_array[value]
            elif key == "source":
                value = source_remap[value]
            out[key].append(value)
        hands += len(part["board"])
        tokens += len(part["tokens"])
    merged = {key: np.concatenate(value) for key, value in out.items()}
    merged["models"] = np.asarray(models)
    if "source" in merged:
        merged["sources"] = np.asarray(sources)
    return merged


def load(data: Path) -> dict[str, np.ndarray]:
    paths = sorted(data.glob("arrays-*.npz"))
    if not paths:
        raise SystemExit(f"no arrays-*.npz in {data}: run belief_data.py first")
    parts = []
    for path in paths:
        with np.load(path) as loaded:
            if "s_hand" not in loaded.files:
                continue  # a shard none of whose hands was kept
            parts.append({key: loaded[key] for key in loaded.files})
    return concatenate(parts)


def main(argv: list[str] | None = None) -> int:
    from pokerlab.rl.forking import run_in_copies

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("checkpoints/studies/belief/data"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--hud", action="store_true", help="pack the opponent statistics too (the HUD arms)")
    args = parser.parse_args(argv)
    todo = [p for p in sorted(args.data.glob("hands-*.jsonl.gz"))
            if not p.with_name(p.name.replace("hands-", "arrays-").replace(".jsonl.gz", ".npz")).exists()]

    def work(job: int) -> str:
        source = todo[job]
        target = source.with_name(source.name.replace("hands-", "arrays-").replace(".jsonl.gz", ".npz"))
        arrays = pack_sessions(read_shard(source), hud=args.hud)
        partial = target.with_name("." + target.name + ".partial.npz")
        np.savez(partial, **arrays)
        partial.rename(target)
        return f"{target.name}: {len(arrays['board'])} hands, {len(arrays['s_hand'])} samples"

    for line in run_in_copies(work, len(todo), args.jobs):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
