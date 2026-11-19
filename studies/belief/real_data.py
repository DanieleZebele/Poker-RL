"""Every real no-limit hold'em hand we have, as one data set in our own format.

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/belief python studies/belief/real_data.py \\
        --zip dataset_real_hands/downloads/phh-v3.zip --out dataset_real_hands/unified --jobs 40

Reads the PHH archive (`phh_convert.py`) member by member, without extracting it, keeps the no-limit
Texas hold'em hands of 2-9 players (every other variant, the fixed-limit ACPC, badugi, ... is left
out) and writes, per shard of `--shard-files` members:

- `hands-<source>-NNNN.jsonl.gz`: the sessions, one JSON line each (`{"session", "source", "table",
  "full_information", "hands": [...]}`, the hands as `engine/history.py` writes them, plus
  `known_by`): **the unified data set**, re-packable without the archive;
- `arrays-<source>-NNNN.npz`: the same packed for the belief network (`belief_data.pack_sessions`).

Sources (`--sources`): `handhq` (online, 2009, the cards shown at a showdown), `pluribus` (Pluribus
and five professionals, every card), `wsop` (a televised final table, every card), `acpc`
(heads-up no-limit bots of the Annual Computer Poker Competition, every card; `--acpc-files` at most,
the archive holds hundreds of millions). A session id is `SESSION_BASE + shard * 100000 + i`, apart
from the self-play ones. A rerun skips the shards already written.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from belief_data import pack_sessions
from phh_convert import parse_file, sessions_of_file

SESSION_BASE = 10_000_000
FULL_INFORMATION = {"handhq": False, "pluribus": True, "wsop": True, "acpc": True}


def source_of(member: str) -> str | None:
    """`handhq`, `pluribus`, `wsop` or `acpc` for a member of the archive, None for anything else."""
    parts = member.split("/")
    if len(parts) < 3 or parts[0] != "data" or not member.endswith((".phh", ".phhs")):
        return None
    top = parts[1].lower()
    if top == "annual-computer-poker-competition":
        # heads-up no-limit only, and not the "processed" copies of the same matches
        lowered = member.lower()
        no_limit = "nolimit" in lowered or "2pn" in lowered
        return "acpc" if no_limit and "processed" not in lowered else None
    for name in FULL_INFORMATION:
        if top.startswith(name):
            return name
    return None


def site_of(member: str, source: str) -> str:
    """A finer name for the report: HandHQ's site (PS, PTY, ...), the ACPC's year."""
    parts = member.split("/")
    if source == "handhq" and len(parts) > 2:
        return "handhq-" + parts[2].split("-")[0]
    if source == "acpc" and len(parts) > 3:
        return "acpc-" + parts[3]
    return source


def natural_key(name: str) -> list:
    """`a/10.phh` after `a/9.phh`: numbers compared as numbers."""
    import re

    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def convert_shard(zip_path: Path, members: list[str], source: str, shard: int, out: Path) -> dict:
    target = out / f"arrays-{source}-{shard:04d}.npz"
    if target.exists():
        return {"shard": shard, "skipped_shard": True}
    skipped: dict[str, int] = {}
    played = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in members:
            try:
                hands = parse_file(archive.read(member).decode("utf-8", errors="replace"))
            except Exception as error:  # noqa: BLE001 - a broken file is counted, not fatal
                skipped["file:" + type(error).__name__] = skipped.get("file:" + type(error).__name__, 0) + 1
                continue
            folder = member.rsplit("/", 1)[0]
            for hand in hands:
                # a table's hands can span files, and a file of Pluribus is one hand of a session
                # whose files share a folder: the folder stands for a table where none is recorded
                hand.setdefault("table", member if source == "acpc" else folder)  # an ACPC file is one match
                played.append(hand)
    sessions = sessions_of_file(played, source=site_of(members[0], source) if members else source,
                                full_information=FULL_INFORMATION[source], skipped=skipped)
    for i, session in enumerate(sessions):
        session["session"] = SESSION_BASE + shard * 100_000 + i
    hands = sum(len(s["hands"]) for s in sessions)
    with gzip.open(out / f".hands-{source}-{shard:04d}.partial", "wt", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(json.dumps(session, separators=(",", ":")) + "\n")
    (out / f".hands-{source}-{shard:04d}.partial").rename(out / f"hands-{source}-{shard:04d}.jsonl.gz")
    if sessions:
        arrays = pack_sessions(sessions)
        partial = out / f".arrays-{source}-{shard:04d}.partial.npz"
        np.savez(partial, **arrays)
        partial.rename(target)
        samples = len(arrays["s_hand"])
    else:
        np.savez(target, empty=np.zeros(0))
        samples = 0
    return {"shard": shard, "hands": hands, "sessions": len(sessions), "samples": samples, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    from pokerlab.rl.forking import run_in_copies

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=list(FULL_INFORMATION))
    parser.add_argument("--shard-files", type=int, default=40, help="archive members per shard")
    parser.add_argument("--acpc-files", type=int, default=400, help="ACPC members at most (evenly spread)")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as archive:
        names = archive.namelist()
    by_source: dict[str, list[str]] = {}
    for name in names:
        source = source_of(name)
        if source in args.sources:
            by_source.setdefault(source, []).append(name)
    if "acpc" in by_source:
        members = sorted(by_source["acpc"])
        step = max(1, len(members) // max(args.acpc_files, 1))
        by_source["acpc"] = members[::step][: args.acpc_files]
    jobs = []
    for source, members in sorted(by_source.items()):
        members.sort(key=natural_key)
        if source == "pluribus":
            # one shard per folder (a session of play), its hands in the order played
            folders: dict[str, list[str]] = {}
            for member in members:
                folders.setdefault(member.rsplit("/", 1)[0], []).append(member)
            chunks = list(folders.values())
        else:
            # a shard is files of one site and stake (`site_of`), so a table's hands meet in it
            chunks = []
            for member in members:
                if chunks and len(chunks[-1]) < args.shard_files and member.rsplit("/", 2)[0] == chunks[-1][0].rsplit("/", 2)[0]:
                    chunks[-1].append(member)
                else:
                    chunks.append([member])
        for shard, chunk in enumerate(chunks):
            jobs.append((source, shard, chunk))
    print({source: len(members) for source, members in by_source.items()}, f"-> {len(jobs)} shards", flush=True)
    started = time.time()
    totals: dict[str, dict] = {}

    def work(job: int) -> dict:
        source, shard, members = jobs[job]
        return {"source": source, **convert_shard(args.zip, members, source, shard, args.out)}

    for result in run_in_copies(work, len(jobs), args.jobs):
        total = totals.setdefault(result["source"], {"hands": 0, "sessions": 0, "samples": 0, "skipped": {}})
        for key in ("hands", "sessions", "samples"):
            total[key] += result.get(key, 0)
        for reason, count in result.get("skipped", {}).items():
            total["skipped"][reason] = total["skipped"].get(reason, 0) + count
    for source, total in totals.items():
        print(f"{source}: {total['hands']} hands, {total['sessions']} sessions, {total['samples']} samples; "
              f"left out {dict(sorted(total['skipped'].items(), key=lambda kv: -kv[1])[:8])}", flush=True)
    print(f"{time.time() - started:.0f}s", flush=True)
    (args.out / "summary.json").write_text(json.dumps(totals, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
