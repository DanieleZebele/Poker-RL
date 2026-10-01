"""Publish the current top models to git. Run by hand, nothing imports it.

    python -m pokerlab.rl.push_top_models              # top 5 -> top_models/, commit, push
    python -m pokerlab.rl.push_top_models --dry-run    # only say what it would do
    python -m pokerlab.rl.push_top_models --no-push    # commit, but do not push

`checkpoints/` is gitignored (the store is tens of GB), so the models are copied
into a tracked folder, `top_models/` by default, as `<label>.pt` plus a
`ratings.json`. The folder always holds exactly the current top N: a model that
fell out of the top is removed from it, so the working tree stays a few MB. Git
keeps the old files in its history, so each push with a changed top grows the
repository by up to N x 3 MB.

Only that folder is staged and committed -- whatever else is modified in the
working tree is left exactly as it is. Pushing is the one thing this script does
that others can see, and it happens only when it is run, and only to the remote
and branch you are on.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

DEFAULT_DIR = "top_models"
DEFAULT_LIMIT = 5
RATINGS_FILE = "ratings.json"


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def repo_root(start: Path) -> Path:
    result = git(start, "rev-parse", "--show-toplevel", check=False)
    if result.returncode != 0:
        raise SystemExit(f"{start} non e' dentro un repository git")
    return Path(result.stdout.strip())


def copy_models(top: Sequence[tuple[str, Path, float]], folder: Path) -> list[str]:
    """Make `folder` hold exactly `top`; returns the labels that were removed."""
    folder.mkdir(parents=True, exist_ok=True)
    wanted = {f"{label}.pt" for label, _path, _rating in top}
    removed = []
    for old in sorted(folder.glob("*.pt")):
        if old.name not in wanted:
            old.unlink()
            removed.append(old.stem)
    for label, path, _rating in top:
        target = folder / f"{label}.pt"
        if not target.exists() or target.stat().st_size != Path(path).stat().st_size:
            shutil.copy2(path, target)
    ratings = {label: round(rating, 1) for label, _path, rating in top}
    (folder / RATINGS_FILE).write_text(
        json.dumps({"ratings": ratings}, indent=2),  # no timestamp: a same top means no diff
        encoding="utf-8",
    )
    return removed


def publish(
    top: Sequence[tuple[str, Path, float]],
    repo: Path,
    *,
    folder_name: str = DEFAULT_DIR,
    remote: str = "origin",
    push: bool = True,
    message: str | None = None,
) -> str:
    """Copy, commit and (optionally) push. Returns a one-line outcome."""
    folder = repo / folder_name
    copy_models(top, folder)
    git(repo, "add", "-A", "--", folder_name)
    if git(repo, "diff", "--cached", "--quiet", "--", folder_name, check=False).returncode == 0:
        return "nulla da pubblicare: i top modelli sono gia' quelli dell'ultimo commit"
    title = message or f"Top {len(top)} modelli ({time.strftime('%Y-%m-%d')})"
    body = "\n".join(f"{rating:7.1f}  {label}" for label, _path, rating in top)
    # Only this folder is committed, whatever else is staged or modified.
    git(repo, "commit", "-m", f"{title}\n\n{body}", "--", folder_name)
    if not push:
        return "commit fatto, push saltato (--no-push)"
    pushed = git(repo, "push", remote, "HEAD", check=False)
    if pushed.returncode != 0:
        # The commit is already made and stays local; say why the push failed
        # instead of a traceback that hides git's own message.
        raise SystemExit(
            f"commit fatto, ma il push a {remote} e' fallito:\n{pushed.stderr.strip()}\n"
            "il commit resta in locale: sistema l'accesso e rilancia `git push`."
        )
    return f"commit fatto e inviato a {remote}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copia i modelli meglio classificati in una cartella tracciata da git, "
        "fa il commit e il push di quella cartella soltanto."
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="quanti modelli (default 5)")
    parser.add_argument("--dir", default=DEFAULT_DIR, help="cartella nel repository (default top_models)")
    parser.add_argument("--global-dir", type=Path, default=Path("checkpoints/global"))
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--message", default=None, help="titolo del commit")
    parser.add_argument("--no-push", action="store_true", help="solo il commit, niente push")
    parser.add_argument("--dry-run", action="store_true", help="mostra cosa farebbe e basta")
    args = parser.parse_args(argv)

    from pokerlab.cli.play import discover_global_top_models

    top = discover_global_top_models(args.global_dir, limit=args.limit)
    if not top:
        raise SystemExit("nessun modello nella classifica globale")
    repo = repo_root(Path.cwd())
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    print(f"top {len(top)} modelli -> {repo / args.dir} (branch {branch}, remote {args.remote}):")
    for label, path, rating in top:
        print(f"  {rating:7.1f}  {label}  ({Path(path).stat().st_size / 1e6:.1f} MB)")
    if args.dry_run:
        print("DRY RUN: nessun file copiato, nessun commit, nessun push.")
        return 0
    print(publish(top, repo, folder_name=args.dir, remote=args.remote,
                  push=not args.no_push, message=args.message))
    return 0


if __name__ == "__main__":
    sys.exit(main())
