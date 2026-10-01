"""Publishing the top models, against a throwaway repository and a local remote."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pokerlab.rl.push_top_models import git, publish


def run(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    remote = tmp_path / "remote.git"
    run("git", "init", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    run("git", "init", "-b", "main", cwd=work)
    run("git", "config", "user.email", "t@example.com", cwd=work)
    run("git", "config", "user.name", "t", cwd=work)
    run("git", "remote", "add", "origin", str(remote), cwd=work)
    (work / "README.md").write_text("x", encoding="utf-8")
    run("git", "add", "README.md", cwd=work)
    run("git", "commit", "-m", "init", cwd=work)
    run("git", "push", "origin", "main", cwd=work)
    return work, remote


def models(tmp_path, *names):
    out = []
    for rating, name in names:
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode() * 10)
        out.append((name, path, rating))
    return out


def remote_files(remote: Path) -> set[str]:
    listing = subprocess.run(
        ["git", f"--git-dir={remote}", "ls-tree", "-r", "--name-only", "main"],
        capture_output=True, text=True, check=True,
    ).stdout
    return set(listing.split())


def test_the_top_models_are_committed_and_pushed(repo, tmp_path):
    work, remote = repo
    top = models(tmp_path, (1700.0, "a"), (1650.0, "b"))
    outcome = publish(top, work)
    assert "inviato" in outcome
    assert remote_files(remote) == {"README.md", "top_models/a.pt", "top_models/b.pt", "top_models/ratings.json"}


def test_a_model_that_left_the_top_is_removed_from_the_folder(repo, tmp_path):
    work, remote = repo
    publish(models(tmp_path, (1700.0, "a"), (1650.0, "b")), work)
    publish(models(tmp_path, (1710.0, "c"), (1700.0, "a")), work)
    assert remote_files(remote) == {"README.md", "top_models/a.pt", "top_models/c.pt", "top_models/ratings.json"}


def test_nothing_is_committed_when_the_top_has_not_changed(repo, tmp_path):
    work, _remote = repo
    top = models(tmp_path, (1700.0, "a"))
    publish(top, work)
    before = git(work, "rev-list", "--count", "HEAD").stdout
    outcome = publish(top, work)
    assert "nulla da pubblicare" in outcome
    assert git(work, "rev-list", "--count", "HEAD").stdout == before


def test_only_the_models_folder_is_committed_and_other_work_is_left_alone(repo, tmp_path):
    work, remote = repo
    (work / "README.md").write_text("changed", encoding="utf-8")
    (work / "wip.txt").write_text("not ready", encoding="utf-8")
    publish(models(tmp_path, (1700.0, "a")), work)
    assert "wip.txt" not in remote_files(remote)
    status = git(work, "status", "--porcelain").stdout
    assert " M README.md" in status and "?? wip.txt" in status


def test_no_push_leaves_the_commit_local(repo, tmp_path):
    work, remote = repo
    outcome = publish(models(tmp_path, (1700.0, "a")), work, push=False)
    assert "saltato" in outcome
    assert "top_models/a.pt" not in remote_files(remote)
    assert "top_models/a.pt" in git(work, "ls-files").stdout


def test_a_failed_push_keeps_the_commit_and_says_why(repo, tmp_path):
    work, _remote = repo
    run("git", "remote", "set-url", "origin", str(tmp_path / "does-not-exist.git"), cwd=work)
    with pytest.raises(SystemExit) as raised:
        publish(models(tmp_path, (1700.0, "a")), work)
    assert "push" in str(raised.value) and "fallito" in str(raised.value)
    assert "top_models/a.pt" in git(work, "ls-files").stdout  # the commit is there
