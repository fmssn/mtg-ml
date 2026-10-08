"""Exercise shared writers, PR refresh races, resource lifetimes and handoffs."""

import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def coordination(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    return importlib.import_module("progress")


@pytest.fixture
def tracker(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"title": "Existing tracker", "tracks": [], "custom": {"preserve": True}}))
    return path


def call(module, tracker, *args):
    return module.main(["--file", str(tracker), *args])


def claim(module, tracker, task_id="first", owner="workspace-a", path="src/engine"):
    assert call(module, tracker, "track", task_id, "--name", task_id, "--status", "running", "--owner", owner,
                "--workspace", str(tracker.parent / owner), "--branch", owner, "--write-path", path) == 0


def test_legacy_updates_preserve_unknown_fields_and_timestamp(coordination, tracker):
    assert call(coordination, tracker, "track", "legacy", "--name", "Legacy", "--step", "tests=done") == 0
    assert call(coordination, tracker, "result", "score", "0.5") == 0
    assert call(coordination, tracker, "log", "Existing", "caller") == 0
    data = json.loads(tracker.read_text())
    assert data["custom"] == {"preserve": True}
    assert data["tracks"][0]["updated_at"] and data["tracks"][0]["steps"][0]["done"]
    assert data["results"] == [{"name": "score", "value": "0.5"}]


def test_claims_reject_overlaps_and_wrong_owner_atomically(coordination, tracker):
    claim(coordination, tracker)
    before = tracker.read_bytes()
    assert call(coordination, tracker, "track", "second", "--name", "Second", "--status", "running", "--owner", "b",
                "--workspace", "/b", "--branch", "b", "--write-path", "src/engine/rules.py") == 1
    assert tracker.read_bytes() == before
    assert call(coordination, tracker, "track", "first", "--owner", "b", "--note", "steal") == 1
    assert tracker.read_bytes() == before
    assert call(coordination, tracker, "track", "first", "--owner", "workspace-a", "--status", "done") == 0
    claim(coordination, tracker, "second", "workspace-b", "src/engine/rules.py")


@pytest.mark.parametrize("path", ["../other", "/absolute", ".", "src/**/*.py", ""])
def test_claim_paths_are_bounded(coordination, path):
    with pytest.raises(ValueError):
        coordination.write_paths([path])


def test_parallel_legacy_writers_do_not_lose_updates(tracker):
    children = [subprocess.Popen([sys.executable, str(TOOLS / "progress.py"), "--file", str(tracker),
                                 "track", str(i), "--name", f"Task {i}"]) for i in range(8)]
    assert [child.wait(timeout=10) for child in children] == [0] * 8
    assert {t["id"] for t in json.loads(tracker.read_text())["tracks"]} == {str(i) for i in range(8)}


def test_github_refresh_preserves_task_work_and_concurrent_edits(coordination, tracker, monkeypatch):
    call(coordination, tracker, "track", "one", "--name", "One", "--status", "running", "--pr", "1", "--note", "Training outstanding")
    call(coordination, tracker, "track", "two", "--name", "Two", "--pr", "2")
    def github(*args):
        if args[2] == "1":
            call(coordination, tracker, "track", "one", "--note", "Newer local work")
            return {"number": 1, "url": "https://github.com/o/r/pull/1", "state": "MERGED", "isDraft": False}
        call(coordination, tracker, "track", "two", "--pr", "3")
        return {"number": 2, "url": "https://github.com/o/r/pull/2", "state": "CLOSED", "isDraft": False}
    monkeypatch.setattr(coordination, "github", github)
    assert coordination.reconcile(tracker, "o/r") is False
    one, two = json.loads(tracker.read_text())["tracks"]
    assert one["github"]["state"] == "MERGED" and one["status"] == "running"
    assert one["note"] == "Newer local work"
    assert two["pr"] == "3" and "github" not in two


def test_refresh_failure_marks_old_evidence_without_losing_it(coordination, tracker, monkeypatch):
    call(coordination, tracker, "track", "one", "--name", "One", "--pr", "1")
    with coordination.transaction(tracker) as data:
        data["tracks"][0]["github"] = {"state": "OPEN"}
    def fail(*args):
        raise subprocess.CalledProcessError(1, ["gh"])
    monkeypatch.setattr(coordination, "github", fail)
    assert coordination.reconcile(tracker, "o/r") is True
    item = json.loads(tracker.read_text())["tracks"][0]
    assert item["github"]["state"] == "OPEN" and item["github_error"]


def test_resource_exclusion_and_failed_command_release(coordination, tracker):
    claim(coordination, tracker)
    claim(coordination, tracker, "second", "workspace-b", "other")
    coordination.reserve(tracker, "first", "workspace-a", ["local:heavy"])
    with pytest.raises(ValueError, match="busy"):
        coordination.reserve(tracker, "second", "workspace-b", ["local:heavy", "gpu:0"])
    with pytest.raises(ValueError, match="another task"):
        coordination.release(tracker, "second", "workspace-b", ["local:heavy"])
    assert len(json.loads(tracker.read_text())["resources"]) == 1
    coordination.release(tracker, "first", "workspace-a", ["local:heavy"])
    assert coordination.run_reserved(tracker, "first", "workspace-a", ["local:heavy"],
                                     [sys.executable, "-c", "raise SystemExit(7)"]) == 7
    assert json.loads(tracker.read_text())["resources"] == []


def test_terminated_runner_releases_resource_and_stops_child(coordination, tracker, tmp_path):
    claim(coordination, tracker)
    child_pid = tmp_path / "child-pid"
    code = f"import os,time; from pathlib import Path; Path({str(child_pid)!r}).write_text(str(os.getpid())); time.sleep(30)"
    runner = subprocess.Popen([sys.executable, str(TOOLS / "progress.py"), "--file", str(tracker), "run", "first",
                               "--owner", "workspace-a", "--resource", "local:heavy", "--", sys.executable, "-c", code])
    try:
        deadline = time.monotonic() + 5
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_pid.exists()
        runner.send_signal(signal.SIGTERM)
        assert runner.wait(timeout=12) == 130
        assert json.loads(tracker.read_text())["resources"] == []
        with pytest.raises(ProcessLookupError):
            os.kill(int(child_pid.read_text()), 0)
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait()


@pytest.fixture
def handoff_repo(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    module = importlib.import_module("handoff")
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    git("init", "-q", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    (repo / ".gitignore").write_text(".context/\n")
    (repo / "source.py").write_text("original\n")
    git("add", ".")
    git("commit", "-qm", "base")
    git("branch", "base")
    (repo / ".context/plans").mkdir(parents=True)
    (repo / ".context/plans/approved.md").write_text("Implement the agreed feature.\n")
    monkeypatch.chdir(repo)
    assert module.main(["activate", "--plan", ".context/plans/approved.md", "--task", "task", "--owner", "workspace",
                        "--base", "base", "--write-path", "source.py", "--check", "make lint"]) == 0
    return module, repo, git


@pytest.mark.parametrize("change", ["plan", "tracked", "untracked", "branch", "head", "base", "missing-plan", "dependency"])
def test_handoff_detects_changed_starting_state(handoff_repo, change, monkeypatch):
    module, repo, git = handoff_repo
    assert module.main(["check"]) == 0
    if change == "plan":
        (repo / ".context/plans/approved.md").write_text("Different plan")
    elif change == "missing-plan":
        (repo / ".context/plans/approved.md").unlink()
    elif change == "tracked":
        (repo / "source.py").write_text("changed")
    elif change == "untracked":
        (repo / "new.py").write_text("new")
    elif change == "branch":
        git("switch", "-qc", "other")
    elif change in ("head", "base"):
        git("commit", "--allow-empty", "-qm", "later")
        if change == "base":
            new = git("rev-parse", "HEAD")
            git("reset", "--hard", "HEAD~1")
            git("branch", "-f", "base", new)
    else:
        path = repo / ".context/handoff.json"
        record = json.loads(path.read_text())
        record["repo"] = "o/r"
        record["dependencies"] = [{"number": 1, "headRefOid": "old"}]
        path.write_text(json.dumps(record))
        monkeypatch.setattr(module, "dependency", lambda *args: {"number": 1, "headRefOid": "new"})
    assert module.main(["check"]) == 1


def test_handoff_ignores_logs_and_selects_one_plan(handoff_repo):
    module, repo, _ = handoff_repo
    (repo / ".context/plans/old.md").write_text("Old plan")
    (repo / ".context/log.txt").write_text("Diagnostic output")
    assert module.main(["check"]) == 0
    record = json.loads((repo / ".context/handoff.json").read_text())
    assert record["plan"] == ".context/plans/approved.md"
    assert record["acceptance_checks"] == ["make lint"]
