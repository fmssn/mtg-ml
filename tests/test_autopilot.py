"""Reviewer control-flow regressions; no provider calls or GitHub writes."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

AP = Path(__file__).resolve().parents[1] / ".github/autopilot"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(AP))
    return tuple(importlib.import_module(name) for name in ("review_result", "runner", "context", "finish", "native"))


def report(**kw):
    return {"verdict": "clean", "summary": "Reviewed the change.", "findings": [],
            "escalation_reason": "", "reviewed_files": ["a.py"], "pending_checks": [], **kw}


def envelope(result=None, **kw):
    return {"type": "result", "subtype": "success", "is_error": False,
            "result": "```json\n" + json.dumps(report() if result is None else result) + "\n```", **kw}


@pytest.mark.parametrize("bad", [[], {"verdict": "unknown"}, report(pending_checks="no"),
                                  report(findings=[{"file": "a.py", "line": 1, "issue": "Bug", "status": "open"}])])
def test_invalid_contract_is_rejected(modules, bad):
    results, *_ = modules
    with pytest.raises(results.InvalidResult):
        results.parse_result(envelope(bad))


def test_legacy_and_structured_contracts_and_coverage(modules):
    results, *_ = modules
    old = {k: v for k, v in report().items() if k not in ("reviewed_files", "pending_checks")}
    parsed = results.parse_result(envelope(old))
    assert not results.completion_problems(parsed)
    manifest = {"version": 2, "files": [{"path": "a.py", "excluded": None}, {"path": "tests/data/x.json", "excluded": "generated"}]}
    assert results.completion_problems(parsed, manifest)
    assert not results.completion_problems(results.parse_result({"structured_output": report()}), manifest)
    assert "unreviewed files: a.py" in results.completion_problems(report(reviewed_files=[]), manifest)
    with pytest.raises(results.InvalidResult):
        results.parse_result(envelope(subtype="error_max_turns", is_error=True))


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    script = tmp_path / "claude"
    script.write_text(f"#!{sys.executable}\n" + """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
resumed = '--resume' in args
prompt = sys.stdin.read()
with Path(os.environ['TRACE']).open('a') as f:
    f.write(json.dumps({'args':args, 'prompt':prompt}) + '\\n')
print(json.dumps({'type':'system','subtype':'init','session_id':'same-session'}), flush=True)
scenario = os.environ['SCENARIO']
if scenario == 'api':
    print(json.dumps({'type':'result','subtype':'error_during_execution','is_error':True,'session_id':'same-session','api_error_status':401}))
    sys.exit(1)
if scenario == 'malformed' and not resumed:
    print(json.dumps({'type':'result','subtype':'success','is_error':False,'session_id':'same-session','result':'not JSON'}))
    sys.exit(0)
if not resumed and scenario in ('cap', 'unfinished', 'repeat-cap'):
    print(json.dumps({'type':'result','subtype':'error_max_turns','is_error':True,'session_id':'same-session','total_cost_usd':1}))
    sys.exit(1)
if resumed and scenario == 'repeat-cap':
    print(json.dumps({'type':'result','subtype':'error_max_turns','is_error':True,'session_id':'same-session'}))
    sys.exit(1)
result = {'verdict':'incomplete' if scenario == 'unfinished' else 'clean','summary':'Review result', 'findings':[], 'escalation_reason':'', 'reviewed_files':['a.py'], 'pending_checks':['native test'] if scenario == 'unfinished' else []}
print(json.dumps({'type':'result','subtype':'success','is_error':False,'session_id':'same-session','total_cost_usd':1.5,'result':'```json\\n'+json.dumps(result)+'\\n```'}))
""")
    script.chmod(0o755)
    monkeypatch.setenv("AUTOPILOT_CLAUDE_BIN", str(script))
    monkeypatch.setenv("AUTOPILOT_PROVIDER", "deepseek")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("TRACE", str(tmp_path / "trace.jsonl"))
    (tmp_path / "context.md").write_text("Exact snapshot and review evidence.")
    (tmp_path / "prompt.md").write_text("Review contract.")
    return tmp_path


@pytest.mark.parametrize("scenario,status,continued", [("cap", "complete", True), ("malformed", "complete", True), ("unfinished", "incomplete", True),
                                                        ("repeat-cap", "incomplete", True), ("api", "incomplete", False)])
def test_runner_continuation_and_failures(modules, fake_cli, monkeypatch, scenario, status, continued):
    _, runner, *_ = modules
    monkeypatch.setenv("SCENARIO", scenario)
    receipt = runner.execute(fake_cli / "prompt.md", "sonnet", "3", "5s")
    assert receipt["status"] == status and receipt["continued"] == continued
    trace = [json.loads(x) for x in (fake_cli / "trace.jsonl").read_text().splitlines()]
    assert "--bare" in trace[0]["args"] and "max" in trace[0]["args"]
    assert trace[0]["args"][trace[0]["args"].index("--max-turns") + 1] == "60"
    if continued:
        args = trace[1]["args"]
        assert args[args.index("--resume") + 1] == "same-session"
        assert args[args.index("--tools") + 1] == ""
        assert args[args.index("--max-turns") + 1] == "10"
        assert receipt["conversation_cost_estimate_usd"] in (None, 1.5)  # never 2.5
    if scenario == "api":
        assert receipt["failure_kind"] == "infrastructure"


def test_opus_keeps_oauth_configuration_and_legacy_arguments(modules, fake_cli, monkeypatch):
    _, runner, *_ = modules
    monkeypatch.setenv("AUTOPILOT_PROVIDER", "opus")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("SCENARIO", "success")
    assert runner.execute(fake_cli / "prompt.md", "opus", "10", "5s", "40")["status"] == "complete"
    args = json.loads((fake_cli / "trace.jsonl").read_text())["args"]
    assert "--bare" not in args and "--effort" not in args and "--max-budget-usd" in args


def test_resumed_telemetry_uses_conversation_totals_once(modules, fake_cli, monkeypatch):
    _, runner, *_ = modules
    phases = iter([
        {"exit_code": 1, "timed_out": False, "session_id": "same", "duration_seconds": 1,
         "result": envelope(subtype="error_max_turns", is_error=True, total_cost_usd=1,
                            permission_denials=[{"tool": "Bash"}], modelUsage={"deepseek": {"inputTokens": 100}})},
        {"exit_code": 0, "timed_out": False, "session_id": "same", "duration_seconds": 1,
         "result": envelope(total_cost_usd=1.5, modelUsage={"deepseek": {"inputTokens": 150}})},
    ])
    monkeypatch.setattr(runner, "run_phase", lambda *args: next(phases))
    receipt = runner.execute(fake_cli / "prompt.md", "sonnet", "3", "5s")
    assert receipt["conversation_usage"]["input_tokens"] == 150
    assert receipt["conversation_cost_estimate_usd"] == 1.5
    assert receipt["denied_calls"] == 1


def test_wall_deadline_terminates_running_process(modules, tmp_path):
    _, runner, *_ = modules
    command = [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"]
    phase = runner.run_phase(command, "", 0.2, tmp_path / "log", dict(os.environ))
    assert phase["timed_out"] and phase["exit_code"] != 0 and phase["duration_seconds"] < 2


@pytest.mark.parametrize("code", [
    "import os,time; os.close(1); os.close(2); time.sleep(30)",
    "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])",
    "import time; time.sleep(30)",
])
def test_deadline_survives_closed_stdout_or_exited_parent(modules, tmp_path, code):
    _, runner, *_ = modules
    phase = runner.run_phase([sys.executable, "-c", code], "x" * 200_000, 0.2, tmp_path / "log", dict(os.environ))
    assert phase["timed_out"] and phase["duration_seconds"] < 2


def test_manifest_records_full_patch_exclusions_and_native_need(modules, tmp_path, monkeypatch):
    _, _, context, *_ = modules
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    def git(*args):
        return subprocess.check_output(("git", *args), text=True).strip()
    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (repo / "a.py").write_text("old\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (repo / "a.py").write_text("new\n" * 40_000)
    (repo / "tests/data").mkdir(parents=True)
    (repo / "tests/data/generated.json").write_text("{}")
    git("add", ".")
    git("commit", "-qm", "change")
    tmp = tmp_path / "artifacts"
    manifest = context.write_context(tmp, "review", 1, base, "Title")
    assert manifest["needs_native"] and manifest["diff_bytes"] > manifest["embedded_diff_bytes"]
    assert manifest["files"][1]["excluded"] == "tests/data/**"
    assert "TRUNCATED" in (tmp / "context.md").read_text()
    assert len((tmp / "pr.diff").read_bytes()) > 120_000
    assert "generated.json" not in (tmp / "pr.diff").read_text()
    assert "generated.json" in (tmp / "pr-full.diff").read_text()


@pytest.mark.parametrize("mode", ["ci-fix", "escalate"])
def test_failed_ci_context_survives_escalation(modules, tmp_path, monkeypatch, mode):
    context = modules[2]
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("CI_RUN_ID", "77")
    monkeypatch.setattr(sys, "argv", ["context.py", mode, "1", "base"])
    calls, written = [], []
    def read(args, **kw):
        calls.append(args)
        return json.dumps({"title": "PR", "body": ""}) if args[1] == "pr" else "native\tcompile failed"
    monkeypatch.setattr(context.subprocess, "check_output", read)
    monkeypatch.setattr(context, "write_context", lambda *args: written.append(args))
    context.main()
    assert ("gh", "run", "view", "77", "--log-failed") in calls
    assert written[0][-2:] == ("native\tcompile failed", True)


def setup_guard(tmp_path, monkeypatch, finish, result=None, receipt=None, changed=()):
    monkeypatch.setattr(finish, "require_target", lambda *args, **kw: {})
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setattr(sys, "argv", ["finish.py", "fix", "1", "branch", "1", "base"])
    (tmp_path / "conflicts.txt").write_text("")
    (tmp_path / "claude.json").write_text(json.dumps(envelope(result) if result else {"subtype": "error_max_turns"}))
    if receipt:
        (tmp_path / "runner.json").write_text(json.dumps(receipt))
    calls = []
    monkeypatch.setattr(finish, "gh", lambda *args: calls.append(args))
    def sh(*args, **kw):
        if args[:3] == ("git", "diff", "--name-only"):
            return "\0".join(changed) + ("\0" if changed else "")
        if args[:3] == ("git", "rev-parse", "--git-path"):
            return str(tmp_path / "no-merge")
        if args[:3] == ("git", "rev-parse", "HEAD"):
            return "reviewed-sha"
        return ""
    monkeypatch.setattr(finish, "sh", sh)
    monkeypatch.setattr(finish.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, "", ""))
    return calls


@pytest.mark.parametrize("result,receipt", [(None, {"status": "incomplete", "failure_kind": "capacity"}),
                                             (report(), {"status": "incomplete", "failure_kind": "infrastructure", "reason": "401"}),
                                             (report(verdict="incomplete", pending_checks=["test"]), None)])
def test_incomplete_and_api_failure_never_escalate_or_merge(modules, tmp_path, monkeypatch, result, receipt):
    finish = modules[3]
    calls = setup_guard(tmp_path, monkeypatch, finish, result, receipt)
    assert finish.main() == 0
    assert any("--disable-auto" in c for c in calls)
    assert any("needs-human" in c for c in calls)
    assert not any("--auto" in c or ("--add-label" in c and "needs-opus" in c) for c in calls)


@pytest.mark.parametrize("path", ["tests/test_rules.py", ".github/workflows/ci.yml", "mtg_ml/engine/cards.toml"])
def test_sensitive_edit_still_escalates_without_push(modules, tmp_path, monkeypatch, path):
    finish = modules[3]
    calls = setup_guard(tmp_path, monkeypatch, finish, report(verdict="fixed"), changed=[path])
    assert finish.main() == 0
    assert any("--add-label" in c and "needs-opus" in c for c in calls)
    assert "outcome<<EOF_AUTOPILOT\nescalate" in (tmp_path / "output").read_text()


def test_auto_merge_matches_reviewed_head(modules, tmp_path, monkeypatch):
    finish = modules[3]
    calls = setup_guard(tmp_path, monkeypatch, finish, report())
    assert finish.main() == 0
    merge = next(c for c in calls if "--auto" in c)
    assert merge[-2:] == ("--match-head-commit", "reviewed-sha")


@pytest.fixture
def merge_policy(monkeypatch):
    monkeypatch.syspath_prepend(str(AP))
    policy = importlib.import_module("policy")
    responses = {
        "repo": {"nameWithOwner": "owner/repo", "defaultBranchRef": {"name": "main"}},
        "pr": {"state": "OPEN", "isDraft": False, "baseRefName": "main", "headRefOid": "reviewed-sha"},
        "branch": {"protected": True},
        "rules": [{"type": "pull_request"}, {"type": "required_status_checks", "parameters": {
            "strict_required_status_checks_policy": True,
            "required_status_checks": [{"context": name} for name in ("lint", "python", "native")],
        }}],
    }
    def read(*args):
        if args[0] in ("repo", "pr"):
            return responses[args[0]]
        return responses["rules" if "/rules/" in args[1] else "branch"]
    monkeypatch.setattr(policy, "read_json", read)
    return policy, responses


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("scenario", ["stack", "draft", "closed", "unprotected", "missing-check", "no-pr-rule", "retarget", "new-head"])
def test_merge_policy_rejects_unsafe_targets(merge_policy, scenario, strict):
    policy, data = merge_policy
    data["rules"][1]["parameters"]["strict_required_status_checks_policy"] = strict
    kwargs = {"expected_base": "main", "expected_head": "reviewed-sha"}
    if scenario == "stack":
        data["pr"]["baseRefName"] = "feature/parent"
    elif scenario == "draft":
        data["pr"]["isDraft"] = True
    elif scenario == "closed":
        data["pr"]["state"] = "MERGED"
    elif scenario == "unprotected":
        data["branch"]["protected"] = False
    elif scenario == "missing-check":
        data["rules"][1]["parameters"]["required_status_checks"].pop()
    elif scenario == "no-pr-rule":
        data["rules"].pop(0)
    elif scenario == "retarget":
        kwargs["expected_base"] = "old-parent"
    else:
        data["pr"]["headRefOid"] = "new-head"
    with pytest.raises(policy.PolicyError):
        policy.require_target(1, **kwargs)


@pytest.mark.parametrize("strict", [True, False])
def test_merge_policy_accepts_protected_current_default(merge_policy, strict):
    policy, data = merge_policy
    data["rules"][1]["parameters"]["strict_required_status_checks_policy"] = strict
    assert policy.require_target(1, expected_base="main", expected_head="reviewed-sha")["baseRefName"] == "main"


def test_policy_read_failure_is_closed(monkeypatch):
    monkeypatch.syspath_prepend(str(AP))
    policy = importlib.import_module("policy")
    def fail(*args, **kw):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(policy.subprocess, "check_output", fail)
    with pytest.raises(policy.PolicyError):
        policy.require_target(1)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("scenario", ["allowed", "stack", "draft", "closed", "foreign-author", "fork", "unprotected", "missing-check", "no-pr-rule", "api-failure"])
def test_workflow_gate_blocks_before_review(tmp_path, scenario, strict):
    workflow = (AP.parent / "workflows/pr-autopilot.yml").read_text()
    start = workflow.index('          info=$(gh pr view "$pr" --json state,isDraft')
    end = workflow.index('          labels=', start)
    guard = "\n".join(line[10:] for line in workflow[start:end].splitlines())
    fake = tmp_path / "gh"
    fake.write_text(f"#!{sys.executable}\n" + """import json, os, sys
args = sys.argv[1:]
scenario = os.environ['SCENARIO']
if scenario == 'api-failure':
    sys.exit(1)
if args[0] == 'pr':
    if args[1] == 'view':
        print(json.dumps({
            'state': 'CLOSED' if scenario == 'closed' else 'OPEN',
            'isDraft': scenario == 'draft',
            'baseRefName': 'parent' if scenario == 'stack' else 'main',
            'author': {'login': 'other' if scenario == 'foreign-author' else 'owner'},
            'headRepositoryOwner': {'login': 'other' if scenario == 'fork' else 'owner'},
        }))
        sys.exit(0)
    assert args == ['pr', 'merge', '1', '--disable-auto']
    sys.exit(0)
if '/rules/' in args[1]:
    checks = ['lint', 'python'] + ([] if scenario == 'missing-check' else ['native'])
    rules = [] if scenario == 'no-pr-rule' else [{'type':'pull_request'}]
    rules.append({'type':'required_status_checks', 'parameters': {
        'strict_required_status_checks_policy':os.environ['STRICT'] == 'true',
        'required_status_checks':[{'context':c} for c in checks]}})
    print(json.dumps(rules))
elif '/branches/' in args[1]:
    print('false' if scenario == 'unprotected' else 'true')
else:
    print('main')
""")
    fake.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"], SCENARIO=scenario,
               STRICT="true" if strict else "false", OWNER="owner", GH_REPO="o/r")
    script = 'set -euo pipefail\npr=1\nskip() { echo "held: $1"; exit 0; }\n' + guard + '\necho allowed\n'
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert ("allowed" in result.stdout) == (scenario == "allowed")
    assert (result.returncode != 0) == (scenario == "api-failure")


@pytest.mark.parametrize("fail_at", [1, 2])
def test_retarget_or_protection_change_holds_controller(modules, tmp_path, monkeypatch, fail_at):
    finish = modules[3]
    calls = setup_guard(tmp_path, monkeypatch, finish, report())
    inspected = []
    def require(*args, **kw):
        inspected.append(kw)
        if len(inspected) == fail_at:
            raise finish.PolicyError("Target changed")
    monkeypatch.setattr(finish, "require_target", require)
    assert finish.main() == 0
    assert not any("--auto" in call for call in calls)
    assert any("--disable-auto" in call for call in calls)
    assert "blocked" in (tmp_path / "output").read_text()
    if fail_at == 2:
        assert inspected[-1]["expected_head"] == "reviewed-sha"


def test_target_is_rechecked_before_push(modules, tmp_path, monkeypatch):
    finish = modules[3]
    calls = setup_guard(tmp_path, monkeypatch, finish, report())
    def deny(*args, **kw):
        raise finish.PolicyError("Stacked PR")
    monkeypatch.setattr(finish, "require_target", deny)
    git_calls = []
    monkeypatch.setattr(finish, "sh", lambda *args, **kw: git_calls.append(args))
    assert finish.push("head", "main", "1") == "blocked"
    assert not git_calls
    assert any("--disable-auto" in call for call in calls)


def test_failed_auto_merge_never_emits_success(modules, tmp_path, monkeypatch):
    finish = modules[3]
    calls = setup_guard(tmp_path, monkeypatch, finish, report())
    def gh(*args):
        calls.append(args)
        if "--auto" in args:
            raise subprocess.CalledProcessError(1, ["gh", *args])
    monkeypatch.setattr(finish, "gh", gh)
    with pytest.raises(subprocess.CalledProcessError):
        finish.main()
    assert not (tmp_path / "output").exists()


def test_missing_coverage_never_auto_merges(modules, tmp_path, monkeypatch):
    finish = modules[3]
    calls = setup_guard(tmp_path, monkeypatch, finish, report(reviewed_files=[]))
    (tmp_path / "manifest.json").write_text(json.dumps({"version": 2, "files": [{"path": "a.py"}]}))
    assert finish.main() == 0
    assert any("needs-human" in c for c in calls) and not any("--auto" in c for c in calls)


def test_failed_github_mutation_raises(modules, monkeypatch):
    finish = modules[3]
    def fail(*args, **kw):
        assert kw["check"] is True
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(finish.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        finish.gh("pr", "merge", "1", "--auto")


def test_native_failure_is_saved_and_blocks_completion(modules, tmp_path, monkeypatch):
    native = modules[4]
    (tmp_path / "manifest.json").write_text('{"needs_native":true}')
    monkeypatch.setattr(native.subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(args, 0, "", ""))
    def failed_build(tmp, env):
        (tmp / "native-build.log").write_text("compiler diagnostic")
        return {"exit_code": 1, "timed_out": False}
    monkeypatch.setattr(native, "run_build", failed_build)
    assert native.build(tmp_path)["status"] == "failed"
    assert (tmp_path / "native-build.log").read_text() == "compiler diagnostic"
    calls = setup_guard(tmp_path, monkeypatch, modules[3], report())
    assert modules[3].main() == 0
    assert any("needs-human" in c for c in calls) and not any("--auto" in c for c in calls)


@pytest.mark.parametrize("listed,content,staged", [
    (True, "resolved source\n", True),
    (True, "<<<<<<< HEAD\nold\n=======\nnew\n>>>>>>> base\n", False),
    (False, "resolved source\n", False),
])
def test_native_helper_stages_only_listed_resolved_conflicts(modules, tmp_path, monkeypatch, listed, content, staged):
    native = modules[4]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "manifest.json").write_text('{"needs_native":true}')
    (tmp_path / "conflicts.txt").write_text("source.py\n" if listed else "other.py\n")
    (tmp_path / "source.py").write_text(content)
    calls = []
    def run(args, **kw):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "source.py\0", "")
    monkeypatch.setattr(native.subprocess, "run", run)
    monkeypatch.setattr(native, "run_build", lambda *args: {"exit_code": 0, "timed_out": False})
    result = native.build(tmp_path)
    assert (result["status"] == "ready") == staged
    assert any(args[:2] == ("git", "add") for args in calls) == staged


def test_replay_uses_checked_out_pr_not_dispatch_workflow_sha(monkeypatch):
    monkeypatch.syspath_prepend(str(AP.parents[1] / "tools"))
    replay = importlib.import_module("autopilot_replay")
    head, base = "a" * 40, "b" * 40
    log = f"job\tdate [command]/usr/bin/git log -1 --format=%H\njob\tdate {head}\njob\tdate [command]/usr/bin/git log -1 --format=%H\njob\tdate {base}\n"
    assert replay.checkout_snapshot(log) == (head, base)
    with pytest.raises(ValueError, match="checkout SHAs"):
        replay.checkout_snapshot("No checkout receipt available")


def test_replay_controls_distinguish_separate_hidden_information_defects(monkeypatch):
    monkeypatch.syspath_prepend(str(AP.parents[1] / "tools"))
    replay = importlib.import_module("autopilot_replay")
    assert replay.matches_bug({"file": "mtg_ml/live.py", "issue": "Opponent Delver prompt leaks the top card"}, "hidden-card")
    assert not replay.matches_bug({"file": "mtg_ml/live.py", "issue": "The public seed reveals the hidden shuffled library"}, "hidden-card")
    assert replay.matches_bug({"file": "mtg_ml/live.py", "issue": "Shutdown drops the unsendable game on another thread"}, "native-shutdown")
    assert not replay.matches_bug({"file": "mtg_ml/live.py", "issue": "Browser close loses the last response"}, "native-shutdown")
