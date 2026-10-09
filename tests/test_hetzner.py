"""tools/hetzner.py (no network: `hcloud` is faked) and the deploy/ files."""

import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
DEPLOY = ROOT / "deploy"
SENTINEL = "SENTINEL-TOKEN-7f3a9c0b-do-not-leak"
PUBKEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMq9zLr1N2fakekeyforthetestsuite operator@laptop"
OURS = {"app": "mtg-play", "managed-by": "mtg-ml-hetzner", "stack": "mtg-play-1"}


@pytest.fixture()
def hz():
    sys.path.insert(0, str(TOOLS))
    try:
        yield importlib.import_module("hetzner")
    finally:
        sys.path.remove(str(TOOLS))


SERVER_TYPE = {
    "id": 115, "name": "cx33", "description": "CX 33", "cores": 4, "memory": 8, "disk": 80, "cpu_type": "shared",
    "architecture": "x86", "deprecated": False, "deprecation": None,
    "locations": [{"name": "fsn1", "available": True, "deprecation": None}],
    "prices": [{"location": "fsn1", "price_hourly": {"net": "0.0136", "gross": "0.0162"},
                "price_monthly": {"net": "8.4900", "gross": "10.1031"}}],
}


class FakeCloud:
    """Stands in for `hcloud`: answers reads from in-memory resources and
    records every invocation (argv and env)."""

    def __init__(self, **resources):
        self.res = {"ssh-key": [], "firewall": [], "server": []}
        self.res.update(resources)
        self.calls = []
        self.next_id = 1000

    def mutations(self):
        return [c["argv"][1:3] for c in self.calls if c["argv"][2] not in ("list", "describe")]

    def __call__(self, argv, env=None, capture_output=False, text=False, timeout=None, check=False):
        self.calls.append({"argv": list(argv), "env": dict(env or {})})
        kind, verb, rest = argv[1], argv[2], argv[3:]

        def ok(obj=None):
            return subprocess.CompletedProcess(argv, 0, json.dumps(obj) if obj is not None else "", "")

        def flag(name):
            return rest[rest.index(name) + 1]

        def labels():
            return dict(rest[i + 1].split("=", 1) for i, a in enumerate(rest) if a == "--label")

        if verb == "list":
            return ok(self.res[kind])
        if verb == "describe":
            if kind == "server-type":
                return ok(SERVER_TYPE)
            if kind == "image":
                return ok({"id": 161547269, "name": "ubuntu-24.04", "status": "available", "deprecated": None})
            if kind == "location":
                return ok({"id": 1, "name": "fsn1"})
            if kind == "server":
                return ok(next(s for s in self.res["server"] if str(s["id"]) == rest[0]))
        self.next_id += 1
        if verb == "create":
            item = {"id": self.next_id, "name": flag("--name"), "labels": labels()}
            if kind == "ssh-key":
                item["public_key"] = Path(flag("--public-key-from-file")).read_text().strip()
            elif kind == "firewall":
                item["rules"] = json.loads(Path(flag("--rules-file")).read_text())
            elif kind == "server":
                item.update(status="running", server_type={"name": flag("--type")},
                            public_net={"ipv4": {"ip": "192.0.2.10"}, "ipv6": {"ip": "2001:db8::/64"},
                                        "firewalls": [{"id": int(flag("--firewall"))}]})
                self.res[kind].append(item)
                return ok({"server": item, "root_password": None})
            self.res[kind].append(item)
            return ok({kind.replace("-", "_"): item})
        if verb == "delete":
            self.res[kind] = [r for r in self.res[kind] if str(r["id"]) != rest[0]]
            return ok()
        raise AssertionError(f"unexpected hcloud call {argv}")


@pytest.fixture()
def env(tmp_path, monkeypatch, hz):
    creds = tmp_path / "creds.env"
    creds.write_text(f"# test\nOTHER=1\nexport HETZNER_TOKEN=\"{SENTINEL}\"\n")
    key = tmp_path / "id.pub"
    key.write_text(PUBKEY + "\n")
    state = tmp_path / "cfg" / "hetzner.json"
    cloud = FakeCloud()
    monkeypatch.setattr(hz.subprocess, "run", cloud)
    common = ["--credentials", str(creds), "--state", str(state), "--hcloud", "hcloud"]
    plan_args = ["--ssh-public-key", str(key), "--admin-cidr", "203.0.113.7/32"]
    return {"cloud": cloud, "common": common, "plan": plan_args, "state": state, "creds": creds, "key": key}


def run(hz, capsys, argv):
    code = hz.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def assert_no_leak(env, out, err):
    assert SENTINEL not in out and SENTINEL not in err
    for c in env["cloud"].calls:
        assert all(SENTINEL not in a for a in c["argv"])
        assert c["env"]["HCLOUD_TOKEN"] == SENTINEL  # only via the child environment
    if env["state"].exists():
        assert SENTINEL not in env["state"].read_text()


@pytest.mark.parametrize("extra", [[], ["inspect"], ["plan"]])
def test_default_inspect_and_plan_only_read(hz, env, capsys, extra):
    argv = extra + env["common"] + (env["plan"] if extra == ["plan"] else [])
    code, out, err = run(hz, capsys, argv)
    assert code == 0, err
    assert env["cloud"].calls and env["cloud"].mutations() == []
    assert {c["argv"][2] for c in env["cloud"].calls} <= {"list", "describe"}
    assert "EUR 8.49/month net" in out
    assert not env["state"].exists()
    assert_no_leak(env, out, err)


def test_plan_lists_creations(hz, env, capsys):
    code, out, _ = run(hz, capsys, ["plan", *env["common"], *env["plan"]])
    assert code == 0
    assert out.count("CREATE") == 3 and "inbound tcp/22 from 203.0.113.7/32" in out


def test_apply_requires_explicit_flag(hz, env, capsys):
    code, out, err = run(hz, capsys, ["apply", *env["common"], *env["plan"]])
    assert code == 2 and "--yes-create-paid-resources" in err
    assert env["cloud"].calls == [] and not env["state"].exists()


def test_apply_creates_once_and_rerun_is_idempotent(hz, env, capsys):
    argv = ["apply", *env["common"], *env["plan"], "--yes-create-paid-resources"]
    code, out, err = run(hz, capsys, argv)
    assert code == 0, err
    assert env["cloud"].mutations() == [["ssh-key", "create"], ["firewall", "create"], ["server", "create"]]
    for c in env["cloud"].calls:
        if c["argv"][2] == "create":
            assert {"--label", "app=mtg-play", "managed-by=mtg-ml-hetzner", "stack=mtg-play-1"} <= set(c["argv"])
    fw = env["cloud"].res["firewall"][0]
    assert fw["rules"] == [{"direction": "in", "protocol": "tcp", "port": "22", "source_ips": ["203.0.113.7/32"],
                            "destination_ips": [], "description": "SSH from admin CIDR"}]
    state = json.loads(env["state"].read_text())
    assert set(state["ids"]) == {"ssh-key", "firewall", "server"}
    assert env["state"].stat().st_mode & 0o077 == 0
    assert_no_leak(env, out, err)

    env["cloud"].calls.clear()
    code, out, err = run(hz, capsys, argv)
    assert code == 0, err
    assert env["cloud"].mutations() == []
    assert out.count("KEEP") == 3


def test_unlabelled_same_name_conflict_refused(hz, env, capsys):
    env["cloud"].res["server"] = [{"id": 5, "name": "mtg-play-1", "labels": {}}]
    code, out, err = run(hz, capsys, ["apply", *env["common"], *env["plan"], "--yes-create-paid-resources"])
    assert code == 2 and "REFUSE" in out and "unlabelled server" in out
    assert env["cloud"].mutations() == []
    assert env["cloud"].res["server"] == [{"id": 5, "name": "mtg-play-1", "labels": {}}]


def test_registered_unlabelled_key_refused(hz, env, capsys):
    env["cloud"].res["ssh-key"] = [{"id": 9, "name": "laptop", "labels": {}, "public_key": PUBKEY}]
    code, out, _ = run(hz, capsys, ["plan", *env["common"], *env["plan"]])
    assert code == 2 and "already registered" in out


@pytest.mark.parametrize("cidr", ["0.0.0.0/0", "::/0", "not-a-cidr"])
def test_wide_open_or_invalid_cidr_refused(hz, env, capsys, cidr):
    argv = ["apply", *env["common"], "--ssh-public-key", str(env["key"]), "--admin-cidr", cidr,
            "--yes-create-paid-resources"]
    code, _, err = run(hz, capsys, argv)
    assert code == 2 and "admin-cidr" in err
    assert env["cloud"].mutations() == []


def test_private_key_refused(hz, env, capsys, tmp_path):
    priv = tmp_path / "id"
    priv.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nxxx\n-----END OPENSSH PRIVATE KEY-----\n")
    code, _, err = run(hz, capsys, ["plan", *env["common"], "--ssh-public-key", str(priv), "--admin-cidr", "10.0.0.1/32"])
    assert code == 2 and "private key" in err


def test_teardown_only_touches_labelled_and_saved(hz, env, capsys):
    cloud = env["cloud"]
    cloud.res["server"] = [
        {"id": 1, "name": "mtg-play-1", "labels": OURS},                 # ours + saved: delete
        {"id": 2, "name": "other", "labels": {}},                        # unlabelled: never
        {"id": 3, "name": "mtg-play-1-old", "labels": OURS},             # labelled, not saved: skip
    ]
    cloud.res["firewall"] = [{"id": 11, "name": "mtg-play-1-ssh", "labels": {}}]  # saved id but unlabelled: skip
    env["state"].parent.mkdir(parents=True)
    env["state"].write_text(json.dumps({"stack": "mtg-play-1", "ids": {"server": 1, "firewall": 11}}))

    code, out, err = run(hz, capsys, ["teardown", *env["common"]])
    assert code == 0 and "WOULD DELETE server id 1" in out and cloud.mutations() == []

    code, out, err = run(hz, capsys, ["teardown", *env["common"], "--yes-delete"])
    assert code == 0, err
    deletes = [c["argv"][1:4] for c in cloud.calls if c["argv"][2] == "delete"]
    assert deletes == [["server", "delete", "1"]]
    assert {s["id"] for s in cloud.res["server"]} == {2, 3}
    assert cloud.res["firewall"] == [{"id": 11, "name": "mtg-play-1-ssh", "labels": {}}]
    assert json.loads(env["state"].read_text())["ids"] == {"firewall": 11}
    assert_no_leak(env, out, err)


def test_missing_credentials_clear_error(hz, env, capsys, tmp_path):
    code, _, err = run(hz, capsys, ["--credentials", str(tmp_path / "nope.env"), "--state", str(env["state"])])
    assert code == 2 and "credential file not found" in err and "--credentials" in err
    assert env["cloud"].calls == []
    empty = tmp_path / "empty.env"
    empty.write_text("OTHER=1\n")
    code, _, err = run(hz, capsys, ["--credentials", str(empty)])
    assert code == 2 and "HETZNER_TOKEN is missing" in err


def test_hcloud_errors_are_redacted(hz, env, capsys, monkeypatch):
    def failing(argv, **kw):
        return subprocess.CompletedProcess(argv, 1, "", f"unauthorized: token {SENTINEL} invalid")
    monkeypatch.setattr(hz.subprocess, "run", failing)
    code, out, err = run(hz, capsys, env["common"])
    assert code == 1 and "unauthorized" in err and "[REDACTED]" in err
    assert SENTINEL not in out + err


def test_read_wrapper_refuses_mutating_verbs(hz):
    h = hz.Hcloud(SENTINEL)
    with pytest.raises(hz.Refused):
        h.call(["server", "create", "--name", "x"])
    with pytest.raises(hz.Refused):
        h.call(["server", "delete", "1"], mutate=True)  # allow_mutation is off


# ---------------------------------------------------------------- deploy/ files


def _compose_text():
    return (DEPLOY / "compose.yaml").read_text()


def test_compose_binds_loopback_readonly_models_bounded_logs():
    text = _compose_text()
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is None:  # text-level fallback
        assert '"127.0.0.1:8080:8080"' in text and "0.0.0.0" not in text
        assert "/srv/mtg-play/models:/models:ro" in text
        assert 'max-size: "10m"' in text and 'max-file: "3"' in text
        return
    services = yaml.safe_load(text)["services"]
    for name, svc in services.items():
        for port in svc.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), (name, port)
        log = svc["logging"]
        assert log["driver"] == "json-file" and log["options"]["max-size"] and log["options"]["max-file"]
        assert svc["read_only"] is True and svc["cap_drop"] == ["ALL"]
    app = services["app"]
    assert app["ports"] == ["127.0.0.1:8080:8080"]
    assert "/srv/mtg-play/models:/models:ro" in app["volumes"]
    assert "ports" not in services["cloudflared"]


def test_dockerfile_non_root_with_healthcheck():
    lines = [ln.strip() for ln in (DEPLOY / "Dockerfile").read_text().splitlines()]
    runtime = lines[max(i for i, ln in enumerate(lines) if ln.startswith("FROM ")):]
    users = [ln.split()[1] for ln in runtime if ln.startswith("USER ")]
    assert users and users[-1].split(":")[0] == "10001"
    assert any(ln.startswith("HEALTHCHECK ") for ln in runtime)
    assert any("/healthz" in ln for ln in runtime)
    assert 'CMD ["python", "-m", "mtg_ml.hosted", "serve"]' in runtime


def test_example_env_files_hold_placeholders_only():
    for name in ("app.env.example", "cloudflared.env.example"):
        for ln in (DEPLOY / name).read_text().splitlines():
            if ln and not ln.startswith("#") and "=" in ln:
                key, value = ln.split("=", 1)
                if key in ("MTG_ACCESS_AUD", "TUNNEL_TOKEN"):
                    assert value.startswith("<") and value.endswith(">")
