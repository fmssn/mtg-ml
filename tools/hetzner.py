#!/usr/bin/env python3
"""Hetzner Cloud provisioning for the hosted play server (docs/hosting.md).

    python tools/hetzner.py [inspect]                       read-only: token, catalog, our resources
    python tools/hetzner.py plan --ssh-public-key F --admin-cidr C    dry run of `apply`
    python tools/hetzner.py apply --ssh-public-key F --admin-cidr C --yes-create-paid-resources
    python tools/hetzner.py teardown [--yes-delete]         dry run unless --yes-delete

Creates one SSH key, one firewall (inbound TCP 22 from the admin CIDRs, nothing
else: web traffic arrives through the outbound Cloudflare tunnel) and one
server, each labelled app=mtg-play,managed-by=mtg-ml-hetzner,stack=<name>. Reruns
reuse labelled resources and create nothing twice; resources without our labels
are never changed, and a same-named unlabelled resource stops the run. Created
IDs are saved in a state file outside Git; teardown deletes only resources that
carry our labels and whose IDs are in that file.

The API token is read from a dotenv file (HETZNER_TOKEN, never executed) and
handed to `hcloud` only as HCLOUD_TOKEN in the child environment: never in
argv, never printed, redacted from any captured output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import ipaddress
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CREDENTIALS = Path("~/conductor/workspaces/mtg-ml/.env")
DEFAULT_USER_DATA = ROOT / "deploy" / "cloud-init.yaml"
TOKEN_KEY = "HETZNER_TOKEN"
BASE_LABELS = {"app": "mtg-play", "managed-by": "mtg-ml-hetzner"}
KINDS = ("ssh-key", "firewall", "server")
READ_VERBS = {"list", "describe"}
REDACTED = "[REDACTED]"
PUBLIC_KEY_TYPES = ("ssh-ed25519", "ssh-rsa", "ecdsa-sha2-", "sk-ssh-ed25519@openssh.com", "sk-ecdsa-sha2-")


class Refused(Exception):
    """A policy refusal: nothing was changed."""


class HcloudError(Exception):
    """`hcloud` exited non-zero (message already redacted)."""


# ---------------------------------------------------------------- credentials


def load_token(path: Path) -> str:
    """HETZNER_TOKEN from a dotenv file, parsed as data (no shell, no expansion)."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise Refused(f"credential file not found: {path} (expected a dotenv file with {TOKEN_KEY}=...; "
                      "pass --credentials PATH)")
    token = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep or key.strip() != TOKEN_KEY:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        token = value
    if not token:
        raise Refused(f"{TOKEN_KEY} is missing or empty in {path}")
    return token


# ---------------------------------------------------------------- hcloud wrapper


class Hcloud:
    """Runs `hcloud`. Read verbs only, unless `allow_mutation` was switched on
    by an explicitly confirmed apply/teardown."""

    def __init__(self, token: str, binary: str = "hcloud", allow_mutation: bool = False, timeout: float = 900):
        self._token = token
        self.binary = binary
        self.allow_mutation = allow_mutation
        self.timeout = timeout

    def redact(self, text: str) -> str:
        return text.replace(self._token, REDACTED) if self._token and text else (text or "")

    def _env(self) -> dict:
        env = dict(os.environ)
        env.pop("HCLOUD_CONTEXT", None)  # the token below, not a cli.toml context
        env["HCLOUD_TOKEN"] = self._token
        return env

    def call(self, args: list[str], mutate: bool = False) -> str:
        args = [str(a) for a in args]
        if any(self._token in a for a in args):
            raise Refused("internal error: the API token must never be passed as an argument")
        if mutate:
            if not self.allow_mutation:
                raise Refused(f"internal error: refusing mutating `hcloud {' '.join(args[:2])}` without an explicit flag")
        elif len(args) < 2 or args[1] not in READ_VERBS:
            raise Refused(f"internal error: `hcloud {' '.join(args[:2])}` is not a read-only command")
        try:
            proc = subprocess.run([self.binary, *args], env=self._env(), capture_output=True, text=True,
                                  timeout=self.timeout, check=False)
        except FileNotFoundError:
            raise HcloudError(f"hcloud binary not found: {self.binary} (install it or pass --hcloud PATH)") from None
        except subprocess.TimeoutExpired:
            raise HcloudError(f"hcloud {' '.join(args[:2])} timed out after {self.timeout:.0f}s") from None
        if proc.returncode != 0:
            detail = self.redact((proc.stderr or "") + (proc.stdout or "")).strip()
            raise HcloudError(f"hcloud {' '.join(args[:2])} failed ({proc.returncode}): {detail}")
        return proc.stdout or ""

    def json(self, args: list[str], mutate: bool = False):
        out = self.call([*args, "-o", "json"], mutate=mutate)
        try:
            return json.loads(out) if out.strip() else None
        except json.JSONDecodeError:
            raise HcloudError(f"hcloud {' '.join(args[:2])}: unexpected output: {self.redact(out)[:200]}") from None


# ---------------------------------------------------------------- inputs


def stack_labels(stack: str) -> dict[str, str]:
    return {**BASE_LABELS, "stack": stack}


def label_args(stack: str) -> list[str]:
    out = []
    for k, v in stack_labels(stack).items():
        out += ["--label", f"{k}={v}"]
    return out


def resource_names(stack: str) -> dict[str, str]:
    return {"ssh-key": f"{stack}-admin", "firewall": f"{stack}-ssh", "server": stack}


def is_ours(resource: dict, stack: str) -> bool:
    labels = resource.get("labels") or {}
    return all(labels.get(k) == v for k, v in stack_labels(stack).items())


def parse_cidrs(values: list[str] | None, allow_open: bool) -> list[str]:
    if not values:
        raise Refused("--admin-cidr is required (the only source allowed to reach SSH), e.g. 203.0.113.7/32")
    out = []
    for value in values:
        try:
            net = ipaddress.ip_network(value.strip(), strict=False)
        except ValueError:
            raise Refused(f"--admin-cidr {value!r} is not a valid CIDR") from None
        if net.prefixlen == 0 and not allow_open:
            raise Refused(f"--admin-cidr {net} opens SSH to the whole internet; refused "
                          "(use a narrow CIDR, or --allow-open-ssh if you really mean it)")
        out.append(str(net))
    return sorted(set(out))


def read_public_key(path: str | None) -> tuple[Path, str]:
    if not path:
        raise Refused("--ssh-public-key FILE is required (an OpenSSH public key, e.g. ~/.ssh/id_ed25519.pub)")
    p = Path(path).expanduser()
    if not p.is_file():
        raise Refused(f"SSH public key not found: {p}")
    text = p.read_text().strip()
    if "PRIVATE KEY" in text:
        raise Refused(f"{p} is a private key; pass the .pub file")
    if not text.startswith(PUBLIC_KEY_TYPES) or len(text.splitlines()) != 1:
        raise Refused(f"{p} does not look like a single OpenSSH public key")
    return p, text


def key_body(public_key: str) -> str:
    """Type and base64 part, ignoring the comment."""
    return " ".join(public_key.split()[:2])


def firewall_rules(cidrs: list[str]) -> list[dict]:
    return [{"direction": "in", "protocol": "tcp", "port": "22", "source_ips": list(cidrs),
             "destination_ips": [], "description": "SSH from admin CIDR"}]


def normalise_rules(rules: list[dict] | None) -> list[tuple]:
    out = []
    for r in rules or []:
        ips = sorted(str(ipaddress.ip_network(ip, strict=False)) for ip in r.get("source_ips") or [])
        out.append((r.get("direction"), r.get("protocol"), str(r.get("port") or ""), tuple(ips)))
    return sorted(out)


# ---------------------------------------------------------------- state file


def default_state_path(stack: str) -> Path:
    return Path("~/.config/mtg-ml").expanduser() / f"hetzner-{stack}.json"


def load_state(path: Path, stack: str) -> dict:
    if not path.is_file():
        return {"stack": stack, "ids": {}}
    data = json.loads(path.read_text())
    if data.get("stack") != stack:
        raise Refused(f"state file {path} belongs to stack {data.get('stack')!r}, not {stack!r}")
    data.setdefault("ids", {})
    return data


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    state = {**state, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            os.fchmod(f.fileno(), 0o600)
            json.dump(state, f, indent=1, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------- reads


def catalog(h: Hcloud, server_type: str, location: str, image: str) -> dict:
    st = h.json(["server-type", "describe", server_type])
    locs = {loc.get("name"): loc for loc in st.get("locations") or []}
    loc = locs.get(location)
    price = next((p for p in st.get("prices") or [] if p.get("location") == location), None)
    arch = st.get("architecture") or "x86"
    img = h.json(["image", "describe", "--architecture", arch, image])
    h.json(["location", "describe", location])
    if loc is not None:
        available = bool(loc.get("available", True)) and not loc.get("deprecation")
    else:  # older API shape: no per-location list, a price entry means it is sold there
        available = price is not None
    return {
        "server_type": st.get("name"), "description": st.get("description"), "cores": st.get("cores"),
        "memory_gb": st.get("memory"), "disk_gb": st.get("disk"), "cpu_type": st.get("cpu_type"),
        "architecture": arch, "deprecated": bool(st.get("deprecated") or st.get("deprecation")),
        "location": location, "available": available and price is not None,
        "monthly_net": price and price["price_monthly"]["net"], "monthly_gross": price and price["price_monthly"]["gross"],
        "hourly_net": price and price["price_hourly"]["net"],
        "image": img.get("name"), "image_id": img.get("id"), "image_status": img.get("status"),
        "image_deprecated": bool(img.get("deprecated")),
    }


def catalog_ok(cat: dict) -> list[str]:
    problems = []
    if not cat["available"]:
        problems.append(f"server type {cat['server_type']} is not orderable in {cat['location']} right now")
    if cat["deprecated"]:
        problems.append(f"server type {cat['server_type']} is deprecated")
    if cat["image_status"] != "available" or cat["image_deprecated"]:
        problems.append(f"image {cat['image']} is not available (status {cat['image_status']})")
    return problems


def format_catalog(cat: dict) -> str:
    def eur(x, digits=2):
        return "n/a" if x is None else f"EUR {float(x):.{digits}f}"
    return (f"catalog: {cat['server_type']} ({cat['cores']} vCPU {cat['cpu_type']}, {cat['memory_gb']} GB RAM, "
            f"{cat['disk_gb']} GB disk, {cat['architecture']}) in {cat['location']}: "
            f"{'available' if cat['available'] else 'NOT available'}; "
            f"{eur(cat['monthly_net'])}/month net ({eur(cat['monthly_gross'])} gross, hourly {eur(cat['hourly_net'], 4)} net; "
            f"primary IPv4 billed separately); image {cat['image']} (id {cat['image_id']}, {cat['image_status']})")


def discover(h: Hcloud, stack: str) -> dict:
    names = resource_names(stack)
    found = {}
    for kind in KINDS:
        items = h.json([kind, "list"]) or []
        found[kind] = {
            "all": items,
            "ours": [i for i in items if is_ours(i, stack)],
            "conflict": [i for i in items if i.get("name") == names[kind] and not is_ours(i, stack)],
        }
    return found


# ---------------------------------------------------------------- planning


def make_plan(found: dict, state: dict, stack: str, public_key: str, cidrs: list[str]) -> list[dict]:
    """One step per resource: create, keep, update-rules, attach-firewall or refuse."""
    names = resource_names(stack)
    ids = state.get("ids", {})
    steps = []
    for kind in KINDS:
        ours, conflict = found[kind]["ours"], found[kind]["conflict"]
        step = {"kind": kind, "name": names[kind]}
        if conflict:
            step.update(action="refuse", detail=f"an unlabelled {kind} named {names[kind]!r} exists "
                        f"(id {conflict[0].get('id')}); it is not ours, so nothing will touch it. "
                        "Rename/remove it yourself or use another --stack")
        elif len(ours) > 1:
            step.update(action="refuse", detail=f"{len(ours)} {kind}s carry our labels for stack {stack!r} "
                        f"(ids {[o.get('id') for o in ours]}); resolve by hand")
        elif ours and ids.get(kind) not in (None, ours[0].get("id")):
            step.update(action="refuse", detail=f"labelled {kind} id {ours[0].get('id')} differs from the saved id "
                        f"{ids.get(kind)}; check the state file")
        elif ours:
            step.update(action="keep", id=ours[0].get("id"), detail=f"exists with our labels (id {ours[0].get('id')})")
        else:
            gone = f"; saved id {ids[kind]} no longer exists" if ids.get(kind) else ""
            step.update(action="create", detail=f"will create{gone}")
        steps.append(step)
    by_kind = {s["kind"]: s for s in steps}

    # The SSH key: Hetzner rejects a second key with the same fingerprint.
    key_step = by_kind["ssh-key"]
    wanted = key_body(public_key)
    if key_step["action"] == "keep":
        mine = found["ssh-key"]["ours"][0]
        if key_body(mine.get("public_key", "")) != wanted:
            key_step.update(action="refuse", detail=f"our labelled key (id {mine.get('id')}) holds a different public key; "
                            "rotate it by hand or tear down first")
    elif key_step["action"] == "create":
        same = [k for k in found["ssh-key"]["all"] if key_body(k.get("public_key", "")) == wanted]
        if same:
            key_step.update(action="refuse", detail=f"this public key is already registered as {same[0].get('name')!r} "
                            f"(id {same[0].get('id')}) without our labels; Hetzner allows it once")

    fw_step = by_kind["firewall"]
    if fw_step["action"] == "keep":
        fw = found["firewall"]["ours"][0]
        if normalise_rules(fw.get("rules")) != normalise_rules(firewall_rules(cidrs)):
            fw_step.update(action="update-rules", detail=f"exists (id {fw.get('id')}); rules differ, will replace them "
                           f"with: inbound tcp/22 from {', '.join(cidrs)}")
    if fw_step["action"] in ("create", "update-rules", "keep"):
        fw_step["rules"] = f"inbound tcp/22 from {', '.join(cidrs)}; everything else inbound dropped"

    srv_step = by_kind["server"]
    if srv_step["action"] == "keep" and fw_step["action"] != "refuse":
        srv = found["server"]["ours"][0]
        attached = {a.get("id") for a in (srv.get("public_net") or {}).get("firewalls") or []}
        fw_id = fw_step.get("id")
        if fw_id is not None and fw_id not in attached:
            srv_step.update(action="attach-firewall", detail=f"exists (id {srv.get('id')}) without our firewall; will attach it")
        elif fw_id is None:
            srv_step.update(action="attach-firewall", detail=f"exists (id {srv.get('id')}); will attach the new firewall")
    return steps


def format_plan(steps: list[dict]) -> str:
    lines = []
    for s in steps:
        lines.append(f"  {s['action'].upper():16} {s['kind']:8} {s['name']}: {s['detail']}")
        if s.get("rules"):
            lines.append(f"  {'':16} {'':8} rules: {s['rules']}")
    return "\n".join(lines)


def describe_server(srv: dict) -> str:
    net = srv.get("public_net") or {}
    ipv4 = (net.get("ipv4") or {}).get("ip")
    ipv6 = (net.get("ipv6") or {}).get("ip")
    loc = (srv.get("location") or (srv.get("datacenter") or {}).get("location") or {}).get("name")
    return (f"id {srv.get('id')} {srv.get('name')} status={srv.get('status')} "
            f"type={(srv.get('server_type') or {}).get('name')} location={loc} ipv4={ipv4} ipv6={ipv6}")


# ---------------------------------------------------------------- commands


def cmd_inspect(h: Hcloud, args) -> int:
    print(f"stack {args.stack}: labels {','.join(f'{k}={v}' for k, v in stack_labels(args.stack).items())}")
    found = discover(h, args.stack)  # first API call: proves the token
    print("token: valid (read access confirmed)")
    cat = catalog(h, args.server_type, args.location, args.image)
    print(f"checked {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(format_catalog(cat))
    for p in catalog_ok(cat):
        print(f"WARNING: {p}")
    for kind in KINDS:
        f = found[kind]
        print(f"{kind}: {len(f['all'])} in project, {len(f['ours'])} with our labels")
        for r in f["ours"]:
            print(f"  ours: {describe_server(r) if kind == 'server' else 'id %s %s' % (r.get('id'), r.get('name'))}")
        for r in f["conflict"]:
            print(f"  CONFLICT: unlabelled {kind} named {r.get('name')!r} (id {r.get('id')})")
    state_path = args.state or default_state_path(args.stack)
    state = load_state(state_path, args.stack)
    print(f"state file {state_path}: {'saved ids ' + json.dumps(state['ids']) if state['ids'] else 'none saved'}")
    print("nothing was created or changed (read-only)")
    return 0


def _prepare(h: Hcloud, args):
    _, public_key = read_public_key(args.ssh_public_key)
    cidrs = parse_cidrs(args.admin_cidr, args.allow_open_ssh)
    user_data = Path(args.user_data).expanduser()
    if not user_data.is_file():
        raise Refused(f"user data file not found: {user_data}")
    state_path = args.state or default_state_path(args.stack)
    state = load_state(state_path, args.stack)
    cat = catalog(h, args.server_type, args.location, args.image)
    found = discover(h, args.stack)
    steps = make_plan(found, state, args.stack, public_key, cidrs)
    return cat, found, steps, state, state_path, cidrs, user_data


def cmd_plan(h: Hcloud, args) -> int:
    cat, _, steps, _, state_path, _, user_data = _prepare(h, args)
    print(format_catalog(cat))
    problems = catalog_ok(cat)
    for p in problems:
        print(f"BLOCKED: {p}")
    print(f"plan for stack {args.stack} ({args.server_type} in {args.location}, image {args.image}, "
          f"user data {user_data}, state {state_path}):")
    print(format_plan(steps))
    refused = any(s["action"] == "refuse" for s in steps)
    print("dry run: nothing was created or changed. "
          + ("apply would REFUSE (see above)." if refused or problems
             else "Run `apply ... --yes-create-paid-resources` to execute."))
    return 2 if refused or problems else 0


def cmd_apply(h: Hcloud, args) -> int:
    if not args.yes_create_paid_resources:
        raise Refused("apply creates billed Hetzner resources; rerun with --yes-create-paid-resources "
                      "(run `plan` first to see what it would do)")
    cat, found, steps, state, state_path, cidrs, user_data = _prepare(h, args)
    print(format_catalog(cat))  # rechecked just now
    problems = catalog_ok(cat)
    print(format_plan(steps))
    if problems or any(s["action"] == "refuse" for s in steps):
        for p in problems:
            print(f"BLOCKED: {p}")
        raise Refused("apply refused; nothing was created")
    by_kind = {s["kind"]: s for s in steps}
    names = resource_names(args.stack)
    h.allow_mutation = True
    ids = state["ids"]

    key = by_kind["ssh-key"]
    if key["action"] == "create":
        res = h.json(["ssh-key", "create", "--name", names["ssh-key"], "--public-key-from-file",
                      str(Path(args.ssh_public_key).expanduser()), *label_args(args.stack)], mutate=True)
        key["id"] = (res.get("ssh_key") or res)["id"]
        print(f"created ssh-key {names['ssh-key']} id {key['id']}")
    ids["ssh-key"] = key["id"]
    save_state(state_path, state)

    fw = by_kind["firewall"]
    if fw["action"] in ("create", "update-rules"):
        with tempfile.TemporaryDirectory() as tmp:
            rules_file = Path(tmp) / "rules.json"
            rules_file.write_text(json.dumps(firewall_rules(cidrs)))
            if fw["action"] == "create":
                res = h.json(["firewall", "create", "--name", names["firewall"], "--rules-file", str(rules_file),
                              *label_args(args.stack)], mutate=True)
                fw["id"] = (res.get("firewall") or res)["id"]
                print(f"created firewall {names['firewall']} id {fw['id']}")
            else:
                h.call(["firewall", "replace-rules", "--rules-file", str(rules_file), str(fw["id"])], mutate=True)
                print(f"replaced rules of firewall id {fw['id']}")
    ids["firewall"] = fw["id"]
    save_state(state_path, state)

    srv = by_kind["server"]
    if srv["action"] == "create":
        res = h.json(["server", "create", "--name", names["server"], "--type", args.server_type,
                      "--location", args.location, "--image", args.image, "--ssh-key", str(key["id"]),
                      "--firewall", str(fw["id"]), "--user-data-from-file", str(user_data),
                      *label_args(args.stack)], mutate=True)
        server = res.get("server") or res
        srv["id"] = server["id"]
        print(f"created server {describe_server(server)}")
    elif srv["action"] == "attach-firewall":
        h.call(["firewall", "apply-to-resource", "--type", "server", "--server", str(srv["id"]), str(fw["id"])],
               mutate=True)
        print(f"attached firewall id {fw['id']} to server id {srv['id']}")
    ids["server"] = srv["id"]
    server = h.json(["server", "describe", str(srv["id"])])
    net = server.get("public_net") or {}
    state["server"] = {"ipv4": (net.get("ipv4") or {}).get("ip"), "ipv6": (net.get("ipv6") or {}).get("ip")}
    save_state(state_path, state)
    print(f"server: {describe_server(server)}")
    print(f"state saved to {state_path}. Next: docs/hosting.md 'Server bootstrap'.")
    return 0


def cmd_teardown(h: Hcloud, args) -> int:
    state_path = args.state or default_state_path(args.stack)
    state = load_state(state_path, args.stack)
    found = discover(h, args.stack)
    ids = state["ids"]
    targets = []
    for kind in ("server", "firewall", "ssh-key"):  # dependency order
        for r in found[kind]["ours"]:
            if r.get("id") == ids.get(kind):
                targets.append((kind, r))
            else:
                print(f"  SKIP   {kind} id {r.get('id')} {r.get('name')}: labelled but not in {state_path}")
        if ids.get(kind) is not None and not any(r.get("id") == ids[kind] for r in found[kind]["ours"]):
            print(f"  SKIP   {kind} saved id {ids[kind]}: no resource with that id carries our labels")
    for kind, r in targets:
        print(f"  {'DELETE' if args.yes_delete else 'WOULD DELETE'} {kind} id {r.get('id')} {r.get('name')}")
    if not targets:
        print("nothing to delete")
    if not args.yes_delete:
        print("dry run: nothing was deleted. Rerun with --yes-delete to delete the resources listed above.")
        return 0
    h.allow_mutation = True
    for kind, r in targets:
        for attempt in range(6):  # a firewall stays "in use" briefly after its server is gone
            try:
                h.call([kind, "delete", str(r["id"])], mutate=True)
                break
            except HcloudError as e:
                if attempt == 5 or "in use" not in str(e).lower() and "resource_in_use" not in str(e):
                    raise
                time.sleep(5)
        ids.pop(kind, None)
        save_state(state_path, state)
        print(f"deleted {kind} id {r['id']}")
    return 0


COMMANDS = {"inspect": cmd_inspect, "plan": cmd_plan, "apply": cmd_apply, "teardown": cmd_teardown}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="inspect", choices=list(COMMANDS))
    p.add_argument("--credentials", default=str(DEFAULT_CREDENTIALS), help="dotenv file with HETZNER_TOKEN")
    p.add_argument("--hcloud", default=shutil.which("hcloud") or "hcloud", help="hcloud binary")
    p.add_argument("--stack", default="mtg-play-1", help="server name and `stack` label")
    p.add_argument("--state", type=Path, default=None, help="state file (default ~/.config/mtg-ml/hetzner-<stack>.json)")
    p.add_argument("--server-type", default="cx33")
    p.add_argument("--location", default="fsn1")
    p.add_argument("--image", default="ubuntu-24.04")
    p.add_argument("--user-data", default=str(DEFAULT_USER_DATA), help="cloud-init file passed to the server")
    p.add_argument("--ssh-public-key", default=None, help="OpenSSH public key file (plan/apply)")
    p.add_argument("--admin-cidr", action="append", default=None, help="source CIDR allowed to SSH (repeatable)")
    p.add_argument("--allow-open-ssh", action="store_true", help="permit 0.0.0.0/0 or ::/0 as admin CIDR")
    p.add_argument("--yes-create-paid-resources", action="store_true", help="apply: actually create billed resources")
    p.add_argument("--yes-delete", action="store_true", help="teardown: actually delete")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = None
    try:
        token = load_token(Path(args.credentials))
        h = Hcloud(token, binary=args.hcloud)
        return COMMANDS[args.command](h, args)
    except (Refused, HcloudError) as e:
        msg = str(e).replace(token, REDACTED) if token else str(e)
        print(f"{'refused' if isinstance(e, Refused) else 'error'}: {msg}", file=sys.stderr)
        return 2 if isinstance(e, Refused) else 1


if __name__ == "__main__":
    sys.exit(main())
