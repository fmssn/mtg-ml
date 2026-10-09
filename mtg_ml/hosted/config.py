"""The hosted server's configuration, read from the environment only.

| env | meaning |
|---|---|
| MTG_HOSTED_HOST / MTG_HOSTED_PORT | bind address (default 127.0.0.1 / 8080) |
| MTG_HOSTED_MODELS | read-only checkpoint directory (`--models` layout) |
| MTG_HOSTED_STATE | durable directory: play.sqlite3 and the token key live here |
| MTG_HOSTED_REPLAYS | durable replay directory |
| MTG_HOSTED_PLAY_CONFIG | the play offer (default: the packaged play_config.toml) |
| MTG_ENGINE | python or native |
| MTG_ACCESS_TEAM_DOMAIN | https://<team>.cloudflareaccess.com (the JWT issuer) |
| MTG_ACCESS_AUD | the Access application's audience tag |
| MTG_ALLOWED_EMAILS | comma-separated allowlist |
| MTG_HOSTED_PUBLIC_ORIGIN | https://play.example.com (JSON mutations must come from it) |
| MTG_HOSTED_INSECURE_DEV_USER | local testing only: trust this email without a JWT |
| MTG_HOSTED_GITHUB_FILING | 1: file flags as GitHub issues (default off) |
| MTG_GIT_REV | the code revision games are recorded with (the image has no .git; default: git, else the package version) |
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import urllib.parse
from dataclasses import dataclass, field


class ConfigError(ValueError):
    """The environment does not describe a usable server (one line per problem)."""


def _is_loopback(host: str | None) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


def normalize_origin(url: str, what: str, problems: list[str]) -> str:
    """scheme://host[:port] without a path; https unless the host is loopback."""
    u = urllib.parse.urlsplit(url.strip())
    if u.scheme not in ("https", "http") or not u.hostname or (u.path not in ("", "/")) or u.query or u.fragment:
        problems.append(f"{what} must be an origin like https://host (got {url!r})")
        return ""
    if u.scheme == "http" and not _is_loopback(u.hostname):
        problems.append(f"{what} must use https (http only for a loopback host)")
        return ""
    return f"{u.scheme}://{u.netloc}".lower()


def runtime_revision(env=None) -> str:
    """The code revision a game is recorded with: a restored game must run on the same one."""
    env = str((os.environ if env is None else env).get("MTG_GIT_REV", "") or "").strip()
    if env:
        return env
    try:
        r = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=10, cwd=os.path.dirname(__file__))
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except OSError:
        pass
    try:
        from importlib.metadata import version

        return f"mtg-ml {version('mtg-ml')}"
    except Exception:
        return "unknown"


@dataclass
class HostedConfig:
    models: pathlib.Path
    state: pathlib.Path
    replays: pathlib.Path
    play_config: pathlib.Path | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    engine: str | None = None
    team_domain: str = ""
    aud: str = ""
    allowed_emails: frozenset[str] = field(default_factory=frozenset)
    public_origin: str = ""  # "" only with dev_user: then the request's own host is the origin
    dev_user: str = ""
    github_filing: bool = False
    runtime: str = ""

    @property
    def dev(self) -> bool:
        return bool(self.dev_user)

    @classmethod
    def from_env(cls, env: dict | None = None) -> "HostedConfig":
        env = os.environ if env is None else env
        problems: list[str] = []

        def get(k: str) -> str:
            return str(env.get(k, "") or "").strip()

        def path(k: str) -> pathlib.Path | None:
            v = get(k)
            if not v:
                problems.append(f"{k} is required")
                return None
            return pathlib.Path(v)

        models, state, replays = path("MTG_HOSTED_MODELS"), path("MTG_HOSTED_STATE"), path("MTG_HOSTED_REPLAYS")
        dev_user = get("MTG_HOSTED_INSECURE_DEV_USER").lower()
        aud = get("MTG_ACCESS_AUD")
        team = get("MTG_ACCESS_TEAM_DOMAIN")
        emails = frozenset(e.strip().lower() for e in get("MTG_ALLOWED_EMAILS").split(",") if e.strip())
        origin = get("MTG_HOSTED_PUBLIC_ORIGIN")
        if dev_user:
            if aud:
                problems.append("MTG_HOSTED_INSECURE_DEV_USER is refused when MTG_ACCESS_AUD is set (a real deployment never trusts a fixed user)")
            if "@" not in dev_user:
                problems.append("MTG_HOSTED_INSECURE_DEV_USER must be an email address")
        else:
            for k, v in (("MTG_ACCESS_TEAM_DOMAIN", team), ("MTG_ACCESS_AUD", aud), ("MTG_ALLOWED_EMAILS", emails), ("MTG_HOSTED_PUBLIC_ORIGIN", origin)):
                if not v:
                    problems.append(f"{k} is required (or MTG_HOSTED_INSECURE_DEV_USER for local testing)")
        team = normalize_origin(team, "MTG_ACCESS_TEAM_DOMAIN", problems) if team else ""
        origin = normalize_origin(origin, "MTG_HOSTED_PUBLIC_ORIGIN", problems) if origin else ""
        port = get("MTG_HOSTED_PORT") or "8080"
        if not port.isdigit() or not 0 < int(port) < 65536:
            problems.append(f"MTG_HOSTED_PORT must be a port number (got {port!r})")
            port = "8080"
        engine = get("MTG_ENGINE") or None
        if engine not in (None, "python", "native"):
            problems.append(f"MTG_ENGINE must be python or native (got {engine!r})")
        filing = get("MTG_HOSTED_GITHUB_FILING")
        if filing not in ("", "0", "1"):
            problems.append("MTG_HOSTED_GITHUB_FILING must be 0 or 1")
        if problems:
            raise ConfigError("\n".join(problems))
        pc = get("MTG_HOSTED_PLAY_CONFIG")
        return cls(models=models, state=state, replays=replays, play_config=pathlib.Path(pc) if pc else None, host=get("MTG_HOSTED_HOST") or "127.0.0.1",
                   port=int(port), engine=engine, team_domain=team, aud=aud, allowed_emails=emails, public_origin=origin, dev_user=dev_user,
                   github_filing=filing == "1", runtime=runtime_revision(env))
