"""File play-UI flags as GitHub issues (server side only).

A flag ("Bot played wrong" or "Bug: engine / UI") becomes one issue on the
project's repository, filed by the server: with `gh` when it is
authenticated, else through the REST API with a token from the environment
(`MTG_PLAY_GITHUB_TOKEN`; for hosting, a fine-grained personal access token
with "Issues: write" on this repository only). The token never reaches the
browser. Without either, the flag is only stored with the game's replay.

The repository is public, so an issue filed while the game runs holds only
what the flagging player could see (their own view of the board, the action
as it was shown to them, their description, a pseudonym). The seed and the
full choice list, which would reveal the bot's hand and library, are added
as a comment once the game has ended.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request

REPO = os.environ.get("MTG_PLAY_GITHUB_REPO", "fmssn/mtg-ml")
LABELS = {"bot": ("bot-play", "d93f0b", "A play-UI flag: the bot played wrong"), "bug": ("bug", "d73a4a", "Something isn't working")}
MAX_ISSUES_PER_GAME = 10


class FilingError(RuntimeError):
    pass


class GitHubFiler:
    """Files issues with `gh` (if authenticated) or the REST API (token in env)."""

    def __init__(self, repo: str = REPO, token: str | None = None):
        self.repo = repo
        self.token = token if token is not None else os.environ.get("MTG_PLAY_GITHUB_TOKEN") or None
        self._gh = None

    def _gh_ok(self) -> bool:
        if self._gh is None:
            self._gh = bool(shutil.which("gh")) and subprocess.run(["gh", "auth", "status"], capture_output=True, timeout=20).returncode == 0
        return self._gh

    def available(self) -> bool:
        return bool(self.token) or self._gh_ok()

    # -- REST
    def _api(self, method: str, path: str, body: dict) -> dict:
        req = urllib.request.Request(f"https://api.github.com/repos/{self.repo}{path}", data=json.dumps(body).encode(), method=method,
                                     headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json", "User-Agent": "mtg-ml-play"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code == 422 and path == "/labels":
                return {}  # the label exists
            raise FilingError(f"GitHub answered {e.code}") from None
        except OSError as e:
            raise FilingError(f"GitHub unreachable: {e}") from None

    def _ensure_label(self, category: str) -> str:
        name, color, desc = LABELS[category]
        if self.token:
            self._api("POST", "/labels", {"name": name, "color": color, "description": desc})
        else:
            subprocess.run(["gh", "label", "create", name, "-R", self.repo, "--color", color, "--description", desc], capture_output=True, timeout=30)
        return name

    def create(self, category: str, title: str, body: str) -> str:
        """The new issue's URL."""
        if not self.available():
            raise FilingError("no GitHub access on this server")
        label = self._ensure_label(category)
        if self.token:
            return self._api("POST", "/issues", {"title": title, "body": body, "labels": [label]})["html_url"]
        r = subprocess.run(["gh", "issue", "create", "-R", self.repo, "--title", title, "--body-file", "-", "--label", label],
                           input=body, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise FilingError(f"gh issue create failed: {r.stderr.strip()[:200]}")
        return r.stdout.strip().splitlines()[-1]

    def comment(self, url: str, body: str) -> None:
        if self.token:
            self._api("POST", f"/issues/{url.rstrip('/').rsplit('/', 1)[-1]}/comments", {"body": body})
            return
        r = subprocess.run(["gh", "issue", "comment", url, "--body-file", "-"], input=body, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise FilingError(f"gh issue comment failed: {r.stderr.strip()[:200]}")


def code_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10, cwd=os.path.dirname(__file__))
        return r.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def board_text(state: dict, seat: int, everything: bool = False) -> str:
    """The board as `seat` sees it (their snapshot): life, hands (theirs by
    name, the opponent's by count), battlefield, graveyards, stack."""
    who = lambda p: ("Player" if p == seat else "Bot") if everything else ("You" if p == seat else "Opponent")  # noqa: E731
    lines = []
    for p, P in enumerate(state["players"]):
        hand = ", ".join(c["name"] for c in P["hand"] if not c.get("hidden")) or "-"
        hidden = sum(1 for c in P["hand"] if c.get("hidden"))
        perms = ", ".join(c["name"] + (" (tapped)" if c.get("tapped") else "") + (f" {c['power']}/{c['toughness']}" if c.get("power") is not None else "")
                          for c in state["battlefield"] if c["controller"] == p) or "-"
        grave = ", ".join(c["name"] for c in P["graveyard"]) or "-"
        hand_txt = f"{hand}" if p == seat or everything else (f"{hidden} hidden" + (f" + known: {hand}" if hand != "-" else ""))
        lines += [f"**{who(p)}**: life {P['life']}, library {P['library']}", f"- hand: {hand_txt}", f"- battlefield: {perms}", f"- graveyard: {grave}"]
    if state["stack"]:
        lines.append("**Stack** (top last): " + "; ".join(f"{it['name']} ({who(it['controller'])})" for it in state["stack"]))
    return "\n".join(lines)


def rebuild(matchup: str, seed: int, seat: int, game_no: int, plan: str, start: int, choices: list[int], engine: str | None = None, scenario: str | None = None):
    """The game of a live session, rebuilt from its seed and choice list."""
    from .backend import game_class
    from .live import live_game_args

    if scenario:
        from .live_dev import new_game

        g = new_game(scenario, seed, engine)[0]
    else:
        g = game_class(engine)(**live_game_args(matchup, game_no, seat, plan), seed=seed, log=True, **({} if start is None else {"starting_player": start}))
    for c in choices:
        g.step(c)
    return g


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="mtg_ml.live_issues")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rebuild", help="replay a flagged live game from seed + choices and print how it ended")
    for a, t in (("--matchup", str), ("--seed", int), ("--seat", int), ("--game", int), ("--plan", str), ("--start", int), ("--engine", str), ("--scenario", str)):
        r.add_argument(a, type=t, default=None)
    r.add_argument("--choices", default="")
    args = ap.parse_args(argv)
    choices = [int(x) for x in args.choices.split(",") if x != ""]
    g = rebuild(args.matchup, args.seed, args.seat, args.game or 1, args.plan or "standard", args.start, choices, args.engine, args.scenario)
    print("\n".join(g.log[-30:]))
    print(f"turn {g.turn}, over: {g.over}, winner: {g.winner if g.over else None}")


if __name__ == "__main__":
    main()
