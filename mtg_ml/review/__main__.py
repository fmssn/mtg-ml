"""CLI: record, review, verify, calibrate, report. See docs/game-review.md."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

from . import faults as F
from .prompt import CAUSES, parse_findings, system_prompt, user_prompt
from .record import game_path, record_game
from .transcript import transcript, turn_chunks

MAX_CHARS = {"deepseek": 360_000, "claude": 1_500_000, "prompt": 1_500_000, "file": 1_500_000}


def _findings_path(game: pathlib.Path, backend: str) -> pathlib.Path:
    return game.with_name(game.stem + f".findings-{backend}.json")


def _record(args, fault_specs: list[str], tag: str = "") -> list[pathlib.Path]:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in range(args.seed, args.seed + args.games):
        fs = [F.parse(s) for s in fault_specs]
        rep = record_game(
            args.agents.split(","), seed, matchup=args.matchup, match_game=args.match_game,
            engine=args.engine, greedy=args.greedy, wrap=F.wrapper(fs, seed) if fs else None,
        )
        if getattr(args, "note", None):  # policy files carry no training state; say it here
            rep["meta"]["review"]["policies"] = [args.note if k.startswith("model:") else None for k in rep["meta"]["review"]["specs"]]
        for f in fs:
            if f["kind"] in ("life", "power"):
                F.tamper(rep, f)
        rep["meta"]["review"]["faults"] = fs
        path = game_path(out, rep, tag)
        path.write_text(json.dumps(rep, separators=(",", ":")))
        m = rep["meta"]
        print(f"{path}: {len(rep['frames'])} decisions, {m['turns']} turns, winner {m['winner']} ({m['end_reason']})"
              + "".join(f"; {f['spec']} at {len(f['decisions'])} decisions" for f in fs))
        paths.append(path)
    return paths


def review_file(path: pathlib.Path, backend: str, model: str | None = None, effort: str = "high", focus: str = "all") -> dict | None:
    from . import llm

    rep = json.loads(path.read_text())
    system = system_prompt()
    if backend == "file":
        # --model names whose reply it is: <game>.reply-<model>.txt
        reply = path.with_name(path.stem + (f".reply-{model}.txt" if model else ".reply.txt"))
        result = parse_findings(reply.read_text())
        result.update(backend="file", model=model, usage={})
    else:
        chunks = turn_chunks(rep, MAX_CHARS[backend])
        if backend == "prompt":
            text = user_prompt(transcript(rep), focus=focus)
            p = llm.write_prompt(path.with_name(path.stem + ".x"), system, text)
            print(f"{p}: {len(text) // 4:,} tokens (approx.); put the reply in {path.stem}.reply.txt, then --backend file")
            return None
        result = {"summary": [], "findings": [], "usage": [], "backend": backend, "model": model or llm.DEFAULTS[backend]}
        for part in chunks:
            text = user_prompt(transcript(rep, part if len(chunks) > 1 else None), part, len(chunks), focus=focus)
            reply, usage = llm.complete(backend, system, text, model=model, effort=effort)
            try:
                got = parse_findings(reply)
            except ValueError:
                bad = path.with_name(path.stem + f".unparsed-{model or backend}.txt")
                bad.write_text(reply)
                raise ValueError(f"no findings JSON in the reply (saved to {bad}; usage {usage})") from None
            result["summary"].append(got.get("summary", ""))
            result["findings"] += got["findings"]
            result["usage"].append(usage)
        result["summary"] = " / ".join(s for s in result["summary"] if s)
    result["game"] = path.name
    # a non-default model gets its own file (--backend <model> to verify/report it)
    out = _findings_path(path, model if model and model != llm.DEFAULTS.get(backend) else backend)
    out.write_text(json.dumps(result, indent=1))
    by = Counter(f["cause"] for f in result["findings"])
    print(f"{out}: {len(result['findings'])} findings {dict(by)}")
    return result


def _verify(path: pathlib.Path, backend: str, n: int) -> None:
    from .verify import verify

    fp = _findings_path(path, backend)
    result = json.loads(fp.read_text())
    rep = json.loads(path.read_text())
    verify(rep, result["findings"], n=n)
    fp.write_text(json.dumps(result, indent=1))
    for f in result["findings"]:
        if "rollout" in f:
            r = f["rollout"]
            extra = f" {r['win_chosen']:.2f} -> {r['win_alternative']:.2f} (diff {r['diff']:+.2f} ± {r['se']:.2f}, n {r['n']})" if "diff" in r else ""
            print(f"  d{f['decision']} [{f['cause']}] {r['verdict']}{extra}")


def report(paths: list[pathlib.Path], backend: str) -> str:
    """Markdown roll-up of the findings of many games."""
    rows, by_cause, sev = [], Counter(), Counter()
    games = 0
    for p in paths:
        fp = _findings_path(p, backend)
        if not fp.exists():
            continue
        games += 1
        res = json.loads(fp.read_text())
        for f in res["findings"]:
            by_cause[f["cause"]] += 1
            sev[f["cause"]] += int(f.get("severity") or 1)
            rows.append((p.name, f))
    lines = [f"# Game review: {games} games, {len(rows)} findings ({backend})", "", "| cause | findings | severity sum |", "|---|---|---|"]
    for c in CAUSES:
        if by_cause[c]:
            lines.append(f"| {c} | {by_cause[c]} | {sev[c]} |")
    for c in CAUSES:
        sel = [(g, f) for g, f in rows if f["cause"] == c]
        if not sel:
            continue
        lines += ["", f"## {c}", ""]
        for g, f in sorted(sel, key=lambda x: -int(x[1].get("severity") or 1)):
            ro = f.get("rollout", {})
            v = f" — rollouts: {ro['verdict']}" + (f" ({ro['diff']:+.2f} ± {ro['se']:.2f})" if "diff" in ro else "") if ro else ""
            lines.append(f"- **{g} d{f['decision']}** (P{f.get('seat')}, sev {f.get('severity')}, conf {f.get('confidence')}){v}: {f.get('what')}")
            if f.get("missing_play"):
                lines.append(f"  - missing: {f['missing_play']}")
            lines.append(f"  - evidence: {f.get('evidence')}")
            lines.append(f"  - fix: {f.get('fix')}")
    return "\n".join(lines) + "\n"


def calibration_report(paths: list[pathlib.Path], backend: str) -> str:
    caught, total, noise = Counter(), Counter(), 0
    lines = []
    for p in paths:
        fp = _findings_path(p, backend)
        if not fp.exists():
            continue
        rep = json.loads(p.read_text())
        found = json.loads(fp.read_text())["findings"]
        scored = F.score(rep["meta"]["review"]["faults"], found)
        hits = {id(x) for s in scored for x in s["findings"]}
        noise += sum(1 for f in found if id(f) not in hits)
        for s in scored:
            if s["caught"] is None:
                continue
            total[s["kind"]] += 1
            caught[s["kind"]] += s["caught"]
            lines.append(f"- {p.name}: {s['spec']} at d{s['decisions'][:5]} -> {'CAUGHT' if s['caught'] else 'missed'}")
    head = [f"# Reviewer calibration ({backend})", "", "| fault | caught / seeded |", "|---|---|"]
    head += [f"| {k} | {caught[k]} / {total[k]} |" for k in total]
    head += ["", f"Other findings (not matched to a seeded fault): {noise}", ""]
    return "\n".join(head + lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mtg_ml.review")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def play_args(p):
        p.add_argument("--agents", default="bot,bot", help="seat0,seat1 specs as in mtg_ml.replay (model:<checkpoint>, bot, ...)")
        p.add_argument("--games", type=int, default=1)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--matchup", default="jund_blue")
        p.add_argument("--match-game", type=int, default=1)
        p.add_argument("--engine", default=None)
        p.add_argument("--greedy", action="store_true")
        p.add_argument("--out", default="reviews/games")

    def llm_args(p):
        p.add_argument("--backend", default="deepseek", choices=("deepseek", "claude", "prompt", "file"))
        p.add_argument("--model", default=None, help="deepseek-chat / deepseek-reasoner / claude-opus-5-5 / claude-sonnet-5-5")
        p.add_argument("--effort", default="high", help="Claude effort level")
        p.add_argument("--focus", default="all", choices=("all", "setup"), help="setup: engine, mask, feature and architecture flaws first, with a sheet of what the policy observes")

    r = sub.add_parser("record", help="play games and write review files")
    play_args(r)
    r.add_argument("--fault", action="append", default=[], help="seed a fault (mtg_ml.review.faults); repeatable")
    r.add_argument("--note", default=None, help="describe the model seats' policy for the reviewer (run, iteration, training games, feature set, trunk)")
    v = sub.add_parser("review", help="have an LLM review game files")
    v.add_argument("games", nargs="+", type=pathlib.Path)
    llm_args(v)
    ve = sub.add_parser("verify", help="counterfactual rollouts for findings that name a better option")
    ve.add_argument("games", nargs="+", type=pathlib.Path)
    ve.add_argument("--backend", default="deepseek", help="which findings: the backend, or the --model a review ran with")
    ve.add_argument("--rollouts", type=int, default=32)
    rp = sub.add_parser("report", help="markdown roll-up of findings")
    rp.add_argument("games", nargs="+", type=pathlib.Path)
    rp.add_argument("--backend", default="deepseek", help="which findings: the backend, or the --model a review ran with")
    rp.add_argument("--calibration", action="store_true", help="score findings against the seeded faults")
    c = sub.add_parser("calibrate", help="record games with seeded faults and review them")
    play_args(c)
    llm_args(c)
    c.add_argument("--fault", action="append", default=None, help="default: mask:0:blocks, blunder:1:2, life:1:6:-3 (and power:0:8:+2 on odd seeds)")
    args = ap.parse_args(argv)

    if args.cmd == "record":
        _record(args, args.fault)
    elif args.cmd == "review":
        for p in args.games:
            try:
                review_file(p, args.backend, args.model, args.effort, args.focus)
            except (ValueError, OSError) as e:
                print(f"{p}: FAILED: {e}")
    elif args.cmd == "verify":
        for p in args.games:
            print(p)
            _verify(p, args.backend, args.rollouts)
    elif args.cmd == "report":
        print(calibration_report(args.games, args.backend) if args.calibration else report(args.games, args.backend))
    elif args.cmd == "calibrate":
        if args.engine == "native":
            raise SystemExit("option faults filter the Python engine's option lists; use --engine python")
        args.engine = "python"
        paths = []
        base_seed, n = args.seed, args.games
        for seed in range(base_seed, base_seed + n):
            args.seed, args.games = seed, 1
            specs = args.fault or ["mask:0:blocks", "blunder:1:2", "life:1:6:-3"] + (["power:0:8:2"] if seed % 2 else [])
            paths += _record(args, specs, tag="-faults")
        for p in paths:
            review_file(p, args.backend, args.model, args.effort, args.focus)
        if args.backend != "prompt":
            print(calibration_report(paths, args.model or args.backend))


if __name__ == "__main__":
    main()
