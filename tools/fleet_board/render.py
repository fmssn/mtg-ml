"""Render a fleet snapshot (collect.py output) plus fleet.toml into one self-contained HTML fragment.

The output follows the Artifact page rules: no <html>/<head>/<body>, everything inline, fonts only from
Google Fonts, no JS, charts are inline SVG built here. Pure functions; no ssh and no clock besides the
snapshot's own timestamp.
"""
from __future__ import annotations

import math
import re
from datetime import datetime
from html import escape as _e
from zoneinfo import ZoneInfo

# Role colour keys the CSS knows. A new key needs one entry in CSS_TOKENS (light and dark) and in .gpu/.seg/.p-* rules.
PALETTE = ("pilot", "base", "opt")

CSS = """
:root {
  --bg: #f3f4f1; --panel: #ffffff; --ink: #1d2320; --muted: #5d6762; --line: #d6dbd7;
  --idle: #e7eae7; --idle-ink: #7b857f;
  --pilot: #2f6f5e; --pilot-soft: #d7ebe4;
  --base: #8a5a1c; --base-soft: #f3e4cc;
  --opt: #4b5ea8; --opt-soft: #dfe4f6;
  --foreign: #9a3b3b; --foreign-soft: #f3dddd;
  --dead: #6d6d6d;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  --body: "IBM Plex Sans", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121614; --panel: #1a201d; --ink: #e4e9e6; --muted: #98a39d; --line: #2c3430;
  --idle: #232a26; --idle-ink: #7f8a84;
  --pilot: #5fb79d; --pilot-soft: #1f3a32; --base: #d9a25a; --base-soft: #3a2c18;
  --opt: #8d9fe6; --opt-soft: #252c47; --foreign: #e08585; --foreign-soft: #3d2222; --dead: #8a8a8a; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #121614; --panel: #1a201d; --ink: #e4e9e6; --muted: #98a39d; --line: #2c3430;
  --idle: #232a26; --idle-ink: #7f8a84;
  --pilot: #5fb79d; --pilot-soft: #1f3a32; --base: #d9a25a; --base-soft: #3a2c18;
  --opt: #8d9fe6; --opt-soft: #252c47; --foreign: #e08585; --foreign-soft: #3d2222; --dead: #8a8a8a; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font: 14px/1.5 var(--body); }
.wrap { max-width: 1180px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 48px; display: grid; grid-template-columns: minmax(0, 1fr); gap: 22px; }
h1 { font: 700 28px/1.1 var(--display); margin: 0; letter-spacing: .01em; text-wrap: balance; }
h2 { font: 600 18px/1.2 var(--display); margin: 0; }
.sub { color: var(--muted); margin: 4px 0 0; }
.mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 16px; font-size: 12.5px; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.streams { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; }
.stream { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; display: grid; gap: 6px; min-width: 0; }
.stream .tag { font: 600 11px/1 var(--display); letter-spacing: .08em; text-transform: uppercase; }
.stream p { margin: 0; color: var(--muted); font-size: 13px; }
.stream b { color: var(--ink); font-weight: 500; }
.machines { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(520px, 100%), 1fr)); gap: 16px; }
.box { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; display: grid; gap: 14px; min-width: 0; }
.boxhead { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 6px 12px; align-items: baseline; }
.role { font: 600 12px/1 var(--display); letter-spacing: .06em; text-transform: uppercase; padding: 5px 8px; border-radius: 4px; }
.label { font: 600 11px/1 var(--display); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.gpus { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; }
.gpu { border-radius: 6px; padding: 7px 8px; min-height: 64px; display: grid; gap: 2px; align-content: start; min-width: 0; border: 1px solid transparent; }
.gpu .bus { font: 500 11px/1 var(--mono); opacity: .8; }
.gpu .who { font: 600 12.5px/1.25 var(--display); overflow-wrap: anywhere; }
.gpu .st { font: 400 11px/1.3 var(--mono); opacity: .85; }
.gpu.idle { background: var(--idle); color: var(--idle-ink); }
.gpu.pilot { background: var(--pilot-soft); color: var(--ink); border-color: var(--pilot); }
.gpu.base { background: var(--base-soft); color: var(--ink); border-color: var(--base); }
.gpu.opt { background: var(--opt-soft); color: var(--ink); border-color: var(--opt); }
.gpu.foreign { background: var(--foreign-soft); color: var(--ink); border-color: var(--foreign); }
.gpu.dead { background: repeating-linear-gradient(135deg, var(--idle), var(--idle) 6px, var(--panel) 6px, var(--panel) 12px); color: var(--dead); }
.cpu { display: grid; gap: 6px; }
.lane { display: flex; height: 26px; border-radius: 5px; overflow: hidden; border: 1px solid var(--line); }
.seg { display: flex; align-items: center; padding-inline: 8px; font: 500 11.5px/1 var(--mono); white-space: nowrap; overflow: hidden; min-width: 0; }
.seg.pilot { background: var(--pilot); color: var(--panel); }
.seg.base { background: var(--base); color: var(--panel); }
.seg.idle { background: var(--idle); color: var(--idle-ink); }
.seg.opt { background: var(--opt); color: var(--panel); }
.ticks { display: flex; justify-content: space-between; font: 400 10.5px/1 var(--mono); color: var(--muted); }
.note { font-size: 12.5px; color: var(--muted); margin: 0; }
.down { border-left: 4px solid var(--foreign); padding: 8px 12px; background: var(--foreign-soft); border-radius: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
td, th { text-align: left; padding: 6px 8px; border-top: 1px solid var(--line); vertical-align: top; }
th { font: 600 11px/1.2 var(--display); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); border-top: 0; }
td.num { font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
.tablewrap { overflow-x: auto; }
.pill { display: inline-block; font: 600 11px/1.3 var(--display); letter-spacing: .04em; padding: 4px 7px; border-radius: 999px; }
.p-run { background: var(--pilot-soft); color: var(--pilot); }
.p-wait { background: var(--idle); color: var(--idle-ink); }
.p-base { background: var(--base-soft); color: var(--base); }
.p-opt { background: var(--opt-soft); color: var(--opt); }
.p-pilot { background: var(--pilot-soft); color: var(--pilot); }
.p-off { background: var(--foreign-soft); color: var(--foreign); }
footer { color: var(--muted); font-size: 12px; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr)); gap: 16px; }
.chart { margin: 0; min-width: 0; display: grid; gap: 6px; --c: var(--base); }
.chart figcaption { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 4px 12px; font-size: 12.5px; color: var(--muted); }
.chart figcaption b { font: 600 14px/1.2 var(--display); color: var(--ink); }
.chart svg { width: 100%; height: auto; display: block; }
.chart .axis { stroke: var(--line); stroke-width: 1; opacity: .5; }
.chart .tick { fill: var(--muted); font: 10.5px var(--mono); }
.chart .band { fill: var(--c); opacity: .18; }
.chart .lineL1 { fill: none; stroke: var(--c); stroke-width: 2; }
.chart .dot { fill: var(--c); }
.chart .ref { stroke: var(--muted); stroke-dasharray: 4 4; stroke-width: 1; }
.chart .reflabel { fill: var(--muted); font: 10.5px var(--body); }
@media (max-width: 600px) { .gpus { grid-template-columns: repeat(2, minmax(0, 1fr)); } h1 { font-size: 24px; } }
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600;700'
         '&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap">')


def e(x) -> str:
    return _e(str(x), quote=True)


def color_key(key: str | None, default: str = "opt") -> str:
    return key if key in PALETTE else default


# ---------------------------------------------------------------- formatting

def fmt_games(n) -> str:
    if n is None:
        return "-"
    if n >= 1e6:
        s = f"{n / 1e6:.2f}".rstrip("0").rstrip(".")
        return f"{s}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return str(int(n))


def fmt_target(n) -> str:
    if n is None:
        return ""
    return fmt_games(n)


def pct(x) -> str:
    return "-" if x is None else f"{round(x * 100)}%"


def run_title(run: dict) -> str:
    return run.get("handle") or run.get("name") or "?"


def describe(run: dict, rules: list[dict]) -> tuple[str, str]:
    """(table text, tile label) for a run from the first matching [[describe]] rule."""
    for rule in rules:
        if re.search(rule["match"], run_title(run)) or re.search(rule["match"], run.get("name", "")):
            return rule.get("text", ""), rule.get("short", "").replace("{name}", run_title(run)) or run_title(run)
    return "", run_title(run)


# ---------------------------------------------------------------- tiles

def gpu_tile(bus: str, gpu: dict | None, owner: tuple | None, note: dict | None, machine_color: str,
             rules: list[dict]) -> str:
    def tile(cls, who, st):
        return (f'<div class="gpu {cls}"><span class="bus">{e(bus)}</span><span class="who">{e(who)}</span>'
                f'<span class="st">{e(st)}</span></div>')

    stats = ""
    if gpu:
        lo, hi, u = gpu.get("util_min"), gpu.get("util_max"), gpu.get("util")
        parts = []
        if u is not None:
            parts.append(f"{lo:.0f}%" if lo == hi else f"{lo:.0f}-{hi:.0f}%")
        if gpu.get("mem_mb") is not None:
            parts.append(f"{gpu['mem_mb'] / 1024:.0f} GB")
        if gpu.get("power_w") is not None:
            parts.append(f"{gpu['power_w']:.0f} W")
        stats = " · ".join(parts)
    if owner:
        run, role = owner
        _, short = describe(run, rules)
        return tile(color_key(run.get("color"), machine_color), f"{short} · {role}", stats)
    if note:
        kind = note.get("kind")
        cls = kind if kind in ("foreign", "dead") else "idle"
        label = note.get("label") or ("Idle" if cls == "idle" else "")
        return tile(cls, label or "Unavailable", note.get("note", ""))
    if gpu is None:
        return tile("dead", "Missing", "not reported by nvidia-smi")
    if gpu.get("foreign"):
        return tile("foreign", f"Other: {gpu['foreign']}", stats)
    if (gpu.get("mem_mb") or 0) < 500 and (gpu.get("util") or 0) < 5:
        return tile("idle", "Idle", "free")
    return tile("idle", "In use", stats)


def cpu_lane(machine: dict, runs: list[dict], nproc: int, busy, color: str, rules: list[dict]) -> str:
    segs = []
    for r in runs:
        rg = r.get("cpus") or []
        if rg:
            segs.append((min(a for a, _ in rg), max(b for _, b in rg), r))
    segs.sort(key=lambda s: s[0])
    cells, pos, i = [], 0, 0
    for lo, hi, r in segs:
        lo = max(lo, pos)
        if hi < lo:
            continue
        if lo > pos:
            cells.append(f'<div class="seg idle" style="flex:{lo - pos}"></div>')
        _, short = describe(r, rules)
        c = color_key(r.get("color"), color)
        op = ("", ";opacity:.8", ";opacity:.65", ";opacity:.5")[i % 4]
        cells.append(f'<div class="seg {c}" style="flex:{hi - lo + 1}{op}" title="{e(short)} {lo}-{hi}">'
                     f'{e(short)} {lo}-{hi}</div>')
        pos, i = hi + 1, i + 1
    if pos < nproc:
        cells.append(f'<div class="seg idle" style="flex:{nproc - pos}"></div>')
    n = max(nproc - 1, 1)
    ticks = "".join(f"<span>{round(n * k / 4)}</span>" for k in range(5))
    unit = machine.get("cpu_label", "cores")
    load = f" · {busy:.0f}% busy" if busy is not None else ""
    return (f'<div class="cpu"><div class="label">CPU {e(unit)} 0-{nproc - 1}{load}</div>'
            f'<div class="lane">{"".join(cells)}</div><div class="ticks">{ticks}</div></div>')


def machine_panel(mcfg: dict, data: dict, rules: list[dict]) -> str:
    color = color_key(mcfg.get("color"))
    head = (f'<div class="boxhead"><h2>{e(mcfg["name"])}</h2>'
            f'<span class="role" style="background:var(--{color}-soft);color:var(--{color})">{e(mcfg.get("role", ""))}</span></div>')
    if data.get("status") != "ok":
        return (f'<article class="box">{head}<p class="note down"><b>Unreachable.</b> '
                f'{e(data.get("error", "no answer"))}. Showing no data for this machine.</p></article>')
    runs = data.get("runs", [])
    by_bus = {g["bus"]: g for g in data.get("gpus", [])}
    notes = {k.upper(): v for k, v in (mcfg.get("gpus") or {}).items()}
    owners = {}
    for r in runs:
        if r.get("learner_bus"):
            owners[r["learner_bus"]] = (r, "learner")
        for b in r.get("server_buses", []):
            owners.setdefault(b, (r, "server"))
    buses = sorted(set(by_bus) | set(notes) | set(owners))
    tiles = "".join(gpu_tile(b, by_bus.get(b), owners.get(b), notes.get(b), color, rules) for b in buses)
    nproc = int(data.get("nproc") or 0)
    busy = data.get("cpu_busy")
    cpu = ""
    if nproc:
        cpu = cpu_lane(mcfg, [r for r in runs if r.get("live")], nproc, busy, color, rules)
    return (f'<article class="box">{head}<div class="label">GPUs by PCI bus</div>'
            f'<div class="gpus">{tiles}</div>{cpu}</article>')


# ---------------------------------------------------------------- charts

STEPS = (1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200)


def chart_runs(snapshot: dict, config: dict) -> list[tuple[str, dict]]:
    cfg = config.get("charts", {})
    include, exclude = set(cfg.get("include", [])), set(cfg.get("exclude", []))
    out = []
    for mname, m in snapshot["machines"].items():
        for r in m.get("runs", []) if m.get("status") == "ok" else []:
            ids = {r.get("name"), r.get("handle")}
            if ids & exclude:
                continue
            has = any(p.get("elo") is not None for p in r.get("metrics", {}).get("evals", []))
            if has or ids & include:
                out.append((mname, r))
    return out


def svg_chart(run: dict, refs: list[dict]) -> str:
    pts = [p for p in run.get("metrics", {}).get("evals", []) if p.get("elo") is not None and p.get("games")]
    if not pts:
        return '<p class="note">No L1 Elo data yet.</p>'
    W, H, L, R, T, B = 520, 240, 44, 12, 12, 28
    gmax = max(p["games"] for p in pts)
    xmax = next((s * 1e6 for s in STEPS if s * 1e6 >= gmax * 1.02), gmax * 1.1)
    lo = min([p["elo"] - (p.get("se") or 0) for p in pts] + [r["value"] for r in refs])
    hi = max([p["elo"] + (p.get("se") or 0) for p in pts] + [r["value"] for r in refs])
    step = 25 if hi - lo < 120 else 50
    ymin, ymax = math.floor((lo - 5) / step) * step, math.ceil((hi + 5) / step) * step

    def x(g):
        return L + g / xmax * (W - L - R)

    def y(v):
        return T + (ymax - v) / (ymax - ymin) * (H - T - B)

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{e(run_title(run))} ladder L1 Elo over games">']
    v = ymin
    while v <= ymax:
        s.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                 f'<text class="tick" x="{L - 6}" y="{y(v) + 3.5:.1f}" text-anchor="end">{v:g}</text>')
        v += step
    for k in range(5):
        g = xmax * k / 4
        s.append(f'<text class="tick" x="{x(g):.1f}" y="{H - 8}" text-anchor="middle">{g / 1e6:g}M</text>')
    for ref in refs:
        s.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{y(ref["value"]):.1f}" y2="{y(ref["value"]):.1f}"/>'
                 f'<text class="reflabel" x="{W - R}" y="{y(ref["value"]) - 5:.1f}" text-anchor="end">'
                 f'{e(ref["label"])}: {ref["value"]:g}</text>')
    up = [f"{x(p['games']):.1f},{y(p['elo'] + (p.get('se') or 0)):.1f}" for p in pts]
    dn = [f"{x(p['games']):.1f},{y(p['elo'] - (p.get('se') or 0)):.1f}" for p in reversed(pts)]
    s.append(f'<polygon class="band" points="{" ".join(up + dn)}"/>')
    line = " ".join(f"{x(p['games']):.1f},{y(p['elo']):.1f}" for p in pts)
    s.append(f'<polyline class="lineL1" points="{line}"/>')
    for p in pts:
        r = 4 if p is pts[-1] else 2.5
        s.append(f'<circle class="dot" cx="{x(p["games"]):.1f}" cy="{y(p["elo"]):.1f}" r="{r}"/>')
    s.append("</svg>")
    return "".join(s)


def chart_section(snapshot: dict, config: dict) -> str:
    runs = chart_runs(snapshot, config)
    if not runs:
        return ""
    figs = []
    for mname, r in runs:
        # a reference line applies to runs whose name or handle matches its optional `match` regex
        refs = [x for x in config.get("reference", [])
                if re.search(x.get("match", ""), run_title(r)) or re.search(x.get("match", ""), r.get("name", ""))]
        mcolor = next((m.get("color") for m in config["machines"] if m["name"] == mname), None)
        c = color_key(r.get("color"), color_key(mcolor))
        pts = [p for p in r["metrics"].get("evals", []) if p.get("elo") is not None]
        sub = " · ".join(x for x in (f"width {r['hidden']}" if r.get("hidden") else "",
                                      f"lr {r['lr']}" if r.get("lr") else "") if x)
        cap = ""
        if pts:
            last = pts[-1]
            bench = next((p for p in reversed(r["metrics"]["evals"]) if p.get("bench") is not None), None)
            se = f" ± {last['se']:g}" if last.get("se") is not None else ""
            b = f" · bench {pct(bench['bench'])} / {pct(bench.get('greedy'))}" if bench else ""
            cap = f'<span class="mono">L1 {last["elo"]:g}{se} at {fmt_games(last["games"])}{b}</span>'
        figs.append(f'<figure class="chart" style="--c:var(--{c})"><figcaption><span><b>{e(run_title(r))}</b>'
                    f'{" · " + e(sub) if sub else ""}</span>{cap}</figcaption>{svg_chart(r, refs)}</figure>')
    note = config.get("charts", {}).get("note", "")
    return (f'<section class="box" aria-label="Strength"><div class="boxhead"><h2>Strength</h2>'
            f'<span class="label">Ladder L1 Elo vs games · ±1 SE band</span></div>'
            f'<div class="charts">{"".join(figs)}</div>'
            f'{f"<p class=note>{e(note)}</p>" if note else ""}</section>')


# ---------------------------------------------------------------- runs table

def status_cell(run: dict, mcolor: str) -> str:
    m = run.get("metrics", {})
    c = color_key(run.get("color"), mcolor)
    if run.get("live"):
        bits = ["running"]
        if m.get("win_vs_frozen") is not None:
            frm = f" (from {pct(m['win_vs_frozen_first'])})" if m.get("win_vs_frozen_first") is not None else ""
            bits.append(f"{pct(m['win_vs_frozen'])} vs frozen{frm}")
        pts = [p for p in m.get("evals", []) if p.get("elo") is not None]
        if pts:
            bits.append(f"L1 {pts[-1]['elo']:g}")
        if m.get("games_per_h"):
            bits.append(f"~{m['games_per_h'] / 1e6:.2f}M games/h")
        return f'<span class="pill p-{c}">{e(" · ".join(bits))}</span>'
    status = run.get("status") or "stopped"
    return f'<span class="pill {"p-wait" if status in ("finished", "done", "complete", "completed") else "p-off"}">{e(status)}</span>'


def runs_table(snapshot: dict, config: dict) -> str:
    rules = config.get("describe", [])
    rows = []
    for mcfg in config["machines"]:
        m = snapshot["machines"].get(mcfg["name"], {})
        if m.get("status") != "ok":
            rows.append(f'<tr><td colspan="5"><span class="pill p-off">{e(mcfg["name"])} unreachable</span></td></tr>')
            continue
        for r in m.get("runs", []):
            text, _ = describe(r, rules)
            games = fmt_games(r.get("metrics", {}).get("games_total"))
            if r.get("target_games"):
                games += f" / {fmt_target(r['target_games'])}"
            rows.append(f'<tr><td>{e(run_title(r))}</td><td>{e(mcfg["name"])}</td><td>{e(text)}</td>'
                        f'<td class="num">{e(games)}</td><td>{status_cell(r, color_key(mcfg.get("color")))}</td></tr>')
    if not rows:
        rows.append('<tr><td colspan="5">No runs found.</td></tr>')
    return ('<section class="box" aria-label="Runs"><h2>Runs</h2><div class="tablewrap"><table><thead><tr>'
            '<th>Run</th><th>Machine</th><th>What it does</th><th>Games</th><th>Status</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></section>')


# ---------------------------------------------------------------- page

def summary_line(snapshot: dict) -> str:
    ms = snapshot["machines"]
    up = [m for m in ms.values() if m.get("status") == "ok"]
    down = [n for n, m in ms.items() if m.get("status") != "ok"]
    gpus = sum(len(m.get("gpus", [])) for m in up)
    live = sum(1 for m in up for r in m.get("runs", []) if r.get("live"))
    busy = sum(1 for m in up for g in m.get("gpus", []) if (g.get("util") or 0) >= 5)
    s = f"{len(up)}/{len(ms)} machines up · {gpus} GPUs ({busy} busy) · {live} runs live"
    return s + (f" · unreachable: {', '.join(down)}" if down else "")


def render(snapshot: dict, config: dict) -> str:
    rules = config.get("describe", [])
    tz = ZoneInfo(config.get("timezone", "UTC"))
    when = datetime.fromisoformat(snapshot["collected_at"]).astimezone(tz)
    ms = snapshot["machines"]
    up = [m for m in ms.values() if m.get("status") == "ok"]
    gpus = sum(len(m.get("gpus", [])) for m in up)
    down = len(ms) - len(up)
    sub = (f'Generated <span class="mono">{when:%Y-%m-%d %H:%M} {e(tz.key.split("/")[-1])}</span>'
           f' · {len(up)} of {len(ms)} machines reachable{f" ({down} down)" if down else ""} · {gpus} GPUs · '
           'read from <span class="mono">nvidia-smi</span>, <span class="mono">/proc</span> and the run logs; a snapshot, not live')
    legend = "".join(f'<span><i class="sw" style="background:var(--{color_key(r["key"])})"></i>{e(r["label"])}</span>'
                     for r in config.get("roles", []))
    legend += ('<span><i class="sw" style="background:var(--foreign)"></i>Not ours</span>'
               '<span><i class="sw" style="background:var(--idle);border:1px solid var(--line)"></i>Idle</span>')
    streams = []
    for s in config.get("streams", []):
        c = s.get("color")
        edge, tagc = (f"var(--{c})", f"var(--{c})") if c in PALETTE else ("var(--line)", "var(--muted)")
        streams.append(f'<div class="stream" style="border-left:4px solid {edge}"><span class="tag" style="color:{tagc}">'
                       f'{e(s.get("title", ""))}</span><p><b>{e(s.get("headline", ""))}</b>'
                       f'{" " + e(s["text"]) if s.get("text") else ""}</p></div>')
    panels = "".join(machine_panel(m, ms.get(m["name"], {"status": "unreachable", "error": "not collected"}), rules)
                     for m in config["machines"])
    body = "\n".join([
        f'<header><h1>{e(config.get("title", "mtg-ml fleet board"))}</h1><p class="sub">{sub}</p></header>',
        f'<div class="legend" aria-label="Legend">{legend}</div>',
        f'<section class="streams" aria-label="Workstreams">{"".join(streams)}</section>' if streams else "",
        f'<section class="machines">{panels}</section>',
        chart_section(snapshot, config),
        runs_table(snapshot, config),
        f'<footer>{e(config["footer"])}</footer>' if config.get("footer") else "",
    ])
    return f'<title>mtg-ml Fleet Board</title>\n{FONTS}\n<style>{CSS}</style>\n<div class="wrap">\n{body}\n</div>\n'
