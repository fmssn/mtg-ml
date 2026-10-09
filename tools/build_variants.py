"""Freeze the deck-variant manifest (mtg_ml/engine/variants.toml).

Source: the local Pauper-Research snapshot's dashboard
(`reports/dashboard/index.html`, the embedded JSON), period 90d
(2026-07-05 to 2026-10-02), scope "combined" (all events): per archetype,
"Builds and decisions" lists the main-deck builds and the sideboard builds
separately, each with a typical list, its number of lists and its share.
These are the same numbers `decks.py` cites (e.g. Jund sideboard "Stock
list", 124 of 428 lists).

Rules (docs/deck-variants.md):
- Names are normalised through ALIASES only; nothing is substituted.
- A build with any card not in `cards.CARDS` is rejected (with its missing
  cards); so is every combination with it. Nothing is dropped from a list.
- Each accepted main build is combined with each accepted sideboard build.
  Main and sideboard are reported independently, so every combination is
  `constructed = true` and weighs main share x side share (independence).
- A combination that breaks 60 / 15 / 4-copies is rejected with the reason.
- The stock deck (`decks.DECKS` + `SIDEBOARDS`) is the variant
  `<archetype>:stock`. When it equals a combination it takes that
  combination's weight; otherwise (its main or side is not a snapshot build)
  it gets the largest accepted weight of its archetype (`weight_basis =
  "policy:max"`).
- Split per archetype by the 75's hash (`variants.assign_splits`); stock always train.
- Plans: the stock row of `sideboard_plans.toml` is reused when valid for
  the variant; otherwise an adjusted plan is stored (see `adapt_plan`).

Usage:
    .venv/bin/python tools/build_variants.py            # write the manifest
    .venv/bin/python tools/build_variants.py --check    # fail if it would change
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtg_ml.engine.cards import CARDS  # noqa: E402
from mtg_ml.engine.decks import DECKS, SIDEBOARDS  # noqa: E402
from mtg_ml.engine.sideboard import PLANS, SideboardPlan, check_deck, validate_plan  # noqa: E402
if "mtg_ml.engine.variants" in sys.modules:
    from mtg_ml.engine import variants as V  # noqa: E402
else:  # the manifest may be missing or stale while rebuilding: use the helpers only
    os.environ["MTG_ML_VARIANTS_SKIP_LOAD"] = "1"
    try:
        from mtg_ml.engine import variants as V  # noqa: E402
    finally:
        del os.environ["MTG_ML_VARIANTS_SKIP_LOAD"]
        # Forget the helpers-only module so a later plain import loads the manifest.
        del sys.modules["mtg_ml.engine.variants"]
        delattr(sys.modules["mtg_ml.engine"], "variants")

DEFAULT_SNAPSHOT = "/Users/fabsi/repos/mtg-ml-orchestration/pauper-research"
DASHBOARD = "reports/dashboard/index.html"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mtg_ml", "engine", "variants.toml")

# Snapshot archetype name -> decks.DECKS key.
ARCHETYPES = {
    "Jund Midrange": "jund_wildfire",
    "Mono Blue Terror": "mono_blue_terror",
    "Red Madness": "red_madness",
    "Grixis Affinity": "grixis_affinity",
    "Elves": "elves",
    "Tron": "tron",
}

# Established spellings of the same card (oracle name on the right). The
# snapshot already folds Snow-Covered basics into the basic.
ALIASES = {
    "Lórien Revealed": "Lorien Revealed",
}

# Functionally equivalent sideboard cards, used only to adapt a stock plan to
# a variant whose sideboard has the other one (never to change a list).
PLAN_EQUIVALENTS = {
    "Red Elemental Blast": ("Pyroblast",),
    "Pyroblast": ("Red Elemental Blast",),
    "Blue Elemental Blast": ("Hydroblast",),
    "Hydroblast": ("Blue Elemental Blast",),
    "Annul": ("Steel Sabotage",),  # both counter an artifact spell
}


def normalise_name(name: str) -> str:
    name = ALIASES.get(name, name)
    if " // " in name and name not in CARDS:
        name = name.split(" // ")[0]
    return name


def normalise_list(pairs) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, n in pairs:
        c = normalise_name(name)
        out[c] = out.get(c, 0) + int(n)
    return out


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def load_dashboard(root: str) -> dict:
    with open(os.path.join(root, DASHBOARD), encoding="utf-8") as f:
        html = f.read()
    tag = '<script id="data" type="application/json">'
    a = html.index(tag) + len(tag)
    return json.loads(html[a:html.index("</script>", a)])


def snapshot_meta(root: str, dash: dict, period: str, scope: str) -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    with open(os.path.join(root, DASHBOARD), "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    p = dash["periods"][period]
    return {
        "repo": "Pauper-Research (fmssn), local snapshot",
        "path": DASHBOARD,
        "commit": git("log", "-1", "--format=%H", "--", DASHBOARD),
        "commit_date": git("log", "-1", "--format=%cs", "--", DASHBOARD),
        "file_sha256": digest,
        "dashboard_generated": dash.get("generated", "unknown"),
        "period": period,
        "since": p["since"],
        "until": p["until"],
        "scope": scope,
        "section": "Builds and decisions",
    }


def adapt_plan(stock: SideboardPlan, main: Mapping[str, int], side: Mapping[str, int],
               stock_main: Mapping[str, int]) -> tuple[dict, dict, str]:
    """The stock plan fitted to a variant's lists, deterministically:

    - each `in` card up to what the sideboard has, the shortfall from its
      PLAN_EQUIVALENTS (in their order);
    - each `out` card up to what the maindeck has; the shortfall from the
      variant's slot replacements: nonland cards it plays more copies of than
      the stock maindeck (most extra copies first, then by name), on the view
      that they fill the slots of the stock cards the plan would cut;
    - then both trimmed from the end to the same total."""
    side_left = dict(side)
    cards_in: dict[str, int] = {}
    notes = []
    for name, n in stock.cards_in.items():
        take = min(n, side_left.get(name, 0))
        if take:
            cards_in[name] = cards_in.get(name, 0) + take
            side_left[name] -= take
        short = n - take
        for eq in PLAN_EQUIVALENTS.get(name, ()):
            if short <= 0:
                break
            t = min(short, side_left.get(eq, 0))
            if t:
                cards_in[eq] = cards_in.get(eq, 0) + t
                side_left[eq] -= t
                short -= t
                notes.append(f"{t} {eq} for {name}")
        if short > 0:
            notes.append(f"{short} {name} not in this sideboard")
    main_left = dict(main)
    cards_out: dict[str, int] = {}
    for name, n in stock.cards_out.items():
        take = min(n, main_left.get(name, 0))
        if take:
            cards_out[name] = cards_out.get(name, 0) + take
            main_left[name] -= take
        if take < n:
            notes.append(f"only {take} of {n} {name} in this maindeck")
    short = sum(stock.cards_out.values()) - sum(cards_out.values())
    extras = sorted((c for c in main if "Land" not in CARDS[c].types and main[c] > stock_main.get(c, 0)),
                    key=lambda c: (stock_main.get(c, 0) - main[c], c))
    for c in extras:
        if short <= 0:
            break
        t = min(short, main[c] - stock_main.get(c, 0), main_left.get(c, 0))
        if t:
            cards_out[c] = cards_out.get(c, 0) + t
            main_left[c] -= t
            short -= t
            notes.append(f"{t} {c} out in their slot")
    for name in set(cards_in) & set(cards_out):  # cancel out swaps of a card for itself
        k = min(cards_in[name], cards_out[name])
        for d in (cards_in, cards_out):
            d[name] -= k
    cards_in = {k: v for k, v in cards_in.items() if v}
    cards_out = {k: v for k, v in cards_out.items() if v}

    def trim(d: dict, total: int) -> dict:
        d = dict(d)
        for name in reversed(list(d)):
            excess = sum(d.values()) - total
            if excess <= 0:
                break
            cut = min(excess, d[name])
            d[name] -= cut
            if not d[name]:
                del d[name]
        return d

    k = min(sum(cards_in.values()), sum(cards_out.values()))
    if sum(cards_in.values()) != sum(cards_out.values()):
        notes.append(f"trimmed to {k} swaps")
    return trim(cards_in, k), trim(cards_out, k), "; ".join(notes)


def plans_for(arch: str, main: Mapping[str, int], side: Mapping[str, int]) -> dict:
    decks, sides = dict(DECKS), dict(SIDEBOARDS)
    decks[arch], sides[arch] = main, side
    out = {}
    for opp in DECKS:
        stock = PLANS.get((arch, opp))
        if stock is None:
            continue  # no table row: no changes, valid for any variant
        try:
            validate_plan(stock, decks, sides)
            continue  # reused as is
        except ValueError:
            pass
        cin, cout, note = adapt_plan(stock, main, side, DECKS[arch])
        plan = SideboardPlan(arch, opp, cin, cout, "")
        validate_plan(plan, decks, sides)
        out[opp] = {"in": cin, "out": cout, "why": f"stock plan adjusted to this list: {note}. Stock: {stock.why}"}
    return out


def build(root: str, period: str = "90d", scope: str = "combined") -> dict:
    dash = load_dashboard(root)
    meta = snapshot_meta(root, dash, period, scope)
    sc = dash["periods"][period]["scopes"][scope]
    names = [a[0] for a in sc["archetypes"]]
    lists_of = {a[0]: a[2] for a in sc["archetypes"]}
    by_name = {names[int(k)]: v for k, v in sc["decisions"].items()}

    variants: list[dict] = []
    rejected: list[dict] = []
    for snap_name, arch in ARCHETYPES.items():
        dec = by_name[snap_name]
        boards: dict[str, list[dict]] = {}
        for board in ("main", "side"):
            ok = []
            for b in dec[board]["builds"]:
                cards = normalise_list(b["list"])
                missing = sorted(c for c in cards if c not in CARDS)
                rec = {"archetype": arch, "board": board, "build": b["name"], "lists": b["lists"], "share": b["share"]}
                if missing:
                    rejected.append({**rec, "reason": "cards not implemented", "missing": missing})
                    continue
                ok.append({**rec, "cards": cards})
            boards[board] = ok
        combos = []
        for m in boards["main"]:
            for s in boards["side"]:
                try:
                    check_deck(f"{arch} {m['build']} + {s['build']}", m["cards"], s["cards"])
                except ValueError as e:
                    rejected.append({"archetype": arch, "board": "75", "build": f"{m['build']} + {s['build']}",
                                     "lists": 0, "share": round(m["share"] * s["share"], 4), "reason": str(e), "missing": []})
                    continue
                combos.append({
                    "id": f"{arch}:{slug(m['build'])}+{slug(s['build'])}",
                    "archetype": arch, "main": m["cards"], "sideboard": s["cards"],
                    "source": f"Pauper-Research {period} {scope}: main build \"{m['build']}\" x sideboard build \"{s['build']}\"",
                    "weight": round(m["share"] * s["share"], 6),
                    "constructed": True, "stock": False,
                    "provenance": {
                        "snapshot_commit": meta["commit"], "snapshot_date": meta["commit_date"],
                        "dashboard_generated": meta["dashboard_generated"], "period": f"{meta['since']}..{meta['until']}",
                        "scope": scope, "snapshot_archetype": snap_name, "archetype_lists": lists_of[snap_name],
                        "main_build": m["build"], "main_lists": m["lists"], "main_share": m["share"],
                        "side_build": s["build"], "side_lists": s["lists"], "side_share": s["share"],
                        "weight_basis": "main_share*side_share",
                    },
                })
        # Duplicate 75s (same main and side) merge; weights add.
        merged: dict[tuple, dict] = {}
        for c in combos:
            key = V.canonical_75(c["main"], c["sideboard"])
            if key in merged:
                merged[key]["weight"] = round(merged[key]["weight"] + c["weight"], 6)
                merged[key]["provenance"]["also"] = merged[key]["provenance"].get("also", []) + [c["id"]]
            else:
                merged[key] = c
        combos = list(merged.values())
        stock_key = V.canonical_75(DECKS[arch], SIDEBOARDS[arch])
        if stock_key in merged:
            c = merged[stock_key]
            c["provenance"]["snapshot_id"] = c["id"]
            c.update(id=f"{arch}:stock", stock=True, source="decks.DECKS / SIDEBOARDS = " + c["source"])
        else:
            side_match = [s["build"] for s in boards["side"] if s["cards"] == SIDEBOARDS[arch]]
            main_match = [m["build"] for m in boards["main"] if m["cards"] == DECKS[arch]]
            w = max((c["weight"] for c in combos), default=1.0)
            combos.insert(0, {
                "id": f"{arch}:stock", "archetype": arch, "main": dict(DECKS[arch]), "sideboard": dict(SIDEBOARDS[arch]),
                "source": "decks.DECKS / SIDEBOARDS (see the decks.py docstring)",
                "weight": w, "constructed": True, "stock": True,
                "provenance": {
                    "snapshot_commit": meta["commit"], "snapshot_date": meta["commit_date"],
                    "main_build": main_match[0] if main_match else "not a snapshot build",
                    "side_build": side_match[0] if side_match else "not a snapshot build",
                    "weight_basis": "policy:max (largest accepted weight of the archetype; stock 75 is not a snapshot combination)",
                },
            })
        combos.sort(key=lambda c: (not c["stock"], -c["weight"], c["id"]))
        splits = V.assign_splits((arch, V.split_key(c["main"], c["sideboard"]), c["stock"]) for c in combos)
        for c in combos:
            c["split"] = splits[(arch, V.split_key(c["main"], c["sideboard"]))]
            c["plans"] = plans_for(arch, c["main"], c["sideboard"])
        variants.extend(combos)
    return {"snapshot": meta, "variants": variants, "rejected": rejected}


# --- TOML output (tomllib has no writer; the subset needed is small) ---

def q(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def val(x) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return repr(x)
    if isinstance(x, str):
        return q(x)
    if isinstance(x, list):
        return "[" + ", ".join(val(i) for i in x) + "]"
    if isinstance(x, dict):
        return "{ " + ", ".join(f"{q(k)} = {val(v)}" for k, v in x.items()) + " }" if x else "{}"
    raise TypeError(type(x))


def to_toml(data: dict) -> str:
    lines = [
        "# Deck variants: GENERATED by tools/build_variants.py from the local Pauper-Research",
        "# snapshot. Do not edit by hand; rebuild. Loaded and validated by mtg_ml/engine/variants.py;",
        "# rules and counts in docs/deck-variants.md.",
        "",
        "[snapshot]",
    ]
    lines += [f"{k} = {val(v)}" for k, v in data["snapshot"].items()]
    lines += ["", "[policy]",
              f"heldout_fraction = {val(V.HELDOUT_FRACTION)}",
              'split = "per archetype: distinct 75s (sha256 of the sorted composition) in key order; first max(1, round(0.15 n)) test (n >= 2), next as many dev (n >= 3), rest train; stock always train"',
              'weight = "main build share x sideboard build share; duplicates summed; stock not in snapshot: largest accepted weight"',
              f"aliases = {val(ALIASES)}",
              f"plan_equivalents = {val({k: list(v) for k, v in PLAN_EQUIVALENTS.items()})}",
              ""]
    for v in data["variants"]:
        lines.append("[[variant]]")
        for k in ("id", "archetype", "split", "weight", "constructed", "stock", "source"):
            lines.append(f"{k} = {val(v[k])}")
        lines.append(f"provenance = {val(v['provenance'])}")
        lines.append("[variant.main]")
        lines += [f"{q(c)} = {n}" for c, n in v["main"].items()]
        lines.append("[variant.sideboard]")
        lines += [f"{q(c)} = {n}" for c, n in v["sideboard"].items()]
        for opp, p in v["plans"].items():
            lines.append(f"[variant.plans.{opp}]")
            lines += [f"in = {val(p['in'])}", f"out = {val(p['out'])}", f"why = {val(p['why'])}"]
        lines.append("")
    for r in data["rejected"]:
        lines.append("[[rejected]]")
        lines += [f"{k} = {val(r[k])}" for k in ("archetype", "board", "build", "lists", "share", "reason", "missing")]
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, help="Pauper-Research checkout")
    ap.add_argument("--period", default="90d")
    ap.add_argument("--scope", default="combined")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", action="store_true", help="exit 1 if the manifest would change")
    a = ap.parse_args(argv)
    data = build(a.snapshot, a.period, a.scope)
    text = to_toml(data)
    if a.check:
        with open(a.out, encoding="utf-8") as f:
            same = f.read() == text
        print("variants.toml is current" if same else "variants.toml is stale: rebuild")
        return 0 if same else 1
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    from collections import Counter
    cnt = Counter((v["archetype"], v["split"]) for v in data["variants"])
    for arch in DECKS:
        print(f"{arch:18s} " + " ".join(f"{s}={cnt[(arch, s)]}" for s in V.SPLITS))
    print(f"rejected entries: {len(data['rejected'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
