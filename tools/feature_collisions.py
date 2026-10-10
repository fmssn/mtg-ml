"""Hash-collision census for the feature encoding (measurement only).

    python tools/feature_collisions.py --games 40 --workers 4 --out runs/collisions.md

Features are strings hashed with crc32 % 2**16 (state, entity and preview
bags) and % 2**15 (option tokens), see `mtg_ml/rl/features.py`. Two different
strings can share a row. This tool measures how often that happens and
whether it ever matters. It plays on the Python reference engine (the only one
that exposes the pre-hash strings; the native engine is bit-identical on the
hashes, `test_featurize_native`), over every matchup in `match.MATCHUPS`.

1. **Census.** Distinct pre-hash strings seen, per space (state side: state
   features plus every entity's tokens; option side: option tokens plus
   previews); how many share a bucket with another string; and which of those
   pairs ever occur *together* in one scope: the same state bag, the same
   entity bag or the same option's token set, where the two strings are merged
   into one input. A pair that never co-occurs still shares one embedding row
   (the net cannot give them different weights), but it only costs
   information when one string can stand in for the other, so co-occurrence is
   the number that matters first.
   Saturation: distinct strings after 25/50/75/100% of each matchup's games.
2. **Option aliasing.** At every decision with 2+ options, options are
   grouped by what the policy sees (hashed tokens plus the token bags of the
   entities they point at, as `mtg_ml.audit.option_signatures`) and by the
   same thing before hashing. A hashed group that holds more than one
   pre-hash class is a hash-caused alias: step each class in a fork and
   compare the successors by `audit.canonical_view` (object ids removed). Also
   reported: pre-hash-identical groups (blind spots that have nothing to do
   with hashing; `mtg_ml.audit` studies them) and, on a sample of decisions,
   the converse: options with different hashes whose successors are the same
   (distinguished but equivalent, harmless).

Measurement only: nothing here changes a feature. Reference: docs/features.md,
section "Hash collisions".
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import random
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mtg_ml import audit  # noqa: E402
from mtg_ml.agents import RandomAgent, take  # noqa: E402
from mtg_ml.bots import make_bot  # noqa: E402
from mtg_ml.encode import FEATURES, check_features, entity_features, option_object_ids, option_previews, state_features  # noqa: E402
from mtg_ml.engine.game import Game  # noqa: E402
from mtg_ml.match import MATCHUPS, game_args, matchup_decks  # noqa: E402
from mtg_ml.rl.features import OPTION_DIM, STATE_DIM, featurize, option_tokens  # noqa: E402

AGENT_MIX = (("bot", "bot"), ("bot", "random"), ("random", "bot"))  # bots reach real board states, random reaches odd ones
_H: dict[tuple[str, int], int] = {}


def h(s: str, dim: int) -> int:
    k = (s, dim)
    v = _H.get(k)
    if v is None:
        if len(_H) > 2_000_000:
            _H.clear()
        v = _H[k] = zlib.crc32(s.encode()) % dim
    return v


def make_agents(spec: tuple[str, str], matchup: str, seed: int):
    decks = matchup_decks(matchup)
    return [make_bot(i, decks[i]) if k == "bot" else RandomAgent(seed + i) for i, k in enumerate(spec)]


def pair_hits(strings, dim: int, scope: str, acc: collections.Counter) -> bool:
    """Count, per scope instance, every pair of distinct strings sharing a
    bucket; True when the scope had one."""
    hit = False
    by: dict[int, list[str]] = {}
    for s in set(strings):
        by.setdefault(h(s, dim), []).append(s)
    for lst in by.values():
        if len(lst) > 1:
            hit = True
            for a, b in itertools.combinations(sorted(lst), 2):
                acc[(scope, a, b)] += 1
    return hit


def decision_strings(game, player: int, features: int):
    """Pre-hash strings of the current decision: (state scope strings, entity
    bags, per option (token strings, pointed entity indices)), mirroring
    `rl.features.featurize`."""
    d = game.decision
    feats = state_features(game, player, features)
    if features < 7:
        feats.append(f"seat:{player}")
    feats.append(f"decision:{d.kind}")
    ents, index = entity_features(game, player, features)
    previews = option_previews(game, player, features)
    opts = []
    for o, pv in zip(game.legal_options(), previews):
        toks = option_tokens(d.kind, o.key) + pv
        ptr = [index[i] for i in option_object_ids(o, game, d.kind, features) if i in index]
        opts.append((toks, ptr))
    return feats, ents, opts


def signatures(feats, ents, opts):
    """(pre-hash signature, hashed signature) per option, as `audit.option_signatures`."""
    hent = [tuple(sorted({h(t, STATE_DIM) for t in e})) for e in ents]
    pre, hashed = [], []
    for toks, ptr in opts:
        pre.append((frozenset(toks), tuple(sorted(tuple(sorted(set(ents[k]))) for k in ptr))))
        hashed.append((tuple(sorted({h(t, OPTION_DIM) for t in toks})), tuple(sorted(hent[k] for k in ptr))))
    return pre, hashed


def run_matchup(task: dict) -> dict:
    matchup, games, seed0, features = task["matchup"], task["games"], task["seed"], task["features"]
    sample = task["vv_frac"]
    rng = random.Random(seed0)
    t0 = time.perf_counter()
    state_cnt: collections.Counter = collections.Counter()  # string -> decisions it appears in
    opt_cnt: collections.Counter = collections.Counter()  # string -> options it appears in
    co: collections.Counter = collections.Counter()  # (scope, a, b) -> scope instances with both
    n_scope = collections.Counter()
    curve = []
    kinds = collections.defaultdict(collections.Counter)
    examples = []
    checked = 0
    for gi in range(games):
        spec = AGENT_MIX[gi % len(AGENT_MIX)]
        agents = make_agents(spec, matchup, seed0 + gi)
        g = Game(**game_args(1, matchup), seed=seed0 + gi, starting_player=gi % 2)
        while not g.over:
            d = g.decision
            kc = kinds[d.kind]
            kc["decisions"] += 1
            if len(d.options) >= 1:
                feats, ents, opts = decision_strings(g, d.player, features)
                if checked < 25:  # the mirror of featurize() must equal featurize()
                    checked += 1
                    st, ho = featurize(g, d.player, features=features)
                    mine = sorted({h(f, STATE_DIM) for f in feats})
                    assert st[: len(mine)] == mine and [sorted(x for x in o if x < OPTION_DIM) for o in ho] == [sorted({h(t, OPTION_DIM) for t in toks}) for toks, _ in opts], "string mirror differs from featurize"
                state_cnt.update(set(feats))
                merged = n_scope["state_merged"]
                n_scope["state_merged"] += pair_hits(feats, STATE_DIM, "state", co)
                n_scope["state"] += 1
                merged_ent = 0
                for e in ents:
                    state_cnt.update(set(e))
                    hit = pair_hits(e, STATE_DIM, "entity", co)
                    n_scope["entity_merged"] += hit
                    merged_ent += hit
                    n_scope["entity"] += 1
                n_scope["decisions"] += 1
                n_scope["decisions_state_merged"] += n_scope["state_merged"] > merged
                n_scope["decisions_entity_merged"] += merged_ent > 0
                any_opt = False
                for toks, _ in opts:
                    opt_cnt.update(set(toks))
                    hit = pair_hits(toks, OPTION_DIM, "option", co)
                    n_scope["option_merged"] += hit
                    any_opt |= hit
                    n_scope["option"] += 1
                n_scope["decisions_option_merged"] += any_opt
                if len(opts) >= 2:
                    kc["multi"] += 1
                    _option_aliasing(g, d, kc, feats, ents, opts, features, rng.random() < sample, examples, matchup, seed0 + gi)
            take(g, agents, agents[d.player].act(g))
        curve.append((len(state_cnt), len(opt_cnt)))
    q = [max(0, int(games * f) - 1) for f in (0.25, 0.5, 0.75, 1.0)]
    return {
        "matchup": matchup, "games": games, "seconds": round(time.perf_counter() - t0, 1),
        "state_strings": dict(state_cnt), "option_strings": dict(opt_cnt), "co": dict(co), "n_scope": dict(n_scope),
        "curve": [curve[i] for i in q], "kinds": {k: dict(v) for k, v in kinds.items()}, "examples": examples,
    }  # fmt: skip


def _option_aliasing(g, d, kc, feats, ents, opts, features, vv: bool, examples, matchup, seed) -> None:
    pre, hashed = signatures(feats, ents, opts)
    by_h: dict = collections.defaultdict(list)
    for i, s in enumerate(hashed):
        by_h[s].append(i)
    alias_decision = defect_decision = same_decision = False
    for grp in by_h.values():
        if len(grp) < 2:
            continue
        classes: dict = collections.defaultdict(list)
        for i in grp:
            classes[pre[i]].append(i)
        if len(classes) == 1:
            same_decision = True  # pre-hash identical: not a hash problem (mtg_ml.audit)
            continue
        alias_decision = True
        reps = [c[0] for c in classes.values()]
        views = audit.successor_views(g, reps, d.player)
        digests = {audit._digest(v) for v in views}
        if len(digests) > 1:
            defect_decision = True
            if len(examples) < 40:
                examples.append({
                    "matchup": matchup, "seed": seed, "decision": len(g.actions), "turn": g.turn, "kind": d.kind,
                    "options": [{"index": i, "label": d.options[i].label, "key": [str(x) for x in d.options[i].key]} for i in reps],
                    "differs": audit._diff_fields(views),
                })  # fmt: skip
    kc["alias"] += alias_decision
    kc["alias_defect"] += defect_decision
    kc["same_string_groups"] += same_decision
    if vv and len(opts) <= 12:
        kc["vv_sampled"] += 1
        views = audit.successor_views(g, list(range(len(opts))), d.player)
        digests = [audit._digest(v) for v in views]
        equiv_pairs = diff_hash = 0
        for a, b in itertools.combinations(range(len(opts)), 2):
            if digests[a] == digests[b]:
                equiv_pairs += 1
                diff_hash += hashed[a] != hashed[b]
        kc["vv_decisions_with_equivalent_distinct"] += diff_hash > 0
        kc["vv_equivalent_pairs"] += equiv_pairs
        kc["vv_equivalent_pairs_distinct_hash"] += diff_hash


# ---------------------------------------------------------------------------
# Merging and the report
# ---------------------------------------------------------------------------


def buckets(strings, dim: int) -> dict[int, list[str]]:
    by: dict[int, list[str]] = {}
    for s in strings:
        by.setdefault(h(s, dim), []).append(s)
    return {k: sorted(v) for k, v in by.items() if len(v) > 1}


def merge(results: list[dict]) -> dict:
    state, opt, co, kinds = collections.Counter(), collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    n_scope, examples = collections.Counter(), []
    for r in results:
        state.update(r["state_strings"])
        opt.update(r["option_strings"])
        co.update(r["co"])
        n_scope.update(r["n_scope"])
        examples += r["examples"]
        for k, v in r["kinds"].items():
            kinds[k].update(v)
    out = {"kinds": {k: dict(v) for k, v in kinds.items()}, "examples": examples[:40], "n_scope": dict(n_scope)}
    for name, cnt, dim in (("state", state, STATE_DIM), ("option", opt, OPTION_DIM)):
        bk = buckets(cnt, dim)
        in_coll = {s for lst in bk.values() for s in lst}
        total_occ = sum(cnt.values())
        pairs = sum(len(l) * (len(l) - 1) // 2 for l in bk.values())
        n = len(cnt)
        scopes = {"state": ("state", "entity"), "option": ("option",)}[name]
        cop = {(k[1], k[2]): v for k, v in co.items() if k[0] in scopes}
        out[name] = {
            "dim": dim, "distinct": n, "occupied_buckets": len({h(s, dim) for s in cnt}),
            "colliding_buckets": len(bk), "strings_in_collision": len(in_coll), "colliding_pairs": pairs,
            "expected_pairs_uniform": round(n * (n - 1) / 2 / dim, 1),
            "occurrences": total_occ, "occurrences_in_collision": sum(cnt[s] for s in in_coll),
            "cooccurring_pairs": len(cop), "cooccurrences": sum(cop.values()),
            "top_cooccurring": [{"a": a, "b": b, "scopes": v, "count_a": cnt[a], "count_b": cnt[b]} for (a, b), v in sorted(cop.items(), key=lambda kv: -kv[1])[:15]],
            "largest_bucket": max((len(v) for v in bk.values()), default=1),
        }  # fmt: skip
    return out


def pct(a, b) -> str:
    return f"{100 * a / b:.2f}%" if b else "-"


def render(meta: dict, results: list[dict], m: dict) -> str:
    L = [f"# Hash-collision census, feature set {meta['features']}", ""]
    L += [f"Python reference engine, {meta['games']} games per matchup x {len(results)} matchups (bots and random agents mixed), seeds from {meta['seed']}, "
          f"vv sample {meta['vv_frac']}. Measured on the Mac ({os.cpu_count()} cores, {meta['workers']} workers, {meta['seconds']:.0f} s).", ""]
    L += ["## 1. Census of pre-hash strings", "", "| space | dim | distinct strings | buckets used | colliding buckets | strings in a collision | colliding pairs | pairs expected if uniform | largest bucket |", "|---|---|---|---|---|---|---|---|---|"]
    for name in ("state", "option"):
        s = m[name]
        L.append(f"| {name} | 2^{s['dim'].bit_length() - 1} | {s['distinct']:,} | {s['occupied_buckets']:,} | {s['colliding_buckets']:,} | {s['strings_in_collision']:,} ({pct(s['strings_in_collision'], s['distinct'])}) | {s['colliding_pairs']:,} | {s['expected_pairs_uniform']:,} | {s['largest_bucket']} |")
    L += ["", "State space = state features plus every entity's tokens (one table). Option space = option key tokens plus previews.", "",
          "Frequency-weighted: how much of the traffic involves a string that shares its bucket.", "", "| space | occurrences | in a colliding bucket | co-occurring pairs (same scope) |", "|---|---|---|---|"]
    for name in ("state", "option"):
        s = m[name]
        L.append(f"| {name} | {s['occurrences']:,} | {pct(s['occurrences_in_collision'], s['occurrences'])} | {s['cooccurring_pairs']:,} |")
    ns = m["n_scope"]
    L += ["", "Scopes with a merged pair, by kind (a scope is one state bag, one entity bag or one option's token set; two strings of one scope in the same bucket become one input):", "",
          "| scope | instances | with a merged pair |", "|---|---|---|"]
    for k in ("state", "entity", "option"):
        L.append(f"| {k} | {ns.get(k, 0):,} | {ns.get(k + '_merged', 0):,} ({pct(ns.get(k + '_merged', 0), ns.get(k, 0))}) |")
    d = ns.get("decisions", 0)
    L += ["", f"Decisions with a merged pair in the state bag: {ns.get('decisions_state_merged', 0):,} of {d:,} ({pct(ns.get('decisions_state_merged', 0), d)}); "
          f"in any entity bag: {pct(ns.get('decisions_entity_merged', 0), d)}; in any option's tokens: {pct(ns.get('decisions_option_merged', 0), d)}.", "",
          "Top co-occurring pairs (scopes = number of scope instances holding both):", ""]
    for name in ("state", "option"):
        L += [f"**{name}**", "", "| string a | string b | scopes | n(a) | n(b) |", "|---|---|---|---|---|"]
        for t in m[name]["top_cooccurring"] or []:
            L.append(f"| `{t['a']}` | `{t['b']}` | {t['scopes']:,} | {t['count_a']:,} | {t['count_b']:,} |")
        if not m[name]["top_cooccurring"]:
            L.append("| (none) | | | | |")
        L.append("")
    L += ["### Saturation", "", "Distinct strings (state, option) after a fraction of each matchup's games.", "", "| matchup | games | 25% | 50% | 75% | 100% | growth last quarter |", "|---|---|---|---|---|---|---|"]
    for r in results:
        c = r["curve"]
        L.append(f"| {r['matchup']} | {r['games']} | " + " | ".join(f"{a:,} / {b:,}" for a, b in c) + f" | {pct(c[3][0] - c[2][0], c[3][0])} / {pct(c[3][1] - c[2][1], c[3][1])} |")
    L += ["", "## 2. Option aliasing per decision kind", "",
          "Decisions with 2+ options. *Hash alias*: two options with different pre-hash features share one hashed signature. *Defect*: their successors differ (ids removed). "
          "*Same-string groups*: options identical even before hashing (not a hashing effect). The vv columns are a sample.", "",
          "| kind | decisions | with 2+ options | hash alias | alias defects | same-string groups | vv sampled | vv: decisions with equivalent options told apart | vv: equivalent pairs, hashes differ / all |", "|---|---|---|---|---|---|---|---|---|"]
    tot = collections.Counter()
    for k, c in sorted(m["kinds"].items(), key=lambda kv: -kv[1].get("multi", 0)):
        tot.update(c)
        L.append(f"| {k} | {c.get('decisions', 0):,} | {c.get('multi', 0):,} | {c.get('alias', 0):,} | {c.get('alias_defect', 0):,} | {c.get('same_string_groups', 0):,} | {c.get('vv_sampled', 0):,} | {c.get('vv_decisions_with_equivalent_distinct', 0):,} | {c.get('vv_equivalent_pairs_distinct_hash', 0):,} / {c.get('vv_equivalent_pairs', 0):,} |")
    L.append(f"| **all** | {tot['decisions']:,} | {tot['multi']:,} | {tot['alias']:,} | {tot['alias_defect']:,} | {tot['same_string_groups']:,} | {tot['vv_sampled']:,} | {tot['vv_decisions_with_equivalent_distinct']:,} | {tot['vv_equivalent_pairs_distinct_hash']:,} / {tot['vv_equivalent_pairs']:,} |")
    L += ["", f"Hash-alias rate: {pct(tot['alias'], tot['multi'])} of multi-option decisions; defect rate: {pct(tot['alias_defect'], tot['multi'])}.", ""]
    if m["examples"]:
        L += ["### Alias defect examples", ""]
        for e in m["examples"][:15]:
            L.append(f"- {e['matchup']} seed {e['seed']} decision {e['decision']} (turn {e['turn']}, {e['kind']}): " + " vs ".join(f"`{o['label']}`" for o in e["options"]) + f"; differs in {', '.join(e['differs'])}")
        L.append("")
    L += ["## Per matchup", "", "| matchup | games | seconds | distinct state strings | distinct option strings |", "|---|---|---|---|---|"]
    L += [f"| {r['matchup']} | {r['games']} | {r['seconds']} | {len(r['state_strings']):,} | {len(r['option_strings']):,} |" for r in results]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", type=int, default=40, help="games per matchup")
    ap.add_argument("--matchups", default="all", help="comma list, or 'all' (every matchup incl. mirrors)")
    ap.add_argument("--features", type=int, default=FEATURES)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--vv-frac", type=float, default=0.05, help="share of multi-option decisions where all options are stepped for the converse check")
    ap.add_argument("--out", default="feature_collisions.md")
    ap.add_argument("--json", default=None, help="also write the merged numbers here")
    a = ap.parse_args(argv)
    check_features(a.features)
    names = sorted(MATCHUPS) if a.matchups == "all" else a.matchups.split(",")
    tasks = [{"matchup": n, "games": a.games, "seed": a.seed + 1000 * i, "features": a.features, "vv_frac": a.vv_frac} for i, n in enumerate(names)]
    t0 = time.perf_counter()
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as pool:
            results = list(pool.map(run_matchup, tasks))
    else:
        results = [run_matchup(t) for t in tasks]
    m = merge(results)
    meta = {"features": a.features, "games": a.games, "seed": a.seed, "vv_frac": a.vv_frac, "workers": a.workers, "seconds": time.perf_counter() - t0}
    with open(a.out, "w") as f:
        f.write(render(meta, results, m))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"meta": meta, **m}, f, indent=1)
    print(f"wrote {a.out} in {meta['seconds']:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
