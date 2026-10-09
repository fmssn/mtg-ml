#!/usr/bin/env python3
"""Write a development BenchmarkManifest for the tactical puzzle bundle.

.venv/bin/python tools/puzzle_manifest.py --out .context/benchmark/dev.json
.venv/bin/python tools/puzzle_manifest.py --blocks 400 --id release-dev-v1 --out release/dev.json
.venv/bin/python tools/puzzle_manifest.py --split power-pilot --blocks 400 --out release/power-pilot.json

A committed manifest cannot pin its own commit, so this pins HEAD (the checkout
must be clean under mtg_ml/ and native/), cards.toml, both deck maps and the two
specialists, and refers to the bundle relative to the manifest.
"""

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import subprocess

from mtg_ml.benchmark.artifacts import FAIR
from mtg_ml.benchmark.jsonio import digest, file_digest, write_json
from mtg_ml.benchmark.tactics import SPECIALISTS, specialist_registry
from mtg_ml.engine import JUND_WILDFIRE, MONO_BLUE_TERROR
from mtg_ml.engine.cards import SPEC_PATH

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "benchmark" / "puzzles" / "dev" / "bundle.json"


def manifest(out, bundle=BUNDLE, revision=None, blocks=1, modes=("greedy", "sampled"), split="dev", id=None):
    """`power-pilot` reuses the dev puzzles on its own seed stream, for sizing the final run."""
    assert split in {"dev", "power-pilot"}
    revision = revision or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    registry = specialist_registry(revision)
    bots = [{k: v for k, v in asdict(registry.metadata(b)).items() if k not in {"kind", "information_contract"}} for b in SPECIALISTS]
    decks = {name: {"cards": dict(cards), "sha256": digest(dict(cards))} for name, cards in
             (("jund_wildfire", JUND_WILDFIRE), ("mono_blue_terror", MONO_BLUE_TERROR))}
    deck_bot = {b["deck"]: b["id"] for b in bots}
    short = {"jund_wildfire": "jund", "mono_blue_terror": "blue"}
    cells = [dict(id=f"{short[a]}_vs_{short[b]}", learner_deck=a, opponent=deck_bot[b]) for a in short for b in short]
    rel = os.path.relpath(Path(bundle).resolve(), Path(out).resolve().parent)
    return dict(format="BenchmarkManifest", version=1, id=id or ("tactical-dev" if split == "dev" else "tactical-power-pilot"), suite_version="0.1.0", split=split,
                stream="benchmark-v1/" + split, information_contract=FAIR,
                freeze={"code_revision": revision, "card_spec_sha256": file_digest(SPEC_PATH), "deck_bundle_sha256": digest(decks)},
                decks=decks, bots=bots, cells=cells, puzzles={"path": rel, "sha256": file_digest(bundle)}, modes=list(modes),
                blocks_per_cell=blocks, puzzle_repetitions=1,
                engine_settings=dict(max_turns=100, max_decisions=10000, auto_single=False, auto_mana=False, auto_pass=False),
                bootstrap=dict(replicates=10000, confidence=0.95))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--blocks", type=int, default=1, help="blocks per cell (four games each)")
    parser.add_argument("--split", choices=["dev", "power-pilot"], default="dev")
    parser.add_argument("--id", help="manifest id; it seeds the bootstrap, so give each release its own")
    args = parser.parse_args(argv)
    write_json(args.out, manifest(args.out, args.bundle, blocks=args.blocks, split=args.split, id=args.id))
    print(args.out)


if __name__ == "__main__":
    main()
