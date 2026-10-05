"""Golden trace digests: the reference engine still plays recorded games
exactly as before (decisions, prompts, labels, keys, views, logs, outcomes).

Regenerate only for an intentional rules change:
    python -m mtg_ml.trace record --games 210
"""

import json
import os

import pytest

from mtg_ml.trace import Scenario, digest

with open(os.path.join(os.path.dirname(__file__), "data", "golden_digests.json")) as f:
    GOLDEN = json.load(f)


@pytest.mark.parametrize("entry", GOLDEN[:70], ids=lambda e: f"seed{e['scenario']['seed']}")
def test_golden_digest(entry):
    assert digest(Scenario.from_json(entry["scenario"]), engine="python") == entry["digest"]
