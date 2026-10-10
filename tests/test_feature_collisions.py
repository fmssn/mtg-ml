"""tools/feature_collisions.py: the pre-hash strings mirror `featurize`, collisions
are found within a scope, and a tiny census runs and renders."""

import collections
import os
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import feature_collisions as F  # noqa: E402


def _colliding_pair(dim: int) -> tuple[str, str]:
    seen: dict[int, str] = {}
    for i in range(10**6):
        s = f"s{i}"
        k = zlib.crc32(s.encode()) % dim
        if k in seen:
            return seen[k], s
        seen[k] = s
    raise AssertionError("no collision found")


def test_pair_hits_counts_only_distinct_strings_sharing_a_bucket():
    a, b = _colliding_pair(1 << 10)
    acc: collections.Counter = collections.Counter()
    assert F.pair_hits([a, b, a, "zzz-unrelated"], 1 << 10, "state", acc)
    assert list(acc) == [("state", *sorted((a, b)))] and acc.most_common(1)[0][1] == 1
    assert not F.pair_hits([a, a], 1 << 10, "state", acc)


def test_tiny_census_runs_and_renders():
    # run_matchup asserts that its string mirror equals featurize() on the first decisions
    task = {"matchup": "jund_blue", "games": 1, "seed": 3, "features": 7, "vv_frac": 1.0}
    r = F.run_matchup(task)
    assert r["state_strings"] and r["option_strings"] and r["kinds"]["priority"]["multi"] > 0
    m = F.merge([r])
    assert m["state"]["distinct"] == len(r["state_strings"])
    text = F.render({"features": 7, "games": 1, "seed": 3, "vv_frac": 1.0, "workers": 1, "seconds": 0.0}, [r], m)
    assert "## 1. Census" in text and "## 2. Option aliasing" in text
