"""Round-2 pilot arms (L-Q) of tools/r8_campaign.py: round 1's recipe plus --init and --opponent-models only."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import r8_campaign as rc  # noqa: E402

FINALS = {deck: Path(f"/p/round1/{deck}.pt") for deck in rc.PILOT_MIX}
ROUND1 = {"L": "F", "M": "G", "N": "H", "O": "I", "P": "J", "Q": "K"}


def flags(arm, finals=FINALS):
    return rc.flags(arm, Path("/run"), Path("/root"), {"/root/ladder/r.pt": 0.0}, 0, "/p/r7.pt", None, finals)


@pytest.mark.parametrize("arm", sorted(rc.ROUND2))
def test_round2_differs_from_round1_only_in_init_and_opponents(arm):
    r2, r1 = flags(arm), rc.flags(ROUND1[arm], Path("/run"), Path("/root"), {"/root/ladder/r.pt": 0.0}, 0, "/p/r7.pt", None)
    deck = rc.ROUND2[arm]
    assert r2.pop("init") == f"/p/round1/{deck}.pt" and r1.pop("init") == "/p/r7.pt"
    assert r2.pop("opponent-frac") == 1.0
    opponents = dict(kv.split("=") for kv in r2.pop("opponent-models").split(","))
    assert opponents == {rc.DECKS[d]: f"/p/round1/{d}.pt" for d in rc.PILOT_MIX if d != deck}
    assert r2 == r1  # same matchup mix, games, learning rate, seed, workers, slots


def test_round_arms_are_named_by_the_scheme():
    assert [rc.ARMS[a] for a in "LMNOPQ"] == [f"r9-{d}-pilot" for d in ("jund", "blue", "madness", "affinity", "elves", "tron")]


def test_round1_finals_are_checked_against_the_manifest(tmp_path):
    import hashlib
    for deck in rc.PILOT_MIX:
        (tmp_path / f"{deck}.pt").write_bytes(deck.encode())
    lines = [f"{hashlib.sha256(deck.encode()).hexdigest()}  {deck}.pt" for deck in rc.PILOT_MIX]
    (tmp_path / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    assert set(rc.round1_finals(tmp_path)) == set(rc.PILOT_MIX)
    (tmp_path / "tron.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):
        rc.round1_finals(tmp_path)
