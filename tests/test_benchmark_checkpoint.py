"""Checkpoint-vs-specialist development panel: schedule shape and a tiny end-to-end run."""

import pytest

from mtg_ml.benchmark.artifacts import FAIR
from mtg_ml.benchmark.jsonio import read_json
from tools.benchmark_checkpoint import CELLS, main, schedule


def test_cells_share_deals_and_cover_both_roles():
    _, specs = schedule(2, ["sampled", "greedy"], list(CELLS))
    assert len(specs) == 2 * len(CELLS) * 2 * 4
    for cell, (deck, opponent) in CELLS.items():
        block = [s for s in specs if s.cell == cell and s.block == 0 and s.mode == "greedy"]
        assert len({s.simulator_seed for s in block}) == 1
        assert {(s.learner_seat, s.starting_player) for s in block} == {(0, 0), (0, 1), (1, 1), (1, 0)}
        assert all(s.deck_ids[s.learner_seat] == deck and s.opponent == opponent for s in block)


@pytest.mark.slow
def test_tiny_checkpoint_plays_every_cell(tmp_path):
    torch = pytest.importorskip("torch")
    from mtg_ml.rl.model import PolicyNet
    net = PolicyNet(hidden=16, memory="gru", features=7)
    path = tmp_path / "tiny.pt"
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    assert main([str(path), "--blocks", "1", "--modes", "greedy", "--engine", "python",
                 "--workers", "1", "--out", str(tmp_path / "out")]) == 0
    report = read_json(tmp_path / "out" / "report.json")
    (ck,) = report["checkpoints"]
    assert ck["contract"] == FAIR and ck["features"] == 7
    assert set(ck["results"]["greedy"]) == set(CELLS)
    assert all(v["games"] == 4 and v["errors"] == 0 for v in ck["results"]["greedy"].values())
