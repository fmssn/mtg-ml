"""Set-7 seat mapping, inference layouts and state-dependent shaping bounds."""

from array import array
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from helpers import scenario  # noqa: E402

from mtg_ml.rl import evaluate, rollout  # noqa: E402
from mtg_ml.rl.features import featurize  # noqa: E402
from mtg_ml.rl.model import PolicyNet, collate, pad_split, padded_batch, structure  # noqa: E402
from mtg_ml.rl.rollout import BOT, LEARNER, GameSpec, Job, Result, Trajectory, _finish  # noqa: E402


def checkpoint(tmp_path, features=7, **kw):
    net = PolicyNet(hidden=16, features=features, **kw)
    path = str(tmp_path / "net.pt")
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    return path


def test_swap_maps_every_seat_bound_input_and_preserves_canonical_results(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    original = GameSpec(91, (LEARNER, BOT), starting_player=0, match_game=2)
    swapped = replace(original, swap_seats=True)
    a, b = original.game_args(), swapped.game_args()
    assert swapped.physical_seats == (BOT, LEARNER)
    for key in ("decks", "registered_main", "registered_sideboards"):
        assert a[key] == b[key][::-1]
    assert a["starting_player"] == 0 and b["starting_player"] == 1
    assert b["deck_names"] == ("mono_blue_terror", "jund_wildfire")
    seen = []

    def game(**kw):
        seen.append(kw)
        return SimpleNamespace(over=True, winner=1 if kw["starting_player"] else 0, end_reason="life", turn=1, actions=[])

    monkeypatch.setattr(rollout, "game_class", lambda _: game)
    results = rollout.run_job(Job([original, swapped], path, 0, record=False))
    assert [g[1] for g in results.games] == [0, 0]
    assert [g[0] for g in results.games] == [(LEARNER, BOT)] * 2
    assert seen[1]["registered_main"] == b["registered_main"]


def test_new_evaluation_balances_roles_deck_seats_and_starts():
    specs = evaluate.paired_specs(BOT, 16, balance_seats=True)
    assert len(specs) == 16
    assert {(sp.seats, sp.starting_player, sp.swap_seats) for sp in specs[:8]} == {
        (seats, start, swap) for seats in ((LEARNER, BOT), (BOT, LEARNER)) for start in (0, 1) for swap in (False, True)
    }
    assert len({sp.seed for sp in specs[:8]}) == 1
    legacy = evaluate.paired_specs(BOT, 4)
    assert all(not sp.swap_seats for sp in legacy)


def test_bo3_mapping_stays_fixed_and_loser_starts_next_game(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    rounds = []

    def play(_, specs, job, jobs):
        rounds.append(specs)
        # Canonical player 1 wins twice; results are already normalized by rollout.
        return Result(games=[(sp.seats, 1, "life", 1, 0, sp.seed) for sp in specs])

    monkeypatch.setattr(evaluate, "play", play)
    result = evaluate.head_to_head_bo3(None, path, BOT, 8, 1)
    assert len(rounds) == 2
    assert [sp.swap_seats for sp in rounds[0]] == [sp.swap_seats for sp in rounds[1]]
    assert all(sp.starting_player == 0 for sp in rounds[1])
    assert {sp.game_args()["starting_player"] for sp in rounds[1]} == {0, 1}
    assert result["all"][2] == 8


def test_high_life_potential_shifts_bootstrap_bound_without_clipping_rewards():
    traj = Trajectory(actions=[0], samples=[(array("i"),) * 4], logps=[0.0], values=[-1.6], potentials=[4.0])
    out = Result()
    _finish(traj, -1.0, Job([], "", 0, shaping=0.2, gamma=1.0, lam=1.0), out)
    assert out.returns == pytest.approx([-1.8])
    assert out.advantages == pytest.approx([-0.2])
    assert traj.values == [-1.6]
    traj.values = [99.0]
    out = Result()
    _finish(traj, -1.0, Job([], "", 0, shaping=0.2, gamma=1.0, lam=1.0), out)
    assert out.advantages == pytest.approx([-2.0])  # +0.2 upper bound, not +1.2
    assert out.returns == pytest.approx([-1.8])


def test_shaping_rejects_tanh_head(tmp_path):
    path = checkpoint(tmp_path, value_bound="tanh")
    with pytest.raises(ValueError, match="unbounded value head"):
        rollout.run_job(Job([GameSpec(1, (LEARNER, BOT))], path, 0, shaping=0.2))


def test_more_than_64_entities_fit_attention_structured_and_padded_inference():
    torch.manual_seed(11)
    g = scenario(p0={"battlefield": ["Eldrazi Spawn"] * 70, "hand": ["Lightning Bolt"]}, p1={"battlefield": ["Island"]})
    st, opts = featurize(g, 0, features=7)
    batch = collate([(st, opts, [])])
    net = PolicyNet(hidden=16, memory="none", trunk="entity", entity_attn=1, features=7).eval()
    structured = structure(batch)
    fields, sizes, _ = pad_split(structured, [0, 1], ent_width=True)
    assert sizes["entw"] > 64
    padded = padded_batch({k: v[0] for k, v in fields.items()}, sizes["entw"])
    with torch.no_grad():
        raw = net(batch)
        for b in (structured, padded):
            got = net(b, lengths=[sizes["row"]] if b is padded else None, max_options=len(opts))
            assert torch.allclose(raw[0], got[0][:1], atol=1e-5)
            assert torch.allclose(raw[1], got[1][:1], atol=1e-5)


def test_entity_permutation_remaps_pointers_without_changing_scores():
    torch.manual_seed(18)
    net = PolicyNet(hidden=16, memory="none", trunk="entity", entity_attn=1, features=7).eval()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(0.03 * torch.randn_like(parameter))
    outputs = []
    for lands in (("Forest", "Mountain"), ("Mountain", "Forest")):
        g = scenario(p0={"battlefield": list(lands) + ["Eldrazi Spawn"] * 70, "hand": ["Lightning Bolt"]})
        state, options = featurize(g, 0, features=7)
        with torch.no_grad():
            logits, value, _ = net(collate([(state, options, [])]))
        outputs.append(({o.key: logits[0, i].item() for i, o in enumerate(g.legal_options())}, value))
    assert outputs[0][0] == pytest.approx(outputs[1][0], abs=1e-5)
    assert torch.allclose(outputs[0][1], outputs[1][1], atol=1e-5)
