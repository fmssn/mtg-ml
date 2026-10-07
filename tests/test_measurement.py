"""Measurement: greedy evaluation (local and through the inference server),
the reference ladder's Elo fits, the per-kind PPO statistics and the
rollout statistics of a metrics row."""

import json
import math
from array import array

import pytest

torch = pytest.importorskip("torch")

from mtg_ml.backend import game_class, native_available  # noqa: E402
from mtg_ml.match import match_decks  # noqa: E402
from mtg_ml.rl.features import encode_event_hashes, featurize_flat  # noqa: E402
from mtg_ml.rl import evaluate  # noqa: E402
from mtg_ml.rl.evaluate import _expected, fit_elo, fit_ratings, rung_names  # noqa: E402
from mtg_ml.rl.model import PolicyNet  # noqa: E402
from mtg_ml.rl.ppo import PPOConfig, ppo_update  # noqa: E402
from mtg_ml.rl.rollout import BOT, KIND_ID, KINDS, LEARNER, RANDOM, GameSpec, Job, _batch, _LocalEvaluator, run_job, split_games  # noqa: E402
from mtg_ml.rl.train import Trainer, rollout_stats  # noqa: E402

ENGINE = "native" if native_available() else "python"


class _InProcess:
    map = staticmethod(lambda f, xs: list(map(f, xs)))


def _ckpt(tmp_path, seed=0, name=None):
    torch.manual_seed(seed)
    net = PolicyNet(hidden=16)
    path = str(tmp_path / (name or f"net{seed}.pt"))
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    return net, path


# -- Elo fits ------------------------------------------------------------------


def test_fit_elo_recovers_the_rating_from_expected_scores():
    rungs = [0.0, 150.0, 400.0, 700.0]
    for true in (-100.0, 280.0, 650.0):
        results = [(r, _expected(true - r), 100_000) for r in rungs]
        elo, se = fit_elo(results)
        assert elo == pytest.approx(true, abs=0.5)
        assert 0 < se < 5
    # fewer games: same estimate (no prior), wider error
    elo, se = fit_elo([(r, _expected(280 - r), 200) for r in rungs], prior_games=0)
    assert elo == pytest.approx(280, abs=0.5) and 15 < se < 60


def test_fit_elo_stays_finite_on_a_sweep():
    elo, se = fit_elo([(0.0, 1.0, 200), (100.0, 1.0, 200)])
    assert 500 < elo < 2000 and math.isfinite(se)
    lo, _ = fit_elo([(0.0, 0.0, 200), (100.0, 0.0, 200)])
    assert -2000 < lo < -400
    assert math.isnan(fit_elo([])[0])


def test_fit_ratings_recovers_a_round_robin():
    true = [0.0, 120.0, -80.0, 450.0]
    results = [(i, j, _expected(true[i] - true[j]), 100_000) for i in range(4) for j in range(i + 1, 4)]
    got = fit_ratings(4, results)
    assert got[0] == 0.0
    assert got == pytest.approx(true, abs=1.0)
    # and the learner fit on top of the fitted ladder lands where it should
    elo, _ = fit_elo([(r, _expected(300 - t), 100_000) for r, t in zip(got, true)])
    assert elo == pytest.approx(300, abs=1.5)


def test_rung_names():
    assert rung_names(["a/iter_00010.pt", "b/latest.pt", "c/latest.pt"]) == ["iter_00010", "1_latest", "2_latest"]


# -- greedy play ----------------------------------------------------------------


def test_greedy_games_are_deterministic_and_take_the_argmax(tmp_path):
    net, path = _ckpt(tmp_path)
    _, other = _ckpt(tmp_path, seed=1)
    specs = [GameSpec(s, seats) for s, seats in enumerate([(LEARNER, BOT), (other, LEARNER), (LEARNER, LEARNER), (RANDOM, LEARNER)])]
    job = Job(specs, path, 1, record=False, max_turns=12, engine=ENGINE, greedy=True)
    a, b = run_job(job), run_job(job)
    assert a.games == b.games
    # the local evaluator takes each decision's most likely option
    g = game_class(ENGINE)(match_decks(1), seed=3)
    xs = []
    for _ in range(12):
        p = g.decision.player
        st, ol, of = featurize_flat(g, p)
        xs.append((array("i", st), array("i", ol), array("i", of), array("i", encode_event_hashes([]))))
        g.step(0)
    ev = _LocalEvaluator(Job([], path, 1, greedy=True), 2 * len(xs))
    acts, _, _ = ev.collect([(LEARNER, k, True, x) for k, x in enumerate(xs)])
    with torch.no_grad():
        batch, width = _batch(torch, xs)
        logits, _, _ = net.eval()(batch, torch.zeros(len(xs), net.state_size) if net.memory != "none" else None, max_options=width)
    assert acts == logits.argmax(-1).tolist()
    with pytest.raises(ValueError, match="greedy"):
        run_job(Job(specs, path, 1, record=True, greedy=True))


def test_greedy_head_to_head_repeats_exactly(tmp_path):
    _, path = _ckpt(tmp_path)
    run = lambda: evaluate.head_to_head(_InProcess(), path, BOT, 4, 2, version=1, max_turns=10, jund_only=True, greedy=True)  # noqa: E731
    assert run()["jund"] == run()["jund"]


def test_server_greedy_matches_local_greedy(tmp_path):
    """The server takes the argmax through a constant Gumbel noise: the same
    games as the local evaluator's argmax."""
    from mtg_ml.rl.inference import InferenceServer, ServerConfig

    _, path = _ckpt(tmp_path)
    _, other = _ckpt(tmp_path, seed=1)
    specs = [GameSpec(s, seats) for s, seats in enumerate([(LEARNER, other), (other, LEARNER), (LEARNER, LEARNER), (LEARNER, BOT)] * 2)]
    local = sorted(run_job(Job(specs, path, 1, record=False, max_turns=12, engine=ENGINE, greedy=True)).games, key=lambda g: g[5])
    srv = InferenceServer(2, ServerConfig(device="cpu", max_wait_ms=2.0, min_rows=8, threads=2))
    try:
        with srv.pool() as procs:
            results = procs.map(run_job, [Job(c, path, 1, record=False, max_turns=12, engine=ENGINE, inference="server", greedy=True) for c in split_games(specs, 2)])
    finally:
        srv.close()
    remote = sorted((g for r in results for g in r.games), key=lambda g: g[5])
    assert remote == local


# -- reference ladder ---------------------------------------------------------------


def test_evaluate_policy_with_a_ladder_rates_it_once(tmp_path, monkeypatch):
    _, policy = _ckpt(tmp_path, name="learner.pt")
    rungs = (_ckpt(tmp_path, seed=1, name="r1.pt")[1], _ckpt(tmp_path, seed=2, name="r2.pt")[1])
    ratings = str(tmp_path / "ladder.json")
    args = (policy, "unused-pool0.pt", 1, 2, 2, 0, 0, 0, 8, "local", ("random", "pool0"), 2, rungs, 2, ratings, False)
    out = evaluate.evaluate_policy(_InProcess(), *args)
    assert "eval/random/jund" in out and not any(k.startswith("eval/pool0") for k in out)  # the ladder replaces pool0
    assert out["bench/jund_vs_bot_greedy_n"] == 2 and "bench/jund_vs_bot" not in out
    assert {"ladder/r1", "ladder/r2", "ladder/r1_ci", "ladder/elo", "ladder/elo_se"} <= set(out)
    saved = json.loads(open(ratings).read())
    assert saved["ratings"][rungs[0]] == 0.0 and saved["results"][0][:2] == [0, 1]
    calls = []
    monkeypatch.setattr(evaluate, "rate_ladder", lambda *a, **k: calls.append(a))
    evaluate.evaluate_policy(_InProcess(), *args)
    assert not calls  # rated once, read back afterwards


def test_ladder_cli_writes_ratings(tmp_path, monkeypatch):
    rungs = [_ckpt(tmp_path, seed=s)[1] for s in range(3)]
    monkeypatch.setattr(evaluate, "create_pool", lambda n: _Ctx())
    out = tmp_path / "l.json"
    evaluate.main(["ladder", *rungs, "--games", "2", "--out", str(out), "--workers", "1"])
    res = json.loads(out.read_text())
    assert list(res["ratings"]) == rungs and len(res["results"]) == 3 and res["games"] == 2


class _Ctx(_InProcess):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# -- statistics -----------------------------------------------------------------


def test_rollouts_record_decision_kinds(tmp_path):
    _, path = _ckpt(tmp_path)
    res = run_job(Job([GameSpec(s, (LEARNER, LEARNER)) for s in range(3)] + [GameSpec(5, (BOT, LEARNER))], path, 1, max_turns=12, engine=ENGINE))
    assert len(res.kinds) == len(res.actions) > 0
    assert set(res.kinds) <= set(range(len(KINDS))) and KIND_ID["priority"] in res.kinds and KIND_ID["pay_mana"] in res.kinds


def test_ppo_reports_entropy_and_kl_per_kind(tmp_path):
    net, path = _ckpt(tmp_path)
    res = run_job(Job([GameSpec(s, (LEARNER, LEARNER)) for s in range(4)], path, 1, shaping=0.5, max_turns=12, engine=ENGINE))
    out = {}
    for mode in ("eager", "padded"):
        n = PolicyNet(hidden=16)
        n.load_state_dict(net.state_dict())
        opt = torch.optim.Adam(n.parameters(), lr=1e-3)
        out[mode] = ppo_update(n, opt, res, PPOConfig(epochs=2, minibatch=64, target_kl=None), gen=torch.Generator().manual_seed(0), mode=mode)
    st = out["eager"]
    nontrivial = [math.exp(lp) < 0.99 for lp in res.logps]
    assert st["nt_frac"] == pytest.approx(sum(nontrivial) / len(nontrivial))
    shares = {k: v for k, v in st.items() if k.startswith("kind/") and k.endswith("/share")}
    assert shares and sum(shares.values()) == pytest.approx(1.0)
    assert all(st[k.replace("/share", "/entropy")] > 0 for k in shares)  # non-trivial decisions have entropy
    assert {"approx_kl_first", "approx_kl_last", "nt_approx_kl_first", "nt_approx_kl_last", "nt_entropy", "nt_approx_kl"} <= set(st)
    assert st["approx_kl_first"] < st["approx_kl_last"]  # the second epoch moved further from the behaviour policy
    for k, v in st.items():  # the padded (CUDA graph) path computes the same statistics
        assert out["padded"][k] == pytest.approx(v, rel=1e-3, abs=1e-6), k


def test_rollout_stats():
    games = [
        ((LEARNER, "runs/x/pool/iter_00010.pt"), 0, "life", 5, 40, 1),
        (("runs/x/pool/iter_00010.pt", LEARNER), 0, "life", 5, 60, 2),
        ((LEARNER, "runs/x/pool/iter_00020.pt"), 0, "life", 5, 50, 3),
        ((LEARNER, BOT), 1, "life", 5, 30, 4),
        ((BOT, LEARNER), 1, "life", 5, 30, 5),
        ((LEARNER, LEARNER), 0, "life", 5, 90, 6),
    ]
    kinds = [KIND_ID["pay_mana"]] * 3 + [KIND_ID["priority"]] * 7
    st = rollout_stats(games, kinds)
    assert st["win_vs_pool"] == pytest.approx(2 / 3)  # bot and self-play games are not pool games
    assert st["win_vs_pool_by_opp"] == {"iter_00010": [0.5, 2], "iter_00020": [1.0, 1]}
    assert st["win_vs_bot"] == 0.5
    assert st["decisions_per_game"] == 50
    assert st["pay_mana_share"] == pytest.approx(0.3)


def test_eval_due_by_iterations_and_by_games():
    from types import SimpleNamespace

    t = object.__new__(Trainer)
    t.cfg = SimpleNamespace(eval_every=0, eval_every_games=1000)
    due = []
    for it in range(1, 9):
        t.iteration, t.games_total = it, it * 300
        due.append(t._eval_due(300))
    assert due == [False, False, False, True, False, False, True, False]  # 900->1200, 1800->2100
    t.cfg = SimpleNamespace(eval_every=4, eval_every_games=0)
    t.iteration, t.games_total = 8, 10
    assert t._eval_due(5)
