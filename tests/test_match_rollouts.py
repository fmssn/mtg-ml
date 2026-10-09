"""Best-of-three match rollouts and the knowledge a feature-set-8 network
carries between games (docs/belief-head.md): transfer G1 -> G2 -> G3, reset
on a new match, seat swaps, belief targets aligned with trajectories,
`play_match` / `ModelAgent` and the trainer's match mode."""

import json
import math
import random

import pytest

torch = pytest.importorskip("torch")

from mtg_ml.engine.variants import sample_variant  # noqa: E402
from mtg_ml.knowledge import EMPTY, MatchKnowledge  # noqa: E402
from mtg_ml.match import game_seed, matchup_decks, play_match  # noqa: E402
from mtg_ml.rl import rollout  # noqa: E402
from mtg_ml.rl.belief import BeliefSpec  # noqa: E402
from mtg_ml.rl.model import PolicyNet  # noqa: E402
from mtg_ml.rl.ppo import PPOConfig, ppo_update  # noqa: E402
from mtg_ml.rl.rollout import BOT, LEARNER, GameSpec, Job, registered_lists, run_job  # noqa: E402
from mtg_ml.rl.train import parse_args, Trainer  # noqa: E402


@pytest.fixture(params=["python", "native"])
def engine_name(request):
    if request.param == "native":
        pytest.importorskip("mtg_ml_native")
    return request.param


def belief_ckpt(tmp_path, name="b.pt"):
    net = PolicyNet(hidden=16, features=8, belief=BeliefSpec.default())
    path = str(tmp_path / name)
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    return net, path


def variants_for(matchup, seed):
    rng = random.Random(seed)
    return tuple(sample_variant(d, rng, "train").id for d in matchup_decks(matchup))


def _record_evidence(monkeypatch):
    """Every evidence call of the rollout: (knowledge games, current witnessed)."""
    calls = []
    real = BeliefSpec.evidence

    def spy(self, knowledge, current):
        calls.append((knowledge.games, dict(current)))
        return real(self, knowledge, current)

    monkeypatch.setattr(BeliefSpec, "evidence", spy)
    return calls


def _match(path, engine, seed=3, swap=False, matchup="jund_blue", variants=None, record=True, max_turns=8):
    spec = GameSpec(seed, (LEARNER, BOT), starting_player=0, matchup=matchup, swap_seats=swap, bo3=True, variants=variants)
    return run_job(Job([spec], path, 1, record=record, max_turns=max_turns, engine=engine))


def test_bo3_spec_plays_the_match_and_carries_knowledge(tmp_path, monkeypatch, engine_name):
    _, path = belief_ckpt(tmp_path)
    calls = _record_evidence(monkeypatch)
    res = _match(path, engine_name, variants=variants_for("jund_blue", 1))
    (seats, mseed, matchup, games, winner), = res.matches
    assert (seats, mseed, matchup) == ((LEARNER, BOT), 3, "jund_blue") and 2 <= len(games) <= 3
    assert [g[-1] for g in res.games] == [game_seed(3, n) for n in range(1, len(games) + 1)]
    wins = [sum(w == c for _, w, _ in games) for c in (0, 1)]
    assert len(games) == 3 or max(wins) == 2
    # the loser of a game (or, after a draw, the same player) starts the next one
    for (start, w, _), (nxt, _, _) in zip(games, games[1:]):
        assert nxt == (start if w is None else 1 - w)
    # each game's decisions see exactly the earlier games' records, in order
    by_game = [c[0] for c in calls]
    assert by_game[0] == () and sorted(set(len(k) for k in by_game)) == list(range(len(games)))
    for k in by_game:  # knowledge only grows, by appending whole games
        assert by_game[-1][: len(k)] == k
    # targets: one per recorded trajectory, the opponent's registered 75
    assert len(res.belief_targets) == len(res.lengths) > 0
    spec_b = BeliefSpec.default()
    arch, counts = spec_b.target(*registered_lists("jund_blue", variants_for("jund_blue", 1), 1))
    assert all(a == arch and list(c) == list(counts) for a, c in res.belief_targets)
    assert sum(len(res.samples.evidence(i)) for i in range(len(res.samples))) > 0


def test_knowledge_equals_witnessed_record_of_previous_games(tmp_path, monkeypatch):
    """Game 2's first decision gets game 1's final `witnessed` of that seat."""
    _, path = belief_ckpt(tmp_path)
    finals = []
    real_next = rollout._Live.next_game

    def spy(self, Game, kw):
        p = self.spec.physical_seats.index(LEARNER)
        finals.append(self.game.witnessed(p))
        return real_next(self, Game, kw)

    monkeypatch.setattr(rollout._Live, "next_game", spy)
    calls = _record_evidence(monkeypatch)
    _match(path, "python", seed=7, record=False)
    seen_games = sorted({k for k, _ in calls}, key=len)
    assert len(seen_games) >= 2 and len(finals) == len(seen_games)  # next_game also runs after the last game
    for n, k in enumerate(seen_games[1:]):
        assert dict(k[n]) == {c: v for c, v in finals[n].items() if v}


def test_new_match_resets_and_swapped_seats_keep_their_canonical_knowledge(tmp_path, monkeypatch):
    _, path = belief_ckpt(tmp_path)
    starts, steps = [], []
    real_init, real_next = rollout._Live.__init__, rollout._Live.next_game

    def init(self, *a, **kw):
        real_init(self, *a, **kw)
        starts.append(tuple(self.knowledge))

    def nxt(self, Game, kw):
        learner = self.spec.physical_seats.index(LEARNER)
        seen = {c: v for c, v in self.game.witnessed(learner).items() if v}
        before = len(self.knowledge[0].games)
        going_on = real_next(self, Game, kw)
        if going_on:  # canonical seat 0 (the learner) got its own physical seat's record
            assert dict(self.knowledge[0].games[before]) == seen
        steps.append((self.spec.seed, len(self.knowledge[0].games), going_on))
        return going_on

    monkeypatch.setattr(rollout._Live, "__init__", init)
    monkeypatch.setattr(rollout._Live, "next_game", nxt)
    specs = [GameSpec(s, (LEARNER, BOT), starting_player=0, bo3=True, swap_seats=s % 2 == 1) for s in (11, 12)]
    res = run_job(Job(specs, path, 1, record=False, max_turns=8))
    assert len(res.matches) == 2 and all(m[0] == (LEARNER, BOT) for m in res.matches)
    assert starts == [(EMPTY, EMPTY)] * 2  # every match starts from nothing
    for seed in (11, 12):
        mine = [n for s, n, on in steps if s == seed and on]
        assert mine == list(range(1, len(mine) + 1)) and mine


def test_set7_learner_records_no_evidence_or_targets(tmp_path):
    net = PolicyNet(hidden=16, features=7)
    path = str(tmp_path / "s7.pt")
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    res = _match(path, "python")
    assert res.belief_targets == [] and len(res.matches) == 1
    assert all(len(res.samples.evidence(i)) == 0 for i in range(len(res.samples)))


def test_identical_evidence_gives_identical_belief_whatever_the_hidden_lists(tmp_path):
    """Information boundary: the belief input is a function of witnessed
    evidence only; different (unseen) opponent lists cannot change it."""
    net, _ = belief_ckpt(tmp_path)
    spec_b = net.belief
    k = EMPTY.with_game({"Lightning Bolt": 2, "Mountain": 5})
    ev = spec_b.evidence(k, {"Fireblast": 1})
    # the same evidence built for two opponents with different registered 75s
    a = spec_b.evidence(MatchKnowledge.from_dict(k.to_dict()), {"Fireblast": 1})
    assert list(ev) == list(a)
    t1 = spec_b.target(*registered_lists("jund_madness", None, 1))
    t2 = spec_b.target(*registered_lists("jund_madness", variants_for("jund_madness", 5), 1))
    assert t1[0] == t2[0]  # same archetype; counts are targets only, never inputs


def test_play_match_hands_knowledge_to_model_agents(tmp_path, monkeypatch):
    from mtg_ml.bots import make_bot
    from mtg_ml.rl.agent import ModelAgent

    _, path = belief_ckpt(tmp_path)
    agent = ModelAgent(path, 0, seed=1)
    given = []
    real = agent.set_knowledge
    monkeypatch.setattr(agent, "set_knowledge", lambda k: (given.append(k), real(k)))
    res = play_match([agent, make_bot(1, "mono_blue_terror")], seed=4, max_turns=8)
    assert [len(k.games) for k in given] == list(range(len(res.games)))
    assert given[-1].games == res.knowledge[0].games[: len(given) - 1]
    assert "belief" in agent.last_info and math.isclose(sum(agent.last_info["belief"].values()), 1.0, abs_tol=1e-3)
    # an unrelated game afterwards starts from empty knowledge
    from mtg_ml.engine import Game
    from mtg_ml.match import game_args

    g = Game(**game_args(1), seed=99, max_turns=4)
    agent.act(g)
    assert agent.knowledge is EMPTY


def test_belief_loss_trains_on_match_rollouts(tmp_path):
    net, path = belief_ckpt(tmp_path)
    res = _match(path, "python", variants=variants_for("jund_blue", 2))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    first = last = None
    for _ in range(6):
        stats = ppo_update(net, opt, res, PPOConfig(epochs=1, minibatch=64, target_kl=None))
        keys = [k for k in stats if k.startswith("belief")]
        assert keys
        loss = sum(v for k, v in stats.items() if k.startswith("belief") and "loss" in k)
        first = loss if first is None else first
        last = loss
    assert last < first


def test_trainer_match_mode_with_belief(tmp_path):
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="match rollouts"):
        Trainer(parse_args(["--run", str(run), "--belief", "1", "--hidden", "16"]))
    cfg = parse_args(["--run", str(run), "--iterations", "1", "--games-per-iter", "5", "--workers", "1", "--hidden", "16", "--max-turns", "6",
                      "--belief", "1", "--match-rollouts", "1", "--variants", "train", "--eval-every", "0", "--matchup", "jund_blue,jund_madness"])
    t = Trainer(cfg)
    assert t.net.features == 8 and t.net.belief is not None
    t.train()
    row = json.loads((run / "metrics.jsonl").read_text().splitlines()[-1])
    assert row["matches"] == 2 and row["games"] == t.games_total >= 4
    assert any(k.startswith("belief") for k in row) and row["features"] == 8
    assert t.spec_matchup == {}


def test_init_upgrade_from_set7_keeps_the_parents_play(tmp_path):
    parent = PolicyNet(hidden=16, features=7)
    src = str(tmp_path / "p.pt")
    torch.save({"config": parent.config, "model": parent.state_dict()}, src)
    cfg = parse_args(["--run", str(tmp_path / "run"), "--init", src, "--belief", "1", "--match-rollouts", "1", "--variants", "train", "--hidden", "16"])
    t = Trainer(cfg)
    assert t.net.features == 8 and t.net.belief is not None
    assert all(torch.count_nonzero(p) == 0 for p in t.net.belief_proj.parameters())
