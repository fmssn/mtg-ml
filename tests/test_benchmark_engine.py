"""Engine-parametrized benchmark boundaries and determinism."""

from dataclasses import FrozenInstanceError
import copy
import random

import pytest

from benchmark_fixtures import fixture, game, pass_factory, PassAgent
from mtg_ml.backend import game_class
from mtg_ml.benchmark import ScriptedAdapter, CheckpointAdapter, LegacyAdapter, FAIR, DIAGNOSTIC, episodes, run_episodes, take
from mtg_ml.benchmark.views import inputs, thaw
from mtg_ml.benchmark.jsonio import read_json, file_digest, write_json
from mtg_ml.benchmark.validation import validate, native_build
from mtg_ml.engine import JUND_WILDFIRE, MONO_BLUE_TERROR, expand
from mtg_ml.engine.view import determinize
from mtg_ml.bots import make_bot
from mtg_ml.rl.features import event_hashes


def test_detached_inputs_and_hidden_world(engine):
    g = game(engine, own_library=("Swamp", "Mountain", "Cast Down", "Forest") * 3,
             opponent_hand=("Counterspell",), opponent_library=("Island",)*12 + ("Ponder",))
    view, actions = inputs(g, 0, JUND_WILDFIRE)
    before = thaw(view.state)
    fork = determinize(g, 0, random.Random(123))
    assert [c.name for c in fork.players[0].library] != [c.name for c in g.players[0].library]
    fork.deck_names = ("arbitrary-self", "arbitrary-opponent")
    assert inputs(fork, 0, JUND_WILDFIRE) == (view, actions)
    with pytest.raises(TypeError): view.state["turn"] = 100
    with pytest.raises(FrozenInstanceError): actions[0].index = 20
    g.step(0)
    assert thaw(view.state) == before
    assert "simulator_seed" not in str(view) and "fixture_id" not in str(view)


def test_legitimately_revealed_library_card_changes_input(engine):
    g = game(engine, active=0, step="upkeep", hand=(), board=("Delver of Secrets",), opposing=())
    initial, _ = inputs(g, 0, {})
    assert not initial.state["self"]["library_known"]
    while g.decision.kind != "yes_no": g.step(0)
    view, _ = inputs(g, 0, {})
    assert view.state["self"]["library_known"]
    assert view != initial


def test_structured_decisions_on_legacy_development_lines(engine):
    seen = set()
    for seed in range(3):
        g = game_class(engine)((expand(JUND_WILDFIRE), expand(MONO_BLUE_TERROR)), seed=seed, max_turns=12, auto_single=False, log=True)
        bots = [make_bot(p) for p in (0, 1)]
        while not g.over:
            p = g.decision.player
            _, actions = inputs(g, p, JUND_WILDFIRE if p == 0 else MONO_BLUE_TERROR)
            assert len(actions) == len(g.legal_options())
            assert [a.index for a in actions] == list(range(len(actions)))
            seen.add(g.decision.kind)
            g.step(bots[p].act(g))
    assert {"priority", "target", "pay_mana", "mulligan", "choose_card", "declare_attacker", "declare_blocker"} <= seen


def test_combat_subjects_follow_interleaved_group_choices(engine):
    g = game(engine, step="declare_attackers", hand=(), board=("Delver of Secrets", "Delver of Secrets", "Tolarian Terror"), opposing=())
    while g.decision.kind != "declare_attacker": g.step(0)
    for name in ("Tolarian Terror", "Delver of Secrets", "Delver of Secrets"):
        _, actions = inputs(g, 0, {})
        choice = next(a for a in actions if a.key == ("attack", name))
        oid = choice.data["attacker"]["oid"]
        assert oid not in g.attackers
        g.step(choice.index)
        assert oid in g.attackers
    assert len(g.attackers) == 3
    assert g.copy().combat_subjects == g.combat_subjects


def test_private_menu_events_and_public_perspective(engine):
    g = game(engine, hand=("Brainstorm",), board=("Island",), opposing=(), own_library=("Swamp", "Mountain", "Forest") + ("Swamp",)*12)
    agents = [ScriptedAdapter(PassAgent(), p) for p in (0, 1)]
    for a in agents: a.reset({})
    i = next(i for i, o in enumerate(g.legal_options()) if o.key[:2] == ("cast", "Brainstorm"))
    take(g, agents, i)
    public = agents[1].agent.events[-1]
    assert public.actor == "opponent" and public.action.data["card"]["owner"] == "opponent"
    while g.decision.kind != "choose_card": take(g, agents, 0)
    take(g, agents, 0)
    private = agents[1].agent.events[-1]
    assert private.kind == "choose_card" and private.action is None
    assert all(not s.startswith("  p0 choose_card") for s in private.lines)
    assert private.state["self"].get("hand") == ()
    agents[0].reset({})
    assert agents[0].agent.events == []


def test_search_choice_does_not_include_unknown_order(engine):
    g = game(engine, hand=("Lorien Revealed",), board=("Island",), opposing=(), own_library=("Island", "Swamp", "Island"))
    g.step(next(i for i, o in enumerate(g.legal_options()) if o.key[:2] == ("activate", "Lorien Revealed")))
    while g.decision.kind != "choose_card": g.step(0)
    view, actions = inputs(g, 0, {})
    search = next(a for a in actions if a.key == ("search", "Island"))
    assert search.data["card"]["name"] == "Island" and search.data["card"]["uid"] is None
    assert "Island" in view.cards


def test_public_rule_facts_and_costs_support_ward_and_mana_decisions(engine):
    g = game(engine, opposing=("Tolarian Terror",))
    view, actions = inputs(g, 0, JUND_WILDFIRE)
    cast = next(a for a in actions if a.key[:2] == ("cast", "Cast Down"))
    assert cast.data["mana_cost"]["generic"] == 1 and cast.data["mana_cost"]["colored"]["B"] == 1
    assert view.cards["Tolarian Terror"]["ward"] == 2
    assert view.cards["Drossforge Bridge"]["enters_tapped"] is True
    assert set(view.cards["Drossforge Bridge"]["abilities"][0]["mana"]) == {"B", "R"}
    with pytest.raises(TypeError): view.cards["Tolarian Terror"]["ward"] = 0


def without_time(rows):
    rows = copy.deepcopy(rows)
    for r in rows:
        for field in ("elapsed_seconds", "latency_seconds"): r.pop(field, None)
    return rows


@pytest.mark.slow
def test_panel_repeatability_across_workers_and_draws(tmp_path, engine):
    m, _ = fixture(tmp_path, puzzles=False)
    specs = episodes(m, modes=["greedy"])
    serial = run_episodes(specs, m.data["engine_settings"], pass_factory, pass_factory, engine, workers=1)
    threaded = run_episodes(reversed(specs), m.data["engine_settings"], pass_factory, pass_factory, engine, workers=3)
    assert without_time(serial) == without_time(threaded)
    assert all(r["status"] == "completed" and r["winner"] is None and r["reason"] == "turn limit" for r in serial)
    assert len({(r["cell"], r["slot"]) for r in serial}) == 16


def test_agent_factories_follow_deck_and_identity_across_seat_swaps(tmp_path, engine):
    m, _ = fixture(tmp_path, puzzles=False)
    class DeckAgent(PassAgent):
        def __init__(self, card): self.card = card
        def choose(self, view, actions):
            assert self.card in view.own_deck
            return 0
    def jund(seat, mode): return ScriptedAdapter(DeckAgent("Cast Down"), seat)
    def blue(seat, mode): return ScriptedAdapter(DeckAgent("Ponder"), seat)
    learner = {"jund_wildfire": jund, "mono_blue_terror": blue}
    opponent = {"synthetic-jund@1": jund, "synthetic-blue@1": blue}
    rows = run_episodes(episodes(m, modes=["greedy"]), m.data["engine_settings"], learner, opponent, engine)
    assert len(rows) == 16 and all(r["status"] == "completed" for r in rows)


def test_errors_and_caps_retain_rows(tmp_path, engine):
    m, _ = fixture(tmp_path, puzzles=False)
    specs = episodes(m, cells=["jund_vs_blue"], modes=["greedy"])
    class Invalid(PassAgent):
        def choose(self, view, actions): return True
    def bad(seat, mode): return ScriptedAdapter(Invalid(), seat)
    rows = run_episodes(specs, m.data["engine_settings"], bad, pass_factory, engine, record=True)
    assert len(rows) == 4 and all(r["status"] == "error" for r in rows)
    assert all("illegal action" in r["reason"] for r in rows)
    assert all("visible_outcome" in r and "attempted_action" in r for r in rows)
    settings = {**m.data["engine_settings"], "max_decisions": 1}
    rows = run_episodes(specs, settings, pass_factory, pass_factory, engine)
    assert all(r["reason"] == "runner_decision_cap" and r["decisions"] == 1 for r in rows)
    def broken(seat, mode):
        raise RuntimeError("factory_failure")
    rows = run_episodes(specs, settings, broken, pass_factory, engine, record=True)
    assert all(r["status"] == "error" and r["reason"] == "factory_failure" and r["decisions"] == 0 for r in rows)


def test_witness_validation_and_truncated_line(tmp_path, engine, monkeypatch):
    m, registry = fixture(tmp_path)
    monkeypatch.setattr("mtg_ml.benchmark.validation.check_freeze", lambda m: None)
    report = validate(m.path, engines=(engine,), registry=registry)
    assert report["status"] == "complete", report
    assert report["puzzles"][0]["cases"][0]["witnesses"][0]["actions"] == 17
    puzzle = read_json(tmp_path / "puzzle.json")
    puzzle["cases"][0]["witnesses"][0]["actions"] = puzzle["cases"][0]["witnesses"][0]["actions"][:1]
    (tmp_path / "puzzle.json").unlink()
    write_json(tmp_path / "puzzle.json", puzzle)
    bundle = read_json(tmp_path / "bundle.json")
    bundle["puzzles"][0]["sha256"] = file_digest(tmp_path / "puzzle.json")
    (tmp_path / "bundle.json").unlink()
    write_json(tmp_path / "bundle.json", bundle)
    m.data["puzzles"]["sha256"] = file_digest(tmp_path / "bundle.json")
    m.path.unlink()
    write_json(m.path, m.data)
    failed = validate(m.path, engines=(engine,), registry=registry)
    assert failed["status"] == "invalid" and "boundary" in failed["errors"][0]


def checkpoint(tmp_path, features=7):
    torch = pytest.importorskip("torch")
    from mtg_ml.rl.model import PolicyNet
    net = PolicyNet(hidden=16, memory="gru", features=features)
    path = tmp_path / f"policy-{features}.pt"
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    return path


def test_checkpoint_admission_reset_events_and_byte_identity(tmp_path, engine):
    torch = pytest.importorskip("torch")
    path = checkpoint(tmp_path)
    adapter = CheckpointAdapter(path, 0, mode="sampled", contract=FAIR)
    g = game(engine)
    adapter.reset(JUND_WILDFIRE, 123)
    first = adapter.act(g)
    assert adapter.model.hidden is not None
    mine, _ = event_hashes(g, first)
    opponent = pass_factory(1, "greedy")
    opponent.reset(MONO_BLUE_TERROR)
    take(g, [adapter, opponent], first)
    assert adapter.model.events == mine
    adapter.reset(JUND_WILDFIRE, 123)
    assert adapter.model.hidden is None and adapter.model.events == []
    assert adapter.act(game(engine)) == first
    old = checkpoint(tmp_path, 6)
    with pytest.raises(ValueError, match="not fair"): CheckpointAdapter(old, 0)
    diagnostic = CheckpointAdapter(old, 0, contract=DIAGNOSTIC)
    assert diagnostic.features == 6 and diagnostic.recorded_information_contract == "legacy_archetype_disclosed"
    state = torch.load(path, weights_only=False)
    state["model"] = {k: torch.zeros_like(v) for k, v in state["model"].items()}
    # Replace atomically because loaded checkpoint tensors are memory mapped.
    replacement = tmp_path / "replacement.pt"
    torch.save(state, replacement)
    replacement.replace(path)
    new = CheckpointAdapter(path, 0)
    assert all(torch.count_nonzero(v) == 0 for v in new.model.net.state_dict().values())
    with pytest.raises(ValueError, match="changed"): adapter.reset(JUND_WILDFIRE, 123)


def test_fair_checkpoint_choice_ignores_hidden_world_and_opponent_metadata(tmp_path, engine):
    path = checkpoint(tmp_path)
    adapter = CheckpointAdapter(path, 0, mode="greedy")
    g = game(engine, own_library=("Swamp", "Forest", "Mountain", "Cast Down") * 3,
             opponent_hand=("Counterspell",), opponent_library=("Island", "Ponder") * 6)
    # Construct both completions through setup so simulated previews have the
    # same snapshot availability (determinize deliberately invalidates it).
    shuffled = ["Swamp", "Forest", "Mountain", "Cast Down"] * 3
    random.Random(678).shuffle(shuffled)
    fork = game(engine, own_library=tuple(shuffled), opponent_hand=("Ponder",),
                opponent_library=("Island",)*6 + ("Ponder",)*5 + ("Counterspell",))
    fork.deck_names = ("unknown-self-id", "different-opponent-id")
    adapter.reset(JUND_WILDFIRE, 42)
    action = adapter.act(g)
    info = copy.deepcopy(adapter.model.last_info)
    adapter.reset(JUND_WILDFIRE, 42)
    assert adapter.act(fork) == action
    assert adapter.model.last_info == info


def test_native_build_and_trace_parity(tmp_path, engine, monkeypatch):
    if engine != "native": pytest.skip("cross-engine check runs once")
    assert len(native_build()["source_sha256"]) == 64
    m, registry = fixture(tmp_path)
    monkeypatch.setattr("mtg_ml.benchmark.validation.check_freeze", lambda m: None)
    report = validate(m.path, registry=registry)
    assert report["status"] == "complete", report
    specs = episodes(m, cells=["blue_vs_jund"], modes=["greedy"])
    python = run_episodes(specs, m.data["engine_settings"], pass_factory, pass_factory, "python")
    native = run_episodes(specs, m.data["engine_settings"], pass_factory, pass_factory, "native")
    assert without_time(python) == without_time(native)


def test_legacy_adapter_isolated_smoke(engine):
    adapter = LegacyAdapter(lambda seat: make_bot(seat), 0)
    adapter.reset(JUND_WILDFIRE)
    g = game(engine)
    assert 0 <= adapter.act(g) < len(g.legal_options())
    assert adapter.information_contract == DIAGNOSTIC


@pytest.mark.slow
def test_sampled_and_greedy_checkpoint_worker_reproducibility(tmp_path, engine):
    torch = pytest.importorskip("torch")
    from benchmark_fixtures import ModelFactory
    torch.set_num_threads(1)
    m, _ = fixture(tmp_path, puzzles=False)
    path = checkpoint(tmp_path)
    factory = ModelFactory(str(path))
    specs = episodes(m, cells=["blue_vs_jund"])
    settings = {**m.data["engine_settings"], "max_decisions": 8}
    serial = run_episodes(specs, settings, factory, pass_factory, engine, record=True)
    parallel = run_episodes(reversed(specs), settings, factory, pass_factory, engine, workers=2, record=True)
    assert without_time(serial) == without_time(parallel)
    assert all(r["reason"] == "runner_decision_cap" for r in serial)


def test_sequential_and_small_damage_semantics(engine):
    # Both allocations start through legal declarations, with no injected combat.
    for n in (2, 11):
        g = game(engine, step="declare_attackers", hand=(), board=("Tolarian Terror",), opposing=("Delver of Secrets",) * n)
        while g.decision.kind != "declare_attacker": g.step(0)
        g.step(1)
        g.step(0)
        while g.decision.kind != "declare_blocker": g.step(0)
        for _ in range(n):
            _, actions = inputs(g, 1, {})
            assert actions[0].data["blocker"]["oid"] not in g.blocks
            g.step(1)
        while g.decision.kind not in {"assign_damage", "assign_damage_amount"}: g.step(0)
        view, actions = inputs(g, 0, {})
        if n == 2:
            assert actions[0].kind == "assign_damage" and len(actions[0].data["blockers"]) == n
        else:
            assert actions[0].kind == "assign_damage_amount" and len(view.context["damage"]["blockers"]) == n
            assert actions[0].data["amount"] == 0


def test_labels_do_not_drive_semantic_identity(engine):
    if engine == "native": pytest.skip("display label mutation only available in reference engine")
    g = game(engine, step="declare_attackers", hand=(), board=("Delver of Secrets", "Tolarian Terror"), opposing=())
    _, before = inputs(g, 0, {})
    for o in g.legal_options(): o.label = "same misleading display label"
    _, after = inputs(g, 0, {})
    assert [a.data for a in before] == [a.data for a in after]
