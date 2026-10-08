"""Small synthetic development fixtures, never a released benchmark corpus."""

import copy
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import subprocess

from mtg_ml.backend import game_class
from mtg_ml.benchmark import FAIR, AgentMetadata, AgentRegistry, digest, file_digest, ScriptedAdapter
from mtg_ml.benchmark.artifacts import load_manifest
from mtg_ml.benchmark.jsonio import write_json
from mtg_ml.engine import JUND_WILDFIRE, MONO_BLUE_TERROR
from mtg_ml.engine.cards import SPEC_PATH
from mtg_ml.expert.scenarios import compile_scenario, selector_for


class PassAgent:
    def reset(self, own_deck):
        self.own_deck = own_deck
        self.events = []

    def observe(self, event):
        self.events.append(event)

    def choose(self, view, actions):
        return 0


def pass_factory(seat, mode):
    return ScriptedAdapter(PassAgent(), seat)


@dataclass(frozen=True)
class ModelFactory:
    path: str

    def __call__(self, seat, mode):
        from mtg_ml.benchmark import CheckpointAdapter
        return CheckpointAdapter(self.path, seat, mode=mode)


def revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def registry():
    r = AgentRegistry()
    for deck, name in (("jund_wildfire", "synthetic-jund@1"), ("mono_blue_terror", "synthetic-blue@1")):
        r.register(AgentMetadata(name, deck, revision(), 1, digest({}), FAIR, "synthetic"), PassAgent, {})
    return r


def scenario():
    players = [dict(life=20, drawn=0, hand=["Cast Down"], hand_count=1, library=["Swamp", "Swamp"], library_count=2,
                    graveyard=[], exile=[], battlefield=["Swamp", "Swamp"]),
               dict(life=20, drawn=0, hand=[], hand_count=0, library=["Island", "Island"], library_count=2,
                    graveyard=[], exile=[], battlefield=[{"name": "Delver of Secrets", "id": "delver", "sick": False}])]
    initial = dict(active=0, step="main1", seed=5, match_game=1, turn=8, lands_played=1, spells_cast_this_turn=0, players=players)
    fields = {k: v for k, v in initial.items() if k not in {"players", "seed", "match_game"}}
    fields.update({f"players.{i}.{k}": v for i, p in enumerate(players) for k, v in p.items()})
    review = {"status": "accepted", "reviewer": "synthetic-unit-test", "note": "Explicit unit-test position; not independent release review."}
    spec = dict(format="ExecutableScenario", version=1, id="synthetic-removal", group_id="synthetic-removal", perspective=0,
                decision_time=0, episode_mode="reset", initial=initial, review=review, state_facts={},
                synthetic_fields={k: "Explicit synthetic test completion." for k in fields}, assumptions=["Entire position is synthetic."],
                setup_actions=[], preferred=[{"player": 0, "kind": "priority", "key": ["cast", "Cast Down", "hand", "normal"]}])
    evidence = dict(format="EvidenceGame", version=1, sources=[{"id": "synthetic-removal", "group_id": "synthetic-removal", "type": "synthetic_test"}], records=[], metrics={})
    g, _ = compile_scenario(spec, evidence, "python")
    actions = []
    while not (g.step_name == "end" and not g.stack and g.decision.player == 0):
        if not actions:
            i = next(i for i, o in enumerate(g.legal_options()) if o.key[:2] == ("cast", "Cast Down"))
        else:
            i = 0
        actions.append(selector_for(g, i))
        g.step(i)
        assert len(actions) <= 64
    return spec, evidence, actions


def fixture(tmp_path, puzzles=True):
    root = Path(tmp_path)
    r = registry()
    refs = []
    if puzzles:
        spec, evidence, actions = scenario()
        write_json(root / "scenario.json", spec)
        write_json(root / "evidence.json", evidence)
        puzzle = dict(format="TacticalPuzzle", version=1, id="synthetic-removal", deck="jund_wildfire", category="stack-interaction",
                      template_id="synthetic-removal", source_group_id="synthetic-removal", split="dev", review=copy.deepcopy(spec["review"]),
                      episode_mode="reset", max_decisions=64, claim="success-against-declared-responses",
                      cases=[dict(id="empty-hand", scenario={"path": "scenario.json", "sha256": file_digest(root / "scenario.json")},
                                  evidence={"path": "evidence.json", "sha256": file_digest(root / "evidence.json")}, hidden_completion="no-draw",
                                  response_policy={"id": "synthetic-blue@1", "parameters_sha256": digest({})},
                                  witnesses=[{"actions": actions, "note": "Synthetic successful full removal line."}])],
                      objective={"all": [{"path": "opponent.graveyard", "op": "contains", "value": "Delver of Secrets"},
                                         {"object": "delver", "zone": "battlefield", "present": False}]},
                      stop={"kind": "decision_boundary", "turn": 8, "step": "end", "active": "self", "require_empty_stack": True},
                      acceptable_first_actions=spec["preferred"])
        write_json(root / "puzzle.json", puzzle)
        refs.append({"path": "puzzle.json", "sha256": file_digest(root / "puzzle.json")})
    write_json(root / "bundle.json", {"format": "BenchmarkPuzzleBundle", "version": 1, "puzzles": refs})
    decks = {name: {"cards": copy.deepcopy(cards), "sha256": digest(cards)} for name, cards in
             (("jund_wildfire", JUND_WILDFIRE), ("mono_blue_terror", MONO_BLUE_TERROR))}
    bots = [{k: v for k, v in asdict(r.metadata(name)).items() if k not in {"kind", "information_contract"}}
            for name in ("synthetic-jund@1", "synthetic-blue@1")]
    cells = [dict(id=f"{a}_vs_{b}", learner_deck="jund_wildfire" if a == "jund" else "mono_blue_terror", opponent=f"synthetic-{b}@1")
             for a in ("jund", "blue") for b in ("jund", "blue")]
    manifest = dict(format="BenchmarkManifest", version=1, id="synthetic-foundation", suite_version="0.1.0", split="dev",
                    stream="benchmark-v1/dev", information_contract=FAIR, freeze={"code_revision": revision(), "card_spec_sha256": file_digest(SPEC_PATH), "deck_bundle_sha256": digest(decks)},
                    decks=decks, bots=bots, cells=cells, puzzles={"path": "bundle.json", "sha256": file_digest(root / "bundle.json")}, modes=["sampled", "greedy"],
                    blocks_per_cell=1, puzzle_repetitions=1, engine_settings=dict(max_turns=2, max_decisions=1000, auto_single=False, auto_mana=False, auto_pass=False),
                    bootstrap=dict(replicates=10000, confidence=0.95))
    write_json(root / "manifest.json", manifest)
    return load_manifest(root / "manifest.json"), r


def game(engine=None, *, active=0, step="main1", hand=("Cast Down",), board=("Swamp", "Swamp"), opposing=("Delver of Secrets",),
         own_library=("Swamp",)*12, opponent_hand=(), opponent_library=("Island",)*12):
    def setup(g):
        for name in hand:
            g.add_card(name, 0, "hand")
        for name in board:
            g.add_card(name, 0, "battlefield", sick=False)
        for name in opposing:
            g.add_card(name, 1, "battlefield", sick=False)
        for name in own_library:
            g.add_card(name, 0, "library")
        for name in opponent_hand:
            g.add_card(name, 1, "hand")
        for name in opponent_library:
            g.add_card(name, 1, "library")
    return game_class(engine)(([], []), setup=setup, seed=5, starting_player=active, start_step=step,
                              auto_single=False, log=True, max_turns=4, registered_main=(tuple(JUND_WILDFIRE), tuple(MONO_BLUE_TERROR)))
