"""Trusted engine bridges; the scripted protocol never receives a Game."""

from dataclasses import dataclass
from typing import Callable

from ..encode import information_contract
from ..engine.view import observe
from ..replay import visible_events
from ..rl.features import PUBLIC_KINDS
from .artifacts import FAIR, DIAGNOSTIC, require, sha, integer
from .jsonio import digest, file_digest
from .views import LegalAction, VisibleEvent, freeze, inputs, thaw


@dataclass(frozen=True)
class AgentMetadata:
    id: str
    deck: str
    source_revision: str
    rules_revision: int
    parameters_sha256: str
    information_contract: str = FAIR
    kind: str = "specialist"


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, tuple[AgentMetadata, Callable, object]] = {}

    def register(self, metadata: AgentMetadata, factory: Callable, parameters) -> None:
        require(metadata.id not in self._agents, "registry.id", "already registered")
        require(callable(factory), "registry.factory", "callable required")
        require(metadata.kind in {"specialist", "synthetic", "legacy"}, "registry.kind", "unsupported kind")
        require(metadata.information_contract in {FAIR, DIAGNOSTIC}, "registry.contract", "unsupported contract")
        sha(metadata.source_revision, "registry.source_revision", True)
        integer(metadata.rules_revision, "registry.rules_revision")
        require(metadata.parameters_sha256 == digest(parameters), "registry.parameters", "digest mismatch")
        if metadata.id in {"benchmark-jund@1", "benchmark-blue@1"}:
            require(metadata.kind == "specialist", "registry.id", "reserved specialist identity")
        require(metadata.kind != "legacy" or metadata.information_contract == DIAGNOSTIC, "registry.contract", "legacy adapters are diagnostic")
        self._agents[metadata.id] = metadata, factory, parameters

    def metadata(self, name):
        require(name in self._agents, "registry", f"missing agent {name}")
        require(digest(self._agents[name][2]) == self._agents[name][0].parameters_sha256, "registry.parameters", "parameters changed after registration")
        return self._agents[name][0]

    def create(self, name):
        metadata = self.metadata(name)
        agent = self._agents[name][1]()
        methods = ("reset", "act") if metadata.kind == "legacy" else ("reset", "observe", "choose")
        require(all(callable(getattr(agent, method, None)) for method in methods), "registry.factory", "agent protocol required")
        return agent

    def verify(self, spec, contract):
        m = self.metadata(spec["id"])
        for k in ("deck", "source_revision", "rules_revision", "parameters_sha256"):
            require(getattr(m, k) == spec[k], "registry." + k, f"freeze mismatch for {m.id}")
        require(m.information_contract == contract, "registry.contract", "freeze mismatch")
        self.create(spec["id"])  # check factory/interface without evaluating its strategy


class ScriptedAdapter:
    def __init__(self, agent, seat):
        self.agent, self.seat = agent, seat
        self.own_deck = None

    def reset(self, own_deck, actor_seed=0):
        self.own_deck = freeze(own_deck)
        self.agent.reset(self.own_deck)

    def act(self, game):
        require(self.own_deck is not None, "adapter", "reset required")
        view, actions = inputs(game, self.seat, self.own_deck)
        return self.agent.choose(view, actions)


class CheckpointAdapter:
    def __init__(self, path, seat, mode="sampled", contract=FAIR):
        from ..rl.agent import ModelAgent
        from ..rl.rollout import checkpoint_config
        from ..encode import check_features
        checkpoint_sha = file_digest(path)
        features = check_features(checkpoint_config(str(path)).get("features", 1))
        actual = information_contract(features)
        require(contract in {FAIR, DIAGNOSTIC}, "checkpoint.contract", "unsupported contract")
        require(contract == DIAGNOSTIC or actual == "hidden_list", "checkpoint.contract", f"features {features} are not fair inputs")
        require(mode in {"sampled", "greedy"}, "checkpoint.mode", "unsupported mode")
        self.path, self.sha256 = str(path), checkpoint_sha
        self.features, self.information_contract = features, contract
        self.recorded_information_contract = actual
        self.seat = seat
        self.model = ModelAgent(str(path), seat, sample=mode == "sampled")
        # The legacy process cache is path keyed; freeze identity is byte keyed.
        # Loading directly prevents a replaced path from reusing stale weights.
        from ..rl.rollout import load_net
        self.model.net = load_net(str(path))
        require(self.model.net.features == features and file_digest(path) == checkpoint_sha, "checkpoint", "file changed while loading")
        self.model.features = features

    def reset(self, own_deck, actor_seed=0):
        require(file_digest(self.path) == self.sha256, "checkpoint", "file changed after loading")
        self.model.game = self.model.hidden = self.model.last_info = None
        self.model.events = []
        self.model.gen.manual_seed(actor_seed)

    def act(self, game):
        return self.model.act(game)

    def before_step(self, game, index):
        self.model.observe(game, index)


class LegacyAdapter:
    """Development smoke bridge; explicitly outside the fair protocol."""
    information_contract = DIAGNOSTIC

    def __init__(self, factory, seat):
        self.factory, self.seat = factory, seat

    def reset(self, own_deck, actor_seed=0):
        self.agent = self.factory(self.seat)

    def act(self, game):
        return self.agent.act(game)


def take(game, agents, index):
    require(type(index) is int and 0 <= index < len(game.legal_options()), "agent.index", "illegal action index")
    decider, kind = game.decision.player, game.decision.kind
    seen = len(game.log)
    actions = {}
    for a in agents:
        if isinstance(a, ScriptedAdapter):
            # Only the decider's private choice is translated. The opponent
            # sees a public action or just its kind; never their private menu.
            if a.seat == decider or kind in PUBLIC_KINDS:
                _, choices = inputs(game, decider, agents[decider].own_deck if isinstance(agents[decider], ScriptedAdapter) else {})
                actions[a.seat] = choices[index]
        if hasattr(a, "before_step"):
            a.before_step(game, index)
    game.step(index)
    for a in agents:
        if isinstance(a, ScriptedAdapter):
            action = actions.get(a.seat)
            if action is not None and a.seat != decider:
                def reverse(value):
                    if isinstance(value, dict):
                        return {k: ("opponent" if v == "self" else "self" if v == "opponent" else v)
                                if k in {"owner", "controller", "player"} and isinstance(v, str) else reverse(v) for k, v in value.items()}
                    if isinstance(value, list):
                        return [reverse(v) for v in value]
                    return value
                action = LegalAction(action.index, action.kind, action.key, action.label, freeze(reverse(thaw(action.data))))
            a.agent.observe(VisibleEvent("self" if a.seat == decider else "opponent", kind, action,
                                         tuple(visible_events(game.log[seen:], a.seat)), freeze(observe(game, a.seat))))
