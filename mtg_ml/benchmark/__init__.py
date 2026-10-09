"""Unreleased benchmark foundation; PR62 specialists/corpus are separate."""

from .artifacts import FAIR, DIAGNOSTIC, load_manifest, load_puzzle
from .jsonio import canonical_bytes, digest, file_digest, write_json
from .adapters import AgentMetadata, AgentRegistry, ScriptedAdapter, CheckpointAdapter, LegacyAdapter, take
from .views import BenchmarkAgent, PlayerView, LegalAction, VisibleEvent, inputs
from .schedule import EpisodeSpec, episodes, puzzle_plan, simulator_seed, actor_seed, puzzle_seed, bootstrap_seed
from .runner import play_episode, run_episodes
from .results import build_result, load_result, write_result
from .validation import validate

REGISTRY = AgentRegistry()
