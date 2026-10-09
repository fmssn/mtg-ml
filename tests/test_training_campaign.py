"""Resource-aware campaign preparation must not launch onto occupied hardware."""
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from tools.training_campaign import blockers, prepare  # noqa: E402


@pytest.mark.parametrize("change", [
    {"hours": -2}, {"arms": "bf16,unknown"}, {"arms": "bf16,bf16"},
    {"gpus": "GPU-a,GPU-a"}, {"arms": "learners4", "gpus": "GPU-a,GPU-b"},
])
def test_invalid_campaign_does_not_create_partial_runs(tmp_path, change):
    args = SimpleNamespace(root=str(tmp_path / "runs"), frozen=str(tmp_path / "missing-parent"),
                           code=str(tmp_path / "code"), cpus="0-7", gpus="GPU-a,GPU-b",
                           arms="bf16", hours=0, base_learners=1)
    args.__dict__.update(change)
    with pytest.raises(ValueError):
        prepare(args)
    assert not (tmp_path / "runs").exists()


def test_existing_campaign_is_not_overwritten(tmp_path):
    (tmp_path / "campaign.json").write_text("existing manifest")
    args = SimpleNamespace(root=str(tmp_path), frozen="missing-parent", code="code", cpus="0-7",
                           gpus="GPU-a,GPU-b", arms="bf16", hours=0, base_learners=1)
    with pytest.raises(FileExistsError):
        prepare(args)
    assert (tmp_path / "campaign.json").read_text() == "existing manifest"


@pytest.mark.parametrize("bus,memory,apps,reason", [
    ("18", 0, "GPU-a, 12345\n", "owned by PID 12345"),
    ("18", 2048, "", "occupied (2048 MiB)"),
    ("BE", 0, "", "bus BE is excluded"),
    ("2F", 0, "", "bus 2F is excluded"),
])
def test_campaign_blocks_live_contexts_and_excluded_devices(monkeypatch, bus, memory, apps, reason):
    def output(command, **kwargs):
        return f"GPU-a, 00000000:{bus}:00.0, {memory}\n" if "--query-gpu=uuid,pci.bus_id,memory.used" in command else apps

    monkeypatch.setattr("tools.training_campaign.subprocess.check_output", output)
    assert any(reason in r for r in blockers({"gpus": ["GPU-a"], "cpus": []}))
