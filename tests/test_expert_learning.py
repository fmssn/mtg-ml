import copy

import pytest

torch = pytest.importorskip("torch")

from mtg_ml.expert.artifacts import group_split  # noqa: E402
from mtg_ml.expert.learning import batch_episodes, demonstrations, score, split_episodes, train  # noqa: E402
from mtg_ml.rl.model import PolicyNet  # noqa: E402
from test_expert_scenarios import demonstration_fixture  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def dataset():
    spec, data = demonstration_fixture("python")
    base = demonstrations(spec, data, "python", features=6)
    result = []
    for split in ("train", "test"):
        group = next(f"video-{i}" for i in range(100) if group_split(f"video-{i}") == split)
        episode = copy.deepcopy(base)
        episode.update(group_id=group, scenario_id=group)
        result.append(episode)
    return result


def test_recurrent_training_preserves_complete_local_history():
    episode = dataset()[0]
    episode["rows"][1]["label"] = False
    net = PolicyNet(hidden=16, features=6)
    batch, _, labels, lengths = batch_episodes(net, [episode])
    sequence_logits, _, _ = net(batch, lengths=lengths)
    hidden = None
    for i, row in enumerate(episode["rows"]):
        single = copy.deepcopy(episode)
        single["rows"] = [dict(row, label=True)]
        one, _, _, _ = batch_episodes(net, [single])
        logits, _, hidden = net(one, hidden=hidden)
        assert torch.allclose(sequence_logits[i, :len(row["options"])], logits[0], atol=1e-5)
    assert not labels[1] and len(labels) == len(episode["rows"])


def test_grouped_variants_never_cross_split():
    episodes = dataset()
    variant = copy.deepcopy(episodes[0])
    variant["scenario_id"] += "-variant"
    episodes.append(variant)
    training, heldout = split_episodes(episodes)
    assert not {e["group_id"] for e in training} & {e["group_id"] for e in heldout}
    assert len(training) == 2


def test_training_requires_heldout_groups(tmp_path):
    with pytest.raises(ValueError, match="held-out"):
        train(tmp_path / "absent.pt", dataset()[:1], tmp_path / "output.pt")


def test_feature_version_mismatch_is_rejected():
    with pytest.raises(ValueError, match="features differ"):
        score(PolicyNet(hidden=16, features=4), dataset())


@pytest.mark.parametrize("memory", ["none", "gru"])
def test_supervised_update_improves_labels_and_keeps_parent(tmp_path, memory):
    torch.manual_seed(2)
    net = PolicyNet(hidden=16, memory=memory, features=6)
    parent, out = tmp_path / "parent.pt", tmp_path / "expert.pt"
    torch.save({"config": net.config, "model": net.state_dict()}, parent)
    original = parent.read_bytes()
    report = train(parent, dataset(), out, epochs=12, lr=0.003)
    assert report["history"][-1]["train"]["nll"] < report["before"]["train"]["nll"]
    assert parent.read_bytes() == original
    ckpt = torch.load(out, weights_only=False)
    assert "optimizer" not in ckpt and ckpt["config"]["features"] == 6
    assert not set(report["train_groups"]) & set(report["test_groups"])
