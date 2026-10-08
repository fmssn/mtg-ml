"""Supervised legal-action learning, deliberately separate from on-policy PPO."""

from __future__ import annotations

from dataclasses import fields
import math
from pathlib import Path
import random

from ..engine.view import observe
from ..rl.features import encode_event_hashes, event_hashes, featurize
from .artifacts import VERSION, group_split, require
from .scenarios import compile_scenario, select


def demonstrations(spec: dict, evidence: dict, engine=None, features: int = 6) -> dict:
    """Every player's decision participates in history, only reviewed labels in loss.

    Scenario starts reset memory. Setup actions construct the position but do
    not masquerade as human history. Nothing textual is a policy input.
    """
    g, objects = compile_scenario(spec, evidence, engine)
    p, pending, rows = spec["perspective"], [], []
    opening_turn, opening_drawn = g.turn, g.players[p].cards_drawn_this_turn
    require(bool(spec.get("demonstration")), "scenario has no observed demonstration")
    for step in spec["demonstration"]:
        index = select(g, step["selector"], objects)
        if g.decision.player == p:
            if f"players.{p}.library" in spec["synthetic_fields"]:
                require(g.turn == opening_turn and g.players[p].cards_drawn_this_turn == opening_drawn,
                        "demonstration crossed an unobserved draw/turn; create a new reviewed scenario")
                require(g.decision.kind not in {"order", "choose_card"} or not g.stack,
                        "private library choice needs a separately reviewed scenario with known library state")
            state, options = featurize(g, p, features=features)
            rows.append({"state": state, "options": options, "events": encode_event_hashes(pending),
                         "action": index, "label": bool(step.get("label", False)), "refs": step["refs"],
                         "view": observe(g, p)})
            pending = []
        else:
            require(not step.get("label", False), "only the reviewed perspective can supply labels")
        mine, theirs = event_hashes(g, index)
        pending.extend(mine if g.decision.player == p else theirs)
        g.step(index)
    require(any(r["label"] for r in rows), "demonstration contains no reviewed player labels")
    return {"format": "ExpertEpisode", "version": VERSION, "scenario_id": spec["id"], "group_id": spec["group_id"],
            "episode_mode": "reset", "features": features, "rows": rows}


def validate_episodes(episodes: list[dict]) -> None:
    require(bool(episodes), "no expert episodes")
    ids = set()
    for episode in episodes:
        require(episode.get("format") == "ExpertEpisode" and episode.get("version") == VERSION, "unsupported expert episode")
        require(episode.get("episode_mode") == "reset" and bool(episode.get("group_id")), "episode must declare a reset and leakage group")
        require(episode["scenario_id"] not in ids, "duplicate episode id")
        ids.add(episode["scenario_id"])
        require(bool(episode.get("rows")) and any(r["label"] for r in episode["rows"]), "episode has no labels")
        for row in episode["rows"]:
            require(bool(row["options"]) and 0 <= row["action"] < len(row["options"]), "invalid expert action")
            require(bool(row.get("refs")), "expert sample lacks source references")


def split_episodes(episodes: list[dict], heldout_fraction=0.2) -> tuple[list, list]:
    validate_episodes(episodes)
    return ([e for e in episodes if group_split(e["group_id"], heldout_fraction) == "train"],
            [e for e in episodes if group_split(e["group_id"], heldout_fraction) == "test"])


def batch_episodes(net, episodes: list[dict], device="cpu"):
    from ..rl.model import collate
    import torch

    validate_episodes(episodes)
    require(all(e["features"] == net.features for e in episodes), "episode features differ from checkpoint (re-export for that version)")
    samples, labels, actions, lengths = [], [], [], []
    for episode in episodes:
        lengths.append(len(episode["rows"]))
        for row in episode["rows"]:
            samples.append((row["state"], row["options"], row["events"]))
            actions.append(row["action"])
            labels.append(row["label"])
    batch = collate(samples)
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor):
            setattr(batch, field.name, value.to(device))
    return batch, torch.tensor(actions, device=device), torch.tensor(labels, dtype=torch.bool, device=device), lengths


def score(net, episodes: list[dict], device="cpu") -> dict:
    import torch

    if not episodes:
        return {"decisions": 0, "nll": None, "accuracy": None}
    net.eval()
    total, loss, correct = 0, 0.0, 0
    with torch.no_grad():
        for episode in episodes:
            batch, actions, labels, lengths = batch_episodes(net, [episode], device)
            logits, _, _ = net(batch, lengths=lengths)
            logp = torch.log_softmax(logits, -1)
            loss += float(-logp[labels].gather(1, actions[labels, None]).sum())
            correct += int((logits[labels].argmax(-1) == actions[labels]).sum())
            total += int(labels.sum())
    return {"decisions": total, "nll": loss / total, "accuracy": correct / total}


def train(checkpoint: Path, episodes: list[dict], output: Path, *, epochs=5, lr=1e-5, heldout_fraction=0.2, device="cpu", seed=0) -> dict:
    import torch

    from ..rl.rollout import load_net

    require(checkpoint.resolve() != output.resolve() and not output.exists(), "output must be a new checkpoint path")
    require(isinstance(epochs, int) and epochs > 0 and math.isfinite(lr) and lr > 0, "positive epochs and finite learning rate required")
    train_set, test_set = split_episodes(episodes, heldout_fraction)
    require(bool(train_set), "no training groups; adjust dataset or split")
    require(bool(test_set), "held-out video groups required; never train/evaluate on variants of the same video")
    torch.manual_seed(seed)
    net = load_net(str(checkpoint)).to(device)
    before = {"train": score(net, train_set, device), "test": score(net, test_set, device)}
    # Value-only parameters get no gradients. The shared policy core changes,
    # so a shared value head's calibration must be measured before PPO resumes.
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    rng = random.Random(seed)
    history = []
    for epoch in range(epochs):
        net.train()
        order = list(train_set)
        rng.shuffle(order)
        for episode in order:
            batch, actions, labels, lengths = batch_episodes(net, [episode], device)
            logits, _, _ = net(batch, lengths=lengths)
            loss = -torch.log_softmax(logits, -1)[labels].gather(1, actions[labels, None]).mean()
            require(bool(torch.isfinite(loss)), "nonfinite supervised loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
        history.append({"epoch": epoch + 1, "train": score(net, train_set, device), "test": score(net, test_set, device)})
    report = {"format": "ExpertTrainingReport", "version": VERSION, "parent": str(checkpoint.resolve()), "seed": seed,
              "epochs": epochs, "lr": lr, "heldout_fraction": heldout_fraction, "before": before, "history": history,
              "train_groups": sorted({e["group_id"] for e in train_set}), "test_groups": sorted({e["group_id"] for e in test_set}),
              "features": net.features, "verdict": "requires benchmark and ladder evaluation"}
    output.parent.mkdir(parents=True, exist_ok=True)
    # A policy checkpoint, not a resumable PPO checkpoint with obsolete moments.
    torch.save({"config": net.config, "model": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "expert_training": report}, output)
    return report
