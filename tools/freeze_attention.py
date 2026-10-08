"""Freeze a completed entity checkpoint and add identity-initialized attention.

    .venv/bin/python tools/freeze_attention.py --source RUN/pool/iter_N.pt --out DIR --source-ref COMMIT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def freeze(source, out, source_ref):
    import torch

    from mtg_ml.rl.model import PolicyNet, load_partial

    source, out = Path(source).resolve(), Path(out)
    if source.name == "latest.pt":
        raise ValueError("freeze a completed pool snapshot, not a concurrently replaced latest.pt")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite frozen baseline {out}")
    ck = torch.load(source, map_location="cpu", weights_only=False)
    config = PolicyNet(**ck["config"]).config
    required = dict(hidden=256, trunk="entity", memory="gru", value_net="shared", entity_attn=0, features=6)
    if any(config[k] != v for k, v in required.items()):
        raise ValueError(f"expected width-256 GRU shared-value feature-6 parent: {config}")
    torch.manual_seed(0)
    net = PolicyNet(**{**config, "entity_attn": 1})
    new = load_partial(net, ck["model"])
    out.mkdir(parents=True)
    shutil.copyfile(source, out / "parent.pt")
    torch.save({"config": net.config, "model": net.state_dict()}, out / "attention.pt")
    manifest = dict(source=str(source), source_ref=source_ref, source_sha256=digest(source),
                    parent_sha256=digest(out / "parent.pt"), attention_sha256=digest(out / "attention.pt"),
                    config=net.config, new_parameters=new, optimizer="fresh", attention_heads=4, ffn_width=512)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-ref", required=True)
    args = ap.parse_args()
    print(json.dumps(freeze(args.source, args.out, args.source_ref), indent=2))


if __name__ == "__main__":
    main()
