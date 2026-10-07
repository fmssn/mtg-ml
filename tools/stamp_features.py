"""Copy a checkpoint or policy file with its feature-set version recorded.

    python tools/stamp_features.py runs/x/final.pt runs/x/final.f2.pt --features 2

Models trained on feature set 2 before checkpoints recorded the version
(`config["features"]`, docs/features.md) read as set 1. This writes a copy
of `src` (a full checkpoint like `latest.pt` / `final.pt` with the optimizer
state, or a policy / pool file with only `config` and `model`) whose config
records `--features`; everything else is copied unchanged. `src` is only
read, `dst` must not exist: archives stay append-only. Refuses a source
whose config already records a different version.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mtg_ml.encode import FEATURE_VERSIONS, check_features  # noqa: E402


def stamp(src: str, dst: str, features: int) -> dict:
    """Write the copy and return its config."""
    import torch

    check_features(features)
    if os.path.exists(dst) or os.path.abspath(src) == os.path.abspath(dst):
        raise FileExistsError(f"{dst} exists: stamp_features never overwrites a file")
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or not isinstance(ck.get("config"), dict) or "model" not in ck:
        raise ValueError(f"{src} is not a checkpoint or policy file (a dict with 'config' and 'model')")
    if ck["config"].get("features", features) != features:
        raise ValueError(f"{src} already records feature set {ck['config']['features']}")
    config = {k: v for k, v in ck["config"].items() if k != "features"}
    if features != 1:  # PolicyNet's convention: absent means 1
        config["features"] = features
    ck["config"] = config
    tmp = f"{dst}.tmp{os.getpid()}"
    try:
        torch.save(ck, tmp)
        os.link(tmp, dst)  # fails if dst appeared meanwhile; readers never see a partial file
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--features", type=int, required=True, choices=FEATURE_VERSIONS)
    args = ap.parse_args(argv)
    try:
        config = stamp(args.src, args.dst, args.features)
    except (FileExistsError, ValueError) as e:
        sys.exit(f"stamp_features: {e}")
    print(f"wrote {args.dst} (feature set {config.get('features', 1)})")


if __name__ == "__main__":
    main()
