#!/usr/bin/env python3
"""Build the current package and compare wheel deck data in an isolated environment.

Requires uv. All build output and the test environment live in a temporary
directory; no native extension, Torch, or editable install is used by the wheel.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROBE = """
import json
from mtg_ml.engine.decks import DECKS, SIDEBOARDS
from mtg_ml.engine.variants import VARIANTS
print(json.dumps({
    'decks': DECKS,
    'sideboards': SIDEBOARDS,
    'variants': {key: {
        'archetype': v.archetype, 'stock': v.stock,
        'main': dict(v.main), 'sideboard': dict(v.sideboard),
    } for key, v in VARIANTS.items()},
}, sort_keys=True))
"""


def run(command, cwd):
    environment = os.environ.copy()
    environment.pop("MTG_ML_VARIANTS_SKIP_LOAD", None)
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def main():
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for the isolated wheel check")
    expected = json.loads(run([sys.executable, "-c", PROBE], ROOT))
    with tempfile.TemporaryDirectory(prefix="mtg-wheel-check-") as temporary:
        scratch = Path(temporary)
        source = scratch / "source"
        source.mkdir()
        shutil.copytree(ROOT / "mtg_ml", source / "mtg_ml",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("pyproject.toml", "README.md"):
            shutil.copyfile(ROOT / name, source / name)
        wheels = scratch / "wheels"
        run([uv, "build", "--wheel", "--out-dir", str(wheels), str(source)], scratch)
        wheel, = wheels.glob("*.whl")
        environment = scratch / "venv"
        run([uv, "venv", "--python", sys.executable, str(environment)], scratch)
        python = environment / "bin" / "python"
        run([uv, "pip", "install", "--python", str(python), str(wheel)], scratch)
        actual = json.loads(run([str(python), "-I", "-c", PROBE], scratch))
        if actual != expected:
            raise RuntimeError("Installed wheel deck or variant data differs from the source checkout")
    print(f"Wheel check passed: {len(actual['decks'])} stock decks, {len(actual['variants'])} variants")


if __name__ == "__main__":
    main()
