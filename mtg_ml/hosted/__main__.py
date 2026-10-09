"""python -m mtg_ml.hosted check|serve (configuration from the environment: config.py)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile

from .config import ConfigError, HostedConfig


def check(cfg: HostedConfig) -> int:
    """Validate the play offer's pinned checkpoints and the writable directories."""
    from ..live import load_play_config, validate_offer

    problems = []
    try:
        offer = load_play_config(cfg.play_config)
    except (OSError, ValueError) as e:
        print(f"play config: {e}", file=sys.stderr)
        return 1
    pinned, bad = validate_offer(cfg.models, offer, strict=True)
    problems += bad
    for d in (cfg.state, cfg.replays):
        try:
            d.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=d):
                pass
        except OSError as e:
            problems.append(f"{d}: not writable ({e})")
    for p in problems:
        print(f"problem: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"ok: {len(offer.get('opponents', []))} opponents, models " + (", ".join(f"{k} {v['sha256'][:12]}" for k, v in pinned.items()) or "none")
          + f"; runtime {cfg.runtime}; auth {'INSECURE DEV USER ' + cfg.dev_user if cfg.dev else 'Cloudflare Access ' + cfg.team_domain}")
    return 0


def serve(cfg: HostedConfig) -> None:
    import uvicorn

    from .app import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if cfg.dev:
        logging.getLogger("mtg_ml.hosted").warning("INSECURE: every request is %s (MTG_HOSTED_INSECURE_DEV_USER); never expose this server", cfg.dev_user)
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, server_header=False, proxy_headers=False, timeout_graceful_shutdown=10)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mtg_ml.hosted", description="the hosted play server (configured from the environment)")
    ap.add_argument("cmd", choices=["check", "serve"])
    args = ap.parse_args(argv)
    try:
        cfg = HostedConfig.from_env(os.environ)
    except ConfigError as e:
        for line in str(e).splitlines():
            print(f"config: {line}", file=sys.stderr)
        sys.exit(1)
    if args.cmd == "check":
        sys.exit(check(cfg))
    serve(cfg)


if __name__ == "__main__":
    main()
