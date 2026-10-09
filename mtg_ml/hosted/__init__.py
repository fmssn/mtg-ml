"""The hosted play server: play against the models behind Cloudflare Access.

    python -m mtg_ml.hosted check     # validate the configuration and the pinned checkpoints
    python -m mtg_ml.hosted serve     # the Starlette app under uvicorn

Configured from the environment only (`config.HostedConfig`, docs/hosting.md).
It reuses `live.LiveManager` (one worker thread for every game: native games
are thread-bound) and adds what a public server needs: verified accounts
(`auth`), games and replays bound to them, private seeds, request and
capacity limits, and a SQLite record of every game (`store`) from which an
unfinished game is rebuilt after a restart (`manager`).
"""
