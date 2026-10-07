"""Reviewer backends: DeepSeek (cheap first pass) and Claude (escalation).

    deepseek  DEEPSEEK_API_KEY (environment, or the nearest .env upwards);
              OpenAI-compatible chat API over plain HTTP (no extra dependency). Model `deepseek-chat` by default;
              `deepseek-reasoner` thinks first and costs more.
    claude    the `anthropic` SDK (pip install anthropic) with its usual
              credentials (ANTHROPIC_API_KEY or `ant auth login`).
    prompt    no API call: writes the system and user prompt next to the
              game so any agent (a Claude Code subagent, a person) can answer;
              put the reply in `<game>.reply.txt` and run `review`
              again with `--backend file`.
    file      reads that reply.

The system prompt and the card/deck header are identical across games, so
both APIs serve them from their prompt caches.
"""

from __future__ import annotations

import json
import os
import pathlib
import urllib.request

DEFAULTS = {"deepseek": "deepseek-chat", "claude": "claude-opus-5-5"}


def complete(backend: str, system: str, user: str, model: str | None = None, effort: str = "high") -> tuple[str, dict]:
    """(reply text, usage dict)."""
    model = model or DEFAULTS.get(backend)
    if backend == "deepseek":
        return _deepseek(system, user, model)
    if backend == "claude":
        return _claude(system, user, model, effort)
    raise ValueError(f"backend {backend!r} makes no API calls")


def _dotenv(name: str) -> str | None:
    """`name` from the environment, else from the nearest `.env` in the
    working directory or a parent (worktrees find the main checkout's)."""
    if os.environ.get(name):
        return os.environ[name]
    for d in (pathlib.Path.cwd(), *pathlib.Path.cwd().parents):
        f = d / ".env"
        if f.is_file():
            for line in f.read_text().splitlines():
                k, sep, v = line.strip().removeprefix("export ").partition("=")
                if sep and k.strip() == name:
                    return v.strip().strip("'\"")
    return None


def _deepseek(system: str, user: str, model: str) -> tuple[str, dict]:
    key = _dotenv("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("set DEEPSEEK_API_KEY")
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        # the reasoner's chain of thought counts against max_tokens
        "max_tokens": 32768 if model == "deepseek-reasoner" else 8192,
    }
    if model != "deepseek-reasoner":  # the reasoner rejects json mode and sampling settings
        body["response_format"] = {"type": "json_object"}
        body["temperature"] = 0.0
    req = urllib.request.Request(
        os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"], out.get("usage", {})


def _claude(system: str, user: str, model: str, effort: str) -> tuple[str, dict]:
    import anthropic

    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=model,
        max_tokens=64000,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        betas=["server-side-fallback-2026-07-01"],
        # fallbacks: on a safety decline, rerun on the model the API picks for the category.
        extra_body={"output_config": {"effort": effort}, "fallbacks": "default"},
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"reviewer declined: {msg.stop_details}")
    text = "".join(b.text for b in msg.content if b.type == "text")
    return text, msg.usage.to_dict()


def write_prompt(base: pathlib.Path, system: str, user: str) -> pathlib.Path:
    path = base.with_suffix(".prompt.md")
    path.write_text(f"<system>\n{system}\n</system>\n\n{user}")
    return path
