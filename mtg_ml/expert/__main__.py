"""python -m mtg_ml.expert --help"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

from .artifacts import VERSION, read, require, review, validate_evidence, write


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract, review, compile and learn from narrated MTGO evidence")
    sub = p.add_subparsers(dest="command", required=True)
    acquire = sub.add_parser("acquire", help="download source video, metadata and English captions")
    acquire.add_argument("url")
    acquire.add_argument("--out", type=Path, required=True)
    extract = sub.add_parser("extract", help="decode a source interval; optionally OCR/transcribe")
    extract.add_argument("--source", type=Path, required=True)
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--start", type=float, required=True)
    extract.add_argument("--end", type=float, required=True)
    extract.add_argument("--fps", type=float, default=2)
    extract.add_argument("--crop", help="reviewed log crop x,y,width,height; auto-locate otherwise")
    extract.add_argument("--captions", type=Path)
    extract.add_argument("--ocr", action="store_true")
    extract.add_argument("--asr", action="store_true")
    extract.add_argument("--ocr-device", default="cpu")
    extract.add_argument("--asr-device", choices=("cpu", "cuda"), default="cpu")
    extract.add_argument("--asr-model", default="large-v3")
    extract.add_argument("--aliases", type=Path)
    native = sub.add_parser("import-log", help="import a text game log or mtgo_utils .dat export")
    native.add_argument("evidence", type=Path)
    native.add_argument("log", type=Path)
    native.add_argument("--source-id", required=True)
    native.add_argument("--start", type=float, required=True)
    native.add_argument("--end", type=float, required=True)
    native.add_argument("--parser", help="mtgo_utils file_parser executable for .dat")
    native.add_argument("--out", type=Path, required=True)
    alignment = sub.add_parser("align", help="rank log/commentary links without approving them")
    alignment.add_argument("evidence", type=Path)
    alignment.add_argument("--aliases", type=Path)
    alignment.add_argument("--out", type=Path, required=True)
    vision = sub.add_parser("interpret", help="pending visible-state/commentary extraction via a vision model")
    vision.add_argument("evidence", type=Path)
    vision.add_argument("--frames", type=Path, required=True)
    vision.add_argument("--backend", choices=("local", "openai", "gemini"), default="local")
    vision.add_argument("--model")
    vision.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    vision.add_argument("--start", type=float, required=True)
    vision.add_argument("--end", type=float, required=True)
    vision.add_argument("--viewer", type=int, choices=(0, 1), required=True)
    vision.add_argument("--max-frames", type=int, default=12)
    vision.add_argument("--source-id", help="required when evidence has multiple video sources")
    vision.add_argument("--video", help="YouTube URL for Gemini's direct-video comparison")
    vision.add_argument("--out", type=Path, required=True)
    rev = sub.add_parser("review", help="apply explicit review patches; preserve unreviewed records")
    rev.add_argument("evidence", type=Path)
    rev.add_argument("patch", type=Path)
    rev.add_argument("--out", type=Path, required=True)
    template = sub.add_parser("template", help="draft a scenario from accepted state facts (missing fields stay null)")
    template.add_argument("evidence", type=Path)
    template.add_argument("--id", required=True)
    template.add_argument("--source-id", required=True)
    template.add_argument("--time", type=float, required=True)
    template.add_argument("--viewer", type=int, choices=(0, 1), required=True)
    template.add_argument("--out", type=Path, required=True)
    for name in ("compile", "regression", "export", "replay", "verify"):
        cmd = sub.add_parser(name)
        cmd.add_argument("scenario", type=Path)
        cmd.add_argument("--evidence", type=Path, required=True)
        cmd.add_argument("--engine", choices=("python", "native"), default="python")
        cmd.add_argument("--out", type=Path, required=True)
        if name == "export":
            cmd.add_argument("--features", type=int, default=6)
    reconstruction = sub.add_parser("reconstruct", help="bounded legal replay between reviewed evidence checkpoints")
    reconstruction.add_argument("evidence", type=Path)
    reconstruction.add_argument("--spec", type=Path, required=True)
    reconstruction.add_argument("--engine", choices=("python", "native"), default="native")
    reconstruction.add_argument("--out", type=Path, required=True)
    for field in ("candidates", "expansions", "actions", "seconds", "escalations"):
        reconstruction.add_argument("--" + field, type=int, default=None, help="override this reconstruction search budget")
    checkpoint = sub.add_parser("checkpoint-template", help="pending scenario using only one checkpoint's as-of facts")
    checkpoint.add_argument("evidence", type=Path)
    checkpoint.add_argument("--spec", type=Path, required=True)
    checkpoint.add_argument("--checkpoint", required=True)
    checkpoint.add_argument("--id", required=True)
    checkpoint.add_argument("--out", type=Path, required=True)
    metrics = sub.add_parser("metrics", help="ordered extraction accuracy against a manual gold log")
    metrics.add_argument("evidence", type=Path)
    metrics.add_argument("--gold", type=Path, required=True, help="JSON list of event texts in order")
    metrics.add_argument("--start", type=float, default=0, help="first-seen source time, inclusive")
    metrics.add_argument("--end", type=float, default=float("inf"), help="first-seen source time, inclusive")
    metrics.add_argument("--review-minutes", type=float, help="measured correction time; omitted means unmeasured")
    metrics.add_argument("--api-cost", type=float, help="measured monetary cost; omitted means unmeasured")
    metrics.add_argument("--out", type=Path, required=True)
    score = sub.add_parser("score", help="evaluate a checkpoint on compiled expert episodes")
    score.add_argument("checkpoint", type=Path)
    score.add_argument("episodes", type=Path, nargs="+")
    score.add_argument("--device", default="cpu")
    score.add_argument("--out", type=Path, required=True)
    bc = sub.add_parser("train", help="supervised legal-action warm start; separate from PPO")
    bc.add_argument("checkpoint", type=Path)
    bc.add_argument("episodes", type=Path, nargs="+")
    bc.add_argument("--out", type=Path, required=True)
    bc.add_argument("--epochs", type=int, default=5)
    bc.add_argument("--lr", type=float, default=1e-5)
    bc.add_argument("--heldout-fraction", type=float, default=0.2)
    bc.add_argument("--device", default="cpu")
    bc.add_argument("--seed", type=int, default=0)
    return p


def apply_reviews(data: dict, patches: list[dict]) -> dict:
    validate_evidence(data)
    result = copy.deepcopy(data)
    records = {r["id"]: r for r in result["records"]}
    ids = set()
    for patch in patches:
        rid = patch["id"]
        require(rid in records and rid not in ids, "unknown or duplicate reviewed record id")
        ids.add(rid)
        require(set(patch) <= {"id", "value", "review", "refs", "visibility", "available_at", "classification", "uncertainty"}, "unknown review patch field")
        records[rid].update(patch)
    return validate_evidence(result)


def scenario_template(data: dict, source_id: str, sid: str, time: float, viewer: int) -> dict:
    from .artifacts import admissible

    validate_evidence(data)
    source = next((s for s in data["sources"] if s["id"] == source_id), None)
    require(source is not None, "unknown source id")
    initial = {"players": [{k: None for k in ("life", "drawn", "hand", "hand_count", "battlefield", "graveyard", "exile", "library", "library_count")} for _ in range(2)],
               "active": None, "step": None, "turn": None, "lands_played": None, "spells_cast_this_turn": None,
               "seed": 0, "match_game": 1}
    bindings = {}
    for record in data["records"]:
        if record["kind"] != "state" or not admissible(record, viewer, time) or any(r["source_id"] != source_id for r in record["refs"]):
            continue
        path = record["value"].get("path", "")
        if path in bindings:
            raise ValueError(f"conflicting/duplicate state facts for {path}; resolve them before templating")
        obj = initial
        bits = path.split(".")
        try:
            for bit in bits[:-1]:
                obj = obj[int(bit)] if isinstance(obj, list) else obj[bit]
            require(bits[-1] in obj, "unsupported template field")
            obj[bits[-1]] = record["value"]["value"]
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise ValueError(f"unsupported state path: {path}") from e
        bindings[path] = [record["id"]]
    return {"format": "ExecutableScenario", "version": VERSION, "id": sid, "group_id": source.get("group_id", source_id),
            "perspective": viewer, "decision_time": time, "episode_mode": "reset", "review": review(), "initial": initial,
            "state_facts": bindings, "synthetic_fields": {}, "assumptions": [], "setup_actions": [], "demonstration": [],
            "preferred": [], "continuation": [], "expected_view": {}, "expected_after": {}}


def run(args) -> dict:
    name = args.command
    if name == "acquire":
        from .media import acquire

        return acquire(args.url, args.out)
    if name == "extract":
        from .media import extract

        crop = None
        if args.crop:
            x, y, w, h = map(int, args.crop.split(","))
            crop = (x, y, x + w, y + h)
        return extract(read(args.source), args.out, args.start, args.end, crop=crop, fps=args.fps, captions=args.captions,
                       run_ocr=args.ocr, run_asr=args.asr, ocr_device=args.ocr_device, asr_device=args.asr_device,
                       asr_model=args.asr_model, aliases=read(args.aliases) if args.aliases else None)
    if name in {"import-log", "align", "interpret", "review", "template", "metrics", "reconstruct", "checkpoint-template"}:
        data = validate_evidence(read(args.evidence))
    if name == "reconstruct":
        from .reconstruction import reconstruct

        spec = read(args.spec)
        overrides = {field: getattr(args, field) for field in ("candidates", "expansions", "actions", "seconds", "escalations")
                     if getattr(args, field) is not None}
        spec["budget"] = {**spec.get("budget", {}), **overrides}
        result = reconstruct(data, spec, args.engine,
                             on_progress=lambda status: write(args.out.with_suffix(".progress.json"), status))
    elif name == "checkpoint-template":
        from .reconstruction import checkpoint_template

        result = checkpoint_template(data, read(args.spec), args.checkpoint, args.id)
    elif name == "import-log":
        from .media import import_log

        data["records"].extend(import_log(args.log, args.source_id, args.start, args.end, args.parser))
        result = validate_evidence(data)
    elif name == "align":
        from ..engine.cards import CARDS, TOKENS
        from .extraction import align

        align(data["records"], list(CARDS) + list(TOKENS), read(args.aliases) if args.aliases else None)
        result = data
    elif name == "interpret":
        from .interpret import InterpretationError, interpret

        frames = [f for f in read(args.frames) if args.start <= f["time"] <= args.end]
        require(args.max_frames > 0, "positive max-frames required")
        if len(frames) > args.max_frames:
            frames = [frames[int(i * (len(frames) - 1) / max(1, args.max_frames - 1))] for i in range(args.max_frames)]
        transcript = [r for r in data["records"] if r["kind"] == "commentary" and all(args.start <= t["start"] <= t["end"] <= args.end for t in r["refs"])]
        require(args.source_id or len(data["sources"]) == 1, "multiple sources: select --source-id")
        source_id = args.source_id or data["sources"][0]["id"]
        require(any(s["id"] == source_id for s in data["sources"]), "unknown interpretation source")
        transcript = [r for r in transcript if all(t["source_id"] == source_id for t in r["refs"])]
        try:
            records, metrics = interpret(frames, transcript, source_id, args.viewer, backend=args.backend, model=args.model,
                                         endpoint=args.endpoint, video=args.video, start=args.start, end=args.end)
        except InterpretationError as e:
            write(args.out.with_suffix(".attempt.json"), e.metrics)
            raise
        existing = {r["id"] for r in data["records"]}
        for r in records:
            r["id"] = f"{args.backend}-{len(existing)}-{r['id']}"
            existing.add(r["id"])
        data["records"].extend(records)
        data["metrics"][f"{args.backend}-{args.start}-{args.end}"] = metrics
        result = validate_evidence(data)
    elif name == "review":
        result = apply_reviews(data, read(args.patch))
    elif name == "template":
        result = scenario_template(data, args.source_id, args.id, args.time, args.viewer)
    elif name in {"compile", "regression", "export", "replay", "verify"}:
        spec, data = read(args.scenario), read(args.evidence)
        if name == "verify":
            from .scenarios import verify

            result = verify(spec, data)
        elif name == "export":
            from .learning import demonstrations

            result = demonstrations(spec, data, args.engine, args.features)
        elif name == "regression":
            from .scenarios import regression

            result = regression(spec, data, args.engine)
        elif name == "replay":
            from .scenarios import replay

            result = replay(spec, data, args.engine)
        else:
            from ..engine.view import observe
            from .scenarios import compile_scenario

            g, _ = compile_scenario(spec, data, args.engine)
            result = {"format": "CompiledScenario", "version": VERSION, "scenario_id": spec["id"],
                      "view": observe(g, spec["perspective"]), "engine": args.engine, "episode_mode": "reset"}
    elif name == "metrics":
        from .extraction import event_metrics

        require(all(v is None or v >= 0 for v in (args.review_minutes, args.api_cost)), "cost and review duration cannot be negative")
        require(0 <= args.start <= args.end, "invalid metrics interval")
        result = event_metrics([r["value"]["text"] for r in data["records"] if r["kind"] == "log"
                                and args.start <= r["refs"][0]["start"] <= args.end], read(args.gold))
        result.update(review_minutes=args.review_minutes, api_cost=args.api_cost,
                      accepted_records=sum(r["review"]["status"] == "accepted" for r in data["records"]))
        result["metrics"] = data.get("metrics", {})
    elif name in {"score", "train"}:
        from .learning import score, train

        episodes = [read(path) for path in args.episodes]
        if name == "train":
            result = train(args.checkpoint, episodes, args.out, epochs=args.epochs, lr=args.lr,
                           heldout_fraction=args.heldout_fraction, device=args.device, seed=args.seed)
            write(args.out.with_suffix(".report.json"), result)
            return result
        from ..rl.rollout import load_net

        result = score(load_net(str(args.checkpoint)).to(args.device), episodes, args.device)
    else:
        raise ValueError(f"unknown command {name}")
    write(args.out, result)
    return result


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except (ValueError, RuntimeError, KeyError, OSError) as e:
        print(f"expert: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    # Never print transcript contents, signed URLs, or image payloads.
    metrics = {key: len(value) if isinstance(value, list) else value for key, value in result.get("metrics", {}).items()}
    print(json.dumps({"command": args.command, "out": str(args.out), "format": result.get("format"),
                      "records": len(result.get("records", [])), "metrics": metrics}, default=str))


if __name__ == "__main__":
    main()
