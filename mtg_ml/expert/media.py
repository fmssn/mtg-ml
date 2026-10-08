"""Optional media adapters; neither the rules engine nor RL imports these."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time

from .artifacts import evidence, fact, read, ref, require, write
from .extraction import align, merge_log, parse_vtt


def dependency(name: str, extra: str):
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise RuntimeError(f"{name} is needed for this stage; install mtg-ml[{extra}] in its processing environment") from e


def command(args: list[str], log: Path | None = None) -> None:
    require(shutil.which(args[0]) is not None, f"required executable not found: {args[0]}")
    if log is not None:
        with log.open("w") as f:
            result = subprocess.run(args, stdout=f, stderr=f)
        require(result.returncode == 0, f"{args[0]} failed; see {log}")
    else:
        result = subprocess.run(args, capture_output=True, text=True)
        require(result.returncode == 0, f"{args[0]} failed: {result.stderr[-1500:]}")


def acquire(url: str, directory: Path) -> dict:
    """Download a full source: original timestamps remain unambiguous."""
    yt = dependency("yt_dlp", "expert-media")
    directory.mkdir(parents=True, exist_ok=True)
    opts = {"outtmpl": str(directory / "video.%(ext)s"), "format": "bestvideo+bestaudio/best", "merge_output_format": "mp4",
            "writeautomaticsub": True, "writesubtitles": True, "subtitleslangs": ["en"], "subtitlesformat": "vtt",
            "writeinfojson": True, "noplaylist": True, "quiet": True, "no_warnings": True}
    with yt.YoutubeDL(opts) as downloader:
        info = downloader.extract_info(url, download=True)
    paths = sorted(directory.glob("video.*"))
    media = next((p for p in paths if p.suffix in {".mp4", ".mkv", ".webm"}), None)
    require(media is not None, "downloader produced no supported video")
    source = {"id": info["id"], "url": url, "title": info.get("title"), "media": str(media.resolve()), "media_offset": 0,
              "duration": info.get("duration"), "chapters": info.get("chapters", []), "description": info.get("description", ""),
              "group_id": info["id"]}
    write(directory / "source.json", source)
    return source


def decode_frames(media: Path, directory: Path, start: float, end: float, fps: float = 2) -> list[dict]:
    require(0 <= start < end and 0 < fps <= 30, "invalid clip interval or frame rate")
    directory.mkdir(parents=True, exist_ok=True)
    # Use a separate directory on every run: old frames must not become new
    # observations when the clip/fps changes.
    require(not any(directory.glob("*.png")), "frame directory already contains frames; choose a fresh output directory")
    command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-i", str(media), "-t", str(end - start),
             "-vf", f"fps={fps}", str(directory / "%06d.png")])
    return [{"time": start + i / fps, "frame": str(p.resolve())} for i, p in enumerate(sorted(directory.glob("*.png")))]


class PaddleReader:
    def __init__(self, device: str = "cpu"):
        paddle = dependency("paddleocr", "expert-ocr")
        self.ocr = paddle.PaddleOCR(lang="en", device=device, use_doc_orientation_classify=False,
                                   use_doc_unwarping=False, use_textline_orientation=False)

    def __call__(self, image) -> list[dict]:
        import numpy as np

        result = next(iter(self.ocr.predict(np.asarray(image.convert("RGB")))))
        data = result.json
        if isinstance(data, str):
            data = json.loads(data)
        data = data.get("res", data)
        boxes = data.get("rec_polys", data.get("dt_polys", []))
        return [{"text": text, "score": float(score), "box": [[float(x), float(y)] for x, y in box]}
                for text, score, box in zip(data["rec_texts"], data["rec_scores"], boxes)]


def locate_log(boxes: list[dict], size: tuple[int, int], image=None) -> tuple[int, int, int, int]:
    """Find a right-side cluster of MTGO log sentences; fail closed otherwise."""
    import re

    w, h = size
    pattern = re.compile(r"\b(casts|plays|draws|turn|mulligan|chooses|loses|wins|resolves|attacks)\b", re.I)
    hits = [b for b in boxes if pattern.search(b["text"]) and min(p[0] for p in b["box"]) > w / 2]
    require(len(hits) >= 2, "could not locate a log panel; supply a reviewed --crop x,y,width,height")
    x = max(0, int(min(p[0] for b in hits for p in b["box"])) - 12)
    if image is not None:
        import numpy as np

        # Identify the light log surface around the detected sentences. A
        # webcam or chat below it must not enter the game event stream.
        pixels = np.asarray(image.convert("L"))[:, x + 12:w - 8]
        light = (pixels > 215).mean(axis=1) > 0.65
        y = int(sum(min(p[1] for p in b["box"]) for b in hits) / len(hits))
        if light[y]:
            top, bottom = y, y
            while top > 0 and light[top - 1]:
                top -= 1
            while bottom + 1 < h and light[bottom + 1]:
                bottom += 1
            if bottom - top >= 100:
                return x, top, w, bottom + 1
    raise ValueError("could not establish the log panel boundaries; supply a reviewed --crop")


def text_lines(boxes: list[dict]) -> list[str]:
    ordered = sorted(boxes, key=lambda b: (min(p[1] for p in b["box"]), min(p[0] for p in b["box"])))
    lines = []
    last_y = None
    for box in ordered:
        y = min(p[1] for p in box["box"])
        height = max(p[1] for p in box["box"]) - y
        if last_y is not None and abs(y - last_y) < max(3, height / 3):
            lines[-1] += " " + box["text"]
        else:
            lines.append(box["text"])
            last_y = y
    return lines


def ocr_frames(frames: list[dict], source_id: str, reader, crop: tuple | None = None, threshold: float = 0.5) -> tuple[list[dict], dict]:
    import numpy as np

    Image = dependency("PIL.Image", "expert-media")
    previous, windows, raw = None, [], []
    t0 = time.monotonic()
    for frame in frames:
        with Image.open(frame["frame"]) as image:
            if crop is None:
                crop = locate_log(reader(image), image.size, image)
            require(0 <= crop[0] < crop[2] <= image.width and 0 <= crop[1] < crop[3] <= image.height, "crop outside frame")
            panel = image.crop(crop)
            pixels = np.asarray(panel.convert("L").resize((256, 256)), dtype=np.float32)
            delta = None if previous is None else float(np.abs(pixels - previous).mean())
            previous = pixels
            if delta is not None and delta < threshold:
                continue
            boxes = reader(panel)
            from .extraction import logical_log_lines

            lines = logical_log_lines(text_lines(boxes))
            windows.append({**frame, "lines": lines, "occluded": not lines})
            raw.append({**frame, "crop": list(crop), "boxes": boxes, "change": delta})
    return merge_log(windows, source_id), {"frames": len(frames), "ocr_windows": len(windows), "wall_seconds": time.monotonic() - t0,
                                           "crop": list(crop) if crop else None, "raw_windows": raw}


def transcribe(media: Path, source_id: str, start: float, end: float, directory: Path, model: str = "large-v3", device: str = "cpu", offset: float = 0) -> tuple[list[dict], dict]:
    whisper = dependency("faster_whisper", "expert-asr")
    audio = directory / "speech.wav"
    command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start), "-i", str(media), "-t", str(end - start),
             "-vn", "-ac", "1", "-ar", "16000", str(audio)])
    t0 = time.monotonic()
    net = whisper.WhisperModel(model, device=device, compute_type="float16" if device == "cuda" else "int8")
    segments, _ = net.transcribe(str(audio), language="en", word_timestamps=True, vad_filter=True, condition_on_previous_text=False)
    records = []
    for seg in segments:
        a, b = offset + start + seg.start, offset + start + seg.end
        words = [{"start": offset + start + w.start, "end": offset + start + w.end, "word": w.word, "probability": w.probability} for w in seg.words or []]
        records.append(fact(f"speech-{len(records)}", "commentary", {"text": seg.text.strip(), "words": words},
                            [ref(source_id, a, b, "audio", str(audio.resolve()))], classification="other"))
    return records, {"model": model, "device": device, "wall_seconds": time.monotonic() - t0}


def extract(source: dict, directory: Path, start: float, end: float, *, crop=None, fps=2, captions: Path | None = None,
            run_ocr=False, run_asr=False, ocr_device="cpu", asr_device="cpu", asr_model="large-v3", aliases=None) -> dict:
    from ..engine.cards import CARDS, TOKENS

    directory.mkdir(parents=True, exist_ok=True)
    require(not (directory / "evidence.json").exists(), "extraction output exists; choose a fresh directory to preserve reviews")
    source = dict(source)
    source["segment"] = {"start": start, "end": end}
    data = evidence(source)
    offset = source.get("media_offset", 0)
    media = Path(source["media"])
    manifest = {"media": str(media.resolve()), "bytes": media.stat().st_size, "mtime_ns": media.stat().st_mtime_ns,
                "start": start, "end": end, "offset": offset, "fps": fps}
    manifest_path = directory / "decode-manifest.json"
    if (directory / "frames.json").exists():
        require(manifest_path.exists() and read(manifest_path) == manifest, "existing frames belong to different media/interval; use a fresh directory")
        frames = read(directory / "frames.json")
        require(all(Path(f["frame"]).exists() for f in frames), "cached frame files are missing")
    else:
        write(manifest_path, manifest)
        frames = decode_frames(media, directory / "frames", start - offset, end - offset, fps)
        for frame in frames:
            frame["time"] += offset
        write(directory / "frames.json", frames)
    if run_ocr:
        records, metrics = ocr_frames(frames, source["id"], PaddleReader(ocr_device), crop)
        write(directory / "ocr-windows.json", metrics.pop("raw_windows"))
        data["records"].extend(records)
        data["metrics"]["ocr"] = metrics
    if captions:
        data["records"].extend(parse_vtt(captions.read_text(), source["id"], str(captions.resolve()), start, end))
    if run_asr:
        records, metrics = transcribe(media, source["id"], start - offset, end - offset, directory, asr_model, asr_device, offset)
        data["records"].extend(records)
        data["metrics"]["asr"] = metrics
    align(data["records"], list(CARDS) + list(TOKENS), aliases)
    write(directory / "evidence.json", data)
    return data


def import_log(path: Path, source_id: str, start: float, end: float, parser: str | None = None) -> list[dict]:
    """Accept a text export or invoke mtgo_utils; never decode binary as text.

    A .dat export establishes line order, not video timing. Its broad source
    interval must be narrowed during review before it can supply state facts.
    """
    if path.suffix.lower() == ".dat":
        require(parser is not None and shutil.which(parser) is not None, ".dat input requires --parser pointing to mtgo_utils file_parser")
        result = subprocess.run([parser, str(path)], capture_output=True, text=True)
        require(result.returncode == 0, f"MTGO parser failed: {result.stderr[-500:]}")
        text = result.stdout
    else:
        text = path.read_text(encoding="utf-8-sig")
    return [fact(f"native-log-{i}", "log", {"text": line, "sequence": i, "timing": "unaligned"},
                 [ref(source_id, start, end, "mtgo_log", f"{path.resolve()}#line-{i + 1}")], visibility="public", available_at=end)
            for i, line in enumerate(text.splitlines()) if line.strip()]
