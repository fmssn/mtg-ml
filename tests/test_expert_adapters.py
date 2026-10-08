"""No downloads or credentials: exercise real transport contracts and failure accounting."""
import json

import pytest

from mtg_ml.expert import interpret as vision
from mtg_ml.expert.artifacts import read
from mtg_ml.expert.extraction import logical_log_lines
from mtg_ml.expert.media import extract


def model_output(frame, time=1):
    return {"records": [{"kind": "state", "field": "players.0.life", "value_json": "20", "visibility": "public",
                         "classification": "other", "start": time, "end": time, "locator": str(frame), "uncertainty": ""}]}


@pytest.mark.parametrize("backend", ["local", "openai", "gemini"])
def test_vision_adapter_payload_and_pending_output(tmp_path, monkeypatch, backend):
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"test-image")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    output = json.dumps(model_output(frame))

    def post(url, payload, headers):
        if backend == "gemini":
            assert url.endswith("/interactions") and payload["store"] is False
            assert payload["response_format"]["mime_type"] == "application/json"
            assert payload["input"][-1]["type"] == "image"
            return {"status": "completed", "outputs": [{"type": "text", "text": output}], "usage": {"input_tokens": 100}}
        assert url.endswith("/chat/completions")
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["messages"][0]["content"][-1]["image_url"]["detail"] == ("original" if backend == "openai" else "auto")
        return {"choices": [{"finish_reason": "stop", "message": {"content": output}}], "usage": {"prompt_tokens": 100}}

    monkeypatch.setattr(vision, "_post", post)
    records, metrics = vision.interpret([{"frame": str(frame), "time": 1}], [], "v", 0, backend=backend, start=0, end=2)
    assert records[0]["value"]["value"] == 20 and records[0]["review"]["status"] == "pending"
    assert metrics["accepted"] == 0 and metrics["usage"]


def test_bad_model_value_accounts_for_paid_failed_attempt(monkeypatch):
    monkeypatch.setattr(vision, "_post", lambda *a: {"choices": [{"finish_reason": "stop", "message": {"content": '{"records":[{"kind":"state","field":"turn","value_json":"","visibility":"public","classification":"other","start":1,"end":1,"locator":"frame","uncertainty":""}]}'}}], "usage": {"completion_tokens": 42}})
    monkeypatch.setattr(vision, "image_url", lambda p: "data:image/png;base64,AA==")
    with pytest.raises(vision.InterpretationError) as err:
        vision.interpret([{"frame": "frame", "time": 1}], [], "v", 0, start=0, end=2)
    assert err.value.metrics["usage"]["completion_tokens"] == 42
    assert err.value.metrics["review_status"] == "rejected" and err.value.metrics["wall_seconds"] >= 0


def test_gemini_video_offsets_use_original_timestamps(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def post(url, payload, headers):
        clip = payload["input"][-1]
        assert clip["type"] == "video" and clip["processing"] == {"type": "static", "start_offset": "1359s", "end_offset": "1380s", "fps": 2}
        return {"status": "completed", "outputs": [{"type": "text", "text": '{"records":[]}'}]}

    monkeypatch.setattr(vision, "_post", post)
    assert vision.interpret([], [], "v", 0, backend="gemini", video="https://www.youtube.com/watch?v=test", start=1359, end=1380)[0] == []


def test_wrapped_log_lines_preserve_identical_occurrences():
    lines = ["10:45 AM: p casts", "Ponder.", "10:45 AM: p casts", "Ponder."]
    assert logical_log_lines(lines) == ["10:45 AM: p casts Ponder."] * 2


def test_decode_cache_never_relabels_frames_from_another_clip(tmp_path, monkeypatch):
    from mtg_ml.expert import media

    video = tmp_path / "video.mp4"
    video.write_bytes(b"media")
    source = {"id": "v", "media": str(video), "media_offset": 10}
    output = tmp_path / "output"

    def decode(path, directory, start, end, fps):
        assert (start, end) == (0, 2)
        directory.mkdir()
        frame = directory / "1.png"
        frame.write_bytes(b"frame")
        return [{"frame": str(frame), "time": 0}]

    monkeypatch.setattr(media, "decode_frames", decode)
    extract(source, output, 10, 12)
    assert read(output / "frames.json")[0]["time"] == 10
    with pytest.raises(ValueError, match="preserve reviews"):
        extract(source, output, 10, 12)
    (output / "evidence.json").unlink()
    with pytest.raises(ValueError, match="different media/interval"):
        extract(source, output, 11, 12)
    monkeypatch.setattr(media, "decode_frames", lambda *a: pytest.fail("cached frames should be reused"))
    extract(source, output, 10, 12)
