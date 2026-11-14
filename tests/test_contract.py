"""Freeze the HTTP JSON contract the TypeScript Transcriber adapter consumes."""

from __future__ import annotations

REQUIRED_TRANSCRIBE_KEYS = {
    "text",
    "lang",
    "segments",
    "sendsDataOffHost",
    "model",
    "durationMs",
    "needsReview",
    "vtt",
    "alignment",
    "vttPath",
}

REQUIRED_SEGMENT_KEYS = {"start", "end", "text"}
REQUIRED_HEALTH_KEYS = {"ok", "ready", "model", "version"}
REQUIRED_ERROR_KEYS = {"error"}


def test_transcribe_response_shape(client, silent_wav_bytes):
    res = client.post(
        "/transcribe?lang=en",
        files={"file": ("clip.wav", silent_wav_bytes, "audio/wav")},
    )
    assert res.status_code == 200
    body = res.json()
    assert REQUIRED_TRANSCRIBE_KEYS <= set(body.keys())
    assert body["sendsDataOffHost"] is False
    assert body["needsReview"] is True
    assert body["lang"] == "en"
    assert isinstance(body["text"], str)
    assert isinstance(body["durationMs"], int)
    assert isinstance(body["segments"], list) and len(body["segments"]) >= 1
    seg = body["segments"][0]
    assert REQUIRED_SEGMENT_KEYS <= set(seg.keys())
    assert isinstance(seg["start"], (int, float))
    assert isinstance(seg["end"], (int, float))
    assert isinstance(seg["text"], str)
    assert body["alignment"] == "ctc"
    assert isinstance(body["vtt"], str) and body["vtt"].startswith("WEBVTT")
    assert "-->" in body["vtt"]


def test_health_response_shape(client):
    body = client.get("/health").json()
    assert REQUIRED_HEALTH_KEYS <= set(body.keys())
    assert body["ok"] is True
    assert body["model"] == "facebook/mms-1b-all"
    assert body["version"] == "0.1.0"


def test_error_response_shape(client, silent_wav_bytes):
    res = client.post(
        "/transcribe?lang=fr",
        files={"file": ("clip.wav", silent_wav_bytes, "audio/wav")},
    )
    assert res.status_code == 400
    body = res.json()
    assert REQUIRED_ERROR_KEYS <= set(body.keys())
    assert set(body["error"].keys()) == {"code", "message"}
    assert body["error"]["code"] == "UNSUPPORTED_LANG"
