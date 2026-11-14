from __future__ import annotations


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["ready"] is True
    assert body["model"] == "facebook/mms-1b-all"
    assert body["version"] == "0.1.0"


def test_ready_ok(client):
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json()["ready"] is True


def test_ready_503(client_not_ready):
    res = client_not_ready.get("/ready")
    assert res.status_code == 503
    body = res.json()
    assert body["ready"] is False
    assert body["error"]["code"] == "MODEL_NOT_READY"


def test_v1_info(client):
    res = client.get("/v1/info")
    assert res.status_code == 200
    body = res.json()
    assert body["langs"] == ["am", "en"]
    assert body["langMapping"]["am"] == "amh"
    assert body["langMapping"]["en"] == "eng"
    assert body["sendsDataOffHost"] is False
    assert body["maxUploadMb"] == 5


def test_transcribe_en(client, silent_wav_bytes):
    res = client.post(
        "/transcribe?lang=en",
        files={"file": ("clip.wav", silent_wav_bytes, "audio/wav")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["text"] == "hello world"
    assert body["lang"] == "en"
    assert body["needsReview"] is True
    assert body["sendsDataOffHost"] is False
    assert body["model"] == "facebook/mms-1b-all"
    assert isinstance(body["durationMs"], int)
    assert len(body["segments"]) >= 2
    assert body["segments"][0]["start"] == 0.0
    assert body["segments"][0]["text"] == "hello"
    assert body["vtt"].startswith("WEBVTT")
    assert body["alignment"] == "ctc"


def test_transcribe_am_form_lang(client, silent_wav_bytes):
    res = client.post(
        "/transcribe",
        data={"lang": "am"},
        files={"file": ("clip.wav", silent_wav_bytes, "audio/wav")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["lang"] == "am"
    assert "ሰላም" in body["text"]
    assert body["needsReview"] is True
    assert body["vtt"].startswith("WEBVTT")
    assert body["alignment"] == "ctc"


def test_transcribe_unsupported_lang(client, silent_wav_bytes):
    res = client.post(
        "/transcribe?lang=fr",
        files={"file": ("clip.wav", silent_wav_bytes, "audio/wav")},
    )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "UNSUPPORTED_LANG"


def test_transcribe_missing_lang(client, silent_wav_bytes):
    res = client.post(
        "/transcribe",
        files={"file": ("clip.wav", silent_wav_bytes, "audio/wav")},
    )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "UNSUPPORTED_LANG"


def test_transcribe_model_not_ready(client_not_ready, silent_wav_bytes):
    res = client_not_ready.post(
        "/transcribe?lang=en",
        files={"file": ("clip.wav", silent_wav_bytes, "audio/wav")},
    )
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_NOT_READY"


def test_transcribe_empty_file(client):
    res = client.post(
        "/transcribe?lang=en",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "BAD_AUDIO"


def test_api_key_optional_when_unset(client, silent_wav_bytes):
    res = client.post(
        "/transcribe?lang=en",
        files={"file": ("clip.wav", silent_wav_bytes, "audio/wav")},
        headers={"X-API-Key": "anything"},
    )
    assert res.status_code == 200


def test_api_key_enforced(monkeypatch, silent_wav_bytes):
    monkeypatch.setenv("MODEL_ID", "facebook/mms-1b-all")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("MAX_UPLOAD_MB", "5")
    monkeypatch.setenv("API_KEY", "secret-token")
    from app.config import get_settings
    from app.asr import set_engine
    from app.main import app
    from tests.conftest import FakeEngine

    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr("app.main.MmsAsrEngine", FakeEngine)
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        denied = c.post(
            "/transcribe?lang=en",
            files={"file": ("clip.wav", silent_wav_bytes, "audio/wav")},
        )
        assert denied.status_code == 401
        ok = c.post(
            "/transcribe?lang=en",
            files={"file": ("clip.wav", silent_wav_bytes, "audio/wav")},
            headers={"X-API-Key": "secret-token"},
        )
        assert ok.status_code == 200
    set_engine(None)
    monkeypatch.setenv("API_KEY", "")
    get_settings.cache_clear()
