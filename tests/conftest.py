"""Shared pytest fixtures. Heavy model load is replaced with a fake engine."""

from __future__ import annotations

import io
import threading
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.asr import MmsAsrEngine, TranscriptionResult, set_engine
from app.config import Settings, get_settings
from app.schemas import Segment


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    monkeypatch.setenv("MODEL_ID", "facebook/mms-1b-all")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("MAX_UPLOAD_MB", "5")
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("VTT_OUTPUT_DIR", str(tmp_path / "vtt"))
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def silent_wav_bytes() -> bytes:
    """Minimal 16 kHz mono PCM WAV (silence)."""
    sr = 16_000
    n = int(sr * 0.25)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.zeros(n, dtype=np.int16).tobytes())
    return buf.getvalue()


class FakeEngine(MmsAsrEngine):
    """Ready engine that skips torch inference."""

    def __init__(self, settings: Settings, *, ready: bool = True) -> None:
        self.settings = settings
        self.model_id = settings.model_id
        self.device = "cpu"  # type: ignore[assignment]
        self._ready = ready
        self._load_error: str | None = None if ready else "still loading"
        self._model = object()  # type: ignore[assignment]
        self._processor = object()  # type: ignore[assignment]
        self._current_mms_lang = "eng"
        self._lock = threading.RLock()

    def load(self) -> None:
        self._ready = True
        self._load_error = None

    def transcribe(
        self,
        waveform: np.ndarray,
        *,
        lang: str,
        duration_s: float,
        sampling_rate: int = 16_000,
    ) -> TranscriptionResult:
        if lang == "am":
            text = "ሰላም ዓለም"
            segments = [
                Segment(start=0.0, end=round(duration_s * 0.45, 3), text="ሰላም"),
                Segment(start=round(duration_s * 0.45, 3), end=duration_s, text="ዓለም"),
            ]
        else:
            text = "hello world"
            mid = round(duration_s * 0.5, 3)
            segments = [
                Segment(start=0.0, end=mid, text="hello"),
                Segment(start=mid, end=duration_s, text="world"),
            ]
        return TranscriptionResult(
            text=text,
            lang=lang,
            segments=segments,
            duration_ms=int(round(duration_s * 1000)),
            model=self.model_id,
            alignment="ctc",
        )


class FakeEngineNotReady(FakeEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, ready=False)

    def load(self) -> None:
        # Stay not ready — simulates model still downloading.
        self._ready = False
        self._load_error = "still loading"


@pytest.fixture
def client(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.main.MmsAsrEngine", FakeEngine)
    from app.main import app

    with TestClient(app) as c:
        yield c
    set_engine(None)


@pytest.fixture
def client_not_ready(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.main.MmsAsrEngine", FakeEngineNotReady)
    from app.main import app

    with TestClient(app) as c:
        yield c
    set_engine(None)
