"""HTTP wrapper routes for Meta MMS ASR (Abol Kit transcriber contract)."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, Query, UploadFile
from fastapi.responses import JSONResponse

from app import __version__
from app.asr import (
    MmsAsrEngine,
    ModelNotReadyError,
    UnsupportedLangError,
    get_engine,
    set_engine,
)
from app.audio import AudioError, cleanup, load_waveform, normalize_to_wav16k_mono, write_upload_to_temp
from app.config import Settings, get_settings
from app.schemas import (
    LANG_TO_MMS,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    InfoResponse,
    TranscribeResponse,
)
from app.vtt import segments_to_vtt, write_vtt_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("abol-mms-stt")


def _error(code: str, message: str, status: int) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message))  # type: ignore[arg-type]
    return JSONResponse(status_code=status, content=body.model_dump())


def _check_api_key(settings: Settings, x_api_key: str | None) -> JSONResponse | None:
    expected = settings.api_key
    if not expected:
        return None
    if not x_api_key or x_api_key != expected:
        return _error("INTERNAL", "invalid or missing X-API-Key", 401)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = MmsAsrEngine(settings)
    set_engine(engine)
    app.state.settings = settings
    app.state.engine = engine

    # Load model in a daemon thread so /health answers while weights download.
    import threading

    def _load() -> None:
        try:
            engine.load()
        except Exception:
            logger.exception("background model load failed")

    thread = threading.Thread(target=_load, name="mms-load", daemon=True)
    thread.start()
    logger.info("abol-mms-stt %s starting; model load in background", __version__)
    yield
    set_engine(None)
    logger.info("abol-mms-stt shutting down")


app = FastAPI(
    title="abol-mms-stt",
    description="Self-hosted Meta MMS ASR wrapper for Abol Kit",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings: Settings = app.state.settings
    engine: MmsAsrEngine = app.state.engine
    return HealthResponse(
        ok=True,
        ready=engine.ready,
        model=settings.model_id,
        version=settings.version,
    )


@app.get("/ready")
def ready():
    engine: MmsAsrEngine = app.state.engine
    if engine.ready:
        return {"ready": True}
    return JSONResponse(
        status_code=503,
        content={
            "ready": False,
            "error": {
                "code": "MODEL_NOT_READY",
                "message": engine.load_error or "model is still loading",
            },
        },
    )


@app.get("/v1/info", response_model=InfoResponse)
def info() -> InfoResponse:
    settings: Settings = app.state.settings
    return InfoResponse(
        model=settings.model_id,
        version=settings.version,
        langs=sorted(LANG_TO_MMS.keys()),
        langMapping=dict(LANG_TO_MMS),
        maxUploadMb=settings.max_upload_mb,
        sendsDataOffHost=False,
    )


@app.post("/transcribe")
async def transcribe(
    file: Annotated[UploadFile, File(...)],
    lang: Annotated[str | None, Query(description='App language: "am" | "en"')] = None,
    lang_form: Annotated[str | None, Form(alias="lang")] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
):
    """
    Transcribe uploaded audio with Meta MMS ASR.

    Accepts `lang` as query string or form field (`am` | `en`).
    Maps internally: am→amh, en→eng.
    """
    settings: Settings = app.state.settings
    auth_err = _check_api_key(settings, x_api_key)
    if auth_err is not None:
        return auth_err

    app_lang = (lang or lang_form or "").strip().lower()
    if not app_lang:
        return _error("UNSUPPORTED_LANG", 'lang is required ("am" or "en")', 400)
    if app_lang not in LANG_TO_MMS:
        return _error(
            "UNSUPPORTED_LANG",
            f'unsupported lang "{app_lang}"; supported: am, en',
            400,
        )

    engine: MmsAsrEngine = app.state.engine
    if not engine.ready:
        return _error(
            "MODEL_NOT_READY",
            engine.load_error or "model is still loading",
            503,
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    raw = await file.read()
    if not raw:
        return _error("BAD_AUDIO", "empty upload", 400)
    if len(raw) > max_bytes:
        return _error(
            "BAD_AUDIO",
            f"upload exceeds MAX_UPLOAD_MB={settings.max_upload_mb}",
            400,
        )

    src: Path | None = None
    wav: Path | None = None
    t0 = time.perf_counter()
    try:
        src = write_upload_to_temp(file.filename, raw)
        wav, duration_s = normalize_to_wav16k_mono(src)
        waveform, sr = load_waveform(wav)
        result = engine.transcribe(
            waveform,
            lang=app_lang,
            duration_s=duration_s,
            sampling_rate=sr,
        )
    except UnsupportedLangError as exc:
        return _error("UNSUPPORTED_LANG", str(exc), 400)
    except AudioError as exc:
        return _error("BAD_AUDIO", str(exc), 400)
    except ModelNotReadyError as exc:
        return _error("MODEL_NOT_READY", str(exc), 503)
    except Exception as exc:  # noqa: BLE001
        logger.exception("transcribe failed")
        return _error("INTERNAL", f"transcription failed: {exc}", 500)
    finally:
        cleanup(src, wav)

    wall_ms = int(round((time.perf_counter() - t0) * 1000))
    duration_ms = result.duration_ms if result.duration_ms > 0 else wall_ms

    vtt = segments_to_vtt(
        result.segments,
        header_note=f"abol-mms-stt CTC alignment lang={app_lang} model={result.model}",
    )

    vtt_path: str | None = None
    out_dir = (settings.vtt_output_dir or "").strip()
    if out_dir:
        try:
            stem = Path(file.filename or "audio").stem or "audio"
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = Path(out_dir) / f"{stem}-{app_lang}-{stamp}.vtt"
            write_vtt_file(vtt, path)
            vtt_path = str(path.resolve())
            logger.info("Wrote VTT %s", vtt_path)
        except OSError:
            logger.exception("failed to write VTT file")

    payload = TranscribeResponse(
        text=result.text,
        lang=app_lang,  # type: ignore[arg-type]
        segments=result.segments,
        sendsDataOffHost=False,
        model=result.model,
        durationMs=duration_ms,
        needsReview=True,
        vtt=vtt,
        alignment=result.alignment,
        vttPath=vtt_path,
    )
    return JSONResponse(content=payload.model_dump())
