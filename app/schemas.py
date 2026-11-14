"""Request/response models for the HTTP contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AppLang = Literal["am", "en"]
ErrorCode = Literal["UNSUPPORTED_LANG", "BAD_AUDIO", "MODEL_NOT_READY", "INTERNAL"]

# API lang → MMS ISO 639-3 adapter code
LANG_TO_MMS: dict[str, str] = {
    "am": "amh",  # Amharic
    "en": "eng",  # English
}

MMS_TO_APP: dict[str, str] = {v: k for k, v in LANG_TO_MMS.items()}


class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    text: str
    lang: AppLang
    segments: list[Segment]
    sendsDataOffHost: bool = False
    model: str
    durationMs: int
    needsReview: bool = True
    # CTC-aligned WebVTT document (timed cues)
    vtt: str = ""
    alignment: str = "ctc"
    # Absolute path when written to disk; null if disk write disabled/failed
    vttPath: str | None = None


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    ok: bool = True
    ready: bool
    model: str
    version: str


class InfoResponse(BaseModel):
    model: str
    version: str
    langs: list[str] = Field(default_factory=lambda: ["am", "en"])
    langMapping: dict[str, str] = Field(default_factory=lambda: dict(LANG_TO_MMS))
    maxUploadMb: int
    sendsDataOffHost: bool = False
    sampleRateHz: int = 16000
    acceptedFormats: list[str] = Field(
        default_factory=lambda: ["wav", "mp3", "m4a", "webm", "ogg", "flac"]
    )
