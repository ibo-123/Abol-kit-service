"""ffmpeg helpers: normalize uploads to 16 kHz mono WAV for MMS ASR."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

TARGET_SR = 16_000
ACCEPTED_SUFFIXES = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".aac", ".mp4"}


class AudioError(Exception):
    """Raised when audio cannot be decoded or normalized."""


def ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise AudioError("ffmpeg is not installed or not on PATH")
    return path


def normalize_to_wav16k_mono(
    source: Path,
    *,
    dest: Path | None = None,
) -> tuple[Path, float]:
    """
    Convert any supported audio file to 16 kHz mono PCM WAV.

    Returns (wav_path, duration_seconds).
    """
    if not source.exists():
        raise AudioError(f"audio file not found: {source}")

    ffmpeg = ensure_ffmpeg()
    out = dest or source.with_suffix(".16k.wav")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-c:a",
        "pcm_s16le",
        str(out),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioError("ffmpeg timed out while converting audio") from exc

    if proc.returncode != 0 or not out.exists():
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-5:] if detail else ["unknown ffmpeg error"]
        raise AudioError("failed to decode audio: " + " | ".join(tail))

    duration = wav_duration_seconds(out)
    if duration <= 0:
        raise AudioError("audio has zero duration after conversion")
    return out, duration


def load_waveform(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV as float32 mono waveform and sampling rate."""
    try:
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001 — surface as BAD_AUDIO
        raise AudioError(f"could not read wav: {exc}") from exc

    if getattr(data, "ndim", 1) > 1:
        data = np.mean(data, axis=1)

    if sr != TARGET_SR:
        # Should already be 16 kHz from ffmpeg; refuse silent mismatch.
        raise AudioError(f"expected {TARGET_SR} Hz, got {sr} Hz")

    if data.size == 0:
        raise AudioError("empty audio waveform")

    return data.astype(np.float32, copy=False), sr


def wav_duration_seconds(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def write_upload_to_temp(filename: str | None, data: bytes) -> Path:
    """Persist multipart bytes to a temp file with a safe suffix (default .mp3)."""
    suffix = Path(filename or "audio.mp3").suffix.lower() or ".mp3"
    if suffix not in ACCEPTED_SUFFIXES:
        # Still write it — ffmpeg may handle unknown containers; mark oddly.
        suffix = ".bin"
    fd, name = tempfile.mkstemp(prefix="mms-upload-", suffix=suffix)
    path = Path(name)
    try:
        with open(fd, "wb") as fh:
            fh.write(data)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def generate_silent_mp3(
    dest: Path,
    *,
    duration_s: float = 0.3,
    sample_rate: int = TARGET_SR,
) -> Path:
    """
    Generate a short silent MP3 (16 kHz mono by default) via ffmpeg.

    Used by smoke tests and local fixtures so sample clips match the
    preferred upload format (mp3).
    """
    ffmpeg = ensure_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sample_rate}:cl=mono",
        "-t",
        str(duration_s),
        "-c:a",
        "libmp3lame",
        "-q:a",
        "4",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
    if proc.returncode != 0 or not dest.exists():
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-5:] if detail else ["unknown ffmpeg error"]
        raise AudioError("failed to generate mp3: " + " | ".join(tail))
    return dest


def cleanup(*paths: Path | None) -> None:
    for p in paths:
        if p is None:
            continue
        try:
            p.unlink(missing_ok=True)
        except OSError:
            logger.debug("could not remove temp file %s", p)
