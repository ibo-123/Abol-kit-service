import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from app.audio import (
    TARGET_SR,
    AudioError,
    generate_silent_mp3,
    load_waveform,
    normalize_to_wav16k_mono,
    write_upload_to_temp,
)


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not installed",
)


def _write_wav(path: Path, sr: int = 44_100, duration_s: float = 0.2) -> None:
    n = int(sr * duration_s)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        # simple tone-ish noise so ffmpeg has something to encode
        samples = (np.sin(np.linspace(0, 20, n)) * 1000).astype(np.int16)
        wf.writeframes(samples.tobytes())


def test_normalize_resamples_to_16k(tmp_path: Path):
    src = tmp_path / "in_44k.wav"
    _write_wav(src, sr=44_100)
    out, duration = normalize_to_wav16k_mono(src, dest=tmp_path / "out.wav")
    assert out.exists()
    assert duration > 0
    data, sr = load_waveform(out)
    assert sr == TARGET_SR
    assert data.ndim == 1
    assert data.size > 0


def test_write_upload_roundtrip(silent_wav_bytes: bytes, tmp_path: Path):
    path = write_upload_to_temp("clip.wav", silent_wav_bytes)
    try:
        assert path.exists()
        assert path.suffix == ".wav"
        assert path.stat().st_size == len(silent_wav_bytes)
    finally:
        path.unlink(missing_ok=True)


def test_write_upload_defaults_to_mp3(silent_wav_bytes: bytes):
    path = write_upload_to_temp(None, silent_wav_bytes)
    try:
        assert path.suffix == ".mp3"
    finally:
        path.unlink(missing_ok=True)


def test_generate_silent_mp3_and_normalize(tmp_path: Path):
    mp3 = generate_silent_mp3(tmp_path / "sample.mp3", duration_s=0.25)
    assert mp3.exists()
    assert mp3.suffix == ".mp3"
    assert mp3.stat().st_size > 0
    out, duration = normalize_to_wav16k_mono(mp3, dest=tmp_path / "from_mp3.wav")
    assert duration > 0
    data, sr = load_waveform(out)
    assert sr == TARGET_SR
    assert data.size > 0


def test_bad_audio_raises(tmp_path: Path):
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"not-audio")
    with pytest.raises(AudioError):
        normalize_to_wav16k_mono(junk, dest=tmp_path / "out.wav")
