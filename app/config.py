"""Runtime configuration from environment."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_id: str = "facebook/mms-1b-all"
    device: str = "cpu"  # cpu | cuda
    port: int = 8000
    max_upload_mb: int = 50
    hf_home: str | None = None
    api_key: str | None = None  # optional shared token; off when unset
    version: str = "0.1.0"
    # Preload adapters for these MMS ISO-639-3 codes at startup
    preload_langs: str = "amh,eng"
    # CTC → cue packing
    cue_max_chars: int = 48
    cue_max_seconds: float = 5.0
    # Write .vtt files here (empty = skip disk write; JSON always includes `vtt`)
    vtt_output_dir: str = "outputs/vtt"
    # Long-audio chunking (seconds). Keeps memory bounded for podcast-length clips.
    chunk_seconds: float = 30.0
    chunk_overlap_seconds: float = 1.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
