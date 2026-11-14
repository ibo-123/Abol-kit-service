"""Meta MMS ASR load + inference (STT only — not TTS)."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

from app.align import align_logits_to_segments
from app.config import Settings
from app.schemas import LANG_TO_MMS, Segment

logger = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    pass


class UnsupportedLangError(ValueError):
    pass


@dataclass
class TranscriptionResult:
    text: str
    lang: str  # app lang: am | en
    segments: list[Segment]
    duration_ms: int
    model: str
    alignment: str = "ctc"


class MmsAsrEngine:
    """
    Thin wrapper around facebook/mms-1b-all.

    Language adapters use ISO 639-3 codes (amh, eng). Callers pass app langs
    (am, en); mapping lives in schemas.LANG_TO_MMS.

    Timestamps come from CTC greedy alignment. Long audio is processed in
    overlapping chunks so podcast-length files fit in memory.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = settings.model_id
        self.device = self._resolve_device(settings.device)
        self._lock = threading.RLock()
        self._ready = False
        self._load_error: str | None = None
        self._model: Wav2Vec2ForCTC | None = None
        self._processor: AutoProcessor | None = None
        self._current_mms_lang: str | None = None

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("DEVICE=cuda requested but CUDA unavailable; falling back to cpu")
        return torch.device("cpu")

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        """Blocking model load. Safe to call once from a background thread."""
        if self._ready:
            return

        if self.settings.hf_home:
            os.environ["HF_HOME"] = self.settings.hf_home

        logger.info(
            "Loading MMS ASR model_id=%s device=%s",
            self.model_id,
            self.device,
        )
        t0 = time.perf_counter()
        try:
            processor = AutoProcessor.from_pretrained(self.model_id)
            model = Wav2Vec2ForCTC.from_pretrained(
                self.model_id,
                ignore_mismatched_sizes=True,
            )
            model.to(self.device)
            model.eval()

            with self._lock:
                self._processor = processor
                self._model = model
                self._current_mms_lang = "eng"
                self._ready = True
                self._load_error = None

            for code in self._preload_codes():
                try:
                    self._set_language(code)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to preload adapter %s: %s", code, exc)

            elapsed = time.perf_counter() - t0
            logger.info("MMS ASR ready in %.1fs", elapsed)
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            self._ready = False
            logger.exception("Failed to load MMS ASR")
            raise

    def _preload_codes(self) -> list[str]:
        raw = self.settings.preload_langs or ""
        return [c.strip() for c in raw.split(",") if c.strip()]

    def _require_ready(self) -> tuple[Wav2Vec2ForCTC, AutoProcessor]:
        if not self._ready or self._model is None or self._processor is None:
            raise ModelNotReadyError(
                self._load_error or "MMS ASR model is still loading"
            )
        return self._model, self._processor

    def _set_language(self, mms_lang: str) -> None:
        model, processor = self._require_ready()
        if self._current_mms_lang == mms_lang:
            return
        logger.info("Switching MMS adapter %s → %s", self._current_mms_lang, mms_lang)
        processor.tokenizer.set_target_lang(mms_lang)
        model.load_adapter(mms_lang)
        self._current_mms_lang = mms_lang

    @staticmethod
    def resolve_mms_lang(app_lang: str) -> str:
        key = (app_lang or "").strip().lower()
        if key not in LANG_TO_MMS:
            raise UnsupportedLangError(
                f"unsupported lang '{app_lang}'; supported: {sorted(LANG_TO_MMS)}"
            )
        return LANG_TO_MMS[key]

    def _infer_chunk(
        self,
        waveform: np.ndarray,
        *,
        sampling_rate: int,
        duration_s: float,
    ) -> tuple[str, list[Segment]]:
        model, processor = self._require_ready()
        inputs = processor(
            waveform,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            if attention_mask is not None:
                logits = model(input_values, attention_mask=attention_mask).logits
            else:
                logits = model(input_values).logits

        ids = torch.argmax(logits, dim=-1)[0]
        text = (processor.decode(ids) or "").strip()
        segments = align_logits_to_segments(
            logits,
            processor=processor,
            duration_s=duration_s,
            max_chars=self.settings.cue_max_chars,
            max_seconds=self.settings.cue_max_seconds,
        )
        if not segments and text:
            segments = [
                Segment(start=0.0, end=round(max(duration_s, 0.0), 3), text=text)
            ]
        return text, segments

    def transcribe(
        self,
        waveform: np.ndarray,
        *,
        lang: str,
        duration_s: float,
        sampling_rate: int = 16_000,
    ) -> TranscriptionResult:
        """
        Run CTC ASR + greedy CTC alignment for timed cue segments.

        Long files are split into overlapping windows (CHUNK_SECONDS) so memory
        stays bounded; segment times are offset back onto the full timeline.
        """
        mms_lang = self.resolve_mms_lang(lang)
        app_lang = lang.strip().lower()

        chunk_s = max(5.0, float(self.settings.chunk_seconds or 30.0))
        overlap_s = max(0.0, float(self.settings.chunk_overlap_seconds or 0.0))
        if overlap_s >= chunk_s:
            overlap_s = max(0.0, chunk_s / 10.0)

        n = int(waveform.shape[0])
        chunk_samples = int(chunk_s * sampling_rate)
        overlap_samples = int(overlap_s * sampling_rate)
        step = max(1, chunk_samples - overlap_samples)

        texts: list[str] = []
        all_segments: list[Segment] = []

        with self._lock:
            self._require_ready()
            self._set_language(mms_lang)

            if n <= chunk_samples:
                text, segments = self._infer_chunk(
                    waveform,
                    sampling_rate=sampling_rate,
                    duration_s=duration_s,
                )
                texts.append(text)
                all_segments.extend(segments)
            else:
                starts = list(range(0, n, step))
                logger.info(
                    "Long audio: %d samples (%.1fs) → %d chunks of %.1fs (overlap %.1fs)",
                    n,
                    duration_s,
                    len(starts),
                    chunk_s,
                    overlap_s,
                )
                for i, start in enumerate(starts):
                    end = min(n, start + chunk_samples)
                    if end <= start:
                        continue
                    # Skip tiny trailing scraps already covered by previous overlap
                    if i > 0 and (end - start) < sampling_rate * 0.4:
                        continue
                    piece = waveform[start:end]
                    piece_dur = (end - start) / float(sampling_rate)
                    offset = start / float(sampling_rate)
                    logger.info(
                        "chunk %d/%d offset=%.1fs dur=%.1fs",
                        i + 1,
                        len(starts),
                        offset,
                        piece_dur,
                    )
                    text, segs = self._infer_chunk(
                        piece,
                        sampling_rate=sampling_rate,
                        duration_s=piece_dur,
                    )
                    if text:
                        texts.append(text)
                    for seg in segs:
                        # Drop cues that live mostly in the overlap head (except first chunk)
                        if i > 0 and seg.end <= overlap_s * 0.75:
                            continue
                        all_segments.append(
                            Segment(
                                start=round(seg.start + offset, 3),
                                end=round(min(duration_s, seg.end + offset), 3),
                                text=seg.text,
                            )
                        )
                    if end >= n:
                        break

        text = " ".join(t for t in texts if t).strip()
        if not text and all_segments:
            text = " ".join(s.text for s in all_segments).strip()

        # Sort + light de-dup of identical adjacent cues
        all_segments.sort(key=lambda s: (s.start, s.end))
        cleaned: list[Segment] = []
        for seg in all_segments:
            if seg.end <= seg.start or not (seg.text or "").strip():
                continue
            if cleaned and cleaned[-1].text == seg.text and abs(cleaned[-1].start - seg.start) < 0.35:
                continue
            cleaned.append(seg)

        duration_ms = max(0, int(round(duration_s * 1000)))

        return TranscriptionResult(
            text=text,
            lang=app_lang,
            segments=cleaned,
            duration_ms=duration_ms,
            model=self.model_id,
            alignment="ctc",
        )


_engine: MmsAsrEngine | None = None


def get_engine() -> MmsAsrEngine:
    if _engine is None:
        raise ModelNotReadyError("ASR engine not initialized")
    return _engine


def set_engine(engine: MmsAsrEngine | None) -> None:
    global _engine
    _engine = engine
