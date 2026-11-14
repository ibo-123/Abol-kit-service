"""CTC greedy alignment → timed word/phrase segments for MMS / Wav2Vec2."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from app.schemas import Segment

logger = logging.getLogger(__name__)


@dataclass
class TimedToken:
    token: str
    start: float
    end: float


def _frame_times(n_frames: int, duration_s: float) -> float:
    if n_frames <= 0 or duration_s <= 0:
        return 0.0
    return duration_s / float(n_frames)


def ctc_greedy_token_spans(
    logits: torch.Tensor,
    *,
    blank_id: int,
    duration_s: float,
) -> list[tuple[int, int, int]]:
    """
    Collapse CTC repeats / blanks into (token_id, start_frame, end_frame) spans.

    logits: [T, V] or [1, T, V]
    """
    if logits.dim() == 3:
        emissions = logits[0]
    elif logits.dim() == 2:
        emissions = logits
    else:
        raise ValueError(f"unexpected logits shape: {tuple(logits.shape)}")

    pred_ids = torch.argmax(emissions, dim=-1).detach().cpu().tolist()
    spans: list[tuple[int, int, int]] = []
    prev: int | None = None

    for t, tid in enumerate(pred_ids):
        if tid == blank_id:
            prev = tid
            continue
        if prev == tid and spans:
            tid0, s0, _ = spans[-1]
            spans[-1] = (tid0, s0, t)
        else:
            spans.append((tid, t, t))
        prev = tid

    return spans


def spans_to_words(
    spans: list[tuple[int, int, int]],
    *,
    tokenizer,
    duration_s: float,
    n_frames: int,
) -> list[Segment]:
    """Map CTC token spans to word-level Segment list (space / | delimited)."""
    if not spans or n_frames <= 0:
        return []

    frame_dur = _frame_times(n_frames, duration_s)
    delim_id = getattr(tokenizer, "word_delimiter_token_id", None)
    delim_tok = getattr(tokenizer, "word_delimiter_token", "|")

    words: list[Segment] = []
    buf_ids: list[int] = []
    start_f: int | None = None
    end_f: int | None = None

    def flush() -> None:
        nonlocal buf_ids, start_f, end_f
        if not buf_ids or start_f is None or end_f is None:
            buf_ids = []
            start_f = None
            end_f = None
            return
        # Decode this word's token ids (no grouping — already collapsed).
        text = tokenizer.decode(buf_ids, group_tokens=False).strip()
        # Some tokenizers leave the delimiter artifact; strip it.
        text = text.replace(delim_tok or "|", " ").strip()
        if text:
            start = max(0.0, start_f * frame_dur)
            end = min(duration_s, (end_f + 1) * frame_dur)
            if end <= start:
                end = min(duration_s, start + frame_dur)
            words.append(Segment(start=round(start, 3), end=round(end, 3), text=text))
        buf_ids = []
        start_f = None
        end_f = None

    for tid, sf, ef in spans:
        tok = tokenizer.convert_ids_to_tokens(int(tid))
        is_delim = (delim_id is not None and tid == delim_id) or tok == delim_tok
        if is_delim:
            flush()
            continue
        if start_f is None:
            start_f = sf
        end_f = ef
        buf_ids.append(int(tid))

    flush()
    return words


def pack_cues(
    words: list[Segment],
    *,
    duration_s: float,
    max_chars: int = 48,
    max_seconds: float = 5.0,
) -> list[Segment]:
    """
    Pack word segments into subtitle-sized cues.

    Avoids one-cue-per-word while preserving CTC timing.
    """
    if not words:
        return []

    cues: list[Segment] = []
    buf: list[str] = []
    cue_start = words[0].start
    cue_end = words[0].end

    def flush_cue() -> None:
        nonlocal buf, cue_start, cue_end
        if not buf:
            return
        text = " ".join(buf).strip()
        if text:
            start = max(0.0, cue_start)
            end = min(duration_s, max(cue_end, start + 0.05))
            cues.append(
                Segment(start=round(start, 3), end=round(end, 3), text=text)
            )
        buf = []

    for w in words:
        candidate = (" ".join(buf + [w.text])).strip() if buf else w.text
        dur = w.end - cue_start if buf else (w.end - w.start)
        overflow = buf and (
            len(candidate) > max_chars or (w.end - cue_start) > max_seconds
        )
        if overflow:
            flush_cue()
            cue_start = w.start
            buf = [w.text]
            cue_end = w.end
        else:
            if not buf:
                cue_start = w.start
            buf.append(w.text)
            cue_end = w.end
            _ = dur  # kept for clarity / future heuristics

    flush_cue()

    if not cues:
        return words

    # Ensure last cue reaches clip end if we're very close (subtitle players).
    if cues and duration_s > 0 and duration_s - cues[-1].end < 0.35:
        last = cues[-1]
        cues[-1] = Segment(start=last.start, end=round(duration_s, 3), text=last.text)

    return cues


def align_logits_to_segments(
    logits: torch.Tensor,
    *,
    processor,
    duration_s: float,
    max_chars: int = 48,
    max_seconds: float = 5.0,
) -> list[Segment]:
    """
    Full pipeline: CTC greedy spans → words → packed cue segments.

    Falls back to a single full-clip segment if alignment yields nothing.
    """
    tokenizer = processor.tokenizer
    blank_id = tokenizer.pad_token_id
    if blank_id is None:
        blank_id = 0

    if logits.dim() == 3:
        n_frames = int(logits.shape[1])
    else:
        n_frames = int(logits.shape[0])

    spans = ctc_greedy_token_spans(logits, blank_id=int(blank_id), duration_s=duration_s)
    words = spans_to_words(
        spans,
        tokenizer=tokenizer,
        duration_s=duration_s,
        n_frames=n_frames,
    )
    if not words:
        text = processor.decode(torch.argmax(logits, dim=-1)[0] if logits.dim() == 3 else torch.argmax(logits, dim=-1))
        text = (text or "").strip()
        if not text:
            return []
        return [Segment(start=0.0, end=round(max(duration_s, 0.0), 3), text=text)]

    return pack_cues(
        words,
        duration_s=duration_s,
        max_chars=max_chars,
        max_seconds=max_seconds,
    )
