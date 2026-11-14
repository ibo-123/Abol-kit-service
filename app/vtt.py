"""WebVTT helpers from timed Segment lists."""

from __future__ import annotations

from pathlib import Path

from app.schemas import Segment


def format_vtt_timestamp(seconds: float) -> str:
    """Format seconds as WebVTT timestamp HH:MM:SS.mmm."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def segments_to_vtt(segments: list[Segment], *, header_note: str | None = None) -> str:
    """
    Build a WebVTT document from timed segments.

    Example cue:
      00:00:00.000 --> 00:00:02.400
      ሰላም እኔ ቻፒ ነኝ
    """
    lines: list[str] = ["WEBVTT"]
    if header_note:
        lines.append(f"NOTE {header_note}")
    lines.append("")

    for i, seg in enumerate(segments, start=1):
        start = format_vtt_timestamp(float(seg.start))
        end = format_vtt_timestamp(float(seg.end))
        if float(seg.end) <= float(seg.start):
            end = format_vtt_timestamp(float(seg.start) + 0.05)
        text = (seg.text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_vtt_file(vtt: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(vtt, encoding="utf-8")
    return path
