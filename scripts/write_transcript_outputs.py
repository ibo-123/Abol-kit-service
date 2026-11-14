#!/usr/bin/env python3
"""Write VTT + markdown summary from a /transcribe JSON response file."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "usage: write_transcript_outputs.py <json> <vtt_out> <md_out> [title]",
            file=sys.stderr,
        )
        return 2

    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    vtt_out = Path(sys.argv[2])
    md_out = Path(sys.argv[3])
    title = sys.argv[4] if len(sys.argv) > 4 else "Transcription"

    if "error" in data:
        print(json.dumps(data, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    vtt = data.get("vtt") or ""
    vtt_out.write_text(vtt, encoding="utf-8")

    segs = data.get("segments") or []
    lines = [
        f"# {title}",
        "",
        f"- **lang:** {data.get('lang')}",
        f"- **alignment:** {data.get('alignment')}",
        f"- **model:** {data.get('model')}",
        f"- **durationMs:** {data.get('durationMs')}",
        f"- **segments:** {len(segs)}",
        f"- **needsReview:** {data.get('needsReview')}",
        f"- **vttPath:** {data.get('vttPath')}",
        f"- **copy:** {vtt_out}",
        "",
        "## Full text",
        "",
        data.get("text") or "",
        "",
        "## Timed segments",
        "",
        "| start | end | text |",
        "|-------|-----|------|",
    ]
    for s in segs:
        lines.append(f"| {s['start']} | {s['end']} | {s['text']} |")

    preview = "\n".join(vtt.splitlines()[:60])
    lines += ["", "## WebVTT (preview)", "", "```vtt", preview, "```", ""]
    md_out.write_text("\n".join(lines), encoding="utf-8")

    print(f"segments={len(segs)} vtt={vtt_out} md={md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
