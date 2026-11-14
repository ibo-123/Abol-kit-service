#!/usr/bin/env bash
# Run full bitcoin podcast transcription on a GPU host (e.g. g4dn.xlarge).
# Usage (on the GPU machine, from repo root):
#   ./scripts/gpu-transcribe-bitcoin.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

AUDIO="${1:-tests/bitcoin-revolution-ethiopia-podcast-clip.ogg}"
LANG_CODE="${2:-am}"
OUT_JSON="${OUT_JSON:-/tmp/mms-bitcoin-full-${LANG_CODE}.json}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This script must run on an NVIDIA GPU host."
  exit 1
fi
nvidia-smi

if [[ ! -f "$AUDIO" ]]; then
  echo "ERROR: audio not found: $AUDIO"
  exit 1
fi

cp -n .env.example .env 2>/dev/null || true
sed -i 's/^DEVICE=.*/DEVICE=cuda/' .env

echo "==> Building/starting GPU stack"
if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
else
  COMPOSE=(docker-compose)
fi

"${COMPOSE[@]}" -f docker-compose.yml -f docker-compose.gpu.yml up --build -d

echo "==> Waiting for /ready"
until curl -fsS http://127.0.0.1:8000/ready >/dev/null; do
  sleep 5
  echo "  still loading..."
done
curl -fsS http://127.0.0.1:8000/health
echo

echo "==> Transcribing $AUDIO (lang=$LANG_CODE)"
curl -sS -m 43200 -X POST \
  -F "file=@${AUDIO};type=audio/ogg" \
  "http://127.0.0.1:8000/transcribe?lang=${LANG_CODE}" \
  -o "$OUT_JSON" -w "http=%{http_code} time=%{time_total}\n"

python3 scripts/write_transcript_outputs.py \
  "$OUT_JSON" \
  "tests/bitcoin-revolution-ethiopia-podcast-clip.${LANG_CODE}.vtt" \
  "tests/bitcoin-revolution-ethiopia-podcast-clip-transcript.md" \
  "Transcription — bitcoin-revolution-ethiopia-podcast-clip.ogg (full, lang=${LANG_CODE}, GPU)"

echo "OK. VTT + markdown written under tests/ and outputs/vtt/"
