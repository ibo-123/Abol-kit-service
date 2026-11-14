#!/usr/bin/env bash
# Smoke-test abol-mms-stt against a running instance.
# When no audio path is given, generates a short silent MP3 via ffmpeg.
set -euo pipefail

BASE_URL="${TRANSCRIBER_URL:-http://127.0.0.1:8000}"
LANG_CODE="${1:-en}"
AUDIO_FILE="${2:-}"

echo "==> GET ${BASE_URL}/health"
curl -fsS "${BASE_URL}/health" | tee /tmp/mms-health.json
echo

echo "==> GET ${BASE_URL}/ready (may 503 while loading)"
READY_CODE=$(curl -s -o /tmp/mms-ready.json -w "%{http_code}" "${BASE_URL}/ready" || true)
cat /tmp/mms-ready.json
echo
if [[ "${READY_CODE}" != "200" ]]; then
  echo "Model not ready yet (HTTP ${READY_CODE}). Wait and re-run."
  exit 1
fi

echo "==> GET ${BASE_URL}/v1/info"
curl -fsS "${BASE_URL}/v1/info"
echo

CLEANUP_AUDIO=0
if [[ -z "${AUDIO_FILE}" ]]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is required to generate a sample MP3 (or pass a file path as \$2)"
    exit 1
  fi
  AUDIO_FILE="$(mktemp /tmp/mms-smoke-XXXX.mp3)"
  CLEANUP_AUDIO=1
  # 0.3s silent mono MP3 @ 16 kHz
  ffmpeg -y -f lavfi -i "anullsrc=r=16000:cl=mono" -t 0.3 \
    -c:a libmp3lame -q:a 4 "${AUDIO_FILE}" >/dev/null 2>&1
  echo "Generated sample MP3: ${AUDIO_FILE}"
fi

echo "==> POST ${BASE_URL}/transcribe?lang=${LANG_CODE} file=${AUDIO_FILE}"
curl -fsS -X POST \
  -F "file=@${AUDIO_FILE};type=audio/mpeg" \
  "${BASE_URL}/transcribe?lang=${LANG_CODE}" | tee /tmp/mms-transcribe.json
echo

if [[ "${CLEANUP_AUDIO}" == "1" ]]; then
  rm -f "${AUDIO_FILE}"
fi

echo "Smoke test OK."
