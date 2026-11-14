# abol-mms-stt

Self-hosted **Meta MMS ASR** HTTP wrapper for Abol Kit.  
**STT only** — `facebook/mms-1b-all`, not `facebook/mms-tts-*`.

```
Abol Kit worker → Transcriber HTTP adapter → THIS SERVICE :8000 → MMS ASR → JSON → Cue[]
```

## Success criteria

1. Fresh clone → `docker compose up --build` → `/health` and `/ready` succeed  
2. `POST /transcribe` returns stable JSON the TS adapter can consume ([contract](docs/contract.example.json))  
3. EC2 steps are copy-pasteable ([docs/ec2-deploy.md](docs/ec2-deploy.md))  
4. Wire the Abol Kit Transcriber in ≤ 30 minutes ([docs/typescript-client.md](docs/typescript-client.md))

## Quick start

```bash
git clone https://github.com/ibo-123/Abol-kit-service.git
cd Abol-kit-service
cp .env.example .env          # optional but recommended
docker compose up --build
# older Docker: docker-compose up --build
```

`/health` answers while weights download; wait for ready:

```bash
curl -s http://127.0.0.1:8000/health
# {"ok":true,"ready":false|true,"model":"facebook/mms-1b-all","version":"0.1.0"}

until curl -fsS http://127.0.0.1:8000/ready; do sleep 5; done
```

Transcribe:

```bash
curl -s -X POST -F "file=@./sample.mp3;type=audio/mpeg" \
  "http://127.0.0.1:8000/transcribe?lang=en"

curl -s -X POST -F "file=@./amharic.mp3;type=audio/mpeg" \
  "http://127.0.0.1:8000/transcribe?lang=am"

chmod +x scripts/smoke_test.sh && ./scripts/smoke_test.sh en
```

## Communication contract

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/health` | Process up; `ready` reflects model load |
| `GET` | `/ready` | **200** loaded / **503** still loading |
| `POST` | `/transcribe` | multipart `file` + `lang`=`am`\|`en` (query or form) |
| `GET` | `/v1/info` | langs, model id, max upload |

| Request `lang` | MMS adapter |
|----------------|-------------|
| `am` | `amh` |
| `en` | `eng` |

### Success JSON (stable — do not drift)

```json
{
  "text": "full transcript",
  "lang": "am",
  "segments": [
    { "start": 0.0, "end": 2.4, "text": "first cue" },
    { "start": 2.4, "end": 5.1, "text": "second cue" }
  ],
  "sendsDataOffHost": false,
  "model": "facebook/mms-1b-all",
  "durationMs": 5100,
  "needsReview": true,
  "alignment": "ctc",
  "vtt": "WEBVTT\n\n1\n00:00:00.000 --> 00:00:02.400\nfirst cue\n",
  "vttPath": "/app/outputs/vtt/audio-am-….vtt"
}
```

Timestamps come from **CTC greedy alignment** (not fixed chunking).  
`segments` are packed into subtitle-sized cues; `vtt` is a full WebVTT document.  
`.vtt` is also written under `VTT_OUTPUT_DIR` (`outputs/vtt` by default).  
`needsReview` is always `true`.

### Error JSON

```json
{
  "error": {
    "code": "UNSUPPORTED_LANG | BAD_AUDIO | MODEL_NOT_READY | INTERNAL",
    "message": "human readable"
  }
}
```

## TypeScript worker (Abol Kit)

```bash
TRANSCRIBER_URL=http://mms-stt:8000
```

```ts
async function transcribe(filePath: string, lang: "am" | "en") {
  const form = new FormData()
  form.append("file", new Blob([await Deno.readFile(filePath)]), "audio.mp3")
  // (use Node fs/fetch equivalently)
  const res = await fetch(`${process.env.TRANSCRIBER_URL}/transcribe?lang=${lang}`, {
    method: "POST",
    body: form,
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}
```

Drop-in `Transcriber` for `@abolkit/provider-sdk` → [docs/typescript-client.md](docs/typescript-client.md).

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_ID` | `facebook/mms-1b-all` | HF ASR checkpoint |
| `DEVICE` | `cpu` | `cpu` \| `cuda` |
| `PORT` | `8000` | Internal port |
| `MAX_UPLOAD_MB` | `50` | Upload cap |
| `HF_HOME` | `/models/hf` | Weight cache |
| `API_KEY` | _(empty)_ | Optional `X-API-Key`; off by default |
| `PRELOAD_LANGS` | `amh,eng` | Warm adapters |

## Privacy

Self-hosted · `sendsDataOffHost: false` · no third-party STT · no API key required by default.

## AWS EC2 / GPU

This laptop has **no NVIDIA GPU** — use **`g4dn.xlarge`** for full podcasts:

```bash
# on the GPU instance
cp .env.example .env && sed -i 's/^DEVICE=.*/DEVICE=cuda/' .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

Ubuntu + Docker · proxy **80/443** · keep **8000** private · **stop when idle**.  
Copy-paste guide: [docs/ec2-deploy.md](docs/ec2-deploy.md).

## Local Python / tests

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
pytest -q
```

## Layout

```
app/main.py           # routes
app/asr.py            # MMS load + inference
app/audio.py          # ffmpeg → 16 kHz mono
Dockerfile
docker-compose.yml    # service: mms-stt
docs/typescript-client.md
docs/ec2-deploy.md
docs/contract.example.json
scripts/smoke_test.sh
tests/
```
