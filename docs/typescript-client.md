# Wire Abol Kit → `abol-mms-stt` (≤ 30 minutes)

This wrapper is the HTTP backend for Abol Kit’s `Transcriber` adapter (`@abolkit/provider-sdk`).

## Checklist

1. Run this service (`docker compose up --build`) and wait for `/ready` → 200  
2. Set `TRANSCRIBER_URL` on the worker  
3. Paste the drop-in adapter below  
4. Map `segments` → `Cue[]` (shape is frozen; see [contract.example.json](./contract.example.json))  
5. Smoke-test one `en` and one `am` clip  

## Environment

```bash
# Same Docker network as this compose service (recommended)
TRANSCRIBER_URL=http://mms-stt:8000

# Wrapper running on the same machine as the worker (host networking / localhost)
# TRANSCRIBER_URL=http://127.0.0.1:8000

# Behind reverse proxy
# TRANSCRIBER_URL=https://stt.example.com
```

Optional (only if this wrapper has `API_KEY` set):

```bash
TRANSCRIBER_API_KEY=...
# sent as header: X-API-Key
```

### Join Abol Kit’s compose network

In Abol Kit’s `docker-compose.yml`:

```yaml
services:
  worker:
    environment:
      TRANSCRIBER_URL: http://mms-stt:8000
    networks:
      - default
      - mms

networks:
  mms:
    external: true
    # Compose project directory name + "_default", e.g. abol-kit-service_default
    name: abol-kit-service_default
```

## Sequence

```
Abol Kit worker
  → Transcriber HTTP adapter
    → POST multipart  TRANSCRIBER_URL/transcribe?lang=am|en
      → mms-stt :8000  (ffmpeg → MMS ASR)
        → JSON { text, lang, segments[], needsReview: true, sendsDataOffHost: false, ... }
          → Cue[] for review UI
```

## Language mapping

| Abol Kit `lang` | MMS adapter |
|-----------------|-------------|
| `am` | `amh` |
| `en` | `eng` |

## Drop-in `Transcriber` adapter

Matches `@abolkit/provider-sdk`:

```ts
import fs from "node:fs/promises"
import path from "node:path"
import type { Cue } from "@abolkit/core"
import type { LangCode, Transcriber } from "@abolkit/provider-sdk"

type AppLang = "am" | "en"

interface MmsSegment {
  start: number
  end: number
  text: string
}

interface MmsTranscribeResponse {
  text: string
  lang: AppLang
  segments: MmsSegment[]
  sendsDataOffHost: boolean
  model: string
  durationMs: number
  needsReview: boolean
  alignment: "ctc"
  /** Full WebVTT document with timed cues */
  vtt: string
  /** Server path when a .vtt file was written; otherwise null */
  vttPath: string | null
}

/** Stable fields → Cue[] (id?, start, end, text, needsReview?) */
export function toCues(result: MmsTranscribeResponse): Cue[] {
  return result.segments.map((s, i) => ({
    id: `mms-${i}`,
    start: s.start,
    end: s.end,
    text: s.text,
    needsReview: true, // always — human review required
  }))
}

export function createMmsHttpTranscriber(
  baseUrl = process.env.TRANSCRIBER_URL ?? "http://mms-stt:8000",
): Transcriber {
  return {
    id: "mms-http",
    label: "Meta MMS (self-hosted)",
    sendsDataOffHost: false,

    async transcribe(audioPath: string, opts: { lang: LangCode; signal?: AbortSignal }) {
      const lang = opts.lang as AppLang
      if (lang !== "am" && lang !== "en") {
        throw new Error(`mms-http supports am|en only, got: ${opts.lang}`)
      }

      const bytes = await fs.readFile(audioPath)
      const form = new FormData()
      form.append("file", new Blob([bytes]), path.basename(audioPath) || "audio.mp3")

      const headers: HeadersInit = {}
      if (process.env.TRANSCRIBER_API_KEY) {
        headers["X-API-Key"] = process.env.TRANSCRIBER_API_KEY
      }

      const res = await fetch(`${baseUrl.replace(/\/$/, "")}/transcribe?lang=${lang}`, {
        method: "POST",
        body: form,
        headers,
        signal: opts.signal,
      })

      if (!res.ok) {
        throw new Error(`mms-http ${res.status}: ${await res.text()}`)
      }

      const json = (await res.json()) as MmsTranscribeResponse
      return toCues(json)
    },
  }
}
```

Register wherever Abol Kit selects providers (e.g. `TRANSCRIBER=mms-http` → `createMmsHttpTranscriber()`).

### Minimal fetch-only example

```ts
async function transcribe(filePath: string, lang: "am" | "en") {
  const form = new FormData()
  form.append("file", new Blob([await Deno.readFile(filePath)]), "audio.mp3")
  // Node: form.append("file", new Blob([await fs.readFile(filePath)]), "audio.mp3")
  const res = await fetch(`${process.env.TRANSCRIBER_URL}/transcribe?lang=${lang}`, {
    method: "POST",
    body: form,
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}
```

## Contract (do not drift)

Success body — also in [contract.example.json](./contract.example.json):

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
  "vtt": "WEBVTT\\n\\n1\\n00:00:00.000 --> 00:00:02.400\\nfirst cue\\n",
  "vttPath": "/app/outputs/vtt/audio-am-….vtt"
}
```

- Timestamps from **CTC alignment**, packed into subtitle-sized cues  
- Persist captions with `await fs.writeFile("captions.vtt", result.vtt, "utf8")`  
- `needsReview` is **always** `true`  
- `sendsDataOffHost` is **always** `false` in default self-hosted mode  

Error body:

```json
{
  "error": {
    "code": "UNSUPPORTED_LANG | BAD_AUDIO | MODEL_NOT_READY | INTERNAL",
    "message": "human readable"
  }
}
```

Wait for `GET /ready` → 200 before production jobs (first boot downloads weights).
