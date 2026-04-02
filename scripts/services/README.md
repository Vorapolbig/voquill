# Server Services

Systemd service configs for the Voquill home server (Linux, GTX 1070).

| Service | Port | Description |
|---|---|---|
| `whisper-gpu` | 7772 | Rust Whisper transcription sidecar (Vulkan GPU) |
| `llama-server` | 8766 | llama.cpp with Qwen3.5-2B for LLM cleanup |
| `diarize` | 7773 | Python diarization + transcription pipeline (pyannote.audio) |

All three are exposed externally via Cloudflare Tunnel with CF Access service token auth.

## Managing services

Use `scripts/server-services.sh` from the repo root on the server:

```bash
./scripts/server-services.sh install    # deploy config changes and restart everything
./scripts/server-services.sh status     # check all services at a glance
./scripts/server-services.sh restart [service]  # restart one or all
./scripts/server-services.sh logs <service>     # follow journald logs
```

## Client usage

```bash
# Basic transcription
./scripts/transcribe.sh recording.m4a

# With LLM cleanup, debug output, and glossary corrections
./scripts/transcribe.sh recording.m4a -d -g ~/.whisper-glossary.json

# Multi-speaker diarization
./scripts/transcribe.sh recording.m4a --diarize --num-speakers 2

# Re-run LLM cleanup on a saved raw transcript
./scripts/transcribe.sh recording.m4a --from-raw recording.m4a.raw.txt

# See all options
./scripts/transcribe.sh --help
```

The diarize server returns streaming NDJSON — each segment is sent as soon as it's ready, so the client can show progress in real-time.

## Changing a service config

1. Edit the relevant file in `scripts/services/`
2. Commit and push
3. On the server: `git pull && ./scripts/server-services.sh install`

Never edit `/etc/systemd/system/` directly — those are deployed copies.

## Environment variables

| Variable | Default | Used by |
|---|---|---|
| `DIARIZE_LOG` | `/tmp/diarize.log` | `diarize` service — log file path |
| `DIARIZE_HOST` | `0.0.0.0` | `diarize` service — bind host |
| `DIARIZE_PORT` | `7773` | `diarize` service — bind port |
| `WHISPER_URL` | `http://localhost:7772` | `diarize` service — upstream Whisper |
| `HF_TOKEN` | — | `diarize` service — HuggingFace token for pyannote |

## Secrets

| File | Used by |
|---|---|
| `~/.whisper.env` | `transcribe.sh` client (CF_ID, CF_SECRET) |
| `~/.whisper-secrets` | `diarize` service (HF_TOKEN for pyannote) |

## First-time setup

```bash
# Install Python deps for diarize service
pip install -r scripts/requirements.txt

# Deploy and start all services
./scripts/server-services.sh install
```

The diarize service downloads the `pyannote/speaker-diarization-3.1` model (~1 GB) on first start. Follow progress with `./scripts/server-services.sh logs diarize`.
