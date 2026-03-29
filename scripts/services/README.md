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

## Changing a service config

1. Edit the relevant file in `scripts/services/`
2. Commit and push
3. On the server: `git pull && ./scripts/server-services.sh install`

Never edit `/etc/systemd/system/` directly — those are deployed copies.

## Secrets

| File | Used by |
|---|---|
| `~/.whisper.env` | `transcribe.sh` client (CF_ID, CF_SECRET) |
| `~/.whisper-secrets` | `diarize` service (HF_TOKEN for pyannote) |

## First-time setup

```bash
# Install Python deps for diarize service
pip install fastapi uvicorn pyannote.audio torch

# Deploy and start all services
./scripts/server-services.sh install
```

The diarize service downloads the `pyannote/speaker-diarization-3.1` model (~1 GB) on first start. Follow progress with `./scripts/server-services.sh logs diarize`.
