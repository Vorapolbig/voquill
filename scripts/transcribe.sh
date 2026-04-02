#!/usr/bin/env bash
# Transcribe an audio file using the Voquill GPU Whisper server (single-shot),
# then optionally clean up filler words using the local LLM server.
#
# Usage: ./transcribe.sh <audio-file> [options]
#
# Options:
#   -m, --model     Whisper model to use: tiny, base, small, medium, large, turbo (default: turbo)
#   -l, --language  Language code e.g. en, fi, fr (default: auto-detect)
#   -p, --prompt    Initial prompt to guide transcription
#   -u, --url       Whisper server URL (default: https://whisper.vorapol.cv)
#   --llm-url       LLM cleanup server URL (default: https://llm.vorapol.cv)
#   --no-cleanup    Skip LLM filler word cleanup
#   -j, --json      Output full JSON response instead of just the text
#   -d, --debug     Print per-step latency and raw vs cleaned text comparison
#   --local         Connect directly to home server (192.168.86.27) bypassing Cloudflare
#   -c, --context   App/context hint injected into LLM prompt e.g. "Slack", "VS Code", "email"
#   -t, --tone      Tone hint e.g. "casual", "professional", "technical" (default: auto)
#   -s, --single    Send audio as one request instead of chunks (simpler, no streaming)
#   --diarize       Detect multiple speakers via the server-side diarize pipeline
#                   (see scripts/services/README.md for server setup)
#   --diarize-url   Diarize service URL (default: https://diarize.vorapol.cv)
#   --num-speakers  Exact number of speakers (helps diarization accuracy)
#   --min-speakers  Minimum number of speakers
#   --max-speakers  Maximum number of speakers
#   --from-raw      Skip transcription and run LLM on an existing raw transcript file
#                   (raw transcripts are auto-saved to <audio-file>.raw.txt)
#   -g, --glossary  Path to JSON glossary file for find-and-replace corrections
#                   (default: ~/.whisper-glossary.json if it exists)
#                   Format: {"wrong phrase": "correct phrase", ...}
#
# Credentials (in priority order):
#   1. CF_ID / CF_SECRET environment variables
#   2. ~/.whisper.env or ~/.env.whisper file with CF_ID=... and CF_SECRET=...
#
# Prerequisites: ffmpeg (brew install ffmpeg), python3, curl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \?//'
  exit 1
}

AUDIO_FILE=""
MODEL="turbo"
LANGUAGE=""
PROMPT=""
WHISPER_URL="https://whisper.vorapol.cv"
LLM_URL="https://llm.vorapol.cv"
DIARIZE_URL="https://diarize.vorapol.cv"
CLEANUP=true
JSON_OUTPUT=false
DEBUG=false
LOCAL=false
SINGLE=false
DIARIZE=false
NUM_SPEAKERS=""
MIN_SPEAKERS=""
MAX_SPEAKERS=""
FROM_RAW=""
CONTEXT=""
TONE=""
GLOSSARY=""
SAMPLE_RATE=16000
LOCAL_IP="192.168.86.27"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)      MODEL="$2";       shift 2 ;;
    -l|--language)   LANGUAGE="$2";    shift 2 ;;
    -p|--prompt)     PROMPT="$2";      shift 2 ;;
    -u|--url)        WHISPER_URL="$2"; shift 2 ;;
    --llm-url)       LLM_URL="$2";     shift 2 ;;
    --no-cleanup)    CLEANUP=false;    shift ;;
    -j|--json)       JSON_OUTPUT=true; shift ;;
    -d|--debug)      DEBUG=true;       shift ;;
    -s|--single)     SINGLE=true;      shift ;;
    --diarize)       DIARIZE=true;          shift ;;
    --diarize-url)   DIARIZE_URL="$2";     shift 2 ;;
    --num-speakers)  NUM_SPEAKERS="$2";    shift 2 ;;
    --min-speakers)  MIN_SPEAKERS="$2";    shift 2 ;;
    --max-speakers)  MAX_SPEAKERS="$2";    shift 2 ;;
    --from-raw)      FROM_RAW="$2";        shift 2 ;;
    --local)         LOCAL=true;       shift ;;
    -c|--context)    CONTEXT="$2";     shift 2 ;;
    -t|--tone)       TONE="$2";        shift 2 ;;
    -g|--glossary)   GLOSSARY="$2";    shift 2 ;;
    -h|--help)       usage ;;
    -*)              echo "Unknown option: $1" >&2; usage ;;
    *)               AUDIO_FILE="$1";  shift ;;
  esac
done

if [ -z "$FROM_RAW" ]; then
  [ -z "$AUDIO_FILE" ] && usage
  if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: file not found: $AUDIO_FILE" >&2
    exit 1
  fi
else
  if [ ! -f "$FROM_RAW" ]; then
    echo "Error: raw transcript file not found: $FROM_RAW" >&2
    exit 1
  fi
fi

# Load credentials
for ENV_FILE in "$HOME/.whisper.env" "$HOME/.env.whisper"; do
  if [ -f "$ENV_FILE" ]; then
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    break
  fi
done

CF_ID="${CF_ID:-}"
CF_SECRET="${CF_SECRET:-}"

if [ -z "$CF_ID" ] || [ -z "$CF_SECRET" ]; then
  echo "Error: CF_ID and CF_SECRET must be set in ~/.whisper.env, ~/.env.whisper, or as environment variables." >&2
  exit 1
fi

# Resolve glossary file path
if [ -z "$GLOSSARY" ] && [ -f "$HOME/.whisper-glossary.json" ]; then
  GLOSSARY="$HOME/.whisper-glossary.json"
fi

# Local mode: try direct LAN connection, fall back to Cloudflare
if $LOCAL; then
  if python3 -c "import socket; s=socket.create_connection(('$LOCAL_IP', 7772), timeout=1); s.close()" 2>/dev/null; then
    WHISPER_URL="http://$LOCAL_IP:7772"
    LLM_URL="http://$LOCAL_IP:8766"
    DIARIZE_URL="http://$LOCAL_IP:7773"
    $DEBUG && echo "[debug] local: connected to $LOCAL_IP, using LAN" >&2
  else
    $DEBUG && echo "[debug] local: $LOCAL_IP unreachable, falling back to Cloudflare" >&2
  fi
fi

# Convert audio to PCM for non-diarize, non-from-raw paths
TMP_PCM=""
if ! $DIARIZE && [ -z "$FROM_RAW" ]; then
  TMP_PCM=$(mktemp /tmp/whisper_XXXXXX.f32)
  trap "rm -f $TMP_PCM" EXIT

  if $DEBUG; then
    T0=$(python3 -c "import time; print(int(time.monotonic()*1000))")
  fi

  ffmpeg -i "$AUDIO_FILE" -ar $SAMPLE_RATE -ac 1 -f f32le "$TMP_PCM" -y -loglevel quiet

  if $DEBUG; then
    T1=$(python3 -c "import time; print(int(time.monotonic()*1000))")
    echo "[debug] ffmpeg convert:    $((T1 - T0))ms" >&2
  fi
fi

# Build Python args
PY_ARGS=(
  --audio-file "$AUDIO_FILE"
  --model "$MODEL"
  --sample-rate "$SAMPLE_RATE"
  --whisper-url "$WHISPER_URL"
  --llm-url "$LLM_URL"
  --diarize-url "$DIARIZE_URL"
)

[ -n "$TMP_PCM" ]       && PY_ARGS+=(--pcm-file "$TMP_PCM")
[ -n "$LANGUAGE" ]       && PY_ARGS+=(--language "$LANGUAGE")
[ -n "$PROMPT" ]         && PY_ARGS+=(--prompt "$PROMPT")
[ -n "$NUM_SPEAKERS" ]   && PY_ARGS+=(--num-speakers "$NUM_SPEAKERS")
[ -n "$MIN_SPEAKERS" ]   && PY_ARGS+=(--min-speakers "$MIN_SPEAKERS")
[ -n "$MAX_SPEAKERS" ]   && PY_ARGS+=(--max-speakers "$MAX_SPEAKERS")
[ -n "$FROM_RAW" ]       && PY_ARGS+=(--from-raw "$FROM_RAW")
[ -n "$CONTEXT" ]        && PY_ARGS+=(--context "$CONTEXT")
[ -n "$TONE" ]           && PY_ARGS+=(--tone "$TONE")
[ -n "$GLOSSARY" ]       && PY_ARGS+=(--glossary "$GLOSSARY")
$CLEANUP                 && PY_ARGS+=(--cleanup)
$JSON_OUTPUT             && PY_ARGS+=(--json-output)
$DEBUG                   && PY_ARGS+=(--debug)
$SINGLE                  && PY_ARGS+=(--single)
$DIARIZE                 && PY_ARGS+=(--diarize)

export CF_ID CF_SECRET
export LOCAL=$( $LOCAL && echo true || echo false )

exec python3 "$SCRIPT_DIR/transcribe_client.py" "${PY_ARGS[@]}"
