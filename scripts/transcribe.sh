#!/usr/bin/env bash
# Transcribe an audio file using the Voquill GPU Whisper server (single-shot),
# then optionally clean up filler words using the local LLM server.
#
# Usage: ./transcribe.sh <audio-file> [options]
#
# Options:
#   -m, --model     Whisper model to use: tiny, base, small, medium, large (default: base)
#   -l, --language  Language code e.g. en, fi, fr (default: auto-detect)
#   -p, --prompt    Initial prompt to guide transcription
#   -u, --url       Whisper server URL (default: https://whisper.vorapol.cv)
#   --llm-url       LLM cleanup server URL (default: http://localhost:8766)
#   --no-cleanup    Skip LLM filler word cleanup
#   -j, --json      Output full JSON response instead of just the text
#
# Credentials (in priority order):
#   1. CF_ID / CF_SECRET environment variables
#   2. ~/.whisper.env file with CF_ID=... and CF_SECRET=...
#
# Prerequisites: ffmpeg (brew install ffmpeg), python3, curl

set -euo pipefail

usage() {
  sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \?//'
  exit 1
}

AUDIO_FILE=""
MODEL="base"
LANGUAGE=""
PROMPT=""
WHISPER_URL="https://whisper.vorapol.cv"
LLM_URL="http://localhost:8766"
CLEANUP=true
JSON_OUTPUT=false
SAMPLE_RATE=16000

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)      MODEL="$2";       shift 2 ;;
    -l|--language)   LANGUAGE="$2";    shift 2 ;;
    -p|--prompt)     PROMPT="$2";      shift 2 ;;
    -u|--url)        WHISPER_URL="$2"; shift 2 ;;
    --llm-url)       LLM_URL="$2";     shift 2 ;;
    --no-cleanup)    CLEANUP=false;    shift ;;
    -j|--json)       JSON_OUTPUT=true; shift ;;
    -h|--help)       usage ;;
    -*)              echo "Unknown option: $1" >&2; usage ;;
    *)               AUDIO_FILE="$1";  shift ;;
  esac
done

[ -z "$AUDIO_FILE" ] && usage

if [ ! -f "$AUDIO_FILE" ]; then
  echo "Error: file not found: $AUDIO_FILE" >&2
  exit 1
fi

ENV_FILE="$HOME/.whisper.env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck source=/dev/null
  source "$ENV_FILE"
fi

CF_ID="${CF_ID:-}"
CF_SECRET="${CF_SECRET:-}"

if [ -z "$CF_ID" ] || [ -z "$CF_SECRET" ]; then
  echo "Error: CF_ID and CF_SECRET must be set in $ENV_FILE or as environment variables." >&2
  exit 1
fi

TMP_PCM=$(mktemp /tmp/whisper_XXXXXX.f32)
trap "rm -f $TMP_PCM" EXIT

ffmpeg -i "$AUDIO_FILE" -ar $SAMPLE_RATE -ac 1 -f f32le "$TMP_PCM" -y -loglevel quiet

SAMPLES=$(python3 -c "
import struct, json
data = open('$TMP_PCM', 'rb').read()
print(json.dumps(list(struct.unpack_from(f'{len(data)//4}f', data))))
")

BODY=$(python3 -c "
import json
body = {'model': '$MODEL', 'samples': $SAMPLES, 'sampleRate': $SAMPLE_RATE}
if '$LANGUAGE': body['language'] = '$LANGUAGE'
if '$PROMPT':   body['initialPrompt'] = '$PROMPT'
print(json.dumps(body))
")

RESPONSE=$(curl -sf -X POST "$WHISPER_URL/v1/transcriptions" \
  -H "CF-Access-Client-Id: $CF_ID" \
  -H "CF-Access-Client-Secret: $CF_SECRET" \
  -H "Content-Type: application/json" \
  -d "$BODY")

RAW_TEXT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])")

if $CLEANUP && curl -sf "$LLM_URL/health" > /dev/null 2>&1; then
  CLEANED=$(python3 - "$RAW_TEXT" "$LLM_URL" <<'PYEOF'
import sys, json
import urllib.request

raw_text = sys.argv[1]
llm_url  = sys.argv[2]

payload = json.dumps({
  "model": "qwen3.5",
  "chat_template_kwargs": {"enable_thinking": False},
  "messages": [
    {"role": "system", "content": (
      "Remove filler words (um, uh, like, you know, kind of, sort of, basically, "
      "actually, literally, right) from the transcription. "
      "Fix punctuation and capitalisation. Return only the cleaned text, no explanation."
    )},
    {"role": "user", "content": raw_text}
  ],
  "temperature": 0.1,
  "max_tokens": 1024
}).encode()

req = urllib.request.Request(
  f"{llm_url}/v1/chat/completions",
  data=payload,
  headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as resp:
  d = json.load(resp)

content = d["choices"][0]["message"]["content"]
if "</think>" in content:
    content = content.split("</think>")[-1].strip()
print(content)
PYEOF
)

  if $JSON_OUTPUT; then
    python3 -c "
import json, sys
print(json.dumps({'raw': sys.argv[1], 'cleaned': sys.argv[2]}, indent=2))
" "$RAW_TEXT" "$CLEANED"
  else
    echo "$CLEANED"
  fi
else
  $CLEANUP && echo "[warn] LLM server not reachable at $LLM_URL, skipping cleanup" >&2

  if $JSON_OUTPUT; then
    echo "$RESPONSE" | python3 -m json.tool
  else
    echo "$RAW_TEXT"
  fi
fi
