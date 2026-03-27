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

python3 - <<PYEOF
import json, sys, urllib.request

CF_HEADERS = {
    "CF-Access-Client-Id": "$CF_ID",
    "CF-Access-Client-Secret": "$CF_SECRET",
}
CHUNK_BYTES = 16000 * 4  # 1 second of float32

def cf_request(url, data=None, extra_headers=None):
    headers = {**CF_HEADERS, **(extra_headers or {})}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)

# --- Whisper transcription via session API (binary chunks, no size limit) ---
session_body = {"model": "$MODEL", "sampleRate": $SAMPLE_RATE}
if "$LANGUAGE": session_body["language"] = "$LANGUAGE"
if "$PROMPT":   session_body["initialPrompt"] = "$PROMPT"

session = cf_request(
    "$WHISPER_URL/v1/transcriptions/sessions",
    data=json.dumps(session_body).encode(),
    extra_headers={"Content-Type": "application/json"},
)
session_id = session["sessionId"]

with open("$TMP_PCM", "rb") as f:
    while True:
        chunk = f.read(CHUNK_BYTES)
        if not chunk:
            break
        cf_request(
            f"$WHISPER_URL/v1/transcriptions/sessions/{session_id}/chunks",
            data=chunk,
            extra_headers={"Content-Type": "application/octet-stream"},
        )

result = cf_request(
    f"$WHISPER_URL/v1/transcriptions/sessions/{session_id}/finalize",
    data=b"",
    extra_headers={"Content-Type": "application/json"},
)
raw_text = result["text"]

# --- LLM filler cleanup ---
do_cleanup = "$CLEANUP" == "true"
llm_url = "$LLM_URL"
json_output = "$JSON_OUTPUT" == "true"

if do_cleanup:
    try:
        req2 = urllib.request.Request(
            f"{llm_url}/health",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req2, timeout=2)
    except Exception:
        do_cleanup = False
        print("[warn] LLM server not reachable, skipping cleanup", file=sys.stderr)

if do_cleanup:
    payload = json.dumps({
        "model": "qwen3.5",
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": (
                "Remove filler words (um, uh, like, you know, kind of, sort of, basically, "
                "actually, literally, right) from the transcription. "
                "Fix punctuation and capitalisation. Return only the cleaned text, no explanation."
            )},
            {"role": "user", "content": raw_text},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }).encode()
    req2 = urllib.request.Request(
        f"{llm_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req2) as resp:
        d = json.load(resp)
    cleaned = d["choices"][0]["message"]["content"]
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    if json_output:
        print(json.dumps({"raw": raw_text, "cleaned": cleaned}, indent=2))
    else:
        print(cleaned)
else:
    if json_output:
        print(json.dumps({"raw": raw_text}, indent=2))
    else:
        print(raw_text)
PYEOF
