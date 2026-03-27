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
#   --llm-url       LLM cleanup server URL (default: https://llm.vorapol.cv)
#   --no-cleanup    Skip LLM filler word cleanup
#   -j, --json      Output full JSON response instead of just the text
#   -d, --debug     Print per-step latency and raw vs cleaned text comparison
#   --local         Connect directly to home server (192.168.86.27) bypassing Cloudflare
#
# Credentials (in priority order):
#   1. CF_ID / CF_SECRET environment variables
#   2. ~/.whisper.env or ~/.env.whisper file with CF_ID=... and CF_SECRET=...
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
LLM_URL="https://llm.vorapol.cv"
CLEANUP=true
JSON_OUTPUT=false
DEBUG=false
LOCAL=false
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
    --local)         LOCAL=true;       shift ;;
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

export CF_ID CF_SECRET

if $LOCAL; then
  WHISPER_URL="http://$LOCAL_IP:7772"
  LLM_URL="http://$LOCAL_IP:8766"
fi

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

python3 - <<PYEOF
import json, os, sys, time, urllib.request

DEBUG = "$DEBUG" == "true"
LOCAL = "$LOCAL" == "true"
CF_HEADERS = {} if LOCAL else {
    "CF-Access-Client-Id": os.environ["CF_ID"],
    "CF-Access-Client-Secret": os.environ["CF_SECRET"],
    "User-Agent": "curl/8.4.0",
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

t0 = time.monotonic()
session = cf_request(
    "$WHISPER_URL/v1/transcriptions/sessions",
    data=json.dumps(session_body).encode(),
    extra_headers={"Content-Type": "application/json"},
)
session_id = session["sessionId"]
if DEBUG:
    print(f"[debug] whisper session:    {(time.monotonic()-t0)*1000:.0f}ms", file=sys.stderr)

t0 = time.monotonic()
chunk_count = 0
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
        chunk_count += 1
if DEBUG:
    print(f"[debug] whisper upload:     {(time.monotonic()-t0)*1000:.0f}ms  ({chunk_count} chunks)", file=sys.stderr)

t0 = time.monotonic()
result = cf_request(
    f"$WHISPER_URL/v1/transcriptions/sessions/{session_id}/finalize",
    data=b"",
    extra_headers={"Content-Type": "application/json"},
)
raw_text = result["text"]
if DEBUG:
    print(f"[debug] whisper finalize:   {(time.monotonic()-t0)*1000:.0f}ms", file=sys.stderr)

# --- LLM filler cleanup ---
do_cleanup = "$CLEANUP" == "true"
llm_url = "$LLM_URL"
json_output = "$JSON_OUTPUT" == "true"

if do_cleanup:
    try:
        req2 = urllib.request.Request(
            f"{llm_url}/health",
            headers={**CF_HEADERS},
        )
        urllib.request.urlopen(req2, timeout=5)
    except Exception:
        do_cleanup = False
        print("[warn] LLM server not reachable, skipping cleanup", file=sys.stderr)

if do_cleanup:
    if DEBUG:
        print(f"\n[debug] raw text:\n{raw_text}\n", file=sys.stderr)
    payload = json.dumps({
        "model": "qwen3.5",
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": (
                "You are a transcription editor. Clean up the following spoken transcription:\n"
                "- Remove filler words (um, uh, like, you know, kind of, sort of, basically, actually, literally, right)\n"
                "- Fix punctuation and capitalisation\n"
                "- If the speaker lists multiple items or questions, format them as a numbered or bulleted list\n"
                "- If there are distinct topics or sections, add a short bold heading\n"
                "- Preserve the speaker's original meaning and wording — do not paraphrase\n"
                "Return only the cleaned text, no explanation."
            )},
            {"role": "user", "content": raw_text},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }).encode()
    t0 = time.monotonic()
    req2 = urllib.request.Request(
        f"{llm_url}/v1/chat/completions",
        data=payload,
        headers={**CF_HEADERS, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req2) as resp:
        d = json.load(resp)
    cleaned = d["choices"][0]["message"]["content"]
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    if DEBUG:
        print(f"[debug] llm cleanup:        {(time.monotonic()-t0)*1000:.0f}ms", file=sys.stderr)
        print(f"\n[debug] cleaned text:\n{cleaned}\n", file=sys.stderr)

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
