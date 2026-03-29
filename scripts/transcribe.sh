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
CLEANUP=true
JSON_OUTPUT=false
DEBUG=false
LOCAL=false
SINGLE=false
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
    --local)         LOCAL=true;       shift ;;
    -c|--context)    CONTEXT="$2";     shift 2 ;;
    -t|--tone)       TONE="$2";        shift 2 ;;
    -g|--glossary)   GLOSSARY="$2";    shift 2 ;;
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

# Resolve glossary file path
if [ -z "$GLOSSARY" ] && [ -f "$HOME/.whisper-glossary.json" ]; then
  GLOSSARY="$HOME/.whisper-glossary.json"
fi

export CF_ID CF_SECRET CONTEXT TONE GLOSSARY SINGLE

if $LOCAL; then
  if python3 -c "import socket; s=socket.create_connection(('$LOCAL_IP', 7772), timeout=1); s.close()" 2>/dev/null; then
    WHISPER_URL="http://$LOCAL_IP:7772"
    LLM_URL="http://$LOCAL_IP:8766"
    $DEBUG && echo "[debug] local: connected to $LOCAL_IP, using LAN" >&2
  else
    $DEBUG && echo "[debug] local: $LOCAL_IP unreachable, falling back to Cloudflare" >&2
  fi
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
import difflib, json, os, struct, sys, time, urllib.request

def word_diff(a, b):
    """Print a word-level diff of a→b with ANSI colours, git-diff style."""
    RED, GREEN, DIM, RESET = "\033[31m", "\033[32m", "\033[2m", "\033[0m"
    a_words = a.split()
    b_words = b.split()
    matcher = difflib.SequenceMatcher(None, a_words, b_words, autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            out.append(DIM + " ".join(a_words[i1:i2]) + RESET)
        elif op == "replace":
            out.append(RED + " ".join(a_words[i1:i2]) + RESET)
            out.append(GREEN + " ".join(b_words[j1:j2]) + RESET)
        elif op == "delete":
            out.append(RED + " ".join(a_words[i1:i2]) + RESET)
        elif op == "insert":
            out.append(GREEN + " ".join(b_words[j1:j2]) + RESET)
    print(" ".join(out), file=sys.stderr)

DEBUG = "$DEBUG" == "true"
LOCAL = "$LOCAL" == "true"
SINGLE = "$SINGLE" == "true"
CONTEXT = os.environ.get("CONTEXT", "")
TONE = os.environ.get("TONE", "")

def load_glossary():
    path = os.environ.get("GLOSSARY", "")
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

def apply_glossary(text, glossary):
    import re
    hits = {}
    for wrong, correct in glossary.items():
        new_text, count = re.subn(re.escape(wrong), correct, text, flags=re.IGNORECASE)
        if count:
            hits[wrong] = (correct, count)
        text = new_text
    return text, hits
CF_HEADERS = {} if LOCAL else {
    "CF-Access-Client-Id": os.environ["CF_ID"],
    "CF-Access-Client-Secret": os.environ["CF_SECRET"],
    "User-Agent": "curl/8.4.0",
}
CHUNK_BYTES = 16000 * 4 * 4  # 4 seconds of float32

def cf_request(url, data=None, extra_headers=None):
    headers = {**CF_HEADERS, **(extra_headers or {})}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)

# --- Whisper transcription ---
if SINGLE:
    t0 = time.monotonic()
    with open("$TMP_PCM", "rb") as f:
        raw_samples = list(struct.unpack(f'{os.path.getsize("$TMP_PCM")//4}f', f.read()))
    body = {"model": "$MODEL", "sampleRate": $SAMPLE_RATE, "samples": raw_samples}
    if "$LANGUAGE": body["language"] = "$LANGUAGE"
    if "$PROMPT":   body["initialPrompt"] = "$PROMPT"
    result = cf_request(
        "$WHISPER_URL/v1/transcriptions",
        data=json.dumps(body).encode(),
        extra_headers={"Content-Type": "application/json"},
    )
    raw_text = result["text"]
    if DEBUG:
        print(f"[debug] whisper single:     {(time.monotonic()-t0)*1000:.0f}ms", file=sys.stderr)
else:
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
            {"role": "system", "content": "\n".join(filter(None, [
                "You are a highly skilled editor specialising in cleaning up raw speech-to-text transcripts.",
                "Your goal is to produce the clean typed version of what the user intended to say, not a literal transcription.",
                "",
                "Rules:",
                "- WORD CHOICE: Preserve the speaker's word choice and voice",
                "- STRUCTURE: Refine to read like naturally written text without materially changing what the speaker said",
                "- DISFLUENCIES: Remove filler words (um, uh, like, you know, so yeah), false starts, and stutters. Keep meaningful exclamations.",
                "- SELF CORRECTIONS: If the speaker corrects themselves, keep only the final intended version",
                "- INSTRUCTIONS: If the speaker gives a formatting command (e.g. 'make that a bulleted list', 'put that in code'), execute it — do not transcribe it",
                "- TECHNICAL: Preserve and correctly format technical terms, variable names (camelCase, snake_case, PascalCase), filenames, and code snippets in backticks",
                "- SYMBOLS: Convert spoken cues: 'hashtag X' → '#X', 'at name' → '@name'",
                "- LISTS: Format bulleted lists when the speaker enumerates items",
                "- PARAGRAPHS: Split into paragraphs at natural breaks in thought",
                "- EMOJIS: Convert spoken emoji descriptions to actual emoji characters",
                "- Do NOT use em-dashes",
                f"- CONTEXT: The user is writing in {CONTEXT}. Format output appropriately for that context." if CONTEXT else "",
                f"- TONE: Write in a {TONE} tone." if TONE else "",
                "",
                "Output ONLY the cleaned text. No intro, no outro, no explanation.",
            ]))},
            {"role": "user", "content": raw_text},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }).encode()
    t0 = time.monotonic()
    req2 = urllib.request.Request(
        f"{llm_url}/v1/chat/completions",
        data=payload,
        headers={**CF_HEADERS, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req2) as resp:
        d = json.load(resp)
    choice = d["choices"][0]
    cleaned = choice["message"]["content"]
    if DEBUG:
        print(f"[debug] llm finish_reason: {choice.get('finish_reason')}", file=sys.stderr)
        print(f"[debug] llm response length: {len(cleaned)} chars", file=sys.stderr)
        print(f"[debug] llm response start: {repr(cleaned[:80])}", file=sys.stderr)
        print(f"[debug] llm response end:   {repr(cleaned[-80:])}", file=sys.stderr)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    # Dedup: find second occurrence of opening phrase (API sometimes returns response twice)
    prefix = cleaned[:30].strip()
    second = cleaned.find(prefix, 50)
    if second > 50:
        cleaned = cleaned[:second].strip()
        if DEBUG:
            print(f"[debug] dedup: trimmed at pos {second}, kept {len(cleaned)} chars", file=sys.stderr)
    glossary = load_glossary()
    if glossary:
        cleaned, hits = apply_glossary(cleaned, glossary)
        if DEBUG:
            if hits:
                print(f"[debug] glossary: {len(hits)}/{len(glossary)} entries matched:", file=sys.stderr)
                for wrong, (correct, count) in hits.items():
                    print(f"[debug]   '{wrong}' → '{correct}' ({count}×)", file=sys.stderr)
            else:
                print(f"[debug] glossary: 0/{len(glossary)} entries matched", file=sys.stderr)

    if DEBUG:
        print(f"[debug] llm cleanup:        {(time.monotonic()-t0)*1000:.0f}ms", file=sys.stderr)
        print(f"\n[debug] word diff (\033[31mremoved\033[0m / \033[32madded\033[0m / \033[2munchanged\033[0m):", file=sys.stderr)
        word_diff(raw_text, cleaned)
        print(f"\n{'─'*60}\n", file=sys.stderr)

    if json_output:
        print(json.dumps({"raw": raw_text, "cleaned": cleaned}, indent=2))
    else:
        print(cleaned)
else:
    glossary = load_glossary()
    if glossary:
        raw_text, _ = apply_glossary(raw_text, glossary)
    if DEBUG:
        print(f"\n{'─'*60}\n", file=sys.stderr)
    if json_output:
        print(json.dumps({"raw": raw_text}, indent=2))
    else:
        print(raw_text)
PYEOF
