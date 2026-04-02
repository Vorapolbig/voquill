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

export CF_ID CF_SECRET CONTEXT TONE GLOSSARY SINGLE DIARIZE

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

python3 - <<PYEOF
import difflib, json, os, struct, sys, threading, time, urllib.request

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
DIARIZE = "$DIARIZE" == "true"
FROM_RAW = "$FROM_RAW"
WHISPER_URL = "$WHISPER_URL"
DIARIZE_URL = "$DIARIZE_URL"
MODEL = "$MODEL"
SAMPLE_RATE = $SAMPLE_RATE
LANGUAGE = "$LANGUAGE"
PROMPT = "$PROMPT"
TMP_PCM = "$TMP_PCM"
AUDIO_FILE = "$AUDIO_FILE"
NUM_SPEAKERS = "$NUM_SPEAKERS"
MIN_SPEAKERS = "$MIN_SPEAKERS"
MAX_SPEAKERS = "$MAX_SPEAKERS"
CONTEXT = os.environ.get("CONTEXT", "")
TONE = os.environ.get("TONE", "")
RAW_OUTPUT = AUDIO_FILE + ".raw.txt" if AUDIO_FILE else ""

if AUDIO_FILE:
    class _Tee:
        def __init__(self, *files): self.files = files
        def write(self, data):
            for f in self.files: f.write(data)
        def flush(self):
            for f in self.files: f.flush()
    _log_file = open(AUDIO_FILE + ".log", "w")
    sys.stderr = _Tee(sys.__stderr__, _log_file)

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

def _build_multipart(fields, file_path):
    boundary = ("----FormBoundary" + str(int(time.time()))).encode()
    crlf = b"\r\n"
    body = []
    for name, value in fields.items():
        body += [b"--" + boundary, f'Content-Disposition: form-data; name="{name}"'.encode(), b"", value.encode()]
    filename = os.path.basename(file_path).encode()
    with open(file_path, "rb") as f:
        file_data = f.read()
    body += [
        b"--" + boundary,
        b'Content-Disposition: form-data; name="file"; filename="' + filename + b'"',
        b"Content-Type: application/octet-stream",
        b"",
        file_data,
    ]
    body.append(b"--" + boundary + b"--")
    return crlf.join(body), boundary.decode()

def multipart_cf_request(url, fields, file_path):
    body_bytes, boundary = _build_multipart(fields, file_path)
    headers = {**CF_HEADERS, "Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(url, data=body_bytes, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)

def multipart_cf_request_stream(url, fields, file_path):
    """Yields parsed JSON objects as NDJSON lines arrive from the server."""
    body_bytes, boundary = _build_multipart(fields, file_path)
    headers = {**CF_HEADERS, "Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(url, data=body_bytes, headers=headers)
    with urllib.request.urlopen(req) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if line:
                yield json.loads(line)

# --- Helpers ---
def whisper_single(samples):
    body = {"model": MODEL, "sampleRate": SAMPLE_RATE, "samples": samples}
    if LANGUAGE: body["language"] = LANGUAGE
    if PROMPT:   body["initialPrompt"] = PROMPT
    return cf_request(f"{WHISPER_URL}/v1/transcriptions", data=json.dumps(body).encode(),
                      extra_headers={"Content-Type": "application/json"})["text"].strip()

def whisper_chunked(samples):
    import concurrent.futures as cf
    session_body = {"model": MODEL, "sampleRate": SAMPLE_RATE}
    if LANGUAGE: session_body["language"] = LANGUAGE
    if PROMPT:   session_body["initialPrompt"] = PROMPT
    t0 = time.monotonic()
    session = cf_request(f"{WHISPER_URL}/v1/transcriptions/sessions",
                         data=json.dumps(session_body).encode(),
                         extra_headers={"Content-Type": "application/json"})
    session_id = session["sessionId"]
    if DEBUG: print(f"[debug] whisper session:    {(time.monotonic()-t0)*1000:.0f}ms", file=sys.stderr)
    t0 = time.monotonic()
    chunks = [samples[i:i+CHUNK_BYTES//4] for i in range(0, len(samples), CHUNK_BYTES//4)]
    def send_chunk(chunk_bytes):
        cf_request(f"{WHISPER_URL}/v1/transcriptions/sessions/{session_id}/chunks",
                   data=chunk_bytes, extra_headers={"Content-Type": "application/octet-stream"})
    for chunk in chunks:
        send_chunk(struct.pack(f"{len(chunk)}f", *chunk))
    if DEBUG: print(f"[debug] whisper upload:     {(time.monotonic()-t0)*1000:.0f}ms  ({len(chunks)} chunks)", file=sys.stderr)
    t0 = time.monotonic()
    result = cf_request(f"{WHISPER_URL}/v1/transcriptions/sessions/{session_id}/finalize",
                        data=b"", extra_headers={"Content-Type": "application/json"})
    if DEBUG: print(f"[debug] whisper finalize:   {(time.monotonic()-t0)*1000:.0f}ms", file=sys.stderr)
    return result["text"].strip()

do_cleanup = "$CLEANUP" == "true"
llm_url = "$LLM_URL"
json_output = "$JSON_OUTPUT" == "true"

# --- Transcription ---
if FROM_RAW:
    with open(FROM_RAW) as f:
        raw_text = f.read().strip()
    llm_applied = False
    if DEBUG:
        print(f"[debug] loaded raw transcript from {FROM_RAW}", file=sys.stderr)

elif DIARIZE:
    # Probe LLM so we can pipeline it server-side
    diarize_llm_url = ""
    if do_cleanup:
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"{llm_url}/health", headers={**CF_HEADERS}), timeout=5)
            diarize_llm_url = llm_url
        except Exception:
            pass

    fields = {"model": MODEL}
    if LANGUAGE:        fields["language"] = LANGUAGE
    if PROMPT:          fields["initial_prompt"] = PROMPT
    if NUM_SPEAKERS:    fields["num_speakers"] = NUM_SPEAKERS
    if MIN_SPEAKERS:    fields["min_speakers"] = MIN_SPEAKERS
    if MAX_SPEAKERS:    fields["max_speakers"] = MAX_SPEAKERS
    if diarize_llm_url: fields["llm_url"] = diarize_llm_url
    if CONTEXT:         fields["context"] = CONTEXT
    if TONE:            fields["tone"] = TONE

    if DEBUG:
        print(f"[debug] sending to diarize pipeline{' + LLM' if diarize_llm_url else ''}...", file=sys.stderr, flush=True)

    t0 = time.monotonic()
    received_segs = []
    meta = {}
    for item in multipart_cf_request_stream(f"{DIARIZE_URL}/v1/diarize", fields, AUDIO_FILE):
        if item["type"] == "segment":
            received_segs.append(item)
            if DEBUG:
                preview = item["text"][:70] + ("…" if len(item["text"]) > 70 else "")
                print(f"[debug] seg {len(received_segs):3d}: [{item['speaker']}] {preview}", file=sys.stderr, flush=True)
        elif item["type"] == "done":
            meta = item

    received_segs.sort(key=lambda x: x["idx"])
    raw_text = "\n".join(f"[{s['speaker']}] {s['text']}" for s in received_segs)
    llm_applied = meta.get("llm_applied", False)
    if DEBUG:
        print(f"[debug] diarize complete:   {(time.monotonic()-t0)*1000:.0f}ms  "
              f"turns={meta.get('total_turns', '?')}  received={len(received_segs)}  llm={'yes' if llm_applied else 'no'}", file=sys.stderr)

else:
    llm_applied = False
    with open(TMP_PCM, "rb") as f:
        raw = f.read()
    all_samples = list(struct.unpack(f"{len(raw)//4}f", raw))

    if SINGLE:
        t0 = time.monotonic()
        raw_text = whisper_single(all_samples)
        if DEBUG:
            print(f"[debug] whisper single:     {(time.monotonic()-t0)*1000:.0f}ms", file=sys.stderr)
    else:
        raw_text = whisper_chunked(all_samples)

# --- Save raw transcript (before LLM, skip if LLM already applied by diarize server) ---
if RAW_OUTPUT and not FROM_RAW:
    with open(RAW_OUTPUT, "w") as f:
        f.write(raw_text)
    if DEBUG:
        print(f"[debug] raw transcript saved to {RAW_OUTPUT}", file=sys.stderr)

# --- LLM filler cleanup ---
do_cleanup = do_cleanup and not llm_applied

# Split text at natural boundaries into word-limited chunks so each fits
# within the LLM context window (input + output both need to fit in 8192 tokens).
LLM_CHUNK_WORDS = 600  # ~800 tokens input, leaves ~1200 tokens for output within 8192 ctx

def split_chunks(text, max_words=LLM_CHUNK_WORDS):
    lines = text.split("\n")
    chunks, current, count = [], [], 0
    for line in lines:
        n = len(line.split())
        if count + n > max_words and current:
            chunks.append("\n".join(current))
            current, count = [line], n
        else:
            current.append(line)
            count += n
    if current:
        chunks.append("\n".join(current))
    return chunks

SYSTEM_PROMPT = "\n".join(filter(None, [
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
]))

def llm_call(text):
    payload = json.dumps({
        "model": "qwen3.5",
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }).encode()
    req = urllib.request.Request(
        f"{llm_url}/v1/chat/completions",
        data=payload,
        headers={**CF_HEADERS, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        d = json.load(resp)
    choice = d["choices"][0]
    result = choice["message"]["content"]
    if "</think>" in result:
        result = result.split("</think>")[-1].strip()
    # Dedup: trim if response is accidentally doubled
    prefix = result[:30].strip()
    second = result.find(prefix, 50)
    if second > 50:
        result = result[:second].strip()
    return result, choice.get("finish_reason")

if do_cleanup:
    try:
        urllib.request.urlopen(
            urllib.request.Request(f"{llm_url}/health", headers={**CF_HEADERS}), timeout=5)
    except Exception:
        do_cleanup = False
        print("[warn] LLM server not reachable, skipping cleanup", file=sys.stderr)

if do_cleanup:
    if DEBUG:
        print(f"\n[debug] raw text:\n{raw_text}\n", file=sys.stderr)
    chunks = split_chunks(raw_text)
    if DEBUG:
        print(f"[debug] llm chunks:         {len(chunks)} (total words: {len(raw_text.split())})", file=sys.stderr)
    t0 = time.monotonic()
    cleaned_parts = []
    for i, chunk in enumerate(chunks):
        if DEBUG:
            print(f"[debug] llm chunk {i+1}/{len(chunks)}: sending {len(chunk.split())} words...", file=sys.stderr, flush=True)
        tc = time.monotonic()
        part, finish_reason = llm_call(chunk)
        cleaned_parts.append(part)
        if DEBUG:
            print(f"[debug] llm chunk {i+1}/{len(chunks)}: done  {len(part.split())} words out  {(time.monotonic()-tc)*1000:.0f}ms  finish={finish_reason}", file=sys.stderr)
    cleaned = "\n\n".join(cleaned_parts) if len(cleaned_parts) > 1 else cleaned_parts[0]

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
