#!/usr/bin/env python3
"""Client-side transcription pipeline: Whisper + optional LLM cleanup.

Called by transcribe.sh after arg parsing and ffmpeg conversion.
Reads configuration from environment variables.
"""

import argparse
import difflib
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.request

from llm_utils import Tee, build_system_prompt, llm_call, setup_log_tee


def word_diff(a: str, b: str):
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


# -- CF Access helpers --

def cf_headers() -> dict[str, str]:
    if os.environ.get("LOCAL") == "true":
        return {}
    return {
        "CF-Access-Client-Id": os.environ["CF_ID"],
        "CF-Access-Client-Secret": os.environ["CF_SECRET"],
        "User-Agent": "curl/8.4.0",
    }


def cf_request(url: str, data=None, extra_headers=None):
    headers = {**cf_headers(), **(extra_headers or {})}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


# -- Multipart helpers --

def _build_multipart(fields: dict[str, str], file_path: str):
    boundary = ("----FormBoundary" + str(int(time.time()))).encode()
    crlf = b"\r\n"
    body = []
    for name, value in fields.items():
        body += [b"--" + boundary,
                 f'Content-Disposition: form-data; name="{name}"'.encode(),
                 b"", value.encode()]
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


def multipart_cf_request_stream(url: str, fields: dict[str, str], file_path: str):
    body_bytes, boundary = _build_multipart(fields, file_path)
    headers = {**cf_headers(), "Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(url, data=body_bytes, headers=headers)
    with urllib.request.urlopen(req) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if line:
                yield json.loads(line)


# -- Whisper helpers --

CHUNK_BYTES = 16000 * 4 * 4  # 4 seconds of float32


def whisper_single(samples: list[float], whisper_url: str, model: str,
                   sample_rate: int, language: str, prompt: str, debug: bool) -> str:
    body: dict = {"model": model, "sampleRate": sample_rate, "samples": samples}
    if language:
        body["language"] = language
    if prompt:
        body["initialPrompt"] = prompt
    return cf_request(f"{whisper_url}/v1/transcriptions",
                      data=json.dumps(body).encode(),
                      extra_headers={"Content-Type": "application/json"})["text"].strip()


def whisper_chunked(samples: list[float], whisper_url: str, model: str,
                    sample_rate: int, language: str, prompt: str, debug: bool) -> str:
    session_body: dict = {"model": model, "sampleRate": sample_rate}
    if language:
        session_body["language"] = language
    if prompt:
        session_body["initialPrompt"] = prompt

    t0 = time.monotonic()
    session = cf_request(f"{whisper_url}/v1/transcriptions/sessions",
                         data=json.dumps(session_body).encode(),
                         extra_headers={"Content-Type": "application/json"})
    session_id = session["sessionId"]
    if debug:
        print(f"[debug] whisper session:    {(time.monotonic()-t0)*1000:.0f}ms", file=sys.stderr)

    t0 = time.monotonic()
    chunk_size = CHUNK_BYTES // 4
    chunks = [samples[i:i+chunk_size] for i in range(0, len(samples), chunk_size)]
    for chunk in chunks:
        cf_request(f"{whisper_url}/v1/transcriptions/sessions/{session_id}/chunks",
                   data=struct.pack(f"{len(chunk)}f", *chunk),
                   extra_headers={"Content-Type": "application/octet-stream"})
    if debug:
        print(f"[debug] whisper upload:     {(time.monotonic()-t0)*1000:.0f}ms  ({len(chunks)} chunks)", file=sys.stderr)

    t0 = time.monotonic()
    result = cf_request(f"{whisper_url}/v1/transcriptions/sessions/{session_id}/finalize",
                        data=b"", extra_headers={"Content-Type": "application/json"})
    if debug:
        print(f"[debug] whisper finalize:   {(time.monotonic()-t0)*1000:.0f}ms", file=sys.stderr)
    return result["text"].strip()


# -- Glossary --

def load_glossary(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def apply_glossary(text: str, glossary: dict) -> tuple[str, dict]:
    hits = {}
    for wrong, correct in glossary.items():
        new_text, count = re.subn(re.escape(wrong), correct, text, flags=re.IGNORECASE)
        if count:
            hits[wrong] = (correct, count)
        text = new_text
    return text, hits


# -- LLM chunking --

LLM_CHUNK_WORDS = 600


def split_chunks(text: str, max_words: int = LLM_CHUNK_WORDS) -> list[str]:
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


# -- Diarize path --

def run_diarize(args, cf_hdrs: dict) -> tuple[str, bool]:
    diarize_llm_url = ""
    if args.cleanup:
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"{args.llm_url}/health", headers=cf_hdrs), timeout=5)
            diarize_llm_url = args.llm_url
        except Exception:
            pass

    fields = {"model": args.model}
    if args.language:
        fields["language"] = args.language
    if args.prompt:
        fields["initial_prompt"] = args.prompt
    if args.num_speakers:
        fields["num_speakers"] = args.num_speakers
    if args.min_speakers:
        fields["min_speakers"] = args.min_speakers
    if args.max_speakers:
        fields["max_speakers"] = args.max_speakers
    if diarize_llm_url:
        fields["llm_url"] = diarize_llm_url
    if args.context:
        fields["context"] = args.context
    if args.tone:
        fields["tone"] = args.tone

    if args.debug:
        print(f"[debug] sending to diarize pipeline{' + LLM' if diarize_llm_url else ''}...",
              file=sys.stderr, flush=True)

    t0 = time.monotonic()
    received_segs = []
    meta = {}
    for item in multipart_cf_request_stream(f"{args.diarize_url}/v1/diarize", fields, args.audio_file):
        if item["type"] == "segment":
            received_segs.append(item)
            if args.debug:
                preview = item["text"][:70] + ("…" if len(item["text"]) > 70 else "")
                print(f"[debug] seg {len(received_segs):3d}: [{item['speaker']}] {preview}",
                      file=sys.stderr, flush=True)
        elif item["type"] == "done":
            meta = item

    received_segs.sort(key=lambda x: x["idx"])
    raw_text = "\n".join(f"[{s['speaker']}] {s['text']}" for s in received_segs)
    llm_applied = meta.get("llm_applied", False)
    if args.debug:
        print(f"[debug] diarize complete:   {(time.monotonic()-t0)*1000:.0f}ms  "
              f"turns={meta.get('total_turns', '?')}  received={len(received_segs)}  "
              f"llm={'yes' if llm_applied else 'no'}", file=sys.stderr)

    return raw_text, llm_applied


# -- Main --

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-file", default="")
    parser.add_argument("--pcm-file", default="")
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--language", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--whisper-url", default="https://whisper.vorapol.cv")
    parser.add_argument("--llm-url", default="https://llm.vorapol.cv")
    parser.add_argument("--diarize-url", default="https://diarize.vorapol.cv")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--json-output", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--diarize", action="store_true")
    parser.add_argument("--num-speakers", default="")
    parser.add_argument("--min-speakers", default="")
    parser.add_argument("--max-speakers", default="")
    parser.add_argument("--from-raw", default="")
    parser.add_argument("--context", default="")
    parser.add_argument("--tone", default="")
    parser.add_argument("--glossary", default="")
    args = parser.parse_args()

    if args.audio_file:
        setup_log_tee(args.audio_file + ".log")

    raw_output = args.audio_file + ".raw.txt" if args.audio_file else ""

    # -- Transcription --
    if args.from_raw:
        with open(args.from_raw) as f:
            raw_text = f.read().strip()
        llm_applied = False
        if args.debug:
            print(f"[debug] loaded raw transcript from {args.from_raw}", file=sys.stderr)

    elif args.diarize:
        raw_text, llm_applied = run_diarize(args, cf_headers())

    else:
        llm_applied = False
        with open(args.pcm_file, "rb") as f:
            raw = f.read()
        all_samples = list(struct.unpack(f"{len(raw)//4}f", raw))

        if args.single:
            t0 = time.monotonic()
            raw_text = whisper_single(all_samples, args.whisper_url, args.model,
                                     args.sample_rate, args.language, args.prompt, args.debug)
            if args.debug:
                print(f"[debug] whisper single:     {(time.monotonic()-t0)*1000:.0f}ms", file=sys.stderr)
        else:
            raw_text = whisper_chunked(all_samples, args.whisper_url, args.model,
                                      args.sample_rate, args.language, args.prompt, args.debug)

    # -- Save raw transcript --
    if raw_output and not args.from_raw:
        with open(raw_output, "w") as f:
            f.write(raw_text)
        if args.debug:
            print(f"[debug] raw transcript saved to {raw_output}", file=sys.stderr)

    # -- LLM cleanup --
    do_cleanup = args.cleanup and not llm_applied

    system_prompt = build_system_prompt(args.context, args.tone)

    if do_cleanup:
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"{args.llm_url}/health", headers=cf_headers()), timeout=5)
        except Exception:
            do_cleanup = False
            print("[warn] LLM server not reachable, skipping cleanup", file=sys.stderr)

    if do_cleanup:
        if args.debug:
            print(f"\n[debug] raw text:\n{raw_text}\n", file=sys.stderr)
        chunks = split_chunks(raw_text)
        if args.debug:
            print(f"[debug] llm chunks:         {len(chunks)} (total words: {len(raw_text.split())})",
                  file=sys.stderr)
        t0 = time.monotonic()
        cleaned_parts = []
        for i, chunk in enumerate(chunks):
            if args.debug:
                print(f"[debug] llm chunk {i+1}/{len(chunks)}: sending {len(chunk.split())} words...",
                      file=sys.stderr, flush=True)
            tc = time.monotonic()
            part, finish_reason = llm_call(chunk, args.llm_url, system_prompt)
            cleaned_parts.append(part)
            if args.debug:
                print(f"[debug] llm chunk {i+1}/{len(chunks)}: done  {len(part.split())} words out  "
                      f"{(time.monotonic()-tc)*1000:.0f}ms  finish={finish_reason}", file=sys.stderr)
        cleaned = "\n\n".join(cleaned_parts) if len(cleaned_parts) > 1 else cleaned_parts[0]

        glossary = load_glossary(args.glossary)
        if glossary:
            cleaned, hits = apply_glossary(cleaned, glossary)
            if args.debug:
                if hits:
                    print(f"[debug] glossary: {len(hits)}/{len(glossary)} entries matched:",
                          file=sys.stderr)
                    for wrong, (correct, count) in hits.items():
                        print(f"[debug]   '{wrong}' → '{correct}' ({count}×)", file=sys.stderr)
                else:
                    print(f"[debug] glossary: 0/{len(glossary)} entries matched", file=sys.stderr)

        if args.debug:
            print(f"[debug] llm cleanup:        {(time.monotonic()-t0)*1000:.0f}ms", file=sys.stderr)
            print(f"\n[debug] word diff (\033[31mremoved\033[0m / \033[32madded\033[0m / "
                  f"\033[2munchanged\033[0m):", file=sys.stderr)
            word_diff(raw_text, cleaned)
            print(f"\n{'─'*60}\n", file=sys.stderr)

        if args.json_output:
            print(json.dumps({"raw": raw_text, "cleaned": cleaned}, indent=2))
        else:
            print(cleaned)
    else:
        glossary = load_glossary(args.glossary)
        if glossary:
            raw_text, _ = apply_glossary(raw_text, glossary)
        if args.debug:
            print(f"\n{'─'*60}\n", file=sys.stderr)
        if args.json_output:
            print(json.dumps({"raw": raw_text}, indent=2))
        else:
            print(raw_text)


if __name__ == "__main__":
    main()
