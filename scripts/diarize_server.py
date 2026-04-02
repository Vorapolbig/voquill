#!/usr/bin/env python3
"""
Diarization + transcription pipeline service.

POST /v1/diarize
  multipart/form-data:
    file            audio file (required)
    model           Whisper model (default: turbo)
    language        language code (default: auto-detect)
    initial_prompt  initial Whisper prompt
    num_speakers    exact speaker count hint for pyannote
    min_speakers    minimum speaker count hint
    max_speakers    maximum speaker count hint
    merge_gap       seconds of silence to merge same-speaker turns (default: 1.5)
    llm_url         if set, clean each segment via LLM concurrently with transcription
    context         context hint for LLM system prompt (e.g. "Slack", "VS Code")
    tone            tone hint for LLM system prompt (e.g. "casual", "technical")
  Returns: NDJSON stream of:
    {"type": "segment", "idx": N, "start": 1.2, "speaker": "Speaker 1", "text": "..."}
    {"type": "done", "total_turns": N, "llm_applied": true/false}

GET /health
  Returns: {"status": "ok", "device": "cuda"}

Dependencies: pip install fastapi uvicorn pyannote.audio torch
Environment variables:
  DIARIZE_HOST    bind host (default: 0.0.0.0)
  DIARIZE_PORT    bind port (default: 7773)
  WHISPER_URL     upstream Whisper server (default: http://localhost:7772)
  HF_TOKEN        HuggingFace token for pyannote model access
"""
import asyncio
import json
import os
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pyannote.audio import Pipeline as DiarizePipeline

WHISPER_URL = os.environ.get("WHISPER_URL", "http://localhost:7772")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HOST = os.environ.get("DIARIZE_HOST", "0.0.0.0")
PORT = int(os.environ.get("DIARIZE_PORT", "7773"))

_log_path = os.environ.get("DIARIZE_LOG", "/tmp/diarize.log")
_log_fh = open(_log_path, "a", buffering=1)

class _Tee:
    def __init__(self, *files): self.files = files
    def write(self, data):
        for f in self.files: f.write(data)
    def flush(self):
        for f in self.files: f.flush()

sys.stderr = _Tee(sys.__stderr__, _log_fh)
SAMPLE_RATE = 16000
MIN_SEGMENT_SAMPLES = SAMPLE_RATE // 2  # skip segments shorter than 0.5s

_pipeline: DiarizePipeline | None = None
_device: torch.device | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _device
    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")
    except Exception:
        _device = torch.device("cpu")
    print(f"Loading pyannote/speaker-diarization-3.1 on {_device} ...", file=sys.stderr, flush=True)
    _pipeline = DiarizePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN or None,
    )
    try:
        _pipeline.to(_device)
    except Exception as e:
        print(f"Warning: {_device} unavailable ({e}), falling back to CPU", file=sys.stderr, flush=True)
        _device = torch.device("cpu")
        _pipeline.to(_device)
    print(f"Pipeline ready on {_device}.", file=sys.stderr, flush=True)
    yield


app = FastAPI(lifespan=lifespan)


def _whisper_segment(samples: list[float], model: str, language: str, initial_prompt: str) -> str:
    body: dict = {"model": model, "sampleRate": SAMPLE_RATE, "samples": samples}
    if language:
        body["language"] = language
    if initial_prompt:
        body["initialPrompt"] = initial_prompt
    req = urllib.request.Request(
        f"{WHISPER_URL}/v1/transcriptions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)["text"].strip()


def _llm_call(text: str, llm_url: str, system_prompt: str) -> str:
    payload = json.dumps({
        "model": "qwen3.5",
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }).encode()
    req = urllib.request.Request(
        f"{llm_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.load(resp)["choices"][0]["message"]["content"]
    if "</think>" in result:
        result = result.split("</think>")[-1].strip()
    return result


def _build_system_prompt(context: str, tone: str) -> str:
    return "\n".join(filter(None, [
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
        f"- CONTEXT: The user is writing in {context}. Format output appropriately for that context." if context else "",
        f"- TONE: Write in a {tone} tone." if tone else "",
        "",
        "Output ONLY the cleaned text. No intro, no outro, no explanation.",
    ]))


def _merge_segments(turns: list, merge_gap: float) -> list:
    merged = []
    for start, end, speaker in turns:
        if merged and merged[-1][2] == speaker and start - merged[-1][1] <= merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end, speaker])
    return merged


@app.get("/health")
def health():
    return {"status": "ok", "device": str(_device)}


@app.post("/v1/diarize")
async def diarize(
    file: UploadFile = File(...),
    model: str = Form("turbo"),
    language: str = Form(""),
    initial_prompt: str = Form(""),
    num_speakers: int = Form(0),
    min_speakers: int = Form(0),
    max_speakers: int = Form(0),
    merge_gap: float = Form(1.5),
    llm_url: str = Form(""),
    context: str = Form(""),
    tone: str = Form(""),
):
    with tempfile.TemporaryDirectory() as tmpdir:
        ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        audio_path = os.path.join(tmpdir, f"input{ext}")
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        wav_path = os.path.join(tmpdir, "converted.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", str(SAMPLE_RATE), "-ac", "1", wav_path],
            check=True, capture_output=True,
        )
        pcm_path = os.path.join(tmpdir, "audio.pcm")
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-f", "f32le", pcm_path],
            check=True, capture_output=True,
        )
        with open(pcm_path, "rb") as f:
            raw = f.read()
        all_samples = list(struct.unpack(f"{len(raw)//4}f", raw))

        diarize_kwargs = {}
        if num_speakers: diarize_kwargs["num_speakers"] = num_speakers
        if min_speakers:  diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers:  diarize_kwargs["max_speakers"] = max_speakers

        t0 = time.monotonic()
        # Run pyannote in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        diarization = await loop.run_in_executor(
            None, lambda: _pipeline(wav_path, **diarize_kwargs))
        turns = [(seg.start, seg.end, spk)
                 for seg, _, spk in diarization.itertracks(yield_label=True)]
        turns = _merge_segments(turns, merge_gap)
        print(f"[diarize] pyannote: {(time.monotonic()-t0)*1000:.0f}ms  {len(turns)} segments  merge_gap={merge_gap}s", file=sys.stderr, flush=True)
        # all_samples is a plain list — safe to use after tmpdir is gone

    use_llm = bool(llm_url)
    system_prompt = _build_system_prompt(context, tone) if use_llm else ""
    speaker_map: dict[str, str] = {}

    def speaker_label(spk: str) -> str:
        if spk not in speaker_map:
            speaker_map[spk] = f"Speaker {len(speaker_map) + 1}"
        return speaker_map[spk]

    result_queue: queue.Queue = queue.Queue()

    def process_segments():
        seg_llm_queue: queue.Queue = queue.Queue()
        llm_done = threading.Event()

        def llm_worker():
            while True:
                item = seg_llm_queue.get()
                if item is None:
                    llm_done.set()
                    return
                idx, start, spk, text = item
                t = time.monotonic()
                cleaned = _llm_call(text, llm_url, system_prompt)
                print(f"[diarize] LLM  seg {idx+1}: {len(text.split()):3d}w → {(time.monotonic()-t)*1000:.0f}ms", file=sys.stderr, flush=True)
                result_queue.put({"type": "segment", "idx": idx, "start": start,
                                  "speaker": speaker_label(spk), "text": cleaned})

        if use_llm:
            threading.Thread(target=llm_worker, daemon=True).start()

        for i, seg in enumerate(turns):
            start, end, spk = seg
            s = int(start * SAMPLE_RATE)
            e = int(end * SAMPLE_RATE)
            samples = all_samples[s:e]
            if len(samples) < MIN_SEGMENT_SAMPLES:
                continue
            t = time.monotonic()
            text = _whisper_segment(samples, model, language, initial_prompt)
            print(f"[diarize] Whisper seg {i+1}/{len(turns)} ({start:.1f}s-{end:.1f}s): {(time.monotonic()-t)*1000:.0f}ms", file=sys.stderr, flush=True)
            if not text:
                continue
            label = speaker_label(spk)
            if use_llm:
                seg_llm_queue.put((i, start, spk, text))
            else:
                result_queue.put({"type": "segment", "idx": i, "start": start,
                                  "speaker": label, "text": text})

        if use_llm:
            seg_llm_queue.put(None)
            llm_done.wait()

        result_queue.put({"type": "done", "total_turns": len(turns), "llm_applied": use_llm})

    threading.Thread(target=process_segments, daemon=True).start()

    async def generate():
        while True:
            item = await loop.run_in_executor(None, result_queue.get)
            yield json.dumps(item) + "\n"
            if item.get("type") == "done":
                return

    return StreamingResponse(generate(), media_type="application/x-ndjson")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
