#!/usr/bin/env python3
"""
Diarization + transcription pipeline service.

POST /v1/diarize
  multipart/form-data: file (audio), model (default: turbo), language, initial_prompt
  Returns: {"text": "[Speaker 1] ...\n[Speaker 2] ..."}

GET /health
  Returns: {"status": "ok", "device": "cuda"}

Dependencies: pip install fastapi uvicorn pyannote.audio torch
Environment variables:
  DIARIZE_HOST        bind host (default: 0.0.0.0)
  DIARIZE_PORT        bind port (default: 7773)
  WHISPER_URL         upstream Whisper server (default: http://localhost:7772)
  HF_TOKEN            HuggingFace token for pyannote model access
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import urllib.request
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pyannote.audio import Pipeline as DiarizePipeline

WHISPER_URL = os.environ.get("WHISPER_URL", "http://localhost:7772")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HOST = os.environ.get("DIARIZE_HOST", "0.0.0.0")
PORT = int(os.environ.get("DIARIZE_PORT", "7773"))
SAMPLE_RATE = 16000
MERGE_GAP = 0.5  # merge same-speaker segments within this many seconds

_pipeline: DiarizePipeline | None = None
_device: torch.device | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _device
    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()  # probe — raises if SM version is unsupported
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


def _merge_segments(turns: list) -> list:
    merged = []
    for start, end, speaker in turns:
        if merged and merged[-1][2] == speaker and start - merged[-1][1] <= MERGE_GAP:
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
):
    with tempfile.TemporaryDirectory() as tmpdir:
        ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        audio_path = os.path.join(tmpdir, f"audio{ext}")
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        # Convert to 16 kHz mono WAV — used by both pyannote and Whisper segment slicing
        wav_path = os.path.join(tmpdir, "audio.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ar", str(SAMPLE_RATE), "-ac", "1", wav_path],
            check=True, capture_output=True,
        )
        # Read as raw float32 PCM for Whisper
        pcm_path = os.path.join(tmpdir, "audio.pcm")
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-f", "f32le", pcm_path],
            check=True, capture_output=True,
        )
        with open(pcm_path, "rb") as f:
            raw = f.read()
        all_samples = list(struct.unpack(f"{len(raw)//4}f", raw))

        # Diarize using the WAV (soundfile-compatible)
        diarization = _pipeline(wav_path)
        turns = [(seg.start, seg.end, spk)
                 for seg, _, spk in diarization.itertracks(yield_label=True)]
        turns = _merge_segments(turns)

        # Assign readable labels in order of first appearance
        speaker_map: dict[str, str] = {}

        def speaker_label(spk: str) -> str:
            if spk not in speaker_map:
                speaker_map[spk] = f"Speaker {len(speaker_map) + 1}"
            return speaker_map[spk]

        # Transcribe segments concurrently — single-shot per segment is fast for short clips
        def transcribe_segment(seg: list) -> tuple | None:
            start, end, spk = seg
            s = int(start * SAMPLE_RATE)
            e = int(end * SAMPLE_RATE)
            samples = all_samples[s:e]
            if len(samples) < 1600:  # skip clips shorter than 0.1s
                return None
            text = _whisper_segment(samples, model, language, initial_prompt)
            return (start, spk, text) if text else None

        results = [transcribe_segment(seg) for seg in turns]

        results = sorted((r for r in results if r), key=lambda x: x[0])
        transcript = "\n".join(f"[{speaker_label(spk)}] {text}" for _, spk, text in results)
        return JSONResponse({"text": transcript})


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
