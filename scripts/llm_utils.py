"""Shared LLM utilities for transcription cleanup."""

import json
import sys
import urllib.request


class Tee:
    """Redirect writes to multiple file-like objects (e.g. stderr + log file)."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)

    def flush(self):
        for f in self.files:
            f.flush()


def setup_log_tee(log_path: str):
    """Tee stderr to a log file. Returns the opened file handle."""
    fh = open(log_path, "a", buffering=1)
    sys.stderr = Tee(sys.__stderr__, fh)
    return fh


def build_system_prompt(context: str = "", tone: str = "") -> str:
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


def llm_call(text: str, llm_url: str, system_prompt: str) -> tuple[str, str | None]:
    """Call the LLM server for transcript cleanup.

    Returns (cleaned_text, finish_reason).
    """
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
