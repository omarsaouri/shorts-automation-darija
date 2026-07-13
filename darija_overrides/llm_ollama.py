"""Swap shorts_generator.local.llm's OpenAI/Gemini client for a local Ollama
model (Atlas-Chat, a Darija fine-tune of Gemma 2) — the base repo's one
paid dependency in --mode local.

`call_local_llm` matches the vendored original's `(prompt: str) -> str`
contract used by shorts_generator.highlights.get_highlights, so it's a
drop-in replacement. `install()` shadow-imports this module in place of
shorts_generator.local.llm (per CLAUDE.md's override mechanism) since
shorts_generator.pipeline imports call_local_llm directly and can't take
a different llm_fn without editing the vendored file.
"""

import json
import os
import sys
import urllib.request
from typing import Optional

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL", "hf.co/QuantFactory/Atlas-Chat-9B-GGUF:Q4_K_M"
)
OLLAMA_TIMEOUT_SECONDS = 300

_VENDOR_LLM_MODULE = "shorts_generator.local.llm"


def _fix_arabic_json_punctuation(text: str) -> str:
    """Atlas-Chat sometimes emits Arabic comma '،' as a JSON structural
    delimiter instead of ASCII ',', which breaks json.loads.
    # ponytail: fixes only this one observed failure mode; highlights.py's
    # own retry-with-stricter-prompt loop (MAX_HIGHLIGHT_API_ATTEMPTS)
    # already covers other malformed-JSON cases.
    """
    return text.replace("،", ",")


def call_local_llm(prompt: str, model: Optional[str] = None) -> str:
    """Send `prompt` to a local Ollama model, return its raw text response.

    Reads/writes no state.db tables — pure passthrough to the local Ollama
    HTTP API (POST {OLLAMA_HOST}/api/generate).
    """
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7},
    }
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as resp:
        data = json.load(resp)
    return _fix_arabic_json_punctuation(data["response"])


def install() -> None:
    """Shadow-import this module over shorts_generator.local.llm.

    Call once, before shorts_generator.pipeline.generate_shorts(mode="local"):
    pipeline._run_local does `from .local.llm import call_local_llm` lazily
    inside the function body, so patching sys.modules before that call is
    made routes it here instead of the vendored OpenAI/Gemini client.
    """
    sys.modules[_VENDOR_LLM_MODULE] = sys.modules[__name__]


if __name__ == "__main__":
    # ponytail: smallest runnable check — hits the real local Ollama
    # server, not a mock, since the whole point is confirming the model
    # is reachable and returns parseable JSON. Not part of the pytest
    # suite (that mocks the HTTP call per CLAUDE.md's testing rules).
    demo_prompt = (
        'Respond ONLY with JSON: {"highlights":[{"title":"string",'
        '"start_time":0.0,"end_time":1.0,"score":50,"hook_sentence":"string",'
        '"virality_reason":"string"}]}'
    )
    raw = call_local_llm(demo_prompt)
    parsed = json.loads(raw)
    assert "highlights" in parsed, f"missing 'highlights' key: {raw}"
    print("OK:", parsed)
