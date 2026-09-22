"""Model adapters: Fable (Anthropic), Astra (OpenAI), Jev (TypeSafe System One), Whisper.

Each region model exposes `call(system, prompt, images) -> dict` (parsed JSON). Mocks live in
tests so no adapter needs a network to be exercised.
"""
from __future__ import annotations

import base64
import json
import os
import re
from typing import Protocol

import httpx


def parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


class RegionModel(Protocol):
    name: str

    def call(self, system: str, prompt: str, images: list[bytes]) -> dict: ...


class Fable:
    name = "fable"

    def __init__(self, model: str = "claude-fable-5-1"):
        import anthropic

        self.client = anthropic.Anthropic()  # ANTHROPIC_API_KEY
        self.model = model

    def call(self, system: str, prompt: str, images: list[bytes]) -> dict:
        content: list[dict] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(i).decode()}} for i in images
        ]
        content.append({"type": "text", "text": prompt})
        r = self.client.messages.create(model=self.model, max_tokens=8000, system=system,
                                        messages=[{"role": "user", "content": content}])
        return parse_json("".join(b.text for b in r.content if b.type == "text"))


class Astra:
    name = "astra"

    def __init__(self, model: str = "gpt-6-astra"):
        from openai import OpenAI

        self.client = OpenAI()  # OPENAI_API_KEY
        self.model = model

    def call(self, system: str, prompt: str, images: list[bytes]) -> dict:
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        for i in images:
            content.append({"type": "input_image",
                            "image_url": "data:image/jpeg;base64," + base64.b64encode(i).decode()})
        r = self.client.responses.create(model=self.model, instructions=system,
                                         input=[{"role": "user", "content": content}])
        return parse_json(r.output_text)


class JevJudge:
    """Text-only System One model: returns a choice + probabilities + confidence, no prose."""

    def __init__(self, url: str = "https://api.typesafe.ai/v1/systemone", timeout: float = 30,
                 model: str = "jev-latest"):
        self.url = url
        self.model = model
        self.http = httpx.Client(timeout=timeout)

    def choose(self, state: dict, instructions: str, options: dict[str, str]) -> dict:
        body = {"model": self.model, "state": state,
                "questions": {"q": {"type": "choice", "instructions": instructions, "criteria": options}}}
        for attempt in range(4):
            r = self.http.post(self.url, json=body,
                               headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"})
            if r.status_code == 429:
                import time
                time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
                continue
            r.raise_for_status()
            a = r.json()["answers"]["q"]
            return {"choice": a["choice"], "confidence": a.get("confidence", 0.0),
                    "probabilities": a.get("probabilities", {})}
        raise RuntimeError("Jev rate-limited")


def transcribe(audio_path: str, model: str = "whisper-1") -> list[dict]:
    """Word-level timestamps via OpenAI transcription (verbose_json + word granularity)."""
    from openai import OpenAI

    with open(audio_path, "rb") as f:
        r = OpenAI().audio.transcriptions.create(model=model, file=f, response_format="verbose_json",
                                                 timestamp_granularities=["word"])
    return [{"word": w.word, "start": w.start, "end": w.end} for w in (r.words or [])]
