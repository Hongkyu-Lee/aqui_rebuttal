"""Tiny OpenAI-compatible chat helper used by the judge / labeler / answer tools.

Works with a local vLLM server (base_url=http://localhost:PORT/v1, api key "EMPTY") or any hosted
OpenAI-compatible endpoint (e.g. the Thinking Machines Lab API: set --base-url and --api-key-env).
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any


class Chat:
    def __init__(self, model: str, base_url: str, api_key_env: str = "OPENAI_API_KEY",
                 max_tokens: int = 1024, temperature: float = 0.0, top_p: float | None = None,
                 timeout: float = 600.0, retries: int = 4) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get(api_key_env) or "EMPTY", base_url=base_url,
                             timeout=timeout, max_retries=0)
        self.model, self.max_tokens, self.temperature, self.top_p = model, max_tokens, temperature, top_p
        self.retries = retries

    def complete(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        last = None
        for attempt in range(self.retries):
            try:
                args = dict(model=self.model, messages=messages,
                            max_tokens=kw.get("max_tokens", self.max_tokens),
                            temperature=kw.get("temperature", self.temperature))
                if self.top_p is not None:
                    args["top_p"] = self.top_p
                return self.client.chat.completions.create(**args)
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"chat call failed after {self.retries} tries: {last}")

    def json(self, messages: list[dict[str, str]], **kw: Any) -> dict[str, Any]:
        resp = self.complete(messages, **kw)
        return parse_json(resp.choices[0].message.content or "")


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            text = text[s:e + 1]
    return json.loads(text)
