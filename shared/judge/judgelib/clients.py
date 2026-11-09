from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

from openai import OpenAI


LOGGER = logging.getLogger(__name__)


class ChatClient:
    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {api_key_env}")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_call_count = 0
        self.text_call_count = 0
        # Token accounting: run-total (thread-safe) for aggregate efficiency, plus a
        # thread-local counter so a caller can attribute one task's calls to that task
        # (each task runs on a single worker thread, so the thread-local is task-scoped).
        self._usage_lock = threading.Lock()
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0
        self._local = threading.local()
        # Explicit timeout + retries: without a timeout the SDK can wait ~10min on a stuck
        # request, and a single hung call had frozen the whole sweep (idle GPUs, no error).
        # A real generation finishes in seconds; anything past this is a dropped connection.
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(os.getenv("ACTMEM_CHAT_TIMEOUT", "300")),
            max_retries=int(os.getenv("ACTMEM_CHAT_RETRIES", "3")),
        )

    def set_stage(self, stage: str) -> None:
        """Tag subsequent calls on this thread with a stage name (for per-call token stats)."""
        self._local.stage = stage

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        with self._usage_lock:
            self.prompt_tokens_total += pt
            self.completion_tokens_total += ct
        cur = getattr(self._local, "usage", None)
        if cur is None:
            cur = [0, 0]
            self._local.usage = cur
        cur[0] += pt
        cur[1] += ct
        log = getattr(self._local, "call_log", None)
        if log is None:
            log = []
            self._local.call_log = log
        log.append({"stage": getattr(self._local, "stage", "default"),
                    "input_tokens": pt, "output_tokens": ct})

    def reset_task_usage(self) -> None:
        """Zero this thread's per-task counters (call before a task's chat calls)."""
        self._local.usage = [0, 0]
        self._local.call_log = []
        self._local.stage = "default"

    def task_usage(self) -> tuple[int, int]:
        """(input_tokens, output_tokens) accumulated on this thread since reset_task_usage."""
        cur = getattr(self._local, "usage", None)
        return (cur[0], cur[1]) if cur else (0, 0)

    def task_calls(self) -> list[dict]:
        """Per-call [{stage, input_tokens, output_tokens}, ...] since reset_task_usage."""
        return list(getattr(self._local, "call_log", []) or [])

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.json_call_count += 1
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        self._record_usage(response)
        content = response.choices[0].message.content or "{}"
        finish_reason = response.choices[0].finish_reason
        from src.utils import safe_json_loads

        try:
            return safe_json_loads(content)
        except json.JSONDecodeError as exc:
            preview = content[-500:]
            if finish_reason == "length":
                raise RuntimeError(
                    "Model returned truncated JSON. Increase model.chat.max_tokens "
                    "or reduce extraction.dialogs_per_batch / extraction.max_facts_per_dialog. "
                    f"Trailing response preview: {preview}"
                ) from exc
            raise RuntimeError(f"Model returned invalid JSON. Trailing response preview: {preview}") from exc

    def complete_text(self, system: str, user: str) -> str:
        self.text_call_count += 1
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self._record_usage(response)
        return response.choices[0].message.content or ""

    @property
    def call_count(self) -> int:
        return self.json_call_count + self.text_call_count


class EmbeddingClient:
    def __init__(
        self,
        model: str,
        api_key_env: str = "QWEN_EMBEDDING_API_KEY",
        base_url_env: str = "QWEN_EMBEDDING_BASE_URL",
        batch_size: int = 32,
        max_chars_per_text: int | None = None,
    ) -> None:
        api_key = os.getenv(api_key_env, "EMPTY")
        base_url = os.getenv(base_url_env)
        if not base_url:
            raise RuntimeError(f"Missing embedding base URL environment variable: {base_url_env}")
        self.model = model
        self.batch_size = batch_size
        self.max_chars_per_text = max_chars_per_text
        self.call_count = 0
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(os.getenv("ACTMEM_EMBED_TIMEOUT", "120")),
            max_retries=int(os.getenv("ACTMEM_EMBED_RETRIES", "3")),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        texts = [self._prepare_text(text) for text in texts]
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            LOGGER.info(
                "Embedding batch start=%s size=%s max_chars=%s",
                start,
                len(batch),
                max((len(text) for text in batch), default=0),
            )
            self.call_count += 1
            response = self.client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            embeddings.extend([item.embedding for item in ordered])
        return embeddings

    def _prepare_text(self, text: str) -> str:
        text = text or " "
        if self.max_chars_per_text is None or self.max_chars_per_text <= 0:
            return text
        return text[: self.max_chars_per_text]
