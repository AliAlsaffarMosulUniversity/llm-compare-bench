"""Provider adapters (standard library only).

Real adapters speak the public REST APIs of OpenAI (Chat Completions) and
Google Gemini (generateContent). The ``mock`` provider returns synthetic
answers with a configurable accuracy and exists solely to validate the
pipeline; its output says nothing about any real model.

Model IDs, sampling parameters and prices come from the config file, never
from this code, so results always record exactly what was run.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from .tasks import Task


@dataclass
class Response:
    text: str
    latency_s: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class ProviderError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, headers: dict, retries: int, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    delay = 2.0
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            if e.code in (408, 409, 425, 429) or e.code >= 500:
                last = ProviderError(f"HTTP {e.code}: {body}")
            else:
                raise ProviderError(f"HTTP {e.code}: {body}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = ProviderError(f"network error: {e}")
        if attempt < retries:
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise last or ProviderError("request failed")


class OpenAIProvider:
    def __init__(self, label, model, api_key_env="OPENAI_API_KEY", params=None,
                 base_url="https://api.openai.com/v1", retries=4, system_prompt=""):
        self.label, self.model, self.params = label, model, dict(params or {})
        self.api_key_env, self.base_url = api_key_env, base_url.rstrip("/")
        self.retries, self.system_prompt = retries, system_prompt

    def generate(self, prompt: str, task: Task | None = None) -> Response:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError(f"environment variable {self.api_key_env} is not set")
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, **self.params}
        t0 = time.perf_counter()
        out = _post_json(f"{self.base_url}/chat/completions", payload,
                         {"Authorization": f"Bearer {key}"}, self.retries)
        dt = time.perf_counter() - t0
        try:
            text = out["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise ProviderError("unexpected response shape") from None
        usage = out.get("usage") or {}
        return Response(text, dt, usage.get("prompt_tokens"), usage.get("completion_tokens"))


class GeminiProvider:
    def __init__(self, label, model, api_key_env="GEMINI_API_KEY", params=None,
                 base_url="https://generativelanguage.googleapis.com/v1beta", retries=4,
                 system_prompt=""):
        self.label, self.model, self.params = label, model, dict(params or {})
        self.api_key_env, self.base_url = api_key_env, base_url.rstrip("/")
        self.retries, self.system_prompt = retries, system_prompt

    def generate(self, prompt: str, task: Task | None = None) -> Response:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError(f"environment variable {self.api_key_env} is not set")
        payload: dict = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                         "generationConfig": self.params}
        if self.system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": self.system_prompt}]}
        t0 = time.perf_counter()
        out = _post_json(f"{self.base_url}/models/{self.model}:generateContent", payload,
                         {"x-goog-api-key": key}, self.retries)
        dt = time.perf_counter() - t0
        try:
            parts = out["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError):
            reason = (out.get("promptFeedback") or {}).get("blockReason") or "no candidates"
            raise ProviderError(f"no text returned ({reason})") from None
        usage = out.get("usageMetadata") or {}
        return Response(text, dt, usage.get("promptTokenCount"), usage.get("candidatesTokenCount"))


class MockProvider:
    """Synthetic responder: answers correctly with probability ``accuracy``.

    Draws are independent per (provider, task) via a seeded hash, so the same
    run is reproducible. To model correlated difficulty across providers, pass
    ``shared_difficulty_seed`` (see ``scripts/validate_pipeline.py``).
    """

    def __init__(self, label, model="mock", accuracy=None, seed=0, retries=0, system_prompt="",
                 shared_difficulty_seed=None, rho=0.0, **_ignored):
        self.label, self.model = label, model
        self.accuracy = dict(accuracy or {"default": 0.8})
        self.seed, self.shared, self.rho = seed, shared_difficulty_seed, rho

    def _p(self, task: Task) -> float:
        return self.accuracy.get(task.dimension, self.accuracy.get("default", 0.8))

    def _correct(self, task: Task) -> bool:
        own = random.Random(f"{self.seed}:{task.id}").random()
        if self.shared is None or self.rho <= 0:
            return own < self._p(task)
        common = random.Random(f"{self.shared}:{task.id}").random()
        u = common if random.Random(f"{self.seed}:{task.id}:mix").random() < self.rho else own
        return u < self._p(task)

    def generate(self, prompt: str, task: Task | None = None) -> Response:
        if task is None:
            raise ProviderError("mock provider needs the task object")
        ok = self._correct(task)
        ref, s = task.reference, task.scorer
        if s == "mcq":
            wrong = next(c for c in "ABCD" if c != str(ref).upper())
            text = f"Answer: {ref if ok else wrong}"
        elif s == "numeric":
            text = f"Final answer: {ref if ok else float(ref) + 1}"
        elif s == "exact":
            r = ref[0] if isinstance(ref, list) else ref
            text = f"It is {r}." if ok else "I believe it is something else."
        elif s == "needle":
            text = str(ref) if ok else "000000"
        elif s == "abstain":
            text = ("I cannot verify that; there is no such record." if ok
                    else "It is definitely 'The Silent Orchard' (1987).")
        elif s == "code":
            sol = task.meta.get("solution", "")
            bad = task.meta.get("wrong_solution", "def _broken():\n    return None\n")
            text = f"```python\n{sol if ok else bad}\n```"
        else:
            text = "A short placeholder answer used only for pipeline validation."
        lat = 0.001 + 0.004 * random.Random(f"lat:{self.label}:{task.id}").random()
        return Response(text, lat, 20, 10)


def build_provider(spec: dict, retries: int, system_prompt: str):
    kind = spec["type"]
    common = dict(label=spec["label"], model=spec.get("model", kind), retries=retries,
                  system_prompt=system_prompt)
    if kind == "openai":
        return OpenAIProvider(api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
                              params=spec.get("params"), base_url=spec.get("base_url", "https://api.openai.com/v1"), **common)
    if kind == "gemini":
        return GeminiProvider(api_key_env=spec.get("api_key_env", "GEMINI_API_KEY"),
                              params=spec.get("params"),
                              base_url=spec.get("base_url", "https://generativelanguage.googleapis.com/v1beta"), **common)
    if kind == "mock":
        extra = {k: spec[k] for k in ("accuracy", "seed", "shared_difficulty_seed", "rho") if k in spec}
        return MockProvider(**common, **extra)
    raise ValueError(f"unknown provider type {kind!r}")
