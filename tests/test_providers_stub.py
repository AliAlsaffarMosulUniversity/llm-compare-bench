"""Exercise the OpenAI / Gemini adapters against a local stub server.

This verifies request construction, header use, retry behaviour and response
parsing against the documented response shapes. It does NOT contact the real
services.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llmbench.providers import GeminiProvider, OpenAIProvider, ProviderError

SEEN = []
FAIL_FIRST = {"n": 0}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        if self.path.endswith("/flaky/chat/completions") and FAIL_FIRST["n"] < 1:
            FAIL_FIRST["n"] += 1
            self.send_response(503); self.end_headers(); self.wfile.write(b"{}"); return
        if self.path.endswith("/chat/completions"):
            out = {"choices": [{"message": {"content": "Answer: B"}}],
                   "usage": {"prompt_tokens": 11, "completion_tokens": 3}}
        elif ":generateContent" in self.path:
            if "blocked" in self.path:
                out = {"promptFeedback": {"blockReason": "SAFETY"}}
            else:
                out = {"candidates": [{"content": {"parts": [{"text": "Final "}, {"text": "answer: 5"}]}}],
                       "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4}}
        else:
            self.send_response(404); self.end_headers(); return
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


@pytest.fixture(scope="module")
def base():
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_openai_request_and_parse(base, monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")
    p = OpenAIProvider("gpt", "model-x", "TEST_OPENAI_KEY", {"temperature": 0}, base_url=base + "/v1",
                       retries=0, system_prompt="be brief")
    r = p.generate("hello")
    assert r.text == "Answer: B" and (r.input_tokens, r.output_tokens) == (11, 3)
    path, headers, body = SEEN[-1]
    assert path == "/v1/chat/completions"
    assert headers["authorization"] == "Bearer sk-test"
    assert body["model"] == "model-x" and body["temperature"] == 0
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["messages"][-1] == {"role": "user", "content": "hello"}


def test_gemini_request_and_parse(base, monkeypatch):
    monkeypatch.setenv("TEST_GEMINI_KEY", "g-test")
    p = GeminiProvider("gem", "model-y", "TEST_GEMINI_KEY", {"temperature": 0}, base_url=base + "/v1beta", retries=0)
    r = p.generate("hi")
    assert r.text == "Final answer: 5" and (r.input_tokens, r.output_tokens) == (9, 4)
    path, headers, body = SEEN[-1]
    assert path == "/v1beta/models/model-y:generateContent"
    assert headers["x-goog-api-key"] == "g-test"
    assert "key=" not in path                       # key travels in a header, not the URL
    assert body["generationConfig"] == {"temperature": 0}


def test_gemini_blocked_prompt_raises(base, monkeypatch):
    monkeypatch.setenv("TEST_GEMINI_KEY", "g-test")
    p = GeminiProvider("gem", "blocked", "TEST_GEMINI_KEY", base_url=base + "/v1beta", retries=0)
    with pytest.raises(ProviderError, match="SAFETY"):
        p.generate("x")


def test_missing_key_is_reported_without_network(monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    with pytest.raises(ProviderError, match="NOPE_KEY"):
        OpenAIProvider("g", "m", "NOPE_KEY").generate("x")


def test_retry_on_5xx(base, monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")
    monkeypatch.setattr("time.sleep", lambda s: None)
    FAIL_FIRST["n"] = 0
    p = OpenAIProvider("gpt", "m", "TEST_OPENAI_KEY", base_url=base + "/flaky", retries=2)
    assert p.generate("x").text == "Answer: B"
