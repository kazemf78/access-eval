"""
Any OpenAI-compatible chat endpoint (`vllm serve`, an infer-stack lease, Ollama, SGLang, ...), selected by naming a
model `endpoint:<served model name>`, e.g. `endpoint:Qwen/Qwen3.5-9B`.

  base URL   $ACCESS_EVAL_ENDPOINT_BASE_URL (e.g. http://localhost:8000/v1), or per model via
             $ACCESS_EVAL_ENDPOINTS='{"Qwen/Qwen3.5-9B": "http://host:8000/v1"}'
  API key    $ACCESS_EVAL_ENDPOINT_API_KEY (default "EMPTY", which local servers accept)

Uses chat.completions (the common denominator across servers). Thinking stays off unless the model name carries
[with_thinking]; it is requested through chat_template_kwargs, which vLLM/SGLang forward to the chat template and
other servers ignore. A request that still fails after retries RAISES -- an error string is never returned as if it
were the model's answer (it would be cached and graded as one).
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

ENDPOINT_PREFIX = "endpoint:"


def is_endpoint_model(name: str) -> bool:
    return name.startswith(ENDPOINT_PREFIX)


def endpoint_base_url(served_name: str) -> str:
    per_model = json.loads(os.environ.get("ACCESS_EVAL_ENDPOINTS", "{}") or "{}")
    url = per_model.get(served_name) or os.environ.get("ACCESS_EVAL_ENDPOINT_BASE_URL")
    if not url:
        raise ValueError(f"no endpoint URL for {served_name!r}: set $ACCESS_EVAL_ENDPOINT_BASE_URL or $ACCESS_EVAL_ENDPOINTS")
    return url


class OpenAICompatLLM:
    def __init__(self, model: str, max_tokens: int = 256, temperature: float = 0.0, retries: int = 6,
                 timeout: float = 600.0, **kwargs):
        self.served_name = model[len(ENDPOINT_PREFIX):] if is_endpoint_model(model) else model
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.timeout = timeout
        self.enable_thinking = False
        self._client = None

    def prepare(self, ready_timeout: float = 1800.0):
        from openai import OpenAI
        self._client = OpenAI(base_url=endpoint_base_url(self.served_name),
                              api_key=os.environ.get("ACCESS_EVAL_ENDPOINT_API_KEY", "EMPTY"), timeout=self.timeout,
                              max_retries=0)
        # Wait for the server to come up (a lease or `vllm serve` may still be loading weights).
        start, delay = time.time(), 5
        while True:
            try:
                served = [m.id for m in self._client.models.list().data]
                if self.served_name not in served:
                    raise ValueError(f"endpoint serves {served}, not {self.served_name!r}")
                return
            except ValueError:
                raise
            except Exception as e:
                if time.time() - start > ready_timeout:
                    raise RuntimeError(f"endpoint for {self.served_name!r} not ready after {ready_timeout}s: {e}")
                time.sleep(delay)

    def _one(self, prompt: str, temperature) -> str:
        last = None
        for attempt in range(self.retries):
            try:
                r = self._client.chat.completions.create(
                    model=self.served_name, messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens, temperature=self.temperature if temperature is None else temperature,
                    extra_body={"chat_template_kwargs": {"enable_thinking": bool(self.enable_thinking)}})
                return r.choices[0].message.content or ""
            except Exception as e:
                last = e
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"endpoint request failed after {self.retries} attempts: {last}")

    def single(self, prompt: str, temperature=None, **kwargs) -> str:
        return self._one(prompt, temperature)

    def batch(self, prompts: List[str], concurrency: int = 32, temperature=None, **kwargs) -> List[str]:
        if self._client is None:
            raise RuntimeError("Call .prepare() first.")
        with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(prompts)))) as ex:
            return list(ex.map(lambda p: self._one(p, temperature), prompts))
