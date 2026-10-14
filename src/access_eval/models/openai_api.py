from access_eval.models.base import APILLM
from typing import List, Optional, Any, Dict
import os
import asyncio
# --- new imports ---
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Any, Dict
import os


# Concrete OpenAI implementation
class OpenAIAPILLM(APILLM):
    """
    OpenAI implementation using the Responses API.
    - Uses OpenAI + AsyncOpenAI clients.
    - Merges base extra args/kwargs with per-call args/kwargs.
    """
    def __init__(self, model: str, api_key: Optional[str] = None, *args, **kwargs):
        super().__init__(provider="openai", model=model, api_key=api_key, *args, **kwargs)
        self._client = None
        self._aclient = None

    # def prepare(self) -> None:
    #     from openai import OpenAI, AsyncOpenAI
    #     key = self.api_key or os.getenv("OPENAI_API_KEY")
    #     if not key:
    #         raise ValueError("OPENAI_API_KEY not set and no api_key provided.")
    #     self._client = OpenAI(api_key=key)
    #     self._aclient = AsyncOpenAI(api_key=key)
    
    def prepare(self) -> None:
    # use ONLY the sync client
        from openai import OpenAI
        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set and no api_key provided.")
        # add sane networking defaults
        self._client = OpenAI(api_key=key, timeout=60.0, max_retries=3)
        self._aclient = None  # ensure we never touch async paths


    def _sanitize_kwargs(self, kwargs):
        # map old param
        if "max_tokens" in kwargs:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

        # remove unsupported params for GPT-5
        if self.model.startswith("gpt-5"):
            kwargs.pop("temperature", None)

        return kwargs

    def _merge_args(self, call_args: tuple, call_kwargs: Dict[str, Any]):
        """
        Merge init-time *args/**kwargs with per-call *args/**kwargs.
        Returns (args_tuple, kwargs_dict).
        """
        # positional: init extra_args first, then call_args
        args = (*self.extra_args, *call_args)
        # kwargs: per-call overrides init defaults
        merged = {**self.extra_kwargs, **call_kwargs}
        merged = self._sanitize_kwargs(merged)
        return args, merged

    def single(self, prompt: str, *args, **kwargs) -> str:
        if not self._client:
            raise RuntimeError("Call .prepare() first.")
        # Merge extra args/kwargs
        args, merged = self._merge_args(args, kwargs)
        # OpenAI Responses API call
        # Note: most options are keyworded; positional args are ignored by the SDK,
        # but we accept/forward them to honor the requested *args interface.
        resp = self._client.responses.create(model=self.model, input=prompt, **merged)
        # SDK exposes .output_text helper
        return resp.output_text

    def batch(self, prompts: List[str], concurrency: int = 32, *args, **kwargs) -> List[str]:
        if not self._client:
            raise RuntimeError("Call .prepare() first.")

        # merge defaults
        _, base_kwargs = self._merge_args(args, kwargs)

        # thread-pool parallel calls (order-preserving)
        results: List[Optional[str]] = [None] * len(prompts)

        def call_one(i_p):
            i, p = i_p
            try:
                r = self._client.responses.create(model=self.model, input=p, **base_kwargs)
                return i, r.output_text
            except Exception as e:
                return i, f"[Error: {e}]"
        concurrency = base_kwargs.pop("api_conc", concurrency)
        with ThreadPoolExecutor(max_workers=min(max(1, concurrency), len(prompts))) as ex:
            futs = [ex.submit(call_one, (i, p)) for i, p in enumerate(prompts)]
            for fut in as_completed(futs):
                i, text = fut.result()
                results[i] = text

        return [t or "" for t in results]

    # def batch(self, prompts: List[str], concurrency: int = 10, *args, **kwargs) -> List[str]:
    #     if not self._client:
    #         raise RuntimeError("Call .prepare() first.")

    #     # merge defaults
    #     _, base_kwargs = self._merge_args(args, kwargs)

    #     # ---- THREAD-POOL PARALLEL CALLS (order-preserving) ----
    #     results: List[Optional[str]] = [None] * len(prompts)

    #     def call_one(i_p):
    #         i, p = i_p
    #         try:
    #             r = self._client.responses.create(model=self.model, input=p, **base_kwargs)
    #             return i, r.output_text
    #         except Exception as e:
    #             return i, f"[Error: {e}]"

    #     with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
    #         futs = [ex.submit(call_one, (i, p)) for i, p in enumerate(prompts)]
    #         for fut in as_completed(futs):
    #             i, text = fut.result()
    #             results[i] = text

    #     return [t or "" for t in results]
    

    # def batch(self, prompts: List[str], concurrency: int = 10, *args, **kwargs) -> List[str]:
    #     if not (self._client and self._aclient):
    #         raise RuntimeError("Call .prepare() first.")

    #     # merge defaults
    #     _, base_kwargs = self._merge_args(args, kwargs)

    #     # If an event loop is already running (e.g., Jupyter), avoid nested-async headaches
    #     try:
    #         loop = asyncio.get_running_loop()
    #         running = loop.is_running()
    #     except RuntimeError:
    #         running = False

    #     if running:
    #         # ---- THREAD-POOL FALLBACK (order-preserving) ----
    #         from concurrent.futures import ThreadPoolExecutor, as_completed
    #         results: List[Optional[str]] = [None] * len(prompts)

    #         def call_one(i_p):
    #             i, p = i_p
    #             r = self._client.responses.create(model=self.model, input=p, **base_kwargs)
    #             return i, r.output_text

    #         with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
    #             futs = [ex.submit(call_one, (i, p)) for i, p in enumerate(prompts)]
    #             for fut in as_completed(futs):
    #                 i, text = fut.result()
    #                 results[i] = text
    #         return [t or "" for t in results]

    #     # ---- NORMAL ASYNC PATH (no loop running) ----
    #     async def _run():
    #         sem = asyncio.Semaphore(concurrency)
    #         out: List[Optional[str]] = [None] * len(prompts)

    #         async def worker(i: int, p: str):
    #             async with sem:
    #                 r = await self._aclient.responses.create(model=self.model, input=p, **base_kwargs)
    #                 out[i] = r.output_text

    #         await asyncio.gather(*(worker(i, p) for i, p in enumerate(prompts)))
    #         return [o or "" for o in out]

    #     return asyncio.run(_run())
