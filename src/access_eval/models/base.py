# skeleton.py
from typing import List, Optional, Any, Dict, Tuple

# -------- Local (HF) --------

class LocalLLM:
    """
    Minimal local LLM wrapper (HF-backed).
    Accepts arbitrary *args/**kwargs at init and forwards to generate calls.
    """
    def __init__(self, model_id: str, *args, **kwargs):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.extra_args = args
        self.extra_kwargs = dict(kwargs)

    def _merge_args(self, call_args: Tuple[Any, ...], call_kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        # positional: init extras first, then per-call
        args = (*self.extra_args, *call_args)
        # kwargs: per-call overrides init defaults
        merged = {**self.extra_kwargs, **call_kwargs}
        return args, merged

    def prepare(self):
        """Load the local model/tokenizer (HF Transformers)."""
        raise NotImplementedError

    def single(self, prompt: str, *args, **kwargs) -> str:
        """Generate one response for one prompt."""
        raise NotImplementedError

    def batch(self, prompts: List[str], *args, **kwargs) -> List[str]:
        """Generate responses for a batch of prompts (order preserved)."""
        raise NotImplementedError


# -------- API (base) --------

class APILLM:
    """
    Minimal API LLM wrapper.
    Accepts arbitrary *args/**kwargs at init and forwards to each call.
    Subclasses implement prepare(), single(), batch().
    """
    def __init__(self, provider: str, model: str, api_key: Optional[str] = None, *args, **kwargs):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.extra_args = args
        self.extra_kwargs = dict(kwargs)

    def _merge_args(self, call_args: Tuple[Any, ...], call_kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        args = (*self.extra_args, *call_args)
        merged = {**self.extra_kwargs, **call_kwargs}
        return args, merged

    def prepare(self) -> None:
        raise NotImplementedError

    def single(self, prompt: str, *args, **kwargs) -> str:
        raise NotImplementedError

    def batch(self, prompts: List[str], concurrency: int = 10, *args, **kwargs) -> List[str]:
        raise NotImplementedError
