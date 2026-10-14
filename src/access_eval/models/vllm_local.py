# vllm_local.py
import json
import os
from pathlib import Path
from typing import List, Optional
from access_eval.models.base import LocalLLM
from vllm import LLM, SamplingParams
from access_eval.models.engine_profiles import resolve_engine_kwargs

class VLLMLocalLLM(LocalLLM):
    """vLLM-backed local LLM with the same interface as HFLocalLLM.

    LoRA support (`adapter_path` in extra_kwargs, same convention
    HFLocalLLM already uses): NOT wired into `factory.get_llm_from_list()`
    by default -- every existing contamination-pilot run (Track 1, Track
    2) always forces HFLocalLLM whenever an adapter is present, for the
    documented reason that both pre-/post-FT must share an identical
    generation backend for a fair before/after comparison
    (controlled_contamination_plan.md item 1). This class's own LoRA
    support exists for explicit opt-in use (`get_llm_from_list(...,
    prefer_vllm_for_lora=True)`) -- e.g. a broad, exploratory re-
    evaluation of an already-trained adapter where raw HF `.generate()`
    throughput (no continuous batching, no paged attention) is
    prohibitively slow (confirmed directly: a full-pool post-FT pass
    was projected at ~7 hours under HFLocalLLM, with many generations
    hitting the token cap without stopping naturally). Never silently
    changes the default path any existing run relies on.
    """

    def __init__(self, model_id: str, *args, **kwargs):
        super().__init__(model_id, *args, **kwargs)
        self.llm = None
        self._gen_defaults = {}
        self._lora_request = None

    # ---- helpers ----
    def apply_chat(self, p: str) -> str:
        # vLLM does NOT apply chat templates automatically
        # so this stays basically identical
        msgs = [{"role": "user", "content": p}]
        kwargs = {}
        if hasattr(self, "enable_thinking"):
            kwargs["enable_thinking"] = self.enable_thinking # todo: check if this is still a good way to configure this or not?
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                **kwargs
            )
        return p

    def _detect_native_max_len(self):
        """Read the model's own context-length cap from its HF config, if available."""
        from transformers import AutoConfig
        revision_kwargs = {"revision": self.extra_kwargs["revision"]} if self.extra_kwargs.get("revision") else {}
        try:
            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True, **revision_kwargs)
        except Exception as e:
            print(f"WARNING: Could not load config for {self.model_id} to detect native max length: {e}")
            return None
        for attr in ("max_position_embeddings", "model_max_length", "n_positions", "seq_length"):
            val = getattr(config, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        return None

    # ---- PREPARE ----
    def prepare(self):
        # vLLM defaults to forking its TP worker subprocesses. If CUDA has
        # already been touched in this driver process -- e.g. an earlier
        # local model's own prepare() call, or an unrelated torch.cuda import
        # side effect -- forking then re-initializing CUDA in the child dies
        # with "Cannot re-initialize CUDA in forked subprocess". Force spawn
        # (setdefault: an explicit env override still wins) so this stays
        # correct whether this is the only local engine in the process or
        # one of several (e.g. paraphrase.py's resp_model/judge_model pair).
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from transformers import AutoTokenizer
        import torch
        # revision must also reach the tokenizer (LLM() below already gets it
        # via **load_kwargs, since "revision" isn't a SamplingParams field).
        revision_kwargs = {"revision": self.extra_kwargs["revision"]} if self.extra_kwargs.get("revision") else {}
        try:
            print(f"Loading tokenizer for {self.model_id} with trust_remote_code=True and use_fast=True...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, use_fast=True, **revision_kwargs)
        except Exception as e:
            print(f"WARNING: Error loading fast tokenizer for {self.model_id}: {e}")
            print("Retrying with use_fast=False (and trust_remote_code=True)...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, use_fast=False, **revision_kwargs)

        # only keep args SamplingParams actually accepts
        sampling_fields = SamplingParams.__annotations__.keys()
        self._gen_defaults = {
            k: v for k, v in self.extra_kwargs.items()
            if k in sampling_fields
        }
        # Opt-in SamplingParams overrides as a JSON dict (e.g. {"skip_special_tokens": false} to keep
        # end-of-thinking markers that are special tokens); unset = unchanged behavior.
        if os.environ.get("VLLM_EXTRA_SAMPLING_KWARGS"):
            self._gen_defaults.update(json.loads(os.environ["VLLM_EXTRA_SAMPLING_KWARGS"]))

        load_kwargs = {
            k: v for k, v in self.extra_kwargs.items()
            if k not in sampling_fields and k != "adapter_path"
        }
        load_kwargs.setdefault("max_model_len", 16384)
        # Model-specific engine options (text-only loading of multimodal checkpoints, custom code, explicit
        # overrides such as Mamba cache dtypes) are decided in one place: models/engine_profiles.py.
        load_kwargs.update(resolve_engine_kwargs(self.model_id, self.extra_kwargs.get("revision")))

        adapter_path = self.extra_kwargs.get("adapter_path")
        if adapter_path:
            adapter_config = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
            lora_rank = adapter_config["r"]
            print(f"[vLLM] Enabling LoRA: adapter at {adapter_path} (rank={lora_rank})")
            load_kwargs["enable_lora"] = True
            load_kwargs["max_loras"] = 1
            load_kwargs["max_lora_rank"] = lora_rank

        native_max_len = self._detect_native_max_len()
        if native_max_len is not None and load_kwargs["max_model_len"] > native_max_len:
            print(
                f"[vLLM] Capping max_model_len from {load_kwargs['max_model_len']} "
                f"to {self.model_id}'s native limit {native_max_len}"
            )
            load_kwargs["max_model_len"] = native_max_len
        print(load_kwargs)

        num_gpus = torch.cuda.device_count()

        tp_size = load_kwargs.pop("tensor_parallel_size", None) or num_gpus

        print(f"[vLLM] Using tensor_parallel_size={tp_size} (available GPUs={num_gpus})")
        self.llm = LLM(
            model=self.model_id,
            tensor_parallel_size=tp_size,
            **load_kwargs
        )

        if adapter_path:
            from vllm.lora.request import LoRARequest
            self._lora_request = LoRARequest(
                lora_name=Path(adapter_path).name, lora_int_id=1, lora_path=adapter_path,
            )

    # ---- SINGLE PROMPT ----
    def single(self, prompt: str, *args, **kwargs) -> str:
        prompt = self.apply_chat(prompt)

        gen_kwargs = {**self._gen_defaults, **kwargs}

        gen_kwargs.pop("concurrency", None)
        gen_kwargs.setdefault("max_tokens", 256)
        if "max_new_tokens" in gen_kwargs:
            gen_kwargs["max_tokens"] = gen_kwargs["max_new_tokens"]

        sampling = SamplingParams(**gen_kwargs)

        outputs = self.llm.generate([prompt], sampling, lora_request=self._lora_request)

        return outputs[0].outputs[0].text.strip()

    # ---- BATCH PROMPTS ----
    def batch(self, prompts: List[str], *args, **kwargs) -> List[str]:
        prompts = [self.apply_chat(p) for p in prompts]

        gen_kwargs = {**self._gen_defaults, **kwargs}
        
        gen_kwargs.pop("concurrency", None)
        gen_kwargs.setdefault("max_tokens", 256)
        if "max_new_tokens" in gen_kwargs:
            gen_kwargs["max_tokens"] = gen_kwargs["max_new_tokens"]


        sampling = SamplingParams(**gen_kwargs)
        print("*"*100)
        print(sampling)

        outputs = self.llm.generate(prompts, sampling, lora_request=self._lora_request)

        return [o.outputs[0].text.strip() for o in outputs]
