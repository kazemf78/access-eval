from typing import List, Optional
from access_eval.models.base import LocalLLM

_LOAD_ONLY_KEYS = {
    "torch_dtype", "dtype", "device_map", "max_memory", "low_cpu_mem_usage",
    "attn_implementation", "revision", "trust_remote_code", "use_safetensors",
    "load_in_8bit", "load_in_4bit", "quantization_config",
}

class HFLocalLLM(LocalLLM):
    """Simple HF wrapper; supports single/multi-GPU based on device_map."""

    def __init__(self, model_id: str, *args, **kwargs):
        super().__init__(model_id, *args, **kwargs)
        self._gen_defaults = {}
        self._input_device = None

    # ---- helpers ----
    def apply_chat(self, p: str) -> str:
        """Apply tokenizer chat template to a single user prompt."""
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

    def prepare(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

        # Controlled-contamination pilot (controlled_contamination_plan.md v4,
        # item 2): an explicit, already metadata-checked adapter directory
        # from factory.get_llm_from_list(). Excluded from both load_kwargs
        # (not a from_pretrained() arg) and _gen_defaults (not a generate()
        # arg) -- it's consumed directly below instead.
        adapter_path = self.extra_kwargs.get("adapter_path")
        load_kwargs = {k: v for k, v in self.extra_kwargs.items() if k in _LOAD_ONLY_KEYS}
        self._gen_defaults = {
            k: v for k, v in self.extra_kwargs.items()
            if k not in _LOAD_ONLY_KEYS and k != "adapter_path"
        }

        # revision must also reach config/tokenizer, not just the model weights
        # below (see factory.get_llm for why the dict is conditional).
        revision_kwargs = {"revision": load_kwargs["revision"]} if load_kwargs.get("revision") else {}

        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side='left')
        cfg = AutoConfig.from_pretrained(self.model_id, **revision_kwargs)
        context_len = getattr(cfg, "max_position_embeddings", 4096)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            padding_side="left",
            model_max_length=context_len,
            **revision_kwargs,
        )

        load_kwargs.setdefault("dtype", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        load_kwargs.setdefault("device_map", "auto")
        load_kwargs.setdefault("low_cpu_mem_usage", True)

        print(f"[HFLocalLLM] Loading {self.model_id} with dtype={load_kwargs['dtype']} device_map=auto")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        print(f"[HFLocalLLM] Loaded model: {self.model_id}")
        if hasattr(self.model, "hf_device_map"):
            print("[HFLocalLLM] Device map:")
            print(self.model.hf_device_map)
        else:
            print("[HFLocalLLM] No hf_device_map attribute — model fully on one device")

        if adapter_path:
            from peft import PeftModel
            print(f"[HFLocalLLM] Attaching LoRA adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

        # # --- detect whether generate() includes input tokens ---
        # test_text = "Test prompt"
        # inputs = self.tokenizer(test_text, return_tensors="pt")
        # --- detect whether generate() includes input tokens ---
        test_text = self.apply_chat("Test prompt")  # ✅ use your own helper
        inputs = self.tokenizer(test_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            sample_out = self.model.generate(**inputs, max_new_tokens=3)

        input_ids = inputs["input_ids"][0]
        output_ids = sample_out[0]

        includes_input = output_ids[: len(input_ids)].equal(input_ids)
        self.includes_input_in_generate = includes_input

        print(f"[HFLocalLLM] Detected includes_input_in_generate={includes_input} for {self.model_id}")

        device_map = load_kwargs.get("device_map", None)
        if device_map is None or (isinstance(device_map, str) and device_map != "auto"):
            import torch
            if torch.cuda.is_available():
                self.model.to("cuda")
                self._input_device = "cuda"
            else:
                self._input_device = "cpu"
        else:
            first_cuda = next((p.device for p in self.model.parameters() if p.device.type == "cuda"), None)
            self._input_device = first_cuda or "cpu"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prepare_inputs(self, texts, add_special_tokens: bool = True):
        """
        `add_special_tokens` must be False for text that already came out of
        `apply_chat()` -- found by review: `apply_chat_template(tokenize=False)`
        already embeds whatever special tokens the chat template wants (e.g.
        Llama-3.1's `<|begin_of_text|>`), so re-tokenizing that STRING with
        the tokenizer's own default `add_special_tokens=True` inserts a
        second one on top (confirmed directly: Llama-3.1-8B-Instruct
        duplicates `<|begin_of_text|>`; Qwen3-8B happens not to exhibit
        this, so it went unnoticed there). `single()`/`batch()` -- the only
        callers -- always pre-template via `apply_chat()` first, so they
        pass False; the True default is only for a hypothetical caller that
        tokenizes raw, non-templated text directly.
        """
        import torch
        inputs = self.tokenizer(
            texts,                    # str or list[str]
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=add_special_tokens,
        )
        if torch.cuda.is_available():
            inputs = {k: v.to(self._input_device) for k, v in inputs.items()}
        return inputs

    # ---- NEW SHARED POSTPROCESSING ----
    def _postprocess_generation(self, outputs, inputs, gen_kwargs) -> List[str]:
        """Decode and clean up one or more generated sequences."""
        results = []
        batch_size = outputs.shape[0]  # works for both single and multi

        eos_ids = self.tokenizer.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        stop_strings = getattr(self.model.generation_config, "stop_strings", None)

        max_new = gen_kwargs.get("max_new_tokens", None)
        max_time = gen_kwargs.get("max_time", None)

        n_input = inputs["attention_mask"].shape[1]
        for i in range(batch_size):
            # n_input = (inputs["attention_mask"][i] > 0).sum().item()
            # generated_tokens = outputs[i][n_input:]
            decoded = self.tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
            # n_input = (inputs["attention_mask"][i] > 0).sum().item()
            if getattr(self, "includes_input_in_generate", True):
                generated_tokens = outputs[i][n_input:]
            else:
                generated_tokens = outputs[i]
            decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # optional cleanup
            if (
                self.tokenizer.eos_token
                and self.tokenizer.eos_token in decoded
                and not decoded.rstrip().endswith(self.tokenizer.eos_token)
            ):
                decoded = decoded.split(self.tokenizer.eos_token, 1)[0]
            
            stopped_naturally = False
            # Check if EOS token ID is present anywhere in the sequence
            if eos_ids and any(tok in eos_ids for tok in outputs[i].tolist()):
                stopped_naturally = True
            # Or if the text itself ends with known stop strings
            elif stop_strings and any(decoded.rstrip().endswith(s) for s in stop_strings):
                stopped_naturally = True

            # number of generated tokens (after input tokens)
            gen_len = len(generated_tokens)

            # --- determine stopping reason ---
            if stopped_naturally:
                reason = "eos"
            elif max_new is not None and gen_len >= max_new:
                reason = "max_tokens"
            elif max_time is not None:
                reason = "timeout"
            else:
                reason = "unknown"
            
            if not stopped_naturally: # todo: is this working properly????
                print(f"⚠️ Warning: generation #{i} did not stop naturally!")
                print(f"Stop reason: {reason}, max_new_tokens: {max_new}, max_time: {max_time}, #gen_tokens: {gen_len}")
                print(decoded)
                print("*"*100)

            results.append(decoded.strip())
            # results.append({
            #     "text": decoded.strip(),
            #     "stopped_naturally": stopped_naturally,
            # })

        return results

    # ---- SINGLE PROMPT ----
    def single(self, prompt: str, *args, **kwargs) -> str:

        prompt = self.apply_chat(prompt)
        inputs = self._prepare_inputs(prompt, add_special_tokens=False)

        gen_kwargs = {**self._gen_defaults, **kwargs}
        gen_kwargs.setdefault("max_new_tokens", 256)
        if "temperature" in gen_kwargs and gen_kwargs["temperature"] <= 0:
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature", None)

        outputs = self.model.generate(**inputs, **gen_kwargs)
        return self._postprocess_generation(outputs, inputs, gen_kwargs)[0]#["text"]

    # ---- BATCH PROMPTS ----
    def batch(self, prompts: List[str], *args, **kwargs) -> List[str]:

        prompts = [self.apply_chat(p) for p in prompts]
        inputs = self._prepare_inputs(prompts, add_special_tokens=False)

        gen_kwargs = {**self._gen_defaults, **kwargs}
        gen_kwargs.setdefault("max_new_tokens", 256)
        if "temperature" in gen_kwargs and gen_kwargs["temperature"] <= 0:
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature", None)

        outputs = self.model.generate(**inputs, **gen_kwargs)
        return self._postprocess_generation(outputs, inputs, gen_kwargs)

