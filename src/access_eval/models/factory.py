# src/access_eval/models/factory.py
"""
Factory for creating LLM instances based on model name.
Auto-detects whether to use API or local models.
"""

from access_eval.models.openai_compat import OpenAICompatLLM, is_endpoint_model
import json
import re
from pathlib import Path
from typing import Optional, Union


# Model name patterns for API providers
API_MODEL_PATTERNS = {
    "openai": ["gpt-", "o1-", "o3-"],
    "anthropic": ["claude-"],
    # Add more providers as needed
}

# Known API models (can be extended)
KNOWN_API_MODELS = {
    # OpenAI
    "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-3.5-turbo",
    "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo-16k",
    "o1-preview", "o1-mini",
    
    # Anthropic (if you add support)
    "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
    "claude-3-5-sonnet", "claude-3-5-haiku",
}

# Special suffix for enabling thinking mode
THINKING_SUFFIX = "[with_thinking]"


def _parse_revision(model_name: str) -> tuple[str, Optional[str]]:
    """
    Split an optional "@revision" suffix off a model name -- checkpoint/revision
    support, following the same embedded-suffix convention as [with_thinking]
    below (rather than a separate parameter threaded through paths.py /
    answer_generation.py / evaluation.py): whatever full string the caller uses
    as `model_name` already flows verbatim into RunPaths._safe_model() and the
    `model` CSV column, so embedding "@revision" directly gives distinct,
    collision-free filenames/columns per checkpoint with no changes needed
    anywhere else in the pipeline.

    Examples:
        _parse_revision("meta-llama/Llama-3.1-8B-Instruct@refs/pr/3")
            -> ("meta-llama/Llama-3.1-8B-Instruct", "refs/pr/3")
        _parse_revision("Qwen/Qwen3-8B") -> ("Qwen/Qwen3-8B", None)

    Only meaningful for local (HF/vLLM) models -- API models have no revision
    concept and never receive this kwarg. Must be applied AFTER
    _parse_model_name() strips any [with_thinking] suffix, so
    "model@revision[with_thinking]" parses as revision="revision", not
    "revision[with_thinking]".
    """
    if "@" not in model_name:
        return model_name, None
    base, _, revision = model_name.rpartition("@")
    return base, revision


def _parse_model_name(model_name: str) -> tuple[str, bool]:
    """
    Parse model name and extract thinking mode flag.
    
    Args:
        model_name: Model name, optionally with [with_thinking] suffix
        
    Returns:
        (clean_model_name, enable_thinking)
        
    Examples:
        _parse_model_name("gpt-4o") → ("gpt-4o", False)
        _parse_model_name("gpt-4o[with_thinking]") → ("gpt-4o", True)
        _parse_model_name("meta-llama/Llama-3.1-8B[with_thinking]") → ("meta-llama/Llama-3.1-8B", True)
    """
    enable_thinking = model_name.lower().endswith(THINKING_SUFFIX.lower())
    if enable_thinking:
        clean_name = model_name[:-(len(THINKING_SUFFIX))]
    else:
        clean_name = model_name
    return clean_name, enable_thinking


# ---------------------------------------------------------------------------
# Controlled-contamination pilot: [hf] / [lora:<fp>] suffix parsing
# (controlled_contamination_plan.md v4, items 1 and 2). Scoped to
# get_llm_from_list() only -- get_llm()/_parse_model_name()/is_api_model()
# above are unmodified, matching the plan's Files Modified list.
# ---------------------------------------------------------------------------

HF_SUFFIX_TAG = "hf"
LORA_SUFFIX_TAG = "lora"
THINKING_SUFFIX_TAG = "with_thinking"
_KNOWN_SUFFIX_TAGS = (HF_SUFFIX_TAG, LORA_SUFFIX_TAG, THINKING_SUFFIX_TAG)

# Adapter fingerprints are SHA-256 digests truncated to 16 hex chars, prefixed
# "adp_" (mirrors io/identity.py's own "cfg_"/16-char formulation-hash
# conventions, defined here rather than in identity.py since identity.py
# itself is unmodified by this pilot -- see item 3).
ADAPTER_FINGERPRINT_RE = re.compile(r"^adp_[0-9a-f]{16}$")
_TRAILING_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]$")


def parse_adapter_suffixes(model_name: str) -> tuple[str, dict]:
    """
    Strict suffix parser covering `[with_thinking]`, `[hf]`, `[lora:<fp>]`,
    and a trailing `@revision` all at once -- a superset of
    `_parse_model_name`/`_parse_revision` above, needed because those two
    only ever strip ONE bracket type each from the end and would otherwise
    stop at whichever suffix happens to be outermost, silently leaving any
    suffix stacked underneath it unstripped (e.g. stopping at a trailing
    `[with_thinking]` and leaving `[hf][lora:<fp>]` embedded in what
    `_parse_revision` then treats as the revision string). This parser
    instead loops, stripping ANY recognized bracket suffix from the end
    regardless of stacking order, until none remain, then resolves
    `@revision` on what's left.

    Returns (clean_model_name, info) where info has keys:
        enable_thinking (bool), revision (str | None),
        force_hf (bool) -- `[hf]` was present,
        adapter_fingerprint (str | None) -- the validated `<fp>` from
            `[lora:<fp>]`, if present.

    Raises ValueError -- rather than silently absorbing the text into the
    literal HF model id -- for:
        - an unrecognized bracket suffix (e.g. a typo like `[laura:...]`),
        - a duplicated suffix (e.g. `[hf][hf]`),
        - a malformed `[with_thinking]`/`[hf]` payload, or a malformed
          `[lora:...]` payload (missing fingerprint, or one that doesn't
          match the expected `adp_<16 hex chars>` format).
    """
    remaining = model_name
    seen_tags: set[str] = set()
    force_hf = False
    enable_thinking = False
    adapter_fingerprint: Optional[str] = None

    while True:
        m = _TRAILING_BRACKET_RE.search(remaining)
        if not m:
            break
        payload = m.group(1)
        bracket_text = m.group(0)
        tag = payload.split(":", 1)[0]

        if tag not in _KNOWN_SUFFIX_TAGS:
            raise ValueError(
                f"Unrecognized bracket suffix {bracket_text!r} in model name "
                f"{model_name!r}. Valid suffixes: [with_thinking], [hf], "
                f"[lora:<fingerprint>]."
            )
        if tag in seen_tags:
            raise ValueError(
                f"Duplicated {bracket_text!r} suffix in model name {model_name!r}."
            )
        seen_tags.add(tag)

        if tag == THINKING_SUFFIX_TAG:
            if payload != THINKING_SUFFIX_TAG:
                raise ValueError(f"Malformed suffix {bracket_text!r} in {model_name!r}.")
            enable_thinking = True
        elif tag == HF_SUFFIX_TAG:
            if payload != HF_SUFFIX_TAG:
                raise ValueError(f"Malformed suffix {bracket_text!r} in {model_name!r}.")
            force_hf = True
        elif tag == LORA_SUFFIX_TAG:
            _, _, fp = payload.partition(":")
            if not fp or not ADAPTER_FINGERPRINT_RE.match(fp):
                raise ValueError(
                    f"Malformed [lora:...] suffix {bracket_text!r} in {model_name!r}: "
                    f"expected [lora:<fingerprint>] with an adp_<16 hex chars> fingerprint."
                )
            adapter_fingerprint = fp

        remaining = remaining[: -len(bracket_text)]

    # An unterminated/malformed suffix like "Qwen/Qwen3-8B[lora:broken" (no
    # closing "]") never matches _TRAILING_BRACKET_RE's "$"-anchored pattern,
    # so the loop above exits immediately without raising -- found by review
    # to silently fall through into the literal model id instead of being
    # rejected. Any leftover bracket character here means a suffix-shaped
    # fragment failed to parse as a complete, recognized suffix.
    if "[" in remaining or "]" in remaining:
        raise ValueError(
            f"Unterminated or malformed bracket suffix in model name {model_name!r} "
            f"(leftover '[' or ']' after parsing recognized suffixes)."
        )

    clean_model_name, revision = _parse_revision(remaining)

    return clean_model_name, {
        "enable_thinking": enable_thinking,
        "revision": revision,
        "force_hf": force_hf,
        "adapter_fingerprint": adapter_fingerprint,
    }


def _verify_adapter_metadata(
    adapter_path: str, fingerprint: str, base_model: str, revision: Optional[str],
) -> dict:
    """
    Cheap correctness check (item 2, v4): confirm `<adapter_path>/metadata.json`
    was actually written for THIS fingerprint and THIS base model/revision,
    without recomputing a weight digest. Catches "the wrong adapter
    directory got passed in under this fingerprint" -- e.g. a copy-paste
    mistake in an orchestrator script -- cheaply. Raises ValueError on any
    missing file or field mismatch.
    """
    meta_path = Path(adapter_path) / "metadata.json"
    if not meta_path.exists():
        raise ValueError(
            f"No metadata.json found at {adapter_path!r} for adapter fingerprint "
            f"{fingerprint!r} -- refusing to load an unverifiable adapter."
        )
    with open(meta_path) as f:
        metadata = json.load(f)

    if metadata.get("fingerprint") != fingerprint:
        raise ValueError(
            f"metadata.json at {adapter_path!r} records fingerprint "
            f"{metadata.get('fingerprint')!r}, expected {fingerprint!r} -- "
            f"this looks like the wrong adapter directory."
        )
    if metadata.get("base_model") != base_model:
        raise ValueError(
            f"metadata.json at {adapter_path!r} records base_model "
            f"{metadata.get('base_model')!r}, expected {base_model!r}."
        )
    if metadata.get("revision") != revision:
        raise ValueError(
            f"metadata.json at {adapter_path!r} records revision "
            f"{metadata.get('revision')!r}, expected {revision!r}."
        )
    return metadata


def is_api_model(model_name: str) -> bool:
    """
    Determine if a model should use API (vs local inference).
    Handles [with_thinking] suffix automatically.
    
    Args:
        model_name: Name of the model (may include [with_thinking])
        
    Returns:
        True if model should use API, False if local
        
    Examples:
        is_api_model("gpt-4o") → True
        is_api_model("gpt-4o[with_thinking]") → True
        is_api_model("meta-llama/Llama-3.1-8B-Instruct") → False
        is_api_model("Qwen/Qwen2.5-7B-Instruct[with_thinking]") → False
    """
    # Strip thinking suffix for detection
    clean_name, _ = _parse_model_name(model_name)
    if is_endpoint_model(clean_name):
        return True
    model_lower = clean_name.lower()
    
    # Check exact matches first
    if clean_name in KNOWN_API_MODELS:
        return True
    
    # Check patterns
    for provider, patterns in API_MODEL_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in model_lower:
                if "/" not in clean_name:  # Exclude HuggingFace-style names
                    return True
                
    
    # Check for HuggingFace-style naming (org/model)
    # These are typically local models
    if "/" in model_name:
        # Exception: if it's an API-hosted model with slash
        # (you can add exceptions here if needed)
        return False
    
    # Default: assume local if not recognized
    return False


def get_llm(
    model_name: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.85,
    use_vllm: bool = True,
    adapter_path: Optional[str] = None,
    **kwargs
) -> Union['APILLM', 'LocalLLM']: # todo: change this to handle diffusion model as well!
    """
    Factory function to create the appropriate LLM instance.
    Auto-detects whether to use API or local based on model name.
    Supports [with_thinking], [hf], and [lora:<fp>] suffixes (the same
    strict parser as get_llm_from_list() -- see
    controlled_contamination_plan.md v4, item 2: review found this function
    and get_llm_from_list() had drifted into inconsistent suffix handling,
    so both now go through parse_adapter_suffixes()).

    Args:
        model_name: Name of the model (e.g., "gpt-4o", "meta-llama/Llama-3.1-8B")
                   May include [with_thinking], [hf], [lora:<fp>], and/or @revision.
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        gpu_memory_utilization: GPU memory fraction for local models (vLLM)
        use_vllm: Use vLLM for local models (faster) vs HF transformers --
            overridden to HF when [hf] is present or an adapter is loaded,
            same as get_llm_from_list().
        adapter_path: Explicit filesystem path to a trained LoRA adapter
            directory, required exactly when `model_name` carries a
            [lora:<fp>] suffix (see get_llm_from_list()'s docstring for the
            full contract -- identical here).
        **kwargs: Additional arguments passed to LLM constructor

    Returns:
        Initialized LLM instance (not yet prepared!)

    Examples:
        # API model (auto-detected)
        llm = get_llm("gpt-4o")

        # Local model (auto-detected)
        llm = get_llm("meta-llama/Llama-3.1-8B-Instruct")

        # Enable thinking mode
        llm = get_llm("gpt-4o[with_thinking]")
        llm = get_llm("meta-llama/Llama-3.1-8B[with_thinking]")

        # Force specific backend
        llm = get_llm("meta-llama/Llama-3.1-8B-Instruct", use_vllm=False)
    """

    # Strict suffix parsing (item 1/2) -- see parse_adapter_suffixes's
    # docstring for why this single loop replaces the old two-step
    # _parse_model_name/_parse_revision parse (which only ever stripped ONE
    # bracket type from the end and could silently mis-parse a stack of
    # suffixes).
    clean_model_name, suffix_info = parse_adapter_suffixes(model_name)
    enable_thinking = suffix_info["enable_thinking"]
    revision = suffix_info["revision"]
    force_hf = suffix_info["force_hf"]
    adapter_fingerprint = suffix_info["adapter_fingerprint"]

    if adapter_fingerprint is not None and adapter_path is None:
        raise ValueError(
            f"{model_name!r} names adapter fingerprint {adapter_fingerprint!r} "
            f"via [lora:...], but no adapter_path was given to resolve it."
        )
    if adapter_path is not None and adapter_fingerprint is None:
        raise ValueError(
            f"adapter_path={adapter_path!r} was given but {model_name!r} carries "
            f"no [lora:<fp>] suffix to match it against."
        )
    if adapter_fingerprint is not None:
        _verify_adapter_metadata(adapter_path, adapter_fingerprint, clean_model_name, revision)

    if is_endpoint_model(clean_model_name):
        llm = OpenAICompatLLM(model=clean_model_name, max_tokens=max_tokens, temperature=temperature)
    elif is_api_model(clean_model_name):
        # Use API (no revision concept for API models)
        from access_eval.models.openai_api import OpenAIAPILLM
        llm = OpenAIAPILLM(
            model=clean_model_name,
            max_output_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    else:
        # Only forward `revision` when one was actually requested -- passing
        # revision=None explicitly through to from_pretrained()/LLM() differs
        # from omitting it entirely on some versions, so the no-checkpoint case
        # must stay byte-identical to today's behavior.
        revision_kwargs = {"revision": revision} if revision else {}
        adapter_kwargs = {"adapter_path": adapter_path} if adapter_path else {}

        # [hf]/an adapter always forces the HF backend, same as
        # get_llm_from_list() -- an adapter can only be attached there.
        if use_vllm and not force_hf and adapter_fingerprint is None:
            from access_eval.models.vllm_local import VLLMLocalLLM
            kwargs = {k: v for k, v in kwargs.items() if k != "api_conc"}
            llm = VLLMLocalLLM(
                model_id=clean_model_name,
                max_tokens=max_tokens,
                gpu_memory_utilization=gpu_memory_utilization,
                temperature=temperature,
                tensor_parallel_size=kwargs.pop("tensor_parallel_size", None),
                **revision_kwargs,
                **kwargs
            )
        else:
            from access_eval.models.hf_local import HFLocalLLM
            llm = HFLocalLLM(
                model_id=clean_model_name,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                **revision_kwargs,
                **adapter_kwargs,
                **kwargs
            )

    # Set thinking mode if requested
    if enable_thinking:
        llm.enable_thinking = True
    else:
        llm.enable_thinking = False

    return llm


def get_llm_for_evaluation(
    model_name: str,
    max_tokens: int = 32,
    **kwargs
) -> Union['APILLM', 'LocalLLM']:
    """
    Convenience function for getting LLM for evaluation/grading.
    Uses appropriate defaults for evaluation tasks.
    
    Args:
        model_name: Name of the judge model
        max_tokens: Max tokens (evaluation responses are short)
        **kwargs: Additional arguments
        
    Returns:
        LLM instance configured for evaluation
    """
    return get_llm(
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=kwargs.pop("temperature", 0.0),  # Deterministic is the default for grading unless overridden by caller
        **kwargs
    )


# Convenience function for backward compatibility
def get_llm_from_list(
    model_name: str,
    api_models: list,
    max_local_tokens: int = 256,
    max_api_tokens: int = 256,
    adapter_path: Optional[str] = None,
    prefer_vllm_for_lora: bool = False,
    **kwargs
) -> Union['APILLM', 'LocalLLM']:
    """
    Legacy function that matches your current answer_generation.py pattern.
    Can be used as a drop-in replacement for the old get_llm() function.
    Supports [with_thinking], [hf], and [lora:<fp>] suffixes.

    Args:
        model_name: Name of the model (may include [with_thinking], [hf],
            [lora:<fp>], and/or @revision -- see parse_adapter_suffixes()).
        api_models: List of model names that should use API
        max_local_tokens: Max tokens for local models
        max_api_tokens: Max tokens for API models
        adapter_path: Explicit filesystem path to a trained LoRA adapter
            directory (controlled_contamination_plan.md v4, item 2) --
            required exactly when `model_name` carries a `[lora:<fp>]`
            suffix, and forwarded to HFLocalLLM after a cheap metadata.json
            check confirms it matches `<fp>` and this base model/revision.
            None (the default) preserves every existing call site's exact
            behavior.
        prefer_vllm_for_lora: If True AND an adapter is present, route to
            VLLMLocalLLM (with native LoRA support) instead of forcing
            HFLocalLLM. Default False preserves EVERY existing contamination-
            pilot run's behavior exactly (Track 1/Track 2 both rely on
            pre-/post-FT sharing an identical backend, per
            controlled_contamination_plan.md item 1) -- this is an explicit
            opt-in for cases where HFLocalLLM's raw `.generate()` throughput
            (no continuous batching/paged attention) is prohibitively slow
            for a large, exploratory re-evaluation, confirmed directly: a
            5710-formulation full-pool pass was projected at ~7 hours under
            HFLocalLLM.
        **kwargs: Additional arguments

    Returns:
        LLM instance

    Examples:
        llm = get_llm_from_list("gpt-4o", ["gpt-4o", "gpt-4.1"])
        llm = get_llm_from_list("gpt-4o[with_thinking]", ["gpt-4o"])
        llm = get_llm_from_list("Qwen/Qwen3-8B[hf]", [])  # pre-FT, HF-forced
        llm = get_llm_from_list(
            "Qwen/Qwen3-8B[hf][lora:adp_0123456789abcdef]", [],
            adapter_path="run_artifacts/gsm8k_test_v00/contamination/adapters/adp_0123456789abcdef",
        )  # post-FT
    """
    # Strict suffix parsing (item 1): [with_thinking]/[hf]/[lora:<fp>], then
    # @revision -- raises ValueError on anything unrecognized/duplicated/
    # malformed rather than silently absorbing it into the literal model id.
    clean_model_name, suffix_info = parse_adapter_suffixes(model_name)
    enable_thinking = suffix_info["enable_thinking"]
    revision = suffix_info["revision"]
    force_hf = suffix_info["force_hf"]
    adapter_fingerprint = suffix_info["adapter_fingerprint"]

    if adapter_fingerprint is not None and adapter_path is None:
        raise ValueError(
            f"{model_name!r} names adapter fingerprint {adapter_fingerprint!r} "
            f"via [lora:...], but no adapter_path was given to resolve it."
        )
    if adapter_path is not None and adapter_fingerprint is None:
        raise ValueError(
            f"adapter_path={adapter_path!r} was given but {model_name!r} carries "
            f"no [lora:<fp>] suffix to match it against."
        )
    if adapter_fingerprint is not None:
        _verify_adapter_metadata(adapter_path, adapter_fingerprint, clean_model_name, revision)

    # Only forward `revision` when actually requested -- see get_llm() for why
    # revision=None must never be passed through explicitly.
    revision_kwargs = {"revision": revision} if revision else {}
    adapter_kwargs = {"adapter_path": adapter_path} if adapter_path else {}

    gpu_memory_utilization=kwargs.pop("gpu_memory_utilization", 0.85)
    tensor_parallel_size=kwargs.pop("tensor_parallel_size", None)

    if is_endpoint_model(clean_model_name):
        llm = OpenAICompatLLM(model=clean_model_name, max_tokens=max_api_tokens)
    elif clean_model_name in api_models:
        from access_eval.models.openai_api import OpenAIAPILLM
        llm = OpenAIAPILLM(
            model=clean_model_name,
            max_output_tokens=max_api_tokens,
            **kwargs
        )
    elif "gpt-oss" in clean_model_name or (
        not (adapter_fingerprint is not None and prefer_vllm_for_lora)
        and (force_hf or adapter_fingerprint is not None)
    ):
        # prefer_vllm_for_lora=True overrides BOTH the literal [hf] suffix
        # (always present alongside [lora:...] by convention) AND the
        # adapter-forces-HF default -- gpt-oss models are the one
        # exception that always forces HF regardless.
        print(f"Using HFLocalLLM for {clean_model_name}"
              + (" (gpt-oss)" if "gpt-oss" in clean_model_name else "")
              + (" [hf]-forced" if force_hf else "")
              + (f" with adapter {adapter_fingerprint}" if adapter_fingerprint else ""))
        from access_eval.models.hf_local import HFLocalLLM
        llm = HFLocalLLM(
            model_id=clean_model_name,
            max_new_tokens=max_local_tokens,
            **revision_kwargs,
            **adapter_kwargs,
            **kwargs
        )
    else:
        print(f"Using VLLMLocalLLM for {clean_model_name}"
              + (f" with LoRA adapter {adapter_fingerprint} (prefer_vllm_for_lora=True)" if adapter_fingerprint else ""))
        from access_eval.models.vllm_local import VLLMLocalLLM
        llm = VLLMLocalLLM(
            model_id=clean_model_name,
            max_tokens=max_local_tokens,
            # gpu_memory_utilization=kwargs.pop("gpu_memory_utilization", 0.85),
            gpu_memory_utilization=gpu_memory_utilization,
            # tensor_parallel_size=kwargs.pop("tensor_parallel_size", None),
            tensor_parallel_size=tensor_parallel_size,
            **revision_kwargs,
            **adapter_kwargs,
            **kwargs
        )

    # Set thinking mode if requested
    if enable_thinking:
        llm.enable_thinking = True
    else:
        llm.enable_thinking = False

    return llm