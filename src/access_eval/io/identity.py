"""
Generation-config identity hashing -- a lightweight substitute for a persisted
GenerationConfig/PromptTemplate table. Derives a short, stable id from
everything that determines how a response was generated (model incl. any
[with_thinking]/@revision suffix, decoding params, and the *resolved* prompt
text, not just its name). Stored as a plain string column in generation/
grading CSVs; never persisted separately (see io/compat.py for legacy rows).

Deliberately excluded: deployment/hardware settings (gpu_memory_utilization,
dtype, quantization) -- not swept as experimental variables today and don't
change scientific output at temperature=0. Pass anything else that should
distinguish a config via **extra rather than editing this function.
"""
import hashlib
import json
from typing import Any, Optional

HASH_PREFIX = "cfg_"
HASH_LENGTH = 12
FORMULATION_HASH_LENGTH = 16  # matches refilter_paraphrases.py's pre-existing convention


def compute_gen_config_hash(
    model: str,
    revision: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    prompt_name: Optional[str] = None,
    prompt_template_text: Optional[str] = None,
    **extra: Any,
) -> str:
    """Short deterministic id for a generation config, e.g. "cfg_3f9a2b7c1d04".
    Prefixed so it can never collide with the "legacy" sentinel in io/compat.py."""
    payload = {
        "model": model, "revision": revision, "temperature": temperature,
        "max_tokens": max_tokens, "prompt_name": prompt_name,
        "prompt_template_text": prompt_template_text, **extra,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:HASH_LENGTH]
    return f"{HASH_PREFIX}{digest}"


def compute_content_hash(*parts: str, length: int = HASH_LENGTH) -> str:
    """Generic short content hash over one or more text parts."""
    canonical = "|||".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def paraphrase_content_hash(original_question: str, paraphrased_question: str) -> str:
    """Stable identity for one (original, paraphrase) pair, independent of
    `idx` (a dataset row position -- stable for static datasets, not
    guaranteed for unpinned HF-auto-downloaded ones). Canonical implementation
    shared by paraphrase.py (generation time) and refilter_paraphrases.py."""
    return compute_content_hash(
        original_question.strip(), paraphrased_question.strip(),
        length=FORMULATION_HASH_LENGTH,
    )


def original_question_hash(original_question: str) -> str:
    """Stable identity for an inquiry's original wording, independent of idx."""
    return compute_content_hash(original_question.strip(), length=FORMULATION_HASH_LENGTH)


def formulation_hashes(original_questions, paraphrased_questions, idxs=None):
    """Row identity of formulations for resume / judge-cache keys. Text-only (paraphrase_content_hash) by default.
    Pass `idxs` for datasets whose prompt adds per-inquiry context outside the formulation text (a registry
    prompt_builder, e.g. MMLU's fixed answer options): there the same stem text recurs across different questions
    ("Which of the following statements is false?"), so the text alone would merge distinct formulations."""
    if idxs is None:
        return [paraphrase_content_hash(o, p) for o, p in zip(original_questions, paraphrased_questions)]
    return [f"{paraphrase_content_hash(o, p)}#{int(i)}" for o, p, i in zip(original_questions, paraphrased_questions, idxs)]
