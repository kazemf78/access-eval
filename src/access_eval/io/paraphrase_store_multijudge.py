# src/access_eval/io/paraphrase_store_multijudge.py
"""
Multi-judge COPY of the acceptance view in io/paraphrase_store.py (that module is
untouched). Everything else the pipeline needs from io/ (load_generated_paraphrases,
the judge cache, dedup) is imported unchanged from the originals.

A candidate is accepted only if EVERY judge in the list has a cached llm_equiv == True
verdict for it. The cache key is the original's (para_hash, judge_model, prompt_hash); the
generating model is not part of it, so a candidate written by any response model (e.g. a legacy
gemma-3-27b pool row) counts the same as one written by the current paraphraser.

Options-aware cache identity
----------------------------
para_hash is over (original stem, paraphrase) only. When a judge prompt shows the answer
options ({options}), the verdict depends on those options too, and the same stem can occur with
different options blocks. For such prompts the STORED prompt_hash is folded with a hash of the
options block: "<template hash>+o<options hash>". Prompts without {options} keep the plain
template hash, so every pre-existing verdict is read exactly as before. The original cache
module is used unchanged (the composite is just a longer string in its prompt_hash column).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from access_eval.io.identity import compute_content_hash
from access_eval.io.judge_cache import load_judge_cache, prompt_hash_for
from access_eval.io.paraphrase_store import load_generated_paraphrases
from access_eval.prompts import load_prompt

MAX_JUDGES = 3
DEFAULT_PROMPT_NAME = "v1"


def resolve_prompt_path(prompt_name: str) -> str:
    """A bare name (e.g. "v1") lives under prompts/paraphrase/, matching the original
    pipeline. A name containing "/" is a path relative to prompts/ (e.g.
    "paraphrase_multijudge/strict_mcq_v2")."""
    return prompt_name if "/" in prompt_name else f"paraphrase/{prompt_name}"


def judge_kind_for(template: str) -> str:
    """"json2": the prompt asks for {"semantically_equivalent": .., "all_values_preserved": ..};
    "yesno": the original single-token YES/NO protocol."""
    return "json2" if "semantically_equivalent" in template else "yesno"


def options_hash(options: str) -> str:
    return compute_content_hash(str(options))


@dataclass(frozen=True)
class JudgeSpec:
    model: str
    prompt_name: str
    template: str
    prompt_hash: str
    kind: str

    @property
    def key(self):
        return (self.model, self.prompt_hash)

    @property
    def needs_options(self) -> bool:
        return "{options}" in self.template

    @property
    def label(self) -> str:
        return f"{self.model}/{self.prompt_name}"

    def stored_hash(self, options: Optional[str]) -> str:
        """The value stored in / looked up from the judge cache's prompt_hash column."""
        if not self.needs_options:
            return self.prompt_hash
        if options is None:
            raise ValueError(f"judge prompt {self.prompt_name!r} uses {{options}} but no options "
                             f"block was supplied for this question")
        return f"{self.prompt_hash}+o{options_hash(options)}"

    def as_record(self) -> dict:
        return {"model": self.model, "prompt_name": self.prompt_name,
                "prompt_hash": self.prompt_hash, "kind": self.kind,
                "sees_options": self.needs_options}


def make_judge_spec(model: str, prompt_name: str = DEFAULT_PROMPT_NAME) -> JudgeSpec:
    try:
        template = load_prompt(resolve_prompt_path(prompt_name))
    except FileNotFoundError as e:
        raise ValueError(f"judge prompt {prompt_name!r} not found ({e})") from e
    return JudgeSpec(model=model, prompt_name=prompt_name, template=template,
                     prompt_hash=prompt_hash_for(template), kind=judge_kind_for(template))


def parse_judge_specs(items: Sequence) -> List[JudgeSpec]:
    """Validate and build the judge list. Accepts "model:prompt" / "model" (prompt v1) strings,
    (model, prompt) tuples or JudgeSpec objects. 1..MAX_JUDGES entries; the same
    (model, prompt) twice -- or two prompt names resolving to identical text for one
    model -- is rejected. The same model with two DIFFERENT prompts is allowed, and so is the
    same prompt with two different models."""
    items = list(items)
    if not items:
        raise ValueError("judge list is empty: at least 1 judge is required")
    if len(items) > MAX_JUDGES:
        raise ValueError(f"judge list has {len(items)} entries; the maximum is {MAX_JUDGES}")
    specs: List[JudgeSpec] = []
    for it in items:
        if isinstance(it, JudgeSpec):
            spec = it
        elif isinstance(it, (tuple, list)):
            spec = make_judge_spec(*it)
        else:
            model, _, prompt = str(it).rpartition(":")
            if not model:  # no colon at all
                model, prompt = str(it), DEFAULT_PROMPT_NAME
            spec = make_judge_spec(model.strip(), prompt.strip() or DEFAULT_PROMPT_NAME)
        specs.append(spec)
    seen = set()
    for s in specs:
        if (s.model, s.prompt_name) in seen or s.key in seen:
            raise ValueError(f"duplicate judge in list: {s.label}")
        seen.add((s.model, s.prompt_name))
        seen.add(s.key)
    return specs


def row_stored_hashes(spec: JudgeSpec, idxs, options_by_idx: Optional[Dict[int, str]]) -> List[str]:
    """Stored prompt_hash for each pool row (by idx) under `spec`."""
    if not spec.needs_options:
        return [spec.prompt_hash] * len(idxs)
    if options_by_idx is None:
        raise ValueError(f"judge prompt {spec.prompt_name!r} uses {{options}} but no options map was supplied")
    return [spec.stored_hash(options_by_idx[int(i)]) for i in idxs]


def _yes_pairs(cache_df: pd.DataFrame, spec: JudgeSpec) -> Set[Tuple[str, str]]:
    """{(para_hash, stored_prompt_hash)} judged YES by spec.model. Matches both the plain and the
    options-folded storage forms of spec's template hash."""
    if cache_df.empty:
        return set()
    ph = cache_df["prompt_hash"].astype(str)
    m = (cache_df["judge_model"] == spec.model) & (
        (ph == spec.prompt_hash) | ph.str.startswith(spec.prompt_hash + "+o"))
    sub = cache_df[m & (cache_df["llm_equiv"] == True)]  # noqa: E712
    return set(zip(sub["para_hash"], sub["prompt_hash"].astype(str)))


def accepted_all_judges(rp, specs: Sequence[JudgeSpec],
                        options_by_idx: Optional[Dict[int, str]] = None) -> pd.DataFrame:
    """Every generated candidate accepted by ALL judges in `specs`, with each options-aware judge
    matched against the verdict stored for THIS row's options block. Same columns / dedup /
    ordering as io.paraphrase_store.accepted_paraphrases; with one options-free judge the result
    is identical to that function. A candidate with no verdict yet under some judge is excluded
    (not-yet-judged is not accepted)."""
    paras = load_generated_paraphrases(rp)
    if paras.empty:
        return paras
    cache_df = load_judge_cache(rp)
    if cache_df.empty:
        return paras.iloc[0:0]
    keep = pd.Series(True, index=paras.index)
    for spec in specs:
        yes = _yes_pairs(cache_df, spec)
        if not yes:
            return paras.iloc[0:0]
        stored = row_stored_hashes(spec, paras["idx"].tolist(), options_by_idx)
        keep &= pd.Series([(h, s) in yes for h, s in zip(paras["para_hash"], stored)], index=paras.index)
    accepted = paras[keep]
    return accepted.drop_duplicates(subset=["idx", "para_hash"], keep="first").reset_index(drop=True)
