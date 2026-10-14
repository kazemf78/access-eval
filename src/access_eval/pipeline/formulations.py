"""
Build (or top up) a formulation set: K meaning-preserving paraphrases per inquiry, validated before use.

A simplified, batch-oriented successor of pipeline/paraphrase*.py for evaluation workflows (those modules stay
unchanged for the paper). Each round, for every inquiry still short of K accepted paraphrases:

  1. generate   one generator LLM writes `oversample x deficit` candidates per inquiry, all inquiries in one batch
                (the paper's plain / toned prompts with random diversity tips and tones)
  2. diversity  drop candidates whose lexical overlap with the original, an accepted paraphrase or an earlier
                candidate is >= overlap_max (paper: 0.8)
  3. judges     every judge (model + prompt) must accept; judge i only sees judge i-1's accepts; verdicts go to the
                run's judge cache keyed by (para_hash, judge model, prompt hash [+ options hash]), so reruns and
                other configs reuse them
  4. NLI        optional: mean of both entailment directions (original <-> candidate) >= nli_threshold
                (paper: facebook/bart-large-mnli, 0.75)
  5. checks     optional: the candidate keeps the original's numbers (math datasets; the paper only flagged this)

Only one model is on the GPU at a time (models are loaded per phase and released), so generator and judges of
8-20B fit one 48 GB card. Accepted paraphrases are appended to the run's pool file (the file answer_generation
reads); every candidate's fate is logged to formulations/candidates.csv. A seed pool (e.g. UMD's validated sets)
can be installed first and topped up; rerunning with a larger K only generates the deficit.

Paper defaults per dataset for the judge prompt: SimpleQA v1; TruthfulQA, GSM8K, MATH-500 v2; MMLU the
options-aware JSON prompt paraphrase_multijudge/strict_mcq_v2 (the stem is paraphrased; options stay fixed).
"""
import gc
import json
import math
import random
import re
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from access_eval.core.paths import ProjectPaths
from access_eval.datasets.registry import get_dataset_spec
from access_eval.io.identity import compute_gen_config_hash, paraphrase_content_hash
from access_eval.io.judge_cache import append_judge_verdicts, load_judge_cache
from access_eval.io.paraphrase_store import lexical_overlap_ratio
from access_eval.io.paraphrase_store_multijudge import make_judge_spec
from access_eval.prompts.loader import load_prompt

DEFAULT_JUDGE_PROMPT = {"SimpleQA": "v1", "TruthfulQA": "v2", "gsm8k_test": "v2", "math_500_test": "v2",
                        "mmlu": "paraphrase_multijudge/strict_mcq_v2"}
POOL_COLS = ["idx", "original_question", "paraphrased_question", "para_hash", "resp_model", "gen_config_hash",
             "tip_or_tone"]
TIPS = [
    "avoid repeating key nouns verbatim if natural", "prefer different clause ordering",
    "vary syntax; try a different opener", "use synonyms for content words where safe",
    "compress slightly while keeping meaning", "expand slightly with harmless connective words",
    "switch from statement to question form (or vice versa) while preserving intent",
    "try a passive/active voice change if natural"]
TONES = [("formal", "more professional phrasing"), ("casual", "relaxed, conversational feel"),
         ("polite", "courteous tone (include 'please' if natural)"), ("confident", "decisive tone, no hedging"),
         ("neutral", "objective, even tone"), ("supportive", "reassuring, friendly tone"),
         ("inquisitive", "curious tone, optionally as a rhetorical question")]


@dataclass
class FormulationConfig:
    dataset: str
    experiment_flag: str
    target_k: int = 8
    extra_k: Optional[int] = None          # if set: add this many NEW paraphrases to each inquiry's seed count
                                           # (per-inquiry target = count before the first round + extra_k)
    generator: str = "Qwen/Qwen3.5-9B"
    judges: List[str] = field(default_factory=lambda: ["google/gemma-4-12B-it"])   # all must accept
    judge_prompt: Optional[str] = None     # None = the paper's prompt for the dataset (DEFAULT_JUDGE_PROMPT)
    nli: bool = True
    nli_threshold: float = 0.75
    nli_model: str = "facebook/bart-large-mnli"
    overlap_max: float = 0.8
    numbers_check: bool = False
    max_rounds: int = 3
    oversample: float = 2.0
    toned_share: float = 0.4               # paper: ~70% plain / ~40% toned targets
    gen_temperature: float = 1.0
    inquiries: Optional[List[int]] = None  # None = every inquiry passing the dataset's min_length
    seed: int = 0


def _clean(text: str) -> str:
    t = (text or "").split("</think>")[-1].strip()
    line = next((ln.strip() for ln in t.splitlines() if ln.strip()), "")
    if (line[:1] == line[-1:]) and line[:1] in "\"'":
        line = line[1:-1].strip()
    line = re.sub(r"^(?:[-*•]\s+|\d+\.\s+|paraphrase:\s*)", "", line, flags=re.I)
    return re.sub(r"\s+", " ", line)


def _llm_batch(model, prompts, max_tokens, temperature):
    """Run one model over a batch. Runs inside a fresh process for local models (see _run_llm)."""
    from access_eval.models.factory import get_llm, is_api_model
    llm = get_llm(model, max_tokens=max_tokens, temperature=temperature)
    llm.prepare()
    return llm.batch(prompts, **({"concurrency": 32} if is_api_model(model) else {}))


def _run_llm(model, prompts, max_tokens, temperature):
    """API / endpoint models run in-process. A local model runs in a short-lived spawned process: vLLM's engine core
    is a separate process that deleting the Python object does not stop, so the next model (e.g. a judge after the
    generator) would find the GPU still occupied; process exit is the only reliable release."""
    from access_eval.models.factory import is_api_model
    if not prompts:
        return []
    if is_api_model(model):
        return _llm_batch(model, prompts, max_tokens, temperature)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(_llm_batch, model, prompts, max_tokens, temperature).result()


def _release(llm):
    del llm
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _nli_scores(pairs, model_name):
    """Mean of P(entail | orig -> cand) and P(entail | cand -> orig) for each (orig, cand)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    ent = {v.lower(): k for k, v in model.config.id2label.items()}["entailment"]
    out = []
    with torch.inference_mode():
        for s in range(0, len(pairs), 64):
            chunk = pairs[s:s + 64]
            prem = [a for a, b in chunk] + [b for a, b in chunk]
            hyp = [b for a, b in chunk] + [a for a, b in chunk]
            enc = tok(prem, hyp, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
            p = model(**enc).logits.softmax(-1)[:, ent].float().cpu().tolist()
            n = len(chunk)
            out += [(p[i] + p[n + i]) / 2 for i in range(n)]
    del model
    _release(None)
    return out


# Verdict parsing and the number check are copies of paraphrase_multijudge.parse_verdict and
# paraphrase_multijudge_batch_math.numbers_with_words (importing those modules pulls in spaCy / scikit-learn).
def _parse_verdict(kind, raw):
    """True/False, or None when there is no usable verdict (never cached)."""
    if raw is None:
        return None
    t = raw.strip()
    if not t or t.startswith("[Error"):
        return None
    if kind == "yesno":
        return t.upper().startswith("Y")
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    try:
        d = json.loads(t)
    except Exception:
        return None
    se, avp = d.get("semantically_equivalent"), d.get("all_values_preserved")
    return bool(se and avp) if isinstance(se, bool) and isinstance(avp, bool) else None


_NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_WORDNUM_RE = re.compile(r"\b(" + "|".join(sorted(list(_TENS) + list(_ONES), key=len, reverse=True))
                         + r")(?:[- ](one|two|three|four|five|six|seven|eight|nine))?\b", re.I)


def _numbers(text):
    def norm(n):
        n = n.replace(",", "")
        try:
            return format(Decimal(n).normalize(), "f")
        except InvalidOperation:
            return n

    def sub(m):
        v = _TENS.get(m.group(1).lower(), _ONES.get(m.group(1).lower()))
        return str((v or 0) + _ONES[m.group(2).lower()] if m.group(2) else v)
    return sorted(norm(n) for n in _NUM_RE.findall(_WORDNUM_RE.sub(sub, text)))


def _numbers_ok(orig, cand):
    return _numbers(orig) == _numbers(cand)


def install_seed_pool(cfg: FormulationConfig, seed_csv) -> int:
    """Copy a seed pool (pipeline pool format) into the run's pool file, once. Returns rows installed."""
    rp = ProjectPaths().run_paths(cfg.dataset, cfg.experiment_flag)
    rp.ensure_dirs()
    if rp.paraphrases_file().exists():
        return 0
    seed = pd.read_csv(seed_csv)
    if "para_hash" not in seed:
        seed["para_hash"] = [paraphrase_content_hash(o, p) for o, p in zip(seed.original_question, seed.paraphrased_question)]
    seed = seed.reindex(columns=POOL_COLS)
    seed["resp_model"] = seed["resp_model"].fillna("seed")
    seed.to_csv(rp.paraphrases_file(), index=False)
    return len(seed)


def build_formulations(cfg: FormulationConfig) -> dict:
    rng = random.Random(cfg.seed)
    paths = ProjectPaths()
    rp = paths.run_paths(cfg.dataset, cfg.experiment_flag)
    rp.ensure_dirs()
    spec = get_dataset_spec(cfg.dataset)
    data = pd.read_csv(paths.dataset_file(spec["path"]))
    questions = {int(i): str(r[spec["question_col"]]) for i, r in data.iterrows()
                 if len(str(r[spec["question_col"]]).split()) >= spec.get("min_length", 1)}
    if cfg.inquiries is not None:
        questions = {i: questions[i] for i in cfg.inquiries if i in questions}
    options = ({int(i): r["options_block"] for i, r in data.iterrows()} if "options_block" in data.columns else None)
    jprompt = cfg.judge_prompt or DEFAULT_JUDGE_PROMPT.get(cfg.dataset, "v2")
    judges = [make_judge_spec(j, jprompt) for j in cfg.judges]
    plain_t, toned_t = load_prompt("paraphrase/plain"), load_prompt("paraphrase/toned")
    gen_hash = compute_gen_config_hash(model=cfg.generator, revision=None, temperature=cfg.gen_temperature,
                                       max_tokens=512, prompt_name="paraphrase/plain+toned",
                                       prompt_template_text=plain_t + toned_t)
    pool_path, log_path = rp.paraphrases_file(), rp.run_dir / "formulations" / "candidates.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(pool_path) if pool_path.exists() else pd.DataFrame(columns=POOL_COLS)
    summary = {"rounds": 0, "generated": 0, "accepted": 0, "rejected": {}}

    # Per-inquiry targets. With extra_k the baseline counts are stored once, so a resumed run keeps the same targets.
    targets = {i: cfg.target_k for i in questions}
    if cfg.extra_k is not None:
        base_path = log_path.parent / "baseline_counts.json"
        if base_path.exists():
            base = {int(k): v for k, v in json.loads(base_path.read_text()).items()}
        else:
            cnt = pool.groupby("idx").size() if len(pool) else pd.Series(dtype=int)
            base = {i: int(cnt.get(i, 0)) for i in questions}
            base_path.write_text(json.dumps(base))
        targets = {i: base.get(i, 0) + cfg.extra_k for i in questions}

    for rnd in range(1, cfg.max_rounds + 1):
        have = pool.groupby("idx").size() if len(pool) else pd.Series(dtype=int)
        deficits = {i: targets[i] - int(have.get(i, 0)) for i in questions}
        deficits = {i: d for i, d in deficits.items() if d > 0}
        print(f"[formulations] {cfg.dataset} round {rnd}: {len(deficits)}/{len(questions)} inquiries below target "
              f"({'seed+' + str(cfg.extra_k) if cfg.extra_k is not None else 'K=' + str(cfg.target_k)})", flush=True)
        if not deficits:
            break
        summary["rounds"] = rnd

        # 1. generate (generator on the GPU alone)
        reqs = []
        for i, d in deficits.items():
            for _ in range(max(1, math.ceil(cfg.oversample * d))):
                if rng.random() < cfg.toned_share:
                    tone, desc = rng.choice(TONES)
                    reqs.append((i, f"{tone} ({desc})", toned_t.format(q=questions[i], tone=tone, desc=desc)))
                else:
                    tip = rng.choice(TIPS)
                    reqs.append((i, tip, plain_t.format(q=questions[i]) + "\n\n[META -- instruction to you only, "
                                 f"do NOT include this line or its content anywhere in your output]\nDiversity tip: {tip}\n"))
        outs = _run_llm(cfg.generator, [p for _, _, p in reqs], 512, cfg.gen_temperature)
        summary["generated"] += len(reqs)

        # 2. diversity / dedup (CPU)
        log, cands = [], []
        texts = {i: [questions[i]] + pool.loc[pool.idx == i, "paraphrased_question"].tolist() for i in deficits}
        for (i, label, _), raw in zip(reqs, outs):
            c = _clean(raw)
            row = {"round": rnd, "idx": i, "candidate": c, "tip_or_tone": label}
            if not c or any(lexical_overlap_ratio(c, t) >= cfg.overlap_max for t in texts[i]):
                log.append({**row, "outcome": "overlap_or_empty"})
                continue
            texts[i].append(c)
            cands.append(row)

        # 3. judge chain (one judge on the GPU at a time; cached verdicts reused)
        for spec_j in judges:
            if not cands:
                break
            def key(r, spec_j=spec_j):
                return (paraphrase_content_hash(questions[r["idx"]], r["candidate"]),
                        spec_j.stored_hash(options[r["idx"]] if spec_j.needs_options else None))
            cache = load_judge_cache(rp)
            sub = cache[cache["judge_model"] == spec_j.model] if len(cache) else cache
            known = {(h, str(ph)): bool(v) for h, ph, v in zip(sub["para_hash"], sub["prompt_hash"], sub["llm_equiv"])}
            todo = [r for r in cands if key(r) not in known]
            if todo:
                prompts = [spec_j.template.format(orig=questions[r["idx"]], cand=r["candidate"],
                                                  options=options[r["idx"]] if spec_j.needs_options else "")
                           for r in todo]
                raws = _run_llm(spec_j.model, prompts, 256 if spec_j.kind == "json2" else 8, 0.0)
                new = []
                for r, raw in zip(todo, raws):
                    v = _parse_verdict(spec_j.kind, raw)
                    if v is None:          # unusable output: not cached; counts as a rejection this round
                        continue
                    h, ph = key(r)
                    known[(h, ph)] = v
                    new.append({"para_hash": h, "original_question": questions[r["idx"]],
                                "paraphrased_question": r["candidate"], "judge_model": spec_j.model,
                                "prompt_name": spec_j.prompt_name, "prompt_hash": ph, "llm_raw": raw, "llm_equiv": v})
                if new:
                    append_judge_verdicts(rp, new)
            keep = []
            for r in cands:
                if known.get(key(r)) is True:
                    keep.append(r)
                else:
                    log.append({**r, "outcome": f"judge_reject:{spec_j.label}"})
            cands = keep

        # 4. NLI, 5. optional number check
        if cands and cfg.nli:
            scores = _nli_scores([(questions[r["idx"]], r["candidate"]) for r in cands], cfg.nli_model)
            keep = []
            for r, sc in zip(cands, scores):
                if sc >= cfg.nli_threshold:
                    keep.append(r)
                else:
                    log.append({**r, "outcome": f"nli_reject:{sc:.2f}"})
            cands = keep
        if cands and cfg.numbers_check:
            keep = []
            for r in cands:
                if _numbers_ok(questions[r["idx"]], r["candidate"]):
                    keep.append(r)
                else:
                    log.append({**r, "outcome": "numbers_changed"})
            cands = keep

        # accept up to the deficit per inquiry, in generation order
        taken: Dict[int, int] = {}
        new_rows = []
        for r in cands:
            if taken.get(r["idx"], 0) >= deficits[r["idx"]]:
                log.append({**r, "outcome": "surplus"})
                continue
            taken[r["idx"]] = taken.get(r["idx"], 0) + 1
            log.append({**r, "outcome": "accepted"})
            new_rows.append({"idx": r["idx"], "original_question": questions[r["idx"]],
                             "paraphrased_question": r["candidate"],
                             "para_hash": paraphrase_content_hash(questions[r["idx"]], r["candidate"]),
                             "resp_model": cfg.generator, "gen_config_hash": gen_hash, "tip_or_tone": r["tip_or_tone"]})
        if new_rows:
            pool = pd.concat([pool, pd.DataFrame(new_rows)], ignore_index=True)
            pool.reindex(columns=POOL_COLS).to_csv(pool_path, index=False)
        lg = pd.DataFrame(log)
        lg.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
        summary["accepted"] += len(new_rows)
        for o in lg.get("outcome", pd.Series(dtype=str)).str.split(":").str[0]:
            summary["rejected"][o] = summary["rejected"].get(o, 0) + (o != "accepted")
        print(f"[formulations] round {rnd}: {len(reqs)} generated, {len(new_rows)} accepted")

    have = pool.groupby("idx").size() if len(pool) else pd.Series(dtype=int)
    summary.update({"inquiries": len(questions), "complete": int(sum(int(have.get(i, 0)) >= targets[i] for i in questions)),
                    "pool_rows": int(len(pool)), "pool_file": str(pool_path), "candidates_log": str(log_path)})
    return summary
