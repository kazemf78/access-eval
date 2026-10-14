import os
import argparse
import pandas as pd
from typing import List, Optional, Tuple, Union
import time
import gc
from dotenv import load_dotenv
# from access_eval.models.hf_local import HFLocalLLM
from access_eval.models.openai_api import OpenAIAPILLM
from access_eval.core.paths import ProjectPaths
from access_eval.prompts.builders import get_prompt_builder
from access_eval.io.artifacts import save_pipeline_config
from access_eval.io.identity import compute_gen_config_hash, formulation_hashes, paraphrase_content_hash
from access_eval.io.compat import normalize_generation_df
from access_eval.io.paraphrase_store import cross_validated_paraphrases
from access_eval.core.reconcile import reconcile_keyed, AnswerTarget
from access_eval.datasets.registry import get_dataset_spec
from access_eval.prompts import load_prompt
from access_eval.models.factory import get_llm_from_list, is_api_model


import signal


def _free_gpu_memory():
    """Release cached GPU memory after a local model; a no-op without torch (API / endpoint-only installs)."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _print_interrupt(sig_name):
    print(f"[INTERRUPT] {sig_name} received at {time.ctime()}", flush=True)

def _handle_sigint(sig, frame):
    _print_interrupt("SIGINT")   # Ctrl+C
    raise KeyboardInterrupt

def _handle_sigterm(sig, frame):
    _print_interrupt("SIGTERM")  # Slurm timeout, scancel, etc.
    raise SystemExit

signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigterm)

load_dotenv()


# ---------------------------- Utility Functions ----------------------------
def build_both_expanded(df: pd.DataFrame):
    groups = []

    for idx, group in df.groupby("idx"):
        original = group["original_question"].iloc[0]

        # synthetic original-as-paraphrase row
        df_orig = pd.DataFrame({
            "idx": [idx],
            "original_question": [original],
            "paraphrased_question": [original],
        })

        # real paraphrases
        df_para = group[["idx", "original_question", "paraphrased_question"]]

        # combine original FIRST, then its paraphrases
        expanded = pd.concat([df_orig, df_para], ignore_index=True)
        groups.append(expanded)

    # stack all groups preserving order
    return pd.concat(groups, ignore_index=True)

def csv_list(s: str):
    return [x.strip() for x in s.split(",")] if s else []


def chunks(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield i, lst[i:i + n]


def _load_resumable_answers(final_path, partial_path, by_idx: bool = False) -> List[dict]:
    """
    Union final + partial answer rows, deduplicated by (formulation_hash,
    repeat_id, gen_config_hash) keep-first (final rows first) -- instead of
    "final wins outright if it exists, partial only otherwise" (the prior
    behavior), which silently discards a partial run's already-generated
    rows whenever a final also happens to exist. E.g.: run A completes with
    repeats {0} (writes final); run B asks for {0..4}, generates 1 and 2,
    writes partial={0,1,2}, then crashes -- without this merge, resuming
    reads final={0} only and regenerates 1 and 2 for nothing. Mirrors
    evaluation.py's _load_resumable_grades.
    """
    final_rows = pd.read_csv(final_path).to_dict("records") if os.path.exists(final_path) else []
    partial_rows = pd.read_csv(partial_path).to_dict("records") if os.path.exists(partial_path) else []
    if not final_rows and not partial_rows:
        return []
    if not partial_rows:
        return final_rows
    if not final_rows:
        print(f"🔁 Resuming from partial results: {partial_path}")
        return partial_rows

    print(f"🔁 Found both final and partial results -- merging "
          f"({len(final_rows)} final + {len(partial_rows)} partial, deduped) "
          f"rather than discarding the partial's progress: {partial_path}")
    combined = normalize_generation_df(pd.DataFrame(final_rows + partial_rows))
    combined["formulation_hash"] = formulation_hashes(
        combined["original_question"], combined["paraphrased_question"], combined["idx"] if by_idx else None)
    combined = combined.drop_duplicates(
        subset=["formulation_hash", "repeat_id", "gen_config_hash"], keep="first"
    ).drop(columns=["formulation_hash"])
    return combined.to_dict("records")

# todo: remove duplication with factory.py?
def resolve_answer_prompt(dataset: str, answer_prompt_name: Optional[str] = None):
    """(prompt_name, template_text, builder) for a dataset's answers. Datasets whose prompt needs more than the
    formulation text (fixed answer options, few-shot demos) name a prompt builder in their registry spec; its
    identity strings then stand in for the template in gen_config_hash."""
    dataset_cfg = get_dataset_spec(dataset)
    builder = get_prompt_builder(dataset_cfg.get("prompt_builder"))
    if builder is not None:
        if answer_prompt_name not in (None, builder.prompt_name):
            raise ValueError(f"{dataset} builds its prompts with {builder.prompt_name!r}; "
                             f"--answer-prompt-name {answer_prompt_name!r} does not apply")
        return builder.prompt_name, builder.template_text, builder
    name = answer_prompt_name or dataset_cfg["task_type"]
    return name, load_prompt(f"answer/{name}"), None


def answer_config_hash(model_name: str, temperature: float, max_tokens: int, prompt_name: str, template_text: str) -> str:
    """The gen_config_hash answer rows of one generation configuration carry (the cache identity)."""
    return compute_gen_config_hash(
        model=model_name,
        revision=None,  # threaded through once model-checkpoint support lands
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_name=prompt_name,
        prompt_template_text=template_text,
    )


def get_llm(model_name: str, api_models: List[str], max_local_tokens=256, max_local_time=300, max_api_tokens=256):
    SPECIAL_THINKING_SUFFIX = "[with_thinking]"
    if model_name in api_models:
        llm = OpenAIAPILLM(model=model_name.removesuffix(SPECIAL_THINKING_SUFFIX), max_output_tokens=max_api_tokens)
    else:
        # llm = HFLocalLLM(model_id=model_name.removesuffix(SPECIAL_THINKING_SUFFIX), max_new_tokens=max_local_tokens, do_sample=True, max_time=max_local_time)
        from access_eval.models.vllm_local import VLLMLocalLLM   # vLLM only needed for local models
        llm = VLLMLocalLLM(model_id=model_name.removesuffix(SPECIAL_THINKING_SUFFIX), max_tokens=max_local_tokens, gpu_memory_utilization=0.85, 
                        #    enforce_eager=True,
                        #    enable_flash_attn=False,
                        #    flash_attn_version=2,
                        #    max_model_len=2048,
                        #    NOTE: adjust other vLLM params as needed if necessary (e.g., for larger models or different hardware setups)
                           )
    llm.prepare()

    if model_name.lower().endswith(SPECIAL_THINKING_SUFFIX):
        llm.enable_thinking = True # todo: handle openai api usage with thinking!
    else:
        llm.enable_thinking = False
    return llm


# ---------------------------- Main Function ----------------------------
def run_answer_generation(
    dataset: str = "SimpleQA",
    temperature: float = 0.0,
    batch_size: int = 32,
    api_conc: int = 32,
    save_every_n_chunks: int = 1,
    experiment_flag: str = "v0",
    api_models: Optional[List[str]] = None,
    local_models: Optional[List[str]] = None,
    resume: bool = True,
    force: bool = False,
    question_subset: str = "both",
    max_local_tokens: int = 256,
    max_local_time: int = 300,
    max_api_tokens: int = 256,
    answer_prompt_name: str = None,
    tensor_parallel_size: int = None,
    gpu_memory_utilization: float = 0.85,
    repeat_ids: Tuple[int, ...] = (0,),
    paraphrase_judge_model: Optional[Union[str, List[str]]] = None,
    paraphrase_judge_prompt_name: Optional[str] = None,
    paraphrase_resp_model: Optional[str] = None,
    paraphrase_overlap_max: float = 0.8,
    paraphrase_require_different_generation_model: bool = True,
    adapter_paths: Optional[dict] = None,
    prefer_vllm_for_lora: bool = False,
):
    """
    Run the answer generation pipeline for multiple models.

    Args:
        resume (bool): If True, reuse already-generated rows (matched by content
            key, not row position) instead of regenerating them.
        force (bool): If True, ignore all existing final/partial content for a
            model and regenerate it from scratch this run.
        repeat_ids (Tuple[int, ...]): The full set of repeats this call wants to
            exist for every (formulation, model, gen_config). Raising this on a
            later call (e.g. (0,1,2) -> (0,1,2,3,4)) computes and generates only
            the deficit against whatever repeat_ids are already on disk --
            resume no longer requires the caller to know which specific
            repeat_id values are missing, or to loop externally one at a time.
        paraphrase_judge_model: If set, don't answer over the raw
            paraphrases_file() pool -- restrict to paraphrases independently
            validated by this judge_model (or ANY of them, if a list), via
            io.paraphrase_store.cross_validated_paraphrases(): llm_equiv==True
            under (judge_model, paraphrase_judge_prompt_name) in the shared
            judge-verdict cache AND generated under a DIFFERENT judge (self-
            approval at generation time doesn't count), then re-verified with
            the same lexical-overlap dedup paraphrase generation applies.
            Purely a cache lookup -- never calls an LLM to judge a candidate
            with no verdict yet under this (judge_model, prompt_name); such
            candidates are simply excluded, not judged on demand. None (the
            default) keeps the old behavior: every row in paraphrases_file(),
            unfiltered.
        paraphrase_judge_prompt_name: The judge prompt paired with
            paraphrase_judge_model. Optional even when paraphrase_judge_model
            is set -- None means "accepted by this judge_model under ANY
            prompt it's ever been judged with" (see accepted_paraphrases's
            docstring), rather than pinning to one specific prompt version.
        paraphrase_resp_model: If also set, further restricts the selected
            pool to paraphrases whose `resp_model` (the model that generated
            the candidate text, as opposed to `gen_judge_model` -- the one
            that accepted it into the pool, which can differ, e.g. a
            paraphrase_topup.py-generated row) equals this value.
        paraphrase_overlap_max: overlap_max passed through to
            cross_validated_paraphrases()'s dedup_by_overlap pass. Only
            meaningful when paraphrase_judge_model is set.
        paraphrase_require_different_generation_model: When True (default), a
            paraphrase accepted by paraphrase_judge_model is only kept if
            that same model did NOT also write the candidate text (resp_model
            != judge_model) -- a judge validating its own writing isn't
            independent validation. Set False to drop that check and include
            everything paraphrase_judge_model currently accepts, even
            paraphrases it wrote itself. See cross_validated_paraphrases's
            docstring. Only meaningful when paraphrase_judge_model is set.
        adapter_paths: Optional {model_name: adapter_directory_path} dict for
            the controlled-contamination pilot (controlled_contamination_plan.md
            v4, items 2/3). `model_name` here must be a full entry of
            `api_models`/`local_models` that carries a `[lora:<fp>]` suffix
            (e.g. "Qwen/Qwen3-8B[hf][lora:adp_...]"); its path is forwarded
            to `get_llm_from_list()`'s `adapter_path` kwarg for that model
            only. None (the default) preserves today's exact behavior for
            every other call site -- this dict is never consulted otherwise,
            and gen_config_hash/file paths/CSV columns are unaffected since
            the fingerprint already lives in `model_name` itself (item 3).

    Resume is content-keyed on (formulation_hash, repeat_id, gen_config_hash),
    via core.reconcile.reconcile_keyed against an explicitly materialized
    target set (AnswerTarget.expand), rather than row position OR idx -- idx is
    a dataset row position, stable for this project's static datasets but not
    guaranteed for unpinned HF-auto-downloaded ones, so it's deliberately not
    part of the key. `formulation_hash` (io.identity.paraphrase_content_hash)
    is computed fresh from (original_question, paraphrased_question) text on
    both sides of the comparison, so it needs no stored column and works on
    legacy artifacts unchanged. `gen_config_hash` is derived from (model,
    temperature, max_tokens, answer_prompt_name, and the resolved prompt
    template text) via io.identity.compute_gen_config_hash -- changing any of
    those is a distinct config, generated and stored alongside prior configs
    in the same file rather than silently colliding with or overwriting them
    (existing rows under a DIFFERENT gen_config_hash simply never match a
    target row's key, so they can never make that target look satisfied).
    """

    api_models = api_models
    local_models = local_models


    paths = ProjectPaths()
    rp = paths.run_paths(dataset, experiment_flag)
    conf = rp.conf_suffix(temperature=temperature)
    rp.ensure_dirs()
    dataset_cfg = get_dataset_spec(dataset)

    paraphrases_path = rp.paraphrases_file()
    if not paraphrases_path.exists():
        print(f"\nExiting... Paraphrases file not found at: {paraphrases_path}")
        return

    if paraphrase_judge_model is None:
        df = pd.read_csv(paraphrases_path)
        print(f'Reading from path: {paraphrases_path}')
    else:
        df = cross_validated_paraphrases(
            rp, paraphrase_judge_model, paraphrase_judge_prompt_name,
            overlap_max=paraphrase_overlap_max,
            require_different_generation_model=paraphrase_require_different_generation_model,
        )
        if paraphrase_resp_model is not None:
            df = df[df["resp_model"] == paraphrase_resp_model].reset_index(drop=True)
        print(
            f"Selected {len(df)} paraphrase(s) accepted by "
            f"judge_model={paraphrase_judge_model!r}/prompt={paraphrase_judge_prompt_name!r} "
            f"(require_different_generation_model={paraphrase_require_different_generation_model})"
            + (f", resp_model={paraphrase_resp_model!r}" if paraphrase_resp_model else "")
            + " -- cache-only (no live judging); see "
              "io.paraphrase_store.cross_validated_paraphrases."
        )
    print(f"Dataframe shape at the start: {df.shape}")

    answer_prompt_name, ANSWER_PROMPT, builder = resolve_answer_prompt(dataset, answer_prompt_name)
    if builder is not None:
        dataset_rows = pd.read_csv(paths.dataset_file(dataset_cfg["path"]))

    run_config = {
        "dataset": dataset,
        "question_subset": question_subset,
        "temperature": temperature,
        "experiment_flag": experiment_flag,
        "api_models": api_models,
        "local_models": local_models,
        "max_local_tokens": max_local_tokens,
        "max_api_tokens": max_api_tokens,
        "answer_prompt": answer_prompt_name,
        "paraphrase_judge_model": paraphrase_judge_model,
        "paraphrase_judge_prompt_name": paraphrase_judge_prompt_name,
        "paraphrase_resp_model": paraphrase_resp_model,
        "paraphrase_overlap_max": paraphrase_overlap_max,
        "paraphrase_require_different_generation_model": paraphrase_require_different_generation_model,
        # Recorded because BOTH the raw pool (paraphrases_file()) and, more so,
        # cross_validated_paraphrases() are LIVE views over data that only
        # ever grows (the paraphrase pool, and the judge cache it's filtered
        # against) -- not a fixed function of the parameters above over time.
        # The same call can legitimately select more rows if repeated later.
        # This count -- together with save_pipeline_config's own created_at
        # + timestamped history sidecar below -- is what lets a future reader
        # notice "this run actually used N rows on this date", rather than
        # silently assuming a later re-run selects the identical set.
        "paraphrase_pool_size": len(df),
    }

    save_pipeline_config(rp.run_dir, "answer_generation", run_config)


    if question_subset == "original_only":
        df = df[["idx", "original_question"]].drop_duplicates("idx")
        df["paraphrased_question"] = df["original_question"]
        subset_tag = question_subset
        

    elif question_subset == "both":
        df = build_both_expanded(df)
        subset_tag = "both"

    else:
        subset_tag = "paraphrased_only"
    print(f"Dataframe shape after considering question_subset={question_subset}: {df.shape}")
    idxs = df["idx"].tolist()
    origs = df["original_question"].tolist()
    paras = df["paraphrased_question"].tolist()
    # Content identity for resume, independent of idx (a dataset row position --
    # stable for static datasets, not guaranteed for unpinned HF-auto-downloaded
    # ones). Computed from (original_question, paraphrased_question) text, which
    # every answers row already carries regardless of whether the source
    # paraphrases file has a stored formulation_hash column.
    # Datasets with a prompt builder add per-inquiry context outside the text, so the inquiry idx joins the identity.
    by_idx = builder is not None
    formulation_hash_list = formulation_hashes(origs, paras, idxs if by_idx else None)

    for model_name in api_models + local_models:
        print(f"\n=== Processing {model_name} ===")

        final_path = rp.answers_file(subset=subset_tag, model=model_name, conf_suffix=conf)
        partial_path = rp.answers_partial_file(subset=subset_tag, model=model_name, conf_suffix=conf)

        max_tokens_for_model = max_api_tokens if is_api_model(model_name) else max_local_tokens
        gen_config_hash = answer_config_hash(model_name, temperature, max_tokens_for_model,
                                             answer_prompt_name, ANSWER_PROMPT)

        # --- Content-key resume via core.reconcile.reconcile_keyed ---
        # `force` always starts this model from a clean slate (matches its
        # documented meaning unambiguously). Otherwise, when `resume`, reuse
        # whatever's already on disk (final takes precedence over partial) as
        # `existing`, and diff it against the FULL target set -- every
        # (formulation_hash, repeat_id) pair this call wants, for every
        # requested repeat_id in `repeat_ids` -- not a row-count offset into a
        # freshly rebuilt prompt list (misaligns if the paraphrases file's
        # shape changes between runs) and not idx (fragile under dataset
        # drift). The target is exactly enumerable in advance, so this is a
        # keyed-existence reconciliation, not a count/quota one -- it
        # correctly identifies e.g. "repeat_id=1 is the one missing" even when
        # repeats {0,2,3} already exist against a requested {0,1,2,3}.
        existing_rows: List[dict] = []
        if resume and not force:
            existing_rows = _load_resumable_answers(final_path, partial_path, by_idx)

        if existing_rows:
            existing_df = normalize_generation_df(pd.DataFrame(existing_rows))
            existing_rows = existing_df.to_dict("records")
            existing_df["formulation_hash"] = formulation_hashes(
                existing_df["original_question"], existing_df["paraphrased_question"],
                existing_df["idx"] if by_idx else None)
        else:
            existing_df = pd.DataFrame(columns=["formulation_hash", "repeat_id", "gen_config_hash"])

        # formulation_hash uniquely identifies a row here (same assumption the
        # prior single-key done_keys check already made), so target rows can
        # be mapped back to their (idx, paraphrased_question) via this lookup.
        hash_to_i = {h: i for i, h in enumerate(formulation_hash_list)}

        target = AnswerTarget(
            model=model_name, gen_config_hash=gen_config_hash, repeat_ids=tuple(repeat_ids),
        ).expand(formulation_hash_list)
        result = reconcile_keyed(target, existing_df, AnswerTarget.row_key)

        model_rows: List[dict] = list(existing_rows)
        needed = [
            (hash_to_i[row["formulation_hash"]],
             idxs[hash_to_i[row["formulation_hash"]]],
             paras[hash_to_i[row["formulation_hash"]]],
             int(row["repeat_id"]))
            for _, row in result.missing.iterrows()
        ]

        if not needed:
            print(f"⏩ {model_name} already has all {len(paras)} rows for "
                  f"repeat_ids={tuple(repeat_ids)}, config={gen_config_hash} — skipping.")
            # Explicit columns: model_rows can be genuinely empty here (e.g. an
            # empty/filtered paraphrase pool -> paras=[] -> target=[] -> nothing
            # needed and nothing pre-existing). pd.DataFrame([]) with no columns
            # writes a headerless file that a later pd.read_csv (the
            # ALL_models combine step) can't parse.
            pd.DataFrame(model_rows, columns=[
                "model", "idx", "original_question", "paraphrased_question",
                "answer", "repeat_id", "gen_config_hash",
            ]).to_csv(final_path, index=False)
            continue

        print(f"Total questions: {len(paras)} x {len(repeat_ids)} repeat(s) | needed this run: "
              f"{len(needed)} (repeat_ids={tuple(repeat_ids)}, config={gen_config_hash})")

        # llm = get_llm(model_name, api_models, max_local_tokens=max_local_tokens, max_local_time=max_local_time, max_api_tokens=max_api_tokens)
        # Controlled-contamination pilot (v4, item 3): forwarded only for
        # models named in adapter_paths -- every other call site passes
        # adapter_paths=None and this is a no-op, unchanged from before.
        adapter_path = (adapter_paths or {}).get(model_name)
        adapter_kwargs = {"adapter_path": adapter_path} if adapter_path else {}
        # enforce_eager=True for local models only, same fix and same reason
        # as pipeline/paraphrase.py's _get_llm_scoped: CUDA graph capture
        # needs extra host-side staging memory per engine on top of the
        # weights themselves, which is exactly what this loop's large local
        # models (27B/32B, tensor_parallel_size>1) can exceed a job's
        # host-RAM cgroup over -- confirmed directly: job 7484688 (TruthfulQA
        # paper2_cot_prompt) was OOM-killed (MaxRSS 134GB vs a 128G cgroup)
        # mid-load on OLMo-2-0325-32B-Instruct, and job 7484687 (SimpleQA,
        # same run) stalled for 12+ hours at 0% GPU utilization loading
        # gemma-3-27b-it -- both right at/after a "Capturing CUDA graphs"
        # log line. Meaningless for API models, so gated the same way
        # _get_llm_scoped gates its own local-only kwargs.
        eager_kwargs = {} if is_api_model(model_name) else {"enforce_eager": True}
        llm = get_llm_from_list(model_name, api_models, max_local_tokens=max_local_tokens, max_api_tokens=max_api_tokens, gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size, prefer_vllm_for_lora=prefer_vllm_for_lora, **adapter_kwargs, **eager_kwargs)
        llm.prepare()

        if builder is not None:
            needed_prompts = [builder.build(para, dataset_rows.iloc[idx_val]) for (_, idx_val, para, _) in needed]
        else:
            needed_prompts = [ANSWER_PROMPT.format(q=para) for (_, _, para, _) in needed]
        if is_api_model(model_name):
            print(f"Using API model {model_name} for answering.")
            chunk_size = api_conc * 50
        else:
            print(f"Using local model {model_name} for answering.")
            chunk_size = batch_size

        chunk_counter = 0
        for start, batch_prompts in chunks(needed_prompts, chunk_size):
            st = time.time()
            if isinstance(llm, OpenAIAPILLM):
                answers = llm.batch(batch_prompts, concurrency=api_conc, temperature=temperature)
            else:
                answers = llm.batch(batch_prompts, temperature=temperature)

            for j, ans in enumerate(answers):
                i, idx_val, para_val, this_repeat_id = needed[start + j]
                model_rows.append({
                    "model": model_name,
                    "idx": int(idx_val),
                    "original_question": origs[i],
                    "paraphrased_question": para_val,
                    "answer": (ans or "").strip(),
                    "repeat_id": this_repeat_id,
                    "gen_config_hash": gen_config_hash,
                })

            chunk_counter += 1
            if chunk_counter % save_every_n_chunks == 0:
                pd.DataFrame(model_rows).to_csv(partial_path, index=False)
                print(f"💾 Saved progress after {chunk_counter} chunks -> {partial_path}")
                print(time.ctime())
            print("Batch time:", round(time.time() - st, 2), "sec")
            print(time.ctime())

        # save final: union of whatever was already done + newly generated this run
        pd.DataFrame(model_rows).to_csv(final_path, index=False)
        print(f"✅ Completed {model_name}, saved {len(model_rows)} total answers "
              f"({len(needed)} new) -> {final_path}")

        # cleanup
        del llm
        gc.collect()
        _free_gpu_memory()

        print(f"Time: {time.ctime()}, added {len(needed)} new rows for model={model_name}")

    # --- Final combined output ---
    # Rebuilt by globbing every per-model file on disk for this (dataset,
    # subset, conf), not just concatenating rows from THIS invocation's
    # api_models/local_models -- otherwise a model answered in an earlier run
    # but omitted from this run's model lists would silently disappear from
    # the rebuilt combined file. Mirrors evaluation.py's grades_all_judges_file
    # rebuild (same bug, same fix, other side of the pipeline).
    all_path = rp.answers_all_models_file(subset=subset_tag, conf_suffix=conf)
    model_files = [
        p for p in rp.answers_dir.glob(f"{dataset}_answers_{subset_tag}_*_{conf}.csv")
        if p.name != all_path.name
    ]
    if model_files:
        combined_df = pd.concat([pd.read_csv(p) for p in model_files], ignore_index=True)
        combined_df.to_csv(all_path, index=False)
        print(f"🏁 Saved combined results ({len(combined_df)} rows, "
              f"{len(model_files)} model files) -> {all_path}")
    print(time.ctime())


# ---------------------------- CLI Entry Point ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SimpleQA")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--api-conc", type=int, default=32)
    parser.add_argument("--save-every-n-chunks", type=int, default=5)
    parser.add_argument("--experiment-flag", type=str, default="v0")
    parser.add_argument("--question-subset", type=str, default="both", choices=["original_only", "paraphrased_only", "both"]) # todo: maybe None option to handle previous versions?
    parser.add_argument("--max-local-tokens", type=int, default=256)
    parser.add_argument("--max-local-time", type=int, default=300)
    parser.add_argument("--max-api-tokens", type=int, default=256)
    parser.add_argument("--answer-prompt-name", type=str, default=None)
    parser.add_argument("--api-models", type=csv_list,
                        default="gpt-4o,gpt-4.1,gpt-4.1-mini,gpt-3.5-turbo")
    parser.add_argument("--local-models", type=csv_list,
                        default="meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct")

    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume mode (default: resume is enabled)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate even if outputs exist",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="Max GPU memory utilization for local models (e.g., 0.85 for 85%)")
    parser.add_argument("--repeat-ids", type=lambda s: tuple(int(x) for x in csv_list(s)),
                        default=(0,), help="Comma-separated repeat_id values to ensure exist, e.g. 0,1,2,3,4")
    parser.add_argument("--paraphrase-judge-model", type=csv_list, default=None,
                        help="Restrict to paraphrases cross-validated (see "
                             "io.paraphrase_store.cross_validated_paraphrases) by this judge_model, "
                             "or ANY of a comma-separated list. Default: unfiltered paraphrases_file() pool.")
    parser.add_argument("--paraphrase-judge-prompt-name", type=str, default=None,
                        help="Judge prompt name paired with --paraphrase-judge-model. Optional -- "
                             "omit to match this judge_model under ANY prompt it's been judged with.")
    parser.add_argument("--paraphrase-resp-model", type=str, default=None,
                        help="Further restrict the selected pool to this paraphrase-generation resp_model.")
    parser.add_argument("--paraphrase-overlap-max", type=float, default=0.8,
                        help="overlap_max for cross_validated_paraphrases()'s dedup pass.")
    parser.add_argument(
        "--paraphrase-allow-self-generated",
        dest="paraphrase_require_different_generation_model", action="store_false",
        help="Also include paraphrases that --paraphrase-judge-model itself wrote (resp_model == "
             "judge_model), not just ones it judged. Default: require the paraphrase be judged by "
             "this model but NOT generated by it (resp_model != judge_model).",
    )
    parser.set_defaults(resume=True, paraphrase_require_different_generation_model=True)  # ✅ Resume by default
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_answer_generation(
        dataset=args.dataset,
        temperature=args.temperature,
        batch_size=args.batch_size,
        api_conc=args.api_conc,
        save_every_n_chunks=args.save_every_n_chunks,
        experiment_flag=args.experiment_flag,
        api_models=args.api_models,
        local_models=args.local_models,
        resume=args.resume,
        force=args.force,
        question_subset=args.question_subset,
        max_local_tokens=args.max_local_tokens,
        max_local_time=args.max_local_time,
        max_api_tokens=args.max_api_tokens,
        answer_prompt_name=args.answer_prompt_name,
        repeat_ids=args.repeat_ids,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        paraphrase_judge_model=args.paraphrase_judge_model,
        paraphrase_judge_prompt_name=args.paraphrase_judge_prompt_name,
        paraphrase_resp_model=args.paraphrase_resp_model,
        paraphrase_overlap_max=args.paraphrase_overlap_max,
        paraphrase_require_different_generation_model=args.paraphrase_require_different_generation_model,
    )
