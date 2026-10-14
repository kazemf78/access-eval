# src/access_eval/pipeline/evaluation.py
"""
Resumable evaluation pipeline.
Follows the same resume pattern as answer_generation.py.
"""

import os
import time
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
import gc
import multiprocessing

# # Fix for vLLM CUDA multiprocessing issue
# # Must be done before any CUDA initialization
# try:
#     multiprocessing.set_start_method('spawn', force=True)
# except RuntimeError:
#     pass  # Already set

from access_eval.core.paths import ProjectPaths
from access_eval.datasets.registry import get_dataset_spec
from access_eval.evaluators import get_evaluator
from access_eval.data.schema import Question, Answer, df_to_answers, evaluations_to_df
from access_eval.io.artifacts import save_pipeline_config
from access_eval.io.identity import compute_gen_config_hash, formulation_hashes, paraphrase_content_hash
from access_eval.io.compat import normalize_generation_df, normalize_grading_df
from access_eval.core.reconcile import reconcile_keyed, EvalTarget
from access_eval.models.factory import is_api_model

from dotenv import load_dotenv
load_dotenv()


def _free_gpu_memory():
    """Release cached GPU memory after a local model; a no-op without torch (API / endpoint-only installs)."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def default_evaluator_kwargs(dataset: str, tolerance: float = 1e-6) -> Dict:
    """Evaluator kwargs the CLI passes per dataset type (they are part of eval_config_hash, so programmatic callers
    use this to land in the same grade cache)."""
    return {"tolerance": tolerance} if get_dataset_spec(dataset)["task_type"] == "math" else {}


def evaluator_config_hash(evaluator, judge_model: str, temperature: float, evaluator_name: Optional[str] = None,
                          evaluator_kwargs: Optional[Dict] = None) -> str:
    """The eval_config_hash grade rows of one (judge, evaluator, prompt) configuration carry. Hashes the evaluator's
    actual resolved prompt text (not just its name), same rationale as gen_config_hash: a silent prompt-file edit
    under an unchanged evaluator_name must produce a new bucket, not be aliased under the old one. Falls back to
    evaluator.name for evaluator types that don't expose a single `prompt_template` attribute (e.g. ensembles)."""
    return compute_gen_config_hash(
        model=judge_model,
        revision=None,
        temperature=temperature,
        max_tokens=None,
        prompt_name=evaluator_name or evaluator.name,
        prompt_template_text=getattr(evaluator, "prompt_template", None) or evaluator.name,
        **(evaluator_kwargs or {}),
    )


def _resolve_resume_state(
    answers_df: pd.DataFrame,
    evaluations_so_far: List[dict],
    eval_config_hash: str,
    by_idx: bool = False,
):
    """
    Content-key resume + judge-verdict cache split, shared by run_evaluation's
    per-judge loop and _run_custom_evaluator (previously two ~35-line copies
    of the identical logic).

    `answers_df` must already have `formulation_hash`/`repeat_id` columns
    (also `model`/`gen_config_hash`, always present on a real answers_df).
    `evaluations_so_far` must already be normalized (via normalize_grading_df)
    so every row has `repeat_id`/`gen_config_hash`/`eval_config_hash`;
    `formulation_hash` is recomputed here from original_question/
    paraphrased_question, not read as a stored column.

    Two DIFFERENT keys are used, deliberately:
      - row-identity key (formulation_hash, answer, model, gen_config_hash,
        repeat_id, eval_config_hash) decides which answers_df rows still need
        their own materialized output row. `model`/`gen_config_hash` are
        required, not just `answer`/`repeat_id` -- the combined answers file
        spans every
        answer-generating model and every gen_config_hash that's ever been
        run, and two DIFFERENT models (or the same model under two different
        configs) can legitimately produce byte-identical answer text for the
        same question/repeat. Without them in the key, one model's/config's
        already-materialized row would wrongly look like it already covers a
        different model's/config's row for the same text, and that other
        row would never get created.
      - judge-verdict cache key (formulation_hash, answer, eval_config_hash)
        -- no model, no gen_config_hash, no repeat_id -- decides whether an
        actual judge call is needed. Identical answer TEXT only needs judging
        once, REGARDLESS of which model/config/repeat produced it; a
        duplicate's row is materialized by copying the cached verdict
        instead of re-judging -- but see cache_resolved_rows below, its
        identity/provenance fields still come from the row being
        materialized, not from the cache entry.

    Returns (missing_df, cache_resolved_rows, needs_judging_df, to_judge_df,
    text_to_rows):
      - missing_df: answers_df rows without their own row yet this run.
      - cache_resolved_rows: ready-to-append row dicts for missing_df rows
        whose TEXT was already judged (different repeat_id, or a previous
        run) -- copied from the cache, no new judge call needed.
      - needs_judging_df: the rest of missing_df -- TEXT genuinely never
        judged under eval_config_hash.
      - to_judge_df: needs_judging_df deduplicated by (formulation_hash,
        answer) -- what to actually send to evaluate_batch().
      - text_to_rows: {(formulation_hash, answer): [row_label, ...]} over
        needs_judging_df, so one judge call can materialize every row sharing
        that text.
    """
    # Thin wrapper around core.reconcile.reconcile_keyed -- see that module's
    # docstring for the general row-identity-vs-compute-key rationale this
    # function was the original motivating example for. `EvalTarget.row_key`/
    # `.compute_key` reproduce exactly the two key shapes described above.
    target = answers_df.assign(eval_config_hash=eval_config_hash)

    if evaluations_so_far:
        existing = pd.DataFrame(evaluations_so_far)
        # `evaluations_so_far` rows (loaded from a real grades CSV) carry
        # original_question/paraphrased_question (verbose_storage columns),
        # never a stored formulation_hash column -- recomputed fresh here,
        # same as answers_df's own formulation_hash, so both sides of the
        # comparison use the same content-derived identity. (Requires
        # verbose_storage=True -- a pre-existing, documented limitation of
        # this resume path, not new here.)
        existing["formulation_hash"] = formulation_hashes(
            existing["original_question"], existing["paraphrased_question"], existing["idx"] if by_idx else None)
    else:
        existing = pd.DataFrame(columns=[
            "formulation_hash", "answer", "model", "gen_config_hash", "repeat_id", "eval_config_hash",
        ])

    result = reconcile_keyed(target, existing, EvalTarget.row_key, EvalTarget.compute_key)

    missing_df = result.missing.drop(columns=["eval_config_hash"])

    cache_resolved_rows = []
    for label, cached in result.resolved_from_cache.items():
        row = dict(cached)
        row["answer_idx"] = int(label)
        # The cached verdict may have been computed for a DIFFERENT answer
        # row that happened to produce identical text (different model,
        # different gen_config_hash, or a different repeat) -- the verdict
        # itself is correctly reusable (compute_key is text-only), but every
        # identity/provenance field describing WHICH row this is must come
        # from the TARGET row being materialized, never blindly copied from
        # whichever row the cache entry happened to be built from.
        row["repeat_id"] = int(target.loc[label, "repeat_id"])
        row["model"] = target.loc[label, "model"]
        row["gen_config_hash"] = target.loc[label, "gen_config_hash"]
        cache_resolved_rows.append(row)

    needs_judging_df = missing_df.drop(index=list(result.resolved_from_cache.keys()))
    to_judge_df = result.to_compute.drop(columns=["eval_config_hash"])

    # compute_key_to_targets is keyed by EvalTarget.compute_key's 3-tuple
    # (formulation_hash, answer, eval_config_hash) -- but eval_config_hash is
    # a single fixed value for this whole call, and call sites look this dict
    # up with a 2-tuple (formulation_hash, answer) text_key (see run_evaluation's
    # and _run_custom_evaluator's batch loops). Strip the constant third
    # element so the returned dict's key shape matches exactly what it always
    # has -- lossless, since eval_config_hash never varies within one call.
    text_to_rows: Dict[tuple, List] = {
        (formulation_hash, answer): labels
        for (formulation_hash, answer, _cfg), labels in result.compute_key_to_targets.items()
    }

    return missing_df, cache_resolved_rows, needs_judging_df, to_judge_df, text_to_rows


def _load_resumable_grades(final_path, partial_path, by_idx: bool = False) -> List[dict]:
    """
    Union final + partial grades, deduplicated by row identity (keep-first,
    final rows first) -- instead of "final wins outright if it exists,
    partial is only ever consulted otherwise" (the prior behavior), which
    silently discards a partial run's already-computed rows whenever a
    final also happens to exist. Concretely: run A completes with repeats
    {0} (writes final); run B asks for {0,1,2,3,4}, generates 1 and 2,
    writes partial={0,1,2}, then crashes. Without this merge, resuming
    reads final={0} only and regenerates 1 and 2 for nothing -- wasted judge
    calls, not a correctness bug, but exactly the kind of reusable work this
    whole reconciliation effort exists to stop discarding.

    Requires original_question/paraphrased_question to compute the dedup
    key; without them (verbose_storage=False), falls back to no dedup --
    same pre-existing limitation `_resolve_resume_state` already has.
    """
    final_rows = pd.read_csv(final_path).to_dict("records") if final_path.exists() else []
    partial_rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
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
    combined = pd.DataFrame(final_rows + partial_rows)
    if "original_question" in combined.columns and "paraphrased_question" in combined.columns:
        combined["formulation_hash"] = formulation_hashes(
            combined["original_question"], combined["paraphrased_question"], combined["idx"] if by_idx else None)
        combined = normalize_grading_df(combined)
        dedup_cols = [
            c for c in ["formulation_hash", "model", "gen_config_hash", "repeat_id", "eval_config_hash"]
            if c in combined.columns
        ]
        combined = combined.drop_duplicates(subset=dedup_cols, keep="first").drop(columns=["formulation_hash"])
    return combined.to_dict("records")


def _run_custom_evaluator(
    evaluator_file: str,
    judge_model: str,
    answers_df,
    questions: dict,
    rp,
    question_subset: str,
    resume: bool,
    force: bool,
    batch_size: int,
    api_conc: int,
    save_every_n_chunks: int,
    verbose_storage: bool,
    by_idx: bool = False,
) -> None:
    """Run a user-supplied FunctionEvaluator (from evaluator_file) as a single evaluation pass."""
    from access_eval.evaluators.function_evaluator import load_from_file, dry_run
    from access_eval.evaluators.utils import set_judge

    evaluator = load_from_file(evaluator_file)
    set_judge(judge_model)

    final_path   = rp.grades_file(subset=question_subset, judge_model=evaluator.name)
    partial_path = rp.grades_partial_file(subset=question_subset, judge_model=evaluator.name)

    # Hash the evaluator FILE'S ACTUAL SOURCE CONTENT (FunctionEvaluator has no
    # single prompt_template attribute -- its "prompt" is code), so editing the
    # function's logic under an unchanged evaluator_file path is correctly
    # treated as a new config bucket, not silently aliased under the old one.
    with open(evaluator_file, "r") as f:
        evaluator_source = f.read()
    eval_config_hash = compute_gen_config_hash(
        model=judge_model,
        revision=None,
        temperature=None,
        max_tokens=None,
        prompt_name=evaluator_file,
        prompt_template_text=evaluator_source,
    )

    # --- Content-key resume (shared with run_evaluation's main loop via
    # _resolve_resume_state -- see its docstring for the row-identity vs.
    # judge-verdict-cache key rationale) ---
    evaluations_so_far: List[dict] = []
    if resume and not force:
        evaluations_so_far = _load_resumable_grades(final_path, partial_path, by_idx)
    if evaluations_so_far:
        evaluations_so_far = normalize_grading_df(pd.DataFrame(evaluations_so_far)).to_dict("records")

    missing_df, cache_resolved_rows, needs_judging_df, to_judge_df, text_to_rows = \
        _resolve_resume_state(answers_df, evaluations_so_far, eval_config_hash, by_idx)

    if missing_df.empty:
        print(f"⏩ Custom evaluator (config {eval_config_hash}) already has all "
              f"{len(answers_df)} answers graded — skipping.")
        final_df = pd.DataFrame(evaluations_so_far)
        final_df.to_csv(final_path, index=False)
        final_df.to_csv(rp.grades_all_judges_file(subset=question_subset), index=False)
        return

    if not evaluations_so_far:
        dry_run(evaluator, answers_df, questions)

    evaluations_so_far.extend(cache_resolved_rows)

    chunk_size = api_conc * 80 if is_api_model(judge_model) else batch_size
    chunk_counter = 0
    print(f"Total answers: {len(answers_df)} | needed this run: {len(missing_df)}")

    if needs_judging_df.empty:
        final_df = pd.DataFrame(evaluations_so_far)
        final_df.to_csv(final_path, index=False)
        final_df.to_csv(rp.grades_all_judges_file(subset=question_subset), index=False)
        print(f"✅ Saved {len(final_df)} total evaluations ({len(cache_resolved_rows)} new, "
              f"all from cache) → {final_path}")
        return

    for batch_start in range(0, len(to_judge_df), chunk_size):
        st = time.time()
        batch_end    = min(batch_start + chunk_size, len(to_judge_df))
        batch_df     = to_judge_df.iloc[batch_start:batch_end]
        batch_ans    = df_to_answers(batch_df)
        batch_gts    = [questions[a.question_idx] for a in batch_ans]

        batch_evals  = evaluator.evaluate_batch(batch_ans, batch_gts)
        for i, ev in enumerate(batch_evals):
            ev.answer_idx = int(batch_df.index[i])
            row_template = ev.to_dict(verbose=verbose_storage)
            row_template["eval_config_hash"] = eval_config_hash
            text_key = (batch_df["formulation_hash"].iloc[i], batch_df["answer"].iloc[i])
            for row_label in text_to_rows.get(text_key, []):
                row = dict(row_template)
                row["answer_idx"] = int(row_label)
                # `row_template` was built from ONE representative answer
                # (batch_ans[i]) but is fanned out to every target row
                # sharing its TEXT via text_to_rows -- those rows can belong
                # to a different model/gen_config_hash than the
                # representative (that's the whole point of judging text
                # once and reusing it), so every identity field must come
                # from THIS target row, not be left as the representative's.
                target_row = needs_judging_df.loc[row_label]
                row["repeat_id"] = int(target_row["repeat_id"])
                row["model"] = target_row["model"]
                row["gen_config_hash"] = target_row["gen_config_hash"]
                evaluations_so_far.append(row)

        chunk_counter += 1
        if chunk_counter % save_every_n_chunks == 0:
            pd.DataFrame(evaluations_so_far).to_csv(partial_path, index=False)
            print(f"💾 Saved progress after {chunk_counter} chunks → {partial_path}")
        print(f"Batch {batch_start}-{batch_end}: {round(time.time() - st, 2)}s | {time.ctime()}")

    final_df = pd.DataFrame(evaluations_so_far)
    final_df.to_csv(final_path, index=False)
    final_df.to_csv(rp.grades_all_judges_file(subset=question_subset), index=False)
    print(f"✅ Saved {len(final_df)} total evaluations ({len(missing_df)} new) → {final_path}")


def run_evaluation(
    dataset: str = "SimpleQA",
    experiment_flag: str = "v0",
    temperature: float = 0.0,
    question_subset: str = "both",
    judge_models: Optional[List[str]] = None,
    judge_temperature: Optional[float] = None,
    batch_size: int = 32,
    api_conc: int = 32,
    save_every_n_chunks: int = 1,
    resume: bool = True,
    force: bool = False,
    evaluator_kwargs: Optional[Dict] = None,
    verbose_storage: bool = True,
    evaluator_name: Optional[str] = None,
    evaluator_file: Optional[str] = None,
):
    """
    Run evaluation pipeline with resume capability.
    
    Args:
        dataset: Dataset name (e.g., "SimpleQA", "gsm8k_test")
        experiment_flag: Experiment identifier
        temperature: Temperature used in answer generation
        judge_temperature: Sampling temperature of LLM judges. None (default, previous behavior) = `temperature`;
            pass 0.0 to grade sampled (temperature > 0) answers with greedy judges.
        question_subset: Which subset was used ("both", "original_only", etc.)
        judge_models: List of judge models (for LLM-based evaluation)
        batch_size: Batch size for evaluation
        api_conc: API concurrency for LLM judges
        save_every_n_chunks: Save checkpoint every N batches
        resume: If True, resume from partial files
        force: If True, ignore resume and regenerate everything
        evaluator_kwargs: Additional kwargs for evaluator (e.g., tolerance for math)
        verbose_storage: If True, include original_question, paraphrased_question, answer in output
    """
    
    judge_models = judge_models or ["gpt-4.1-mini"]
    judge_t = temperature if judge_temperature is None else judge_temperature
    evaluator_kwargs = evaluator_kwargs or {}
    
    paths = ProjectPaths()
    rp = paths.run_paths(dataset, experiment_flag)
    conf = rp.conf_suffix(temperature=temperature)
    rp.ensure_dirs()
    
    dataset_cfg = get_dataset_spec(dataset)
    
    # Load answers
    answers_path = rp.answers_all_models_file(subset=question_subset, conf_suffix=conf)
    if not answers_path.exists():
        print(f"\n❌ Answers file not found: {answers_path}")
        return
    
    print(f"📖 Loading answers from: {answers_path}")
    # normalize_generation_df backfills repeat_id=0 for answers files written
    # before per-repeat generation was tracked -- needed below for the
    # row-identity key.
    answers_df = normalize_generation_df(pd.read_csv(answers_path))
    print(f"   Loaded {len(answers_df)} answers")
    # Content identity for resume, independent of idx -- computed once here
    # rather than per-judge, since answers_df doesn't change across judges.
    # Datasets with a prompt builder add per-inquiry context outside the text: the inquiry idx joins the identity.
    by_idx = bool(dataset_cfg.get("prompt_builder"))
    answers_df["formulation_hash"] = formulation_hashes(
        answers_df["original_question"], answers_df["paraphrased_question"], answers_df["idx"] if by_idx else None)
    
    # Load dataset with ground truth
    dataset_path = paths.dataset_file(dataset_cfg["path"])
    dataset_df = pd.read_csv(dataset_path)
    print(f"📖 Loading ground truth from: {dataset_path}")
    
    # Create Question objects (with ground truth)
    questions = {}
    for idx, row in dataset_df.iterrows():
        q = Question.from_series(
            row, 
            dataset=dataset,
            question_col=dataset_cfg["question_col"],
            answer_col=dataset_cfg.get("answer_col", "answer"),
        )
        questions[q.idx] = q
    
    # Save run config
    run_config = {
        "dataset": dataset,
        "experiment_flag": experiment_flag,
        "temperature": temperature,
        "judge_temperature": judge_t,
        "question_subset": question_subset,
        "judge_models": judge_models,
        "api_conc": api_conc,
        "batch_size": batch_size,
        "evaluator_kwargs": evaluator_kwargs,
        "evaluator_file": evaluator_file,
    }
    save_pipeline_config(rp.run_dir, "evaluation", run_config)

    if evaluator_file:
        _run_custom_evaluator(
            evaluator_file    = evaluator_file,
            judge_model       = (judge_models or ["gpt-4.1-mini"])[0],
            answers_df        = answers_df,
            questions         = questions,
            rp                = rp,
            question_subset   = question_subset,
            resume            = resume,
            force             = force,
            batch_size        = batch_size,
            api_conc          = api_conc,
            save_every_n_chunks = save_every_n_chunks,
            verbose_storage   = verbose_storage,
            by_idx            = by_idx,
        )
        return
    
    # Process each judge model
    all_evaluations = []
    
    for judge_model in judge_models:
        print(f"\n{'='*60}")
        print(f"🔍 Evaluating with: {judge_model}")
        print('='*60)
        
        # Determine output paths
        final_path = rp.grades_file(subset=question_subset, judge_model=judge_model)
        partial_path = rp.grades_partial_file(subset=question_subset, judge_model=judge_model)

        # Construct the evaluator early (cheap -- just prompt loading + LLM object
        # construction, no network/GPU calls; those happen in .prepare() below,
        # called only if there's actual work) so its RESOLVED prompt text is
        # available for eval_config_hash before deciding what's already done.
        evaluator = get_evaluator(
            dataset,
            evaluator_name=evaluator_name,
            judge_model=judge_model,
            temperature=judge_t,
            api_conc=api_conc,
            **evaluator_kwargs
        )
        eval_config_hash = evaluator_config_hash(evaluator, judge_model, judge_t, evaluator_name, evaluator_kwargs)

        # --- Content-key resume ---
        # `force` starts this judge from a clean slate. Otherwise, when `resume`,
        # reuse whatever's already on disk -- final AND partial, merged and
        # deduped (_load_resumable_grades), not final-only-if-it-exists, so a
        # partial run that got further than the last completed final isn't
        # silently discarded.
        #
        # Two DIFFERENT keys are used below (see _resolve_resume_state / core.reconcile.EvalTarget):
        #   - row-identity key (formulation_hash, answer, model, gen_config_hash,
        #     repeat_id, eval_config_hash) decides which answers_df rows still
        #     need their own materialized output row. model/gen_config_hash are
        #     required, not just repeat_id -- the combined answers file spans
        #     every answer-generating model/config, and two different ones can
        #     legitimately produce identical answer text for the same
        #     question/repeat; repeat_id is required too, since each stochastic
        #     repeat is a separate generation instance needing its own row.
        #   - judge-verdict cache key (formulation_hash, answer, eval_config_hash)
        #     -- no model, no gen_config_hash, no repeat_id -- decides whether
        #     an actual judge call is needed. Identical answer TEXT only needs
        #     judging once, regardless of which model/config/repeat produced
        #     it; a duplicate's row gets materialized by copying the cached
        #     verdict instead of re-judging, with its OWN identity fields
        #     (model/gen_config_hash/repeat_id), not the cache entry's.
        evaluations_so_far: List[dict] = []
        if resume and not force:
            evaluations_so_far = _load_resumable_grades(final_path, partial_path, by_idx)
        if evaluations_so_far:
            evaluations_so_far = normalize_grading_df(pd.DataFrame(evaluations_so_far)).to_dict("records")

        missing_df, cache_resolved_rows, needs_judging_df, to_judge_df, text_to_rows = \
            _resolve_resume_state(answers_df, evaluations_so_far, eval_config_hash, by_idx)

        if missing_df.empty:
            print(f"⏩ {judge_model} (config {eval_config_hash}) already has all "
                  f"{len(answers_df)} answers graded — skipping.")
            pd.DataFrame(evaluations_so_far).to_csv(final_path, index=False)
            all_evaluations.append(pd.DataFrame(evaluations_so_far))
            continue

        print(f"Total answers: {len(answers_df)} | needed this run: {len(missing_df)} "
              f"(config={eval_config_hash})")

        evaluations_so_far.extend(cache_resolved_rows)
        if cache_resolved_rows:
            print(f"   {len(cache_resolved_rows)} row(s) resolved from judge cache "
                  f"(duplicate answer text, e.g. across repeats) — no new judge calls.")

        if needs_judging_df.empty:
            final_df = pd.DataFrame(evaluations_so_far)
            final_df.to_csv(final_path, index=False)
            print(f"✅ Completed {judge_model}, saved {len(final_df)} total evaluations "
                  f"({len(cache_resolved_rows)} new, all from cache) -> {final_path}")
            all_evaluations.append(final_df)
            continue

        print(f"   {len(to_judge_df)} unique text(s) to judge (covering {len(needs_judging_df)} row(s))")

        evaluator.prepare()
        if is_api_model(judge_model):
            print(f"Using API model {judge_model} for evaluation.")
            chunk_size = api_conc * 80
        else:
            print(f"Using local model {judge_model} for evaluation.")
            chunk_size = batch_size

        # Process in batches
        chunk_counter = 0

        for batch_start in range(0, len(to_judge_df), chunk_size):
            st = time.time()
            batch_end = min(batch_start + chunk_size, len(to_judge_df))
            batch_df = to_judge_df.iloc[batch_start:batch_end]

            # Convert to Answer objects
            batch_answers = df_to_answers(batch_df)

            # Get corresponding ground truths
            batch_questions = [questions[ans.question_idx] for ans in batch_answers]

            print("before evaluate batch")
            # Evaluate batch
            batch_evaluations = evaluator.evaluate_batch(batch_answers, batch_questions)

            # Materialize one row per answers_df row sharing this judged TEXT
            # (via text_to_rows), not just the single representative row in
            # `batch_df` -- e.g. several repeat_ids with identical output text
            # all get their own row from this one judge call.
            for i, eval_obj in enumerate(batch_evaluations):
                eval_obj.answer_idx = int(batch_df.index[i])
                row_template = eval_obj.to_dict(verbose=verbose_storage)
                row_template["eval_config_hash"] = eval_config_hash
                text_key = (batch_df["formulation_hash"].iloc[i], batch_df["answer"].iloc[i])
                for row_label in text_to_rows.get(text_key, []):
                    row = dict(row_template)
                    row["answer_idx"] = int(row_label)
                    # See _run_custom_evaluator's identical comment: every
                    # identity field must come from THIS target row, not be
                    # left as whichever representative row produced
                    # `row_template` -- different rows sharing this text can
                    # belong to a different model/gen_config_hash.
                    target_row = needs_judging_df.loc[row_label]
                    row["repeat_id"] = int(target_row["repeat_id"])
                    row["model"] = target_row["model"]
                    row["gen_config_hash"] = target_row["gen_config_hash"]
                    evaluations_so_far.append(row)

            # Save checkpoint periodically
            chunk_counter += 1
            if chunk_counter % save_every_n_chunks == 0:
                pd.DataFrame(evaluations_so_far).to_csv(partial_path, index=False)
                print(f"💾 Saved progress after {chunk_counter} chunks -> {partial_path}")
                print(f"   Time: {time.ctime()}")

            elapsed = round(time.time() - st, 2)
            print(f"Batch {batch_start}-{batch_end}: {elapsed}s")

        # Save final results: union of whatever was already done + newly graded this run
        final_df = pd.DataFrame(evaluations_so_far)
        final_df.to_csv(final_path, index=False)
        print(f"✅ Completed {judge_model}, saved {len(final_df)} total evaluations "
              f"({len(missing_df)} new) -> {final_path}")

        # Cleanup
        if hasattr(evaluator, 'cleanup'):
            evaluator.cleanup()
        del evaluator
        gc.collect()
        _free_gpu_memory()

        all_evaluations.append(final_df)
        print(f"Time: {time.ctime()}")

    # --- Combine all judge results ---
    # Rebuilt by globbing every per-judge file on disk for this (run, subset),
    # not just concatenating `all_evaluations` from THIS invocation's judge_models
    # -- otherwise a judge graded in an earlier run but omitted from this run's
    # judge_models list would silently disappear from the rebuilt combined file.
    # Mirrors filter_artifacts.py's _rebuild_all_judges_file.
    judge_files = [
        p for p in rp.grades_dir.glob(f"graded_{rp.run_id}_{question_subset}_*.csv")
        if "ALL_judges" not in p.name
    ]
    if judge_files:
        combined_path = rp.grades_all_judges_file(subset=question_subset)
        combined_df = pd.concat([pd.read_csv(p) for p in judge_files], ignore_index=True)
        combined_df.to_csv(combined_path, index=False)
        print(f"\n🏁 Saved combined evaluations ({len(combined_df)} rows, "
              f"{len(judge_files)} judge files) -> {combined_path}")

    print(f"\n✅ Evaluation complete! {time.ctime()}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",")] if s else []


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    
    parser.add_argument("--dataset", default="SimpleQA")
    parser.add_argument("--experiment-flag", default="v0")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--judge-temperature", type=float, default=None,
                        help="LLM-judge temperature (default: same as --temperature, the previous behavior)")
    parser.add_argument("--question-subset", default="both",
                       choices=["original_only", "paraphrased_only", "both"])
    
    parser.add_argument("--judge-models", type=csv_list,
                       default="gpt-4.1-mini,gpt-3.5-turbo")
    
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--api-conc", type=int, default=32)
    parser.add_argument("--save-every-n-chunks", type=int, default=5)
    
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                       help="Disable resume mode (default: resume is enabled)")
    parser.add_argument("--force", action="store_true",
                       help="Force regenerate even if outputs exist")
    
    # Storage options
    parser.add_argument("--no-verbose-storage", dest="verbose_storage", action="store_false",
                       help="Don't include questions/answers in output (saves space)")
    
    # Math-specific options
    parser.add_argument("--tolerance", type=float, default=1e-6,
                       help="Tolerance for math evaluation")
    
    parser.set_defaults(resume=True, verbose_storage=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    evaluator_kwargs = default_evaluator_kwargs(args.dataset, tolerance=args.tolerance)
    
    run_evaluation(
        dataset=args.dataset,
        experiment_flag=args.experiment_flag,
        temperature=args.temperature,
        judge_temperature=args.judge_temperature,
        question_subset=args.question_subset,
        judge_models=args.judge_models,
        batch_size=args.batch_size,
        api_conc=args.api_conc,
        save_every_n_chunks=args.save_every_n_chunks,
        resume=args.resume,
        force=args.force,
        evaluator_kwargs=evaluator_kwargs,
        verbose_storage=args.verbose_storage,
    )