# src/access_eval/core/paths.py

from pathlib import Path
import os
import re
from datetime import datetime


class RunPaths:
    def __init__(self, root: Path, dataset: str, experiment_flag: str):
        self.root = root
        self.dataset = dataset
        self.experiment_flag = experiment_flag
        self.run_id = f"{dataset}_{experiment_flag}"
        self.run_dir = root / self.run_id

    # =========================================================================
    # Core directories
    # =========================================================================

    @property
    def paraphrases_dir(self) -> Path:
        return self.run_dir / "paraphrases"

    @property
    def answers_dir(self) -> Path:
        return self.run_dir / "answers"

    @property
    def answer_partials_dir(self) -> Path:
        return self.answers_dir / "partial"

    @property
    def alignment_dir(self) -> Path:
        return self.run_dir / "alignment"

    @property
    def evaluation_dir(self) -> Path:
        return self.run_dir / "evaluation"

    @property
    def grades_dir(self) -> Path:
        return self.evaluation_dir / "grades"

    @property
    def grades_partials_dir(self) -> Path:
        return self.grades_dir / "partial"

    @property
    def grades_cache_dir(self) -> Path:
        return self.grades_dir / "cache"

    # =========================================================================
    # Filtered artifact directories (versioned subdirs — originals untouched)
    # =========================================================================

    def filtered_answers_dir(self, version_tag: str) -> Path:
        """answers/refiltered_v1/"""
        return self.answers_dir / version_tag

    def filtered_grades_dir(self, version_tag: str) -> Path:
        """evaluation/grades/refiltered_v1/"""
        return self.grades_dir / version_tag

    # =========================================================================
    # Helpers
    # =========================================================================

    def _safe_model(self, model: str) -> str:
        """
        Sanitize a model-name string for use as a filename fragment.
        `/` has always been replaced (HF-style "org/model" ids). `:` is
        replaced too -- found by the controlled-contamination pilot's
        `[lora:<fp>]` suffix (item 1/2, controlled_contamination_plan.md
        v4): some cluster filesystems rejects colons
        in filenames outright (`OSError: [Errno 22] Invalid argument`),
        confirmed directly, not just theoretically -- so any model name
        colons must be sanitized before ever reaching a path, not just `/`.
        Every existing model name (no colons) is unaffected; this is
        purely additive.
        """
        return model.replace("/", "_").replace(":", "_")

    def _tag_suffix(self, tag: str | None) -> str:
        return f"_{tag}" if tag else ""

    def conf_suffix(self, *, temperature: float | None = None) -> str:
        parts = []
        if temperature is not None:
            temp_str = str(temperature).replace(".", "")
            parts.append(f"temperature{temp_str}")
        parts.append(self.experiment_flag)
        return "_".join(parts)

    def latest_paraphrase_checkpoint(self) -> tuple[Path | None, int]:
        """Return (path, max_idx) for the latest paraphrase checkpoint."""
        checkpoints = []
        raw_checkpoints = sorted(
            self.paraphrases_dir.glob(f"{self.dataset}_paraphrases_expanded_*.csv")
        )
        for p in raw_checkpoints:
            m = re.search(r"_(\d+)\.csv$", p.name)
            if m:
                checkpoints.append((p, int(m.group(1))))
        if not checkpoints:
            return None, 0
        return max(checkpoints, key=lambda x: x[1])

    # =========================================================================
    # Original canonical files
    # =========================================================================

    def paraphrases_checkpoint_file(self, upto: int) -> Path:
        return self.paraphrases_dir / f"{self.dataset}_paraphrases_expanded_{upto}.csv"

    def paraphrases_file(self) -> Path:
        return self.paraphrases_dir / f"{self.dataset}_paraphrases_expanded.csv"

    def answers_partial_file(self, subset: str, model: str, conf_suffix: str) -> Path:
        return self.answer_partials_dir / (
            f"{self.dataset}_answers_{subset}_{self._safe_model(model)}_{conf_suffix}.partial.csv"
        )

    def answers_file(self, subset: str, model: str, conf_suffix: str) -> Path:
        return self.answers_dir / (
            f"{self.dataset}_answers_{subset}_{self._safe_model(model)}_{conf_suffix}.csv"
        )

    def answers_all_models_file(self, subset: str, conf_suffix: str) -> Path:
        return self.answers_dir / f"{self.dataset}_answers_{subset}_ALL_models_{conf_suffix}.csv"

    def grades_partial_file(self, subset: str, judge_model: str) -> Path:
        return self.grades_partials_dir / (
            f"graded_{self.run_id}_{subset}_{self._safe_model(judge_model)}.partial.csv"
        )

    def grades_file(self, subset: str, judge_model: str) -> Path:
        return self.grades_dir / (
            f"graded_{self.run_id}_{subset}_{self._safe_model(judge_model)}.csv"
        )

    def grades_all_judges_file(self, subset: str) -> Path:
        return self.grades_dir / f"graded_{self.run_id}_{subset}_ALL_judges.csv"

    def alignment_file(self, subset: str = "both") -> Path:
        """Path to the aligned_verdicts pickle for a given question subset."""
        return self.alignment_dir / f"aligned_verdicts_{self.run_id}_{subset}.pkl"

    def preprocessed_grades_cache_file(self, subset: str, collapse_not_attempted: bool = False) -> Path:
        """
        Cache path for analysis.profile_comparison.preprocess_grades()'s output
        -- the expensive per-(model,idx,paraphrased_question) majority-vote
        aggregation. Pass this to preprocess_grades(cache_path=...) to skip
        recomputing it when the source grades CSV hasn't changed.
        """
        suffix = "_collapsed" if collapse_not_attempted else ""
        return self.grades_cache_dir / f"preprocessed_{self.run_id}_{subset}{suffix}.pkl"

    def paired_profiles_cache_file(
        self,
        subset: str,
        model_a: str,
        model_b: str,
        min_agreement_fraction: float = 0.5,
        collapse_not_attempted: bool = False,
    ) -> Path:
        """
        Cache path for analysis.profile_comparison.build_paired_profiles()'s
        output for one model pair. Pass to build_paired_profiles(cache_path=...)
        to skip recomputing it -- validity is tied to the preprocess_grades()
        source fingerprint (via agg.attrs) plus model_a/model_b/
        min_agreement_fraction, so it's namespaced here on all the same axes.
        """
        maf_str = str(min_agreement_fraction).replace(".", "")
        suffix = "_collapsed" if collapse_not_attempted else ""
        return self.grades_cache_dir / (
            f"paired_profiles_{self.run_id}_{subset}"
            f"_{self._safe_model(model_a)}_{self._safe_model(model_b)}"
            f"_maf{maf_str}{suffix}.pkl"
        )

    def single_profile_cache_file(
        self,
        subset: str,
        model: str,
        min_agreement_fraction: float = 0.5,
        collapse_not_attempted: bool = False,
    ) -> Path:
        """
        Cache path for analysis.profile_comparison.build_single_profile()'s
        output for one model. Pass to build_single_profile(cache_path=...) to
        skip recomputing it -- validity is tied to the preprocess_grades()/
        preprocess_math_answers() source fingerprint (via agg.attrs) plus
        model/min_agreement_fraction, same namespacing as
        paired_profiles_cache_file, just for one model instead of a pair.
        """
        maf_str = str(min_agreement_fraction).replace(".", "")
        suffix = "_collapsed" if collapse_not_attempted else ""
        return self.grades_cache_dir / (
            f"single_profile_{self.run_id}_{subset}"
            f"_{self._safe_model(model)}"
            f"_maf{maf_str}{suffix}.pkl"
        )

    def common_profiles_cache_file(
        self,
        subset: str,
        models: list,
        min_agreement_fraction: float = 0.5,
        collapse_not_attempted: bool = False,
    ) -> Path:
        """
        Cache path for analysis.common_profiles.build_common_profiles()'s
        output -- the N-way (2 or more) generalization of
        paired_profiles_cache_file(). `models` can be real model names, or
        any tagged pseudo-model string (e.g. "{model}__cot", "{model}__GEPA")
        -- same convention build_paired_profiles()/load_qa_condition_pair()
        already use for a before/after condition pair.

        A set can be arbitrarily long (a whole Qwen3 scaling family, say),
        so the filename is keyed on a short hash of the sorted tags rather
        than concatenating every one of them -- order doesn't matter for a
        cache hit (sorted before hashing), only membership. A short
        human-readable prefix (first up to 3 tags, safe-named) rides
        alongside the hash purely so `ls` on the cache dir stays legible;
        it plays no role in cache validity, which is entirely the hash +
        the fingerprint build_common_profiles() checks internally.
        """
        import hashlib

        tags = sorted(models)
        tag_hash = hashlib.md5("|".join(tags).encode()).hexdigest()[:12]
        prefix = "_".join(self._safe_model(m) for m in tags[:3])
        if len(tags) > 3:
            prefix += f"_and_{len(tags) - 3}_more"
        maf_str = str(min_agreement_fraction).replace(".", "")
        suffix = "_collapsed" if collapse_not_attempted else ""
        return self.grades_cache_dir / (
            f"common_profiles_{self.run_id}_{subset}_{prefix}_{tag_hash}"
            f"_maf{maf_str}{suffix}.pkl"
        )

    # =========================================================================
    # Refilter paraphrases outputs
    # (new files in paraphrases_dir — originals untouched)
    # =========================================================================

    def paraphrases_filtered_file(
        self, judge_model: str, prompt_name: str, tag: str | None = None
    ) -> Path:
        """
        Clean filtered paraphrases — feed this into filter_artifacts.py.
        e.g. SimpleQA_paraphrases_expanded_filtered_gpt-4.1_detailed.csv
        """
        return self.paraphrases_dir / (
            f"{self.dataset}_paraphrases_expanded"
            f"_filtered_{self._safe_model(judge_model)}_{prompt_name}"
            f"{self._tag_suffix(tag)}.csv"
        )

    def paraphrases_full_judged_file(
        self, judge_model: str, prompt_name: str, tag: str | None = None
    ) -> Path:
        """
        Every row with llm_raw, llm_equiv, judgment_source — for auditing.
        e.g. SimpleQA_paraphrases_expanded_full_judged_gpt-4.1_detailed.csv
        """
        return self.paraphrases_dir / (
            f"{self.dataset}_paraphrases_expanded"
            f"_full_judged_{self._safe_model(judge_model)}_{prompt_name}"
            f"{self._tag_suffix(tag)}.csv"
        )

    def paraphrases_removed_file(
        self, judge_model: str, prompt_name: str, tag: str | None = None
    ) -> Path:
        """
        Only the rows that were filtered out — for prompt iteration / error analysis.
        e.g. SimpleQA_paraphrases_expanded_removed_gpt-4.1_detailed.csv
        """
        return self.paraphrases_dir / (
            f"{self.dataset}_paraphrases_expanded"
            f"_removed_{self._safe_model(judge_model)}_{prompt_name}"
            f"{self._tag_suffix(tag)}.csv"
        )

    def paraphrase_judge_cache_file(self) -> Path:
        """
        One row per (paraphrase, judge, prompt) verdict ever computed for this
        dataset/experiment, regardless of which refilter_paraphrases.py run
        produced it. refilter_paraphrases.py checks this before calling an LLM
        judge, so switching to a stronger judge only re-judges pairs it hasn't
        seen before, and re-running the same (judge, prompt) after adding new
        paraphrases only judges the new rows -- never touches originals, only
        ever grows.
        """
        return self.paraphrases_dir / f"{self.dataset}_paraphrase_judgecache.csv"

    # =========================================================================
    # Filtered artifact accessors
    # (mirrors original accessors — same args, routes into versioned subdir)
    # =========================================================================

    def filtered_answers_file(
        self, version_tag: str, subset: str, model: str, conf_suffix: str
    ) -> Path:
        """Mirrors answers_file() but inside filtered_answers_dir(version_tag)."""
        return self.filtered_answers_dir(version_tag) / (
            f"{self.dataset}_answers_{subset}_{self._safe_model(model)}_{conf_suffix}.csv"
        )

    def filtered_answers_all_models_file(
        self, version_tag: str, subset: str, conf_suffix: str
    ) -> Path:
        """Mirrors answers_all_models_file() but inside filtered_answers_dir(version_tag)."""
        return self.filtered_answers_dir(version_tag) / (
            f"{self.dataset}_answers_{subset}_ALL_models_{conf_suffix}.csv"
        )

    def filtered_grades_file(
        self, version_tag: str, subset: str, judge_model: str
    ) -> Path:
        """Mirrors grades_file() but inside filtered_grades_dir(version_tag)."""
        return self.filtered_grades_dir(version_tag) / (
            f"graded_{self.run_id}_{subset}_{self._safe_model(judge_model)}.csv"
        )

    def filtered_grades_all_judges_file(
        self, version_tag: str, subset: str
    ) -> Path:
        """Mirrors grades_all_judges_file() but inside filtered_grades_dir(version_tag)."""
        return self.filtered_grades_dir(version_tag) / (
            f"graded_{self.run_id}_{subset}_ALL_judges.csv"
        )

    def filter_diff_file(self, version_tag: str) -> Path:
        """Diff report at run_dir level — before/after row counts per file."""
        return self.run_dir / f"filter_diff_{version_tag}.csv"

    # =========================================================================
    # Controlled-contamination pilot
    # (controlled_contamination_plan.md v4, items 2/5)
    # =========================================================================

    @property
    def contamination_dir(self) -> Path:
        return self.run_dir / "contamination"

    @property
    def contamination_adapters_dir(self) -> Path:
        return self.contamination_dir / "adapters"

    def adapter_dir(self, fingerprint: str) -> Path:
        """
        Where one trained LoRA adapter's weights + metadata.json live.
        Only ever called by fine_tune.py (which decides where to save) and
        the contamination orchestrator (which builds the `adapter_paths`
        argument passed into run_answer_generation) -- both have
        dataset/experiment context. factory.py never calls this; it only
        ever receives an already-resolved path as an explicit argument
        (item 2, v4 -- see controlled_contamination_plan.md).
        """
        return self.contamination_adapters_dir / fingerprint

    def contamination_assignments_file(self, flag: str) -> Path:
        """idx, group, cluster_id, assignment_seed, source_pool_fingerprint."""
        return self.contamination_dir / f"assignments_{flag}.csv"

    def contamination_train_examples_file(self, flag: str) -> Path:
        """idx, group, text_used_for_training, target_text, formulation_hash
        -- exact- and paraphrase-group inquiries only."""
        return self.contamination_dir / f"train_examples_{flag}.csv"

    def contamination_eval_formulations_file(self, flag: str) -> Path:
        """idx, role, text, formulation_hash -- role is one of canonical,
        eval_paraphrase, reserved_unused, train_formulation_diagnostic."""
        return self.contamination_dir / f"eval_formulations_{flag}.csv"

    def contamination_provenance_file(self, flag: str) -> Path:
        """Materialized snapshot of the exact source paraphrase-pool rows
        used for selection, plus the filtering config that produced them --
        see 'Provenance, not just a hash' in item 5."""
        return self.contamination_dir / f"provenance_{flag}.json"

    def ensure_contamination_dirs(self):
        """Create the contamination-pilot output directories. Mirrors
        ensure_dirs()/ensure_filtered_dirs() -- never touches other dirs."""
        self.contamination_dir.mkdir(parents=True, exist_ok=True)
        self.contamination_adapters_dir.mkdir(exist_ok=True)

    # =========================================================================
    # Analysis directories
    # =========================================================================

    @property
    def analysis_dir(self) -> Path:
        return self.run_dir / "analysis"

    @property
    def analysis_figs_dir(self) -> Path:
        return self.analysis_dir / "figs"

    def analysis_csv(self, name: str) -> Path:
        """e.g. rp.analysis_csv('agg_dist') → analysis/agg_dist.csv"""
        return self.analysis_dir / f"{name}.csv"

    def ensure_analysis_dirs(self):
        """Create analysis output directories."""
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_figs_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Directory creation
    # =========================================================================

    def ensure_dirs(self):
        """Create all standard pipeline directories."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.paraphrases_dir.mkdir(exist_ok=True)
        self.answers_dir.mkdir(exist_ok=True)
        self.answer_partials_dir.mkdir(exist_ok=True)
        self.evaluation_dir.mkdir(exist_ok=True)
        self.grades_dir.mkdir(exist_ok=True)
        self.grades_partials_dir.mkdir(exist_ok=True)
        self.grades_cache_dir.mkdir(exist_ok=True)
        self.alignment_dir.mkdir(exist_ok=True)

    def ensure_filtered_dirs(self, version_tag: str):
        """
        Create versioned subdirectories for filtered artifacts.
        Call this before writing any filtered outputs.
        Mirrors ensure_dirs() — never touches original directories.
        """
        self.filtered_answers_dir(version_tag).mkdir(parents=True, exist_ok=True)
        self.filtered_grades_dir(version_tag).mkdir(parents=True, exist_ok=True)


PROJECT_ROOT_ENV = "ACCESS_EVAL_ROOT"


def project_root_dir() -> Path:
    """Root holding datasets/ and run_artifacts/. $ACCESS_EVAL_ROOT when set (a separate workspace, e.g. one per
    evaluation-card run, so an installed package never writes into its own source tree); otherwise the
    repository checkout, found by its pyproject.toml anchor."""
    env = os.environ.get(PROJECT_ROOT_ENV)
    if env:
        return Path(env).resolve()
    cur = Path(__file__).resolve()
    while cur != cur.parent:
        if (cur / "pyproject.toml").exists():
            return cur
        cur = cur.parent
    raise RuntimeError(f"Could not locate project root (no pyproject.toml found; set ${PROJECT_ROOT_ENV})")


class ProjectPaths:
    """
    Centralized path resolver.
    Works even if code is moved inside src/.
    """

    def __init__(self, project_root: Path | None = None):
        self.root = project_root if project_root is not None else project_root_dir()

        self.datasets = self.root / "datasets"
        self.run_artifacts = self.root / "run_artifacts"
        self.run_artifacts.mkdir(exist_ok=True)

    def run_paths(self, dataset: str, experiment_flag: str) -> RunPaths:
        return RunPaths(self.run_artifacts, dataset, experiment_flag)

    def dataset_file(self, rel_path: str) -> Path:
        return self.datasets / rel_path

    def new_run_dir(self, run_id: str | None = None) -> Path:
        """
        Legacy / ad-hoc run directory creation.
        Prefer run_paths(dataset, experiment_flag) for structured runs.
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.run_artifacts / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir