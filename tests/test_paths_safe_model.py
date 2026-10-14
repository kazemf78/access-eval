"""
Regression test for RunPaths._safe_model()'s colon-sanitization
(controlled_contamination_plan.md v4): confirmed directly that some cluster filesystems rejects colons in filenames
outright (`OSError: [Errno 22] Invalid argument`), which broke the first
real post-FT evaluation run the moment a `[lora:<fp>]`-suffixed model name
(containing a literal `:`) reached `answers_file()`.
"""

from access_eval.core.paths import ProjectPaths


def _rp(tmp_path):
    return ProjectPaths(project_root=tmp_path).run_paths("gsm8k_test", "unit_test")


class TestSafeModelSanitization:
    def test_colon_is_replaced(self, tmp_path):
        rp = _rp(tmp_path)
        assert ":" not in rp._safe_model("Qwen/Qwen3-8B[hf][lora:adp_0123456789abcdef]")

    def test_slash_still_replaced(self, tmp_path):
        rp = _rp(tmp_path)
        assert "/" not in rp._safe_model("Qwen/Qwen3-8B")

    def test_plain_model_names_unaffected(self, tmp_path):
        """Every existing model name (no colon) must sanitize identically
        to before -- this fix must not change behavior for any currently
        stored file."""
        rp = _rp(tmp_path)
        assert rp._safe_model("gpt-4.1") == "gpt-4.1"
        assert rp._safe_model("gpt-4.1-mini") == "gpt-4.1-mini"

    def test_answers_file_path_actually_writable_with_lora_suffix(self, tmp_path):
        """Sanity check that the derived path is usable end-to-end. Note:
        pytest's tmp_path typically lives on a filesystem that tolerates
        colons (e.g. /tmp), so this alone would NOT have caught the
        original bug -- that's what test_colon_is_replaced (a direct,
        filesystem-independent string assertion) is for."""
        rp = _rp(tmp_path)
        rp.ensure_dirs()
        model_name = "Qwen/Qwen3-8B[hf][lora:adp_0123456789abcdef]"
        path = rp.answers_partial_file(subset="both", model=model_name, conf_suffix="temperature00_unit_test")
        path.write_text("idx,answer\n0,4\n")  # must not raise OSError
        assert path.read_text() == "idx,answer\n0,4\n"

    def test_pre_and_post_ft_model_names_produce_distinct_paths(self, tmp_path):
        rp = _rp(tmp_path)
        pre = rp.answers_file(subset="both", model="Qwen/Qwen3-8B[hf]", conf_suffix="c")
        post = rp.answers_file(subset="both", model="Qwen/Qwen3-8B[hf][lora:adp_0123456789abcdef]", conf_suffix="c")
        assert pre != post
