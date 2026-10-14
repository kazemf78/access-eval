"""
Unit tests for the controlled-contamination pilot's [hf]/[lora:<fp>] suffix
parsing and adapter-path plumbing in src/access_eval/models/factory.py
(controlled_contamination_plan.md v4, items 1 and 2).

No GPU/model loading: HFLocalLLM is monkeypatched with a lightweight stub
that records the kwargs it was constructed with, so these tests only
exercise factory.py's own parsing/validation/branch-selection logic.
"""

import importlib.util
import json

import pytest

# these import models.vllm_local (and so vllm) before stubbing the class
requires_vllm = pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason="vllm not installed")

from access_eval.models.factory import (
    ADAPTER_FINGERPRINT_RE,
    get_llm,
    get_llm_from_list,
    parse_adapter_suffixes,
    _verify_adapter_metadata,
)

VALID_FP = "adp_0123456789abcdef"


# ---------------------------------------------------------------------------
# parse_adapter_suffixes
# ---------------------------------------------------------------------------

class TestParseAdapterSuffixes:
    def test_plain_model_name(self):
        clean, info = parse_adapter_suffixes("Qwen/Qwen3-8B")
        assert clean == "Qwen/Qwen3-8B"
        assert info == {
            "enable_thinking": False, "revision": None,
            "force_hf": False, "adapter_fingerprint": None,
        }

    def test_hf_suffix_alone(self):
        clean, info = parse_adapter_suffixes("Qwen/Qwen3-8B[hf]")
        assert clean == "Qwen/Qwen3-8B"
        assert info["force_hf"] is True
        assert info["adapter_fingerprint"] is None

    def test_hf_and_lora_suffix(self):
        clean, info = parse_adapter_suffixes(f"Qwen/Qwen3-8B[hf][lora:{VALID_FP}]")
        assert clean == "Qwen/Qwen3-8B"
        assert info["force_hf"] is True
        assert info["adapter_fingerprint"] == VALID_FP

    def test_order_of_stacking_does_not_matter(self):
        """Both stacking orders parse identically -- brackets are stripped
        from the end regardless of which recognized tag is at the end."""
        a = parse_adapter_suffixes(f"Qwen/Qwen3-8B[hf][lora:{VALID_FP}]")
        b = parse_adapter_suffixes(f"Qwen/Qwen3-8B[lora:{VALID_FP}][hf]")
        assert a == b

    def test_with_thinking_and_revision_still_work(self):
        clean, info = parse_adapter_suffixes(
            f"meta-llama/Llama-3.1-8B-Instruct@refs/pr/3[hf][lora:{VALID_FP}][with_thinking]"
        )
        assert clean == "meta-llama/Llama-3.1-8B-Instruct"
        assert info["revision"] == "refs/pr/3"
        assert info["enable_thinking"] is True
        assert info["force_hf"] is True
        assert info["adapter_fingerprint"] == VALID_FP

    def test_with_thinking_only_unaffected(self):
        clean, info = parse_adapter_suffixes("gpt-4o[with_thinking]")
        assert clean == "gpt-4o"
        assert info["enable_thinking"] is True
        assert info["force_hf"] is False
        assert info["adapter_fingerprint"] is None

    def test_unrecognized_suffix_raises(self):
        with pytest.raises(ValueError, match="Unrecognized bracket suffix"):
            parse_adapter_suffixes("Qwen/Qwen3-8B[laura:oops]")

    def test_duplicated_hf_suffix_raises(self):
        with pytest.raises(ValueError, match="Duplicated"):
            parse_adapter_suffixes("Qwen/Qwen3-8B[hf][hf]")

    def test_duplicated_lora_suffix_raises(self):
        with pytest.raises(ValueError, match="Duplicated"):
            parse_adapter_suffixes(f"Qwen/Qwen3-8B[lora:{VALID_FP}][lora:{VALID_FP}]")

    def test_duplicated_with_thinking_suffix_raises(self):
        with pytest.raises(ValueError, match="Duplicated"):
            parse_adapter_suffixes("gpt-4o[with_thinking][with_thinking]")

    def test_hf_and_lora_stacked_underneath_with_thinking(self):
        """Regression: [with_thinking] being outermost must not stop
        parsing before [hf]/[lora:<fp>] stacked underneath it are also
        stripped (a prior implementation stopped at the first recognized
        suffix instead of looping until none remain)."""
        clean, info = parse_adapter_suffixes(
            f"meta-llama/Llama-3.1-8B-Instruct@refs/pr/3[hf][lora:{VALID_FP}][with_thinking]"
        )
        assert clean == "meta-llama/Llama-3.1-8B-Instruct"
        assert info["revision"] == "refs/pr/3"
        assert info["enable_thinking"] is True
        assert info["force_hf"] is True
        assert info["adapter_fingerprint"] == VALID_FP

    def test_malformed_lora_missing_fingerprint_raises(self):
        with pytest.raises(ValueError, match="Malformed \\[lora:...\\]"):
            parse_adapter_suffixes("Qwen/Qwen3-8B[lora:]")

    def test_malformed_lora_bad_fingerprint_format_raises(self):
        with pytest.raises(ValueError, match="Malformed \\[lora:...\\]"):
            parse_adapter_suffixes("Qwen/Qwen3-8B[lora:not-a-real-fingerprint]")

    def test_malformed_hf_payload_raises(self):
        with pytest.raises(ValueError, match="Malformed suffix"):
            parse_adapter_suffixes("Qwen/Qwen3-8B[hf:extra]")

    def test_unterminated_lora_suffix_raises(self):
        """Review found: a missing closing ']' (e.g. a truncated/corrupted
        suffix) never matched the trailing-bracket pattern at all, so the
        loop exited immediately and the malformed fragment silently became
        part of the literal model id instead of being rejected."""
        with pytest.raises(ValueError, match="Unterminated or malformed"):
            parse_adapter_suffixes("Qwen/Qwen3-8B[lora:broken")

    def test_unterminated_hf_suffix_raises(self):
        with pytest.raises(ValueError, match="Unterminated or malformed"):
            parse_adapter_suffixes("Qwen/Qwen3-8B[hf")

    def test_stray_bracket_after_valid_suffix_raises(self):
        with pytest.raises(ValueError, match="Unterminated or malformed"):
            parse_adapter_suffixes(f"Qwen/Qwen3-8B[hf][lora:{VALID_FP}][oops")

    def test_fingerprint_regex_matches_intended_format(self):
        assert ADAPTER_FINGERPRINT_RE.match(VALID_FP)
        assert not ADAPTER_FINGERPRINT_RE.match("adp_short")
        assert not ADAPTER_FINGERPRINT_RE.match("wrong_prefix_0123456789abcdef")


# ---------------------------------------------------------------------------
# _verify_adapter_metadata
# ---------------------------------------------------------------------------

class TestVerifyAdapterMetadata:
    def _write_metadata(self, tmp_path, **fields):
        d = tmp_path / "adapter"
        d.mkdir()
        with open(d / "metadata.json", "w") as f:
            json.dump(fields, f)
        return str(d)

    def test_matching_metadata_passes(self, tmp_path):
        path = self._write_metadata(
            tmp_path, fingerprint=VALID_FP, base_model="Qwen/Qwen3-8B", revision=None,
        )
        metadata = _verify_adapter_metadata(path, VALID_FP, "Qwen/Qwen3-8B", None)
        assert metadata["fingerprint"] == VALID_FP

    def test_missing_metadata_file_raises(self, tmp_path):
        d = tmp_path / "empty_adapter"
        d.mkdir()
        with pytest.raises(ValueError, match="No metadata.json"):
            _verify_adapter_metadata(str(d), VALID_FP, "Qwen/Qwen3-8B", None)

    def test_fingerprint_mismatch_raises(self, tmp_path):
        path = self._write_metadata(
            tmp_path, fingerprint="adp_ffffffffffffffff", base_model="Qwen/Qwen3-8B", revision=None,
        )
        with pytest.raises(ValueError, match="fingerprint"):
            _verify_adapter_metadata(path, VALID_FP, "Qwen/Qwen3-8B", None)

    def test_base_model_mismatch_raises(self, tmp_path):
        path = self._write_metadata(
            tmp_path, fingerprint=VALID_FP, base_model="Qwen/Qwen3-14B", revision=None,
        )
        with pytest.raises(ValueError, match="base_model"):
            _verify_adapter_metadata(path, VALID_FP, "Qwen/Qwen3-8B", None)

    def test_revision_mismatch_raises(self, tmp_path):
        path = self._write_metadata(
            tmp_path, fingerprint=VALID_FP, base_model="Qwen/Qwen3-8B", revision="refs/pr/1",
        )
        with pytest.raises(ValueError, match="revision"):
            _verify_adapter_metadata(path, VALID_FP, "Qwen/Qwen3-8B", "refs/pr/2")


# ---------------------------------------------------------------------------
# get_llm_from_list: validation + branch selection (HFLocalLLM stubbed)
# ---------------------------------------------------------------------------

class _StubHFLocalLLM:
    """Records constructor kwargs instead of loading a real model."""
    last_kwargs = None

    def __init__(self, model_id, *args, **kwargs):
        self.model_id = model_id
        type(self).last_kwargs = kwargs
        self.enable_thinking = False


@pytest.fixture
def stub_hf_local(monkeypatch):
    import access_eval.models.hf_local as hf_local_module
    monkeypatch.setattr(hf_local_module, "HFLocalLLM", _StubHFLocalLLM)
    _StubHFLocalLLM.last_kwargs = None
    return _StubHFLocalLLM


class TestGetLlmFromListAdapterPlumbing:
    def test_lora_suffix_without_adapter_path_raises(self):
        with pytest.raises(ValueError, match="no adapter_path was given"):
            get_llm_from_list(f"Qwen/Qwen3-8B[hf][lora:{VALID_FP}]", api_models=[])

    def test_adapter_path_without_lora_suffix_raises(self):
        with pytest.raises(ValueError, match="carries no \\[lora:<fp>\\] suffix"):
            get_llm_from_list("Qwen/Qwen3-8B[hf]", api_models=[], adapter_path="/some/path")

    def test_hf_suffix_forces_hf_backend(self, stub_hf_local, tmp_path):
        llm = get_llm_from_list("Qwen/Qwen3-8B[hf]", api_models=[])
        assert isinstance(llm, stub_hf_local)
        assert "adapter_path" not in stub_hf_local.last_kwargs

    def test_lora_suffix_with_matching_metadata_forwards_adapter_path(self, stub_hf_local, tmp_path):
        adapter_dir = tmp_path / "adp_0123456789abcdef"
        adapter_dir.mkdir()
        with open(adapter_dir / "metadata.json", "w") as f:
            json.dump({"fingerprint": VALID_FP, "base_model": "Qwen/Qwen3-8B", "revision": None}, f)

        llm = get_llm_from_list(
            f"Qwen/Qwen3-8B[hf][lora:{VALID_FP}]", api_models=[], adapter_path=str(adapter_dir),
        )
        assert isinstance(llm, stub_hf_local)
        assert stub_hf_local.last_kwargs["adapter_path"] == str(adapter_dir)

    def test_lora_suffix_with_mismatched_metadata_raises_before_construction(self, stub_hf_local, tmp_path):
        adapter_dir = tmp_path / "adp_0123456789abcdef"
        adapter_dir.mkdir()
        with open(adapter_dir / "metadata.json", "w") as f:
            json.dump({"fingerprint": "adp_ffffffffffffffff", "base_model": "Qwen/Qwen3-8B", "revision": None}, f)

        with pytest.raises(ValueError, match="fingerprint"):
            get_llm_from_list(
                f"Qwen/Qwen3-8B[hf][lora:{VALID_FP}]", api_models=[], adapter_path=str(adapter_dir),
            )
        assert stub_hf_local.last_kwargs is None  # never got far enough to construct

    def test_gpt_oss_still_routes_to_hf_without_adapter(self, stub_hf_local):
        llm = get_llm_from_list("openai/gpt-oss-20b", api_models=[])
        assert isinstance(llm, stub_hf_local)

    def test_lora_suffix_default_still_routes_to_hf(self, stub_hf_local, tmp_path):
        """prefer_vllm_for_lora defaults to False -- every existing call
        site (Track 1, Track 2, everything before this option existed)
        must keep getting HFLocalLLM exactly as before."""
        adapter_dir = tmp_path / "adp_0123456789abcdef"
        adapter_dir.mkdir()
        with open(adapter_dir / "metadata.json", "w") as f:
            json.dump({"fingerprint": VALID_FP, "base_model": "Qwen/Qwen3-8B", "revision": None}, f)
        llm = get_llm_from_list(
            f"Qwen/Qwen3-8B[hf][lora:{VALID_FP}]", api_models=[], adapter_path=str(adapter_dir),
        )
        assert isinstance(llm, stub_hf_local)

    @requires_vllm
    def test_prefer_vllm_for_lora_routes_to_vllm_despite_hf_suffix(self, monkeypatch, tmp_path):
        """The real bug this fixes: model names always carry [hf] alongside
        [lora:...] by convention, so prefer_vllm_for_lora must override
        BOTH the adapter-forces-HF default AND the literal [hf] suffix --
        confirmed directly against a real run before this test existed
        (the first fix attempt routed to HF anyway because it only
        cleared the adapter check, not force_hf)."""
        import access_eval.models.vllm_local as vllm_local_module

        class _StubVLLM:
            last_kwargs = None
            def __init__(self, model_id, *args, **kwargs):
                self.model_id = model_id
                type(self).last_kwargs = kwargs

        monkeypatch.setattr(vllm_local_module, "VLLMLocalLLM", _StubVLLM)
        adapter_dir = tmp_path / "adp_0123456789abcdef"
        adapter_dir.mkdir()
        with open(adapter_dir / "metadata.json", "w") as f:
            json.dump({"fingerprint": VALID_FP, "base_model": "Qwen/Qwen3-8B", "revision": None}, f)

        llm = get_llm_from_list(
            f"Qwen/Qwen3-8B[hf][lora:{VALID_FP}]", api_models=[],
            adapter_path=str(adapter_dir), prefer_vllm_for_lora=True,
        )
        assert isinstance(llm, _StubVLLM)
        assert _StubVLLM.last_kwargs["adapter_path"] == str(adapter_dir)

    def test_prefer_vllm_for_lora_true_without_adapter_has_no_effect(self, stub_hf_local):
        """The flag only matters when an adapter is actually present --
        a plain [hf]-forced, no-adapter call must be unaffected."""
        llm = get_llm_from_list("Qwen/Qwen3-8B[hf]", api_models=[], prefer_vllm_for_lora=True)
        assert isinstance(llm, stub_hf_local)

    def test_gpt_oss_with_prefer_vllm_for_lora_still_forces_hf(self, stub_hf_local):
        """gpt-oss is the one unconditional exception -- prefer_vllm_for_lora
        must not override it even without an adapter present."""
        llm = get_llm_from_list("openai/gpt-oss-20b", api_models=[], prefer_vllm_for_lora=True)
        assert isinstance(llm, stub_hf_local)

    @requires_vllm
    def test_plain_local_model_unaffected_by_new_kwarg_default(self, monkeypatch):
        """A model with none of the new suffixes and adapter_path=None
        (the default) must not even attempt to import hf_local -- confirms
        existing non-contamination call sites are untouched."""
        import access_eval.models.vllm_local as vllm_local_module

        class _StubVLLM:
            def __init__(self, model_id, *args, **kwargs):
                self.model_id = model_id
                self.enable_thinking = False

        monkeypatch.setattr(vllm_local_module, "VLLMLocalLLM", _StubVLLM)
        llm = get_llm_from_list("Qwen/Qwen2.5-7B-Instruct", api_models=[])
        assert isinstance(llm, _StubVLLM)


# ---------------------------------------------------------------------------
# get_llm(): must be consistent with get_llm_from_list() (review found these
# two entry points had drifted -- one supported adapter suffixes/checks, the
# other silently didn't).
# ---------------------------------------------------------------------------

class TestGetLlmAdapterConsistency:
    def test_lora_suffix_without_adapter_path_raises(self):
        with pytest.raises(ValueError, match="no adapter_path was given"):
            get_llm(f"Qwen/Qwen3-8B[hf][lora:{VALID_FP}]")

    def test_adapter_path_without_lora_suffix_raises(self):
        with pytest.raises(ValueError, match="carries no \\[lora:<fp>\\] suffix"):
            get_llm("Qwen/Qwen3-8B[hf]", adapter_path="/some/path")

    def test_unrecognized_suffix_raises(self):
        with pytest.raises(ValueError, match="Unrecognized bracket suffix"):
            get_llm("Qwen/Qwen3-8B[laura:oops]")

    def test_unterminated_suffix_raises(self):
        with pytest.raises(ValueError, match="Unterminated or malformed"):
            get_llm("Qwen/Qwen3-8B[lora:broken")

    def test_hf_suffix_forces_hf_backend_even_with_use_vllm_true(self, stub_hf_local):
        llm = get_llm("Qwen/Qwen3-8B[hf]", use_vllm=True)
        assert isinstance(llm, stub_hf_local)
        assert "adapter_path" not in stub_hf_local.last_kwargs

    def test_lora_suffix_with_matching_metadata_forwards_adapter_path(self, stub_hf_local, tmp_path):
        adapter_dir = tmp_path / "adp_0123456789abcdef"
        adapter_dir.mkdir()
        with open(adapter_dir / "metadata.json", "w") as f:
            json.dump({"fingerprint": VALID_FP, "base_model": "Qwen/Qwen3-8B", "revision": None}, f)

        llm = get_llm(
            f"Qwen/Qwen3-8B[hf][lora:{VALID_FP}]", adapter_path=str(adapter_dir),
        )
        assert isinstance(llm, stub_hf_local)
        assert stub_hf_local.last_kwargs["adapter_path"] == str(adapter_dir)

    def test_lora_suffix_with_mismatched_metadata_raises_before_construction(self, stub_hf_local, tmp_path):
        adapter_dir = tmp_path / "adp_0123456789abcdef"
        adapter_dir.mkdir()
        with open(adapter_dir / "metadata.json", "w") as f:
            json.dump({"fingerprint": "adp_ffffffffffffffff", "base_model": "Qwen/Qwen3-8B", "revision": None}, f)

        with pytest.raises(ValueError, match="fingerprint"):
            get_llm(f"Qwen/Qwen3-8B[hf][lora:{VALID_FP}]", adapter_path=str(adapter_dir))
        assert stub_hf_local.last_kwargs is None

    @requires_vllm
    def test_plain_call_unaffected_by_new_parser(self, monkeypatch):
        """paraphrase.py's existing get_llm(model_name, **kwargs) call
        pattern, with a plain model name and no adapter kwargs, must behave
        exactly as before."""
        import access_eval.models.vllm_local as vllm_local_module

        class _StubVLLM:
            def __init__(self, model_id, *args, **kwargs):
                self.model_id = model_id
                self.enable_thinking = False

        monkeypatch.setattr(vllm_local_module, "VLLMLocalLLM", _StubVLLM)
        llm = get_llm("meta-llama/Llama-3.1-8B-Instruct")
        assert isinstance(llm, _StubVLLM)
        assert llm.model_id == "meta-llama/Llama-3.1-8B-Instruct"

    @requires_vllm
    def test_with_thinking_still_works(self, monkeypatch):
        import access_eval.models.vllm_local as vllm_local_module

        class _StubVLLM:
            def __init__(self, model_id, *args, **kwargs):
                self.model_id = model_id
                self.enable_thinking = False

        monkeypatch.setattr(vllm_local_module, "VLLMLocalLLM", _StubVLLM)
        llm = get_llm("meta-llama/Llama-3.1-8B[with_thinking]")
        assert llm.model_id == "meta-llama/Llama-3.1-8B"
        assert llm.enable_thinking is True
