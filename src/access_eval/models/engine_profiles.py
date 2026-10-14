"""
The single place that decides model-specific vLLM engine options.

    resolve_engine_kwargs(model_id) = auto-detected options  <-  overrides

Auto-detection reads only the checkpoint's config.json (no weights), so it needs no per-model list:
  * a multimodal checkpoint (vision/audio sub-config) used for text  -> language_model_only=True (skips the vision
    tower; only where the installed vLLM supports the option -- vLLM 0.10.x does not and loads the full model)
  * a checkpoint with custom modeling code (config auto_map)         -> trust_remote_code=True

Overrides, applied on top (later wins):
  * $VLLM_EXTRA_ENGINE_KWARGS  JSON dict for every model (kept for the existing job scripts)
  * $ACCESS_EVAL_ENGINE_KWARGS        JSON dict: either flat ({"kv_cache_dtype": "fp8"}, every model) or keyed by model id
                               ({"nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16": {"mamba_ssm_cache_dtype": "float32"}})
Options the installed vLLM does not know are dropped with a warning instead of crashing the engine. Numerics
choices that are not inferable from the config (e.g. Mamba cache dtypes) stay explicit overrides.
"""
import json
import os
from functools import lru_cache


@lru_cache(maxsize=None)
def _engine_fields():
    try:
        from vllm.engine.arg_utils import EngineArgs
        return frozenset(EngineArgs.__dataclass_fields__)
    except Exception:
        return None


def _load_config(model_id: str, revision=None):
    from pathlib import Path
    local = Path(model_id) / "config.json"
    try:
        if local.exists():
            return json.loads(local.read_text())
        from huggingface_hub import hf_hub_download
        return json.loads(Path(hf_hub_download(model_id, "config.json", revision=revision)).read_text())
    except Exception as e:
        print(f"[engine_profiles] could not read config.json for {model_id}: {e} -- no auto-detected options")
        return None


def auto_engine_kwargs(model_id: str, revision=None) -> dict:
    cfg = _load_config(model_id, revision) or {}
    out = {}
    if any(k in cfg for k in ("vision_config", "audio_config", "vision_tower")):
        out["language_model_only"] = True
    if "auto_map" in cfg:
        out["trust_remote_code"] = True
    return out


def _overrides(model_id: str) -> dict:
    out = {}
    if os.environ.get("VLLM_EXTRA_ENGINE_KWARGS"):
        out.update(json.loads(os.environ["VLLM_EXTRA_ENGINE_KWARGS"]))
    if os.environ.get("ACCESS_EVAL_ENGINE_KWARGS"):
        spec = json.loads(os.environ["ACCESS_EVAL_ENGINE_KWARGS"])
        if spec and all(isinstance(v, dict) for v in spec.values()):
            out.update(spec.get(model_id, {}))
        else:
            out.update(spec)
    return out


def resolve_engine_kwargs(model_id: str, revision=None) -> dict:
    kwargs = {**auto_engine_kwargs(model_id, revision), **_overrides(model_id)}
    fields = _engine_fields()
    if fields is not None:
        unknown = [k for k in kwargs if k not in fields]
        for k in unknown:
            print(f"[engine_profiles] {model_id}: this vLLM has no engine option {k!r}; ignoring it")
            kwargs.pop(k)
    return kwargs
