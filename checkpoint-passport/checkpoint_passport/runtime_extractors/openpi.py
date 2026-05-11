"""
OpenPI passport seed extractor.

Runs inside an OpenPI-capable environment to extract the inference contract
from OpenPI's runtime pipeline.  The contract does NOT exist in a static
config.json — it is constructed by cfg.data.create(), transforms, norm stats,
and adapter behavior.

This extractor:
    1. Loads the OpenPI TrainConfig by name.
    2. Instantiates the data pipeline (cfg.data.create) to inspect transforms.
    3. Reads norm stats from the checkpoint assets/ directory.
    4. Emits passport seed sections: stack, input_contract, output_spec,
       model_identity, model_internals (partial).

When --device is provided (runtime enrichment):
    5. Loads the model via the RuntimeAdapter protocol
       (checkpoint_passport.runtime_adapters.openpi).
    6. Collects library versions, parameter count, and smoke results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from checkpoint_passport.runtime_extractors.base import (
    BaseExtractor,
    MissingRuntimeError,
)


_EXTRACTOR_VERSION = "0.2.0"


def _import_openpi_config():
    """Import openpi.training.config, raising MissingRuntimeError on failure."""
    try:
        import openpi.training.config as config_mod
        return config_mod
    except (ImportError, ModuleNotFoundError) as exc:
        missing = getattr(exc, "name", None) or "openpi"
        raise MissingRuntimeError(missing, "openpi") from exc


def _stringify(val: Any) -> Any:
    """Convert enum-like objects to their string value for JSON."""
    if val is None:
        return None
    if hasattr(val, "value"):
        return str(val.value)
    if hasattr(val, "name") and not isinstance(val, type):
        return str(val.name)
    return val


def _safe_getattr(obj: Any, *attrs: str) -> Any:
    """Walk a dotted attribute chain, returning None if any step is missing."""
    val = obj
    for attr in attrs:
        val = getattr(val, attr, None)
        if val is None:
            return None
    return val


def _detect_checkpoint_format(checkpoint_dir: Path) -> str:
    if (checkpoint_dir / "model.safetensors").exists():
        return "model.safetensors"
    if (checkpoint_dir / "params").is_dir():
        return "orbax"
    return "unknown"


def _read_norm_stats(checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
    """Read norm_stats.json from the checkpoint assets/ directory if present."""
    for candidate in [
        checkpoint_dir / "assets" / "norm_stats.json",
        checkpoint_dir / "norm_stats.json",
    ]:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text())
            except (json.JSONDecodeError, OSError):
                pass
    return None


def _extract_input_contract(
    cfg: Any,
    data_config: Any,
    norm_stats: Optional[Dict[str, Any]],
    *,
    default_prompt: Optional[str],
    resize_size: Optional[int],
) -> Dict[str, Any]:
    """Build input_contract from instantiated data config + transforms."""
    contract: Dict[str, Any] = {}

    model = getattr(cfg, "model", None)
    data_factory = getattr(cfg, "data", None)

    # Action spec
    action_horizon = _safe_getattr(model, "action_horizon")
    use_delta = _safe_getattr(data_factory, "use_delta_actions")
    joints_only = _safe_getattr(data_factory, "joints_only")

    # Determine actual robot-facing action dim from data config
    # For joints_only=True, the robot action dim comes from the policy output
    # transforms, not model.action_dim (which is the tokenized dim).
    action_dim = None
    if data_config is not None:
        # The data config's output transforms define the actual action dim
        output_transforms = _safe_getattr(data_config, "data_transforms", "outputs")
        if output_transforms:
            for t in output_transforms:
                ad = getattr(t, "action_dim", None)
                if ad is not None:
                    action_dim = ad
                    break

    actions: Dict[str, Any] = {}
    if action_horizon is not None:
        actions["horizon"] = action_horizon
    if action_dim is not None:
        actions["total_dim"] = action_dim
    if use_delta is not None:
        actions["use_delta_actions"] = use_delta
    if joints_only is not None:
        actions["joints_only"] = joints_only

    # Delta mask from data config
    delta_mask = _safe_getattr(data_factory, "delta_action_mask")
    if delta_mask is not None:
        actions["delta_mask"] = list(delta_mask)

    if actions:
        contract["actions"] = actions

    # Language spec
    prompt = default_prompt or _safe_getattr(data_factory, "default_prompt")
    if prompt is not None:
        contract["language"] = {"default_prompt": prompt}

    # Training datasets
    repo_id = _safe_getattr(data_factory, "repo_id")
    if repo_id is not None:
        if isinstance(repo_id, str) and repo_id.startswith("["):
            repos = [r.strip() for r in repo_id.strip("[]").split(",") if r.strip()]
        elif isinstance(repo_id, (list, tuple)):
            repos = list(repo_id)
        else:
            repos = [repo_id]
        contract["training_datasets"] = [{"repo": r} for r in repos]

    # Image spec (from data transforms if available)
    if data_config is not None:
        input_transforms = _safe_getattr(data_config, "data_transforms", "inputs")
        if input_transforms:
            image_keys = []
            for t in input_transforms:
                ik = getattr(t, "image_keys", None)
                if ik:
                    image_keys.extend(ik)
            if image_keys:
                images = [{"key": k} for k in image_keys]
                if resize_size is not None:
                    for img in images:
                        img["encoder_resize"] = [resize_size, resize_size]
                contract["images"] = images

    return contract


def _extract_output_spec(cfg: Any) -> Dict[str, Any]:
    """Build output_spec from model config."""
    model = getattr(cfg, "model", None)
    spec: Dict[str, Any] = {}

    action_horizon = _safe_getattr(model, "action_horizon")
    if action_horizon is not None:
        spec["actions"] = {"horizon": action_horizon}

    return spec


def _extract_model_identity(cfg: Any) -> Dict[str, Any]:
    """Build model_identity from config (without loading weights)."""
    model = getattr(cfg, "model", None)
    identity: Dict[str, Any] = {}

    model_type = _safe_getattr(model, "model_type")
    if model_type is not None:
        identity["class_name"] = _stringify(model_type)

    identity["resolved_via"] = "openpi.training.config"

    return identity


def _extract_model_internals(cfg: Any) -> Dict[str, Any]:
    """Build partial model_internals from config (no weight loading)."""
    model = getattr(cfg, "model", None)
    internals: Dict[str, Any] = {}

    # Forward graph shape hints from config
    action_dim = _safe_getattr(model, "action_dim")
    action_horizon = _safe_getattr(model, "action_horizon")
    if action_dim is not None and action_horizon is not None:
        internals["forward_graph"] = {
            "sample_output_shapes": {
                "action_tokens": [1, action_horizon, action_dim],
            },
        }

    return internals


def _enrich_with_runtime(
    seed: Dict[str, Any],
    checkpoint_dir: Path,
    *,
    config_name: str,
    default_prompt: Optional[str],
    resize_size: Optional[int],
    device: str,
) -> None:
    """Load the model via the RuntimeAdapter protocol and enrich the seed."""
    from checkpoint_passport.runtime_adapters.openpi import OpenPIRuntimeAdapter

    adapter = OpenPIRuntimeAdapter()
    adapter.load(
        checkpoint_dir,
        device=device,
        config_name=config_name,
        default_prompt=default_prompt,
        resize_size=resize_size,
    )

    seed["extractor"]["device"] = device

    lib_versions = adapter.library_versions()
    if lib_versions:
        seed.setdefault("model_identity", {})
        seed["model_identity"]["library_versions"] = lib_versions

    param_summary = adapter.count_parameters()
    if param_summary is not None:
        seed.setdefault("model_internals", {})
        seed["model_internals"]["parameter_summary"] = param_summary

    smoke = adapter.smoke_inference()
    if smoke is not None:
        seed.setdefault("model_internals", {})
        seed["model_internals"]["numerical_health"] = {"smoke": smoke}


class OpenPIExtractor(BaseExtractor):
    """Extracts a passport seed from an OpenPI checkpoint."""

    def extract_seed(
        self,
        checkpoint_dir: Path,
        *,
        config_name: str,
        default_prompt: Optional[str] = None,
        resize_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        config_mod = _import_openpi_config()
        cfg = config_mod.get_config(config_name)

        # Try to instantiate the data pipeline for deeper introspection
        data_config = None
        try:
            data_factory = getattr(cfg, "data", None)
            model_config = getattr(cfg, "model", None)
            assets_dirs = getattr(cfg, "assets_dirs", None)
            if data_factory is not None and hasattr(data_factory, "create"):
                if assets_dirs is not None and model_config is not None:
                    data_config = data_factory.create(assets_dirs, model_config)
        except Exception:
            pass

        norm_stats = _read_norm_stats(checkpoint_dir)

        input_contract = _extract_input_contract(
            cfg, data_config, norm_stats,
            default_prompt=default_prompt,
            resize_size=resize_size,
        )
        output_spec = _extract_output_spec(cfg)
        model_identity = _extract_model_identity(cfg)
        model_internals = _extract_model_internals(cfg)

        seed: Dict[str, Any] = {
            "extractor": {
                "extractor_name": "openpi",
                "extractor_version": _EXTRACTOR_VERSION,
                "openpi_config_name": config_name,
                "checkpoint_format": _detect_checkpoint_format(checkpoint_dir),
            },
            "stack": "openpi",
        }

        if input_contract:
            seed["input_contract"] = input_contract
        if output_spec:
            seed["output_spec"] = output_spec
        if model_identity:
            seed["model_identity"] = model_identity
        if model_internals:
            seed["model_internals"] = model_internals

        if device is not None:
            _enrich_with_runtime(
                seed,
                checkpoint_dir,
                config_name=config_name,
                default_prompt=default_prompt,
                resize_size=resize_size,
                device=device,
            )

        return seed
