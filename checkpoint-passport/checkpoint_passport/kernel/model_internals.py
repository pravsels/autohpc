"""
Model Internals: "did the model load correctly and is it self-consistent?"

Two layers:
  STATIC_CHECKS   -- CheckpointExtraction-only (config + weight headers).
  PASSPORT_CHECKS -- cross-check the model_internals + model_identity
                     sections of the passport against weight headers and
                     against an attempted import of the declared class.

The identity check is deliberately import-only -- it never loads weights.
It catches silent class swaps (e.g. Qwen3 config loaded as Qwen2.5) before
they become silently degraded outputs at inference time.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List

from ..extraction import CheckpointExtraction
from ..observation import Observation, Status
from ..passport import PassportLoadResult


CATEGORY = "model_internals"


def check_weight_file_integrity(ext: CheckpointExtraction) -> Observation:
    """Are there weight files, and do they contain tensors?"""
    if not ext.weights.files:
        return Observation(
            check="weight_file_integrity",
            status=Status.FAIL,
            message="no weight files found",
            details={"searched_patterns": ["*.safetensors", "*.bin"]},
            category=CATEGORY,
        )

    n_tensors = len(ext.weights.tensors)
    if n_tensors == 0:
        return Observation(
            check="weight_file_integrity",
            status=Status.FAIL,
            message=f"weight file(s) found ({ext.weights.files}) but contain no tensors",
            details={"files": ext.weights.files},
            category=CATEGORY,
        )

    return Observation(
        check="weight_file_integrity",
        status=Status.PASS,
        message=f"{n_tensors} tensors across {len(ext.weights.files)} file(s)",
        details={"files": ext.weights.files, "n_tensors": n_tensors},
        category=CATEGORY,
    )


def check_action_dim_config_vs_weights(ext: CheckpointExtraction) -> Observation:
    """Does the config-declared action dim appear in output layer weight shapes?"""
    action_dim = ext.declared_action_dim
    if action_dim is None:
        return Observation(
            check="action_dim_config_vs_weights",
            status=Status.FAIL,
            message="config does not declare action dim in output_features.action.shape",
            details={"output_features": ext.config.get("output_features")},
            category=CATEGORY,
        )

    output_tensors = {
        name: ext.weights.shape_of(name)
        for name in ext.weights.parameter_names
        if any(k in name.lower() for k in ("output", "head", "final_proj"))
    }

    matching = {
        name: shape for name, shape in output_tensors.items()
        if shape and action_dim in shape
    }

    if matching:
        return Observation(
            check="action_dim_config_vs_weights",
            status=Status.PASS,
            message=f"declared action_dim={action_dim} confirmed in {len(matching)} output tensor(s)",
            details={"action_dim": action_dim, "matching_tensors": matching},
            category=CATEGORY,
        )
    else:
        return Observation(
            check="action_dim_config_vs_weights",
            status=Status.FAIL,
            message=f"declared action_dim={action_dim} not found in any output tensor shape",
            details={"action_dim": action_dim, "output_tensors_found": output_tensors},
            category=CATEGORY,
        )


def check_horizon_consistency(ext: CheckpointExtraction) -> Observation:
    """Do all horizon-related config fields agree with each other?"""
    fields = {}
    for key in ("chunk_size", "horizon", "n_action_steps", "action_horizon"):
        val = ext.config.get(key)
        if val is not None:
            fields[key] = val

    if not fields:
        return Observation(
            check="horizon_consistency",
            status=Status.FAIL,
            message="no horizon-related field found in config",
            details={"searched_fields": ["chunk_size", "horizon", "n_action_steps", "action_horizon"]},
            category=CATEGORY,
        )

    values = set(fields.values())
    if len(values) == 1:
        return Observation(
            check="horizon_consistency",
            status=Status.PASS,
            message=f"all horizon fields agree: {list(values)[0]}",
            details=fields,
            category=CATEGORY,
        )
    else:
        return Observation(
            check="horizon_consistency",
            status=Status.FAIL,
            message=f"horizon fields disagree: {fields}",
            details=fields,
            category=CATEGORY,
        )


# Static checks that run from a CheckpointExtraction (no passport required).
STATIC_CHECKS: List[Callable[[CheckpointExtraction], Observation]] = [
    check_weight_file_integrity,
    check_action_dim_config_vs_weights,
    check_horizon_consistency,
]


# ── Passport-driven checks ──────────────────────────────────────────────


def check_model_identity_resolvable(
    load: PassportLoadResult,
    ext: CheckpointExtraction,
) -> Observation:
    """Import the declared class without loading weights.

    Three resolution paths, in order:
      1. `class_module` + `class_name` declared -> import the module, getattr
         the class, confirm `cls.__name__` matches.
      2. `config_architectures` declared and `transformers` is importable ->
         resolve via `transformers.AutoModel.from_config` lookup table to
         confirm the architecture string maps to the same class HF would
         load. This is the Qwen3/Qwen2.5 catch.
      3. Neither -> NOT_CHECKED, soft signal.

    Crucially: never instantiates the model, never loads weights. If the
    import itself raises, that's a hard fail.
    """
    if not load.has_passport:
        return _not_checked("model_identity_resolvable", "no passport loaded")

    mi = load.passport.model_identity
    findings: Dict[str, Any] = {
        "declared_class": mi.class_name,
        "declared_module": mi.class_module,
        "declared_architectures": list(mi.config_architectures),
    }

    # Path 1: direct import
    if mi.class_module and mi.class_name:
        try:
            mod = importlib.import_module(mi.class_module)
        except Exception as e:
            return Observation(
                check="model_identity_resolvable",
                status=Status.FAIL,
                message=f"cannot import {mi.class_module!r}: {type(e).__name__}",
                details={**findings, "error": str(e)},
                category=CATEGORY,
            )
        cls = getattr(mod, mi.class_name, None)
        if cls is None:
            return Observation(
                check="model_identity_resolvable",
                status=Status.FAIL,
                message=f"module {mi.class_module!r} has no attribute "
                        f"{mi.class_name!r}",
                details=findings,
                category=CATEGORY,
            )
        actual = getattr(cls, "__name__", None)
        if actual != mi.class_name:
            return Observation(
                check="model_identity_resolvable",
                status=Status.FAIL,
                message=f"imported class name {actual!r} does not match "
                        f"declared {mi.class_name!r}",
                details={**findings, "imported_name": actual},
                category=CATEGORY,
            )
        # Cross-check resolved_class_name if the passport recorded it.
        if mi.resolved_class_name and mi.resolved_class_name != actual:
            return Observation(
                check="model_identity_resolvable",
                status=Status.FAIL,
                message=f"resolved_class_name in passport ({mi.resolved_class_name!r}) "
                        f"does not match what direct import produces ({actual!r})",
                details={**findings, "imported_name": actual},
                category=CATEGORY,
            )
        return Observation(
            check="model_identity_resolvable",
            status=Status.PASS,
            message=f"declared class {mi.class_name!r} imports cleanly from "
                    f"{mi.class_module!r}",
            details={**findings, "resolved_via": "direct_import"},
            category=CATEGORY,
        )

    # Path 2: HF AutoModel resolution from architectures
    if mi.config_architectures:
        arch = mi.config_architectures[0]
        try:
            transformers = importlib.import_module("transformers")
        except Exception as e:
            return Observation(
                check="model_identity_resolvable",
                status=Status.SOFT_SIGNAL,
                message="transformers not importable; cannot resolve "
                        f"architecture {arch!r}",
                details={**findings, "error": f"{type(e).__name__}: {e}"},
                category=CATEGORY,
            )
        cls = getattr(transformers, arch, None)
        if cls is None:
            return Observation(
                check="model_identity_resolvable",
                status=Status.FAIL,
                message=f"transformers does not expose class {arch!r}; "
                        "config.architectures may have rotted (silent class swap risk)",
                details=findings,
                category=CATEGORY,
            )
        actual = getattr(cls, "__name__", None)
        if actual != arch:
            return Observation(
                check="model_identity_resolvable",
                status=Status.FAIL,
                message=f"transformers.{arch} resolves to class named {actual!r} "
                        "(silent rename / class swap)",
                details={**findings, "imported_name": actual},
                category=CATEGORY,
            )
        return Observation(
            check="model_identity_resolvable",
            status=Status.PASS,
            message=f"architecture {arch!r} resolves to transformers.{arch}",
            details={**findings, "resolved_via": "transformers"},
            category=CATEGORY,
        )

    return Observation(
        check="model_identity_resolvable",
        status=Status.SOFT_SIGNAL,
        message="passport declares neither class_module/class_name nor "
                "config_architectures; cannot resolve identity",
        details=findings,
        category=CATEGORY,
    )


def check_internals_vs_weight_files(
    load: PassportLoadResult,
    ext: CheckpointExtraction,
) -> Observation:
    """Every parameter the passport claims exists in the weight headers,
    and vice versa, with matching shape and dtype.

    Catches: hand-edited passports, partially-loaded checkpoints, weight
    files swapped between checkpoint dirs.
    """
    if not load.has_passport:
        return _not_checked("internals_vs_weight_files", "no passport loaded")

    declared = load.passport.model_internals.parameters.by_name
    if not declared:
        return _not_checked(
            "internals_vs_weight_files",
            "passport.model_internals.parameters.by_name is empty",
        )

    by_passport = {p.name: p for p in declared}
    by_weight = ext.weights.tensors  # {name: {shape, dtype, ...}}

    missing_in_weights: List[str] = []
    missing_in_passport: List[str] = []
    shape_mismatches: List[Dict[str, Any]] = []
    dtype_mismatches: List[Dict[str, Any]] = []

    for name, p in by_passport.items():
        w = by_weight.get(name)
        if w is None:
            missing_in_weights.append(name)
            continue
        if list(w.get("shape") or []) != list(p.shape):
            shape_mismatches.append({
                "name": name,
                "passport_shape": list(p.shape),
                "weight_shape": list(w.get("shape") or []),
            })
        wdtype = (w.get("dtype") or "").upper()
        pdtype = (p.dtype or "").upper().replace("TORCH.", "")
        if wdtype and pdtype and not _dtype_compatible(wdtype, pdtype):
            dtype_mismatches.append({
                "name": name,
                "passport_dtype": p.dtype,
                "weight_dtype": w.get("dtype"),
            })

    for name in by_weight:
        if name not in by_passport:
            missing_in_passport.append(name)

    n_problems = (
        len(missing_in_weights)
        + len(missing_in_passport)
        + len(shape_mismatches)
        + len(dtype_mismatches)
    )
    if n_problems:
        return Observation(
            check="internals_vs_weight_files",
            status=Status.FAIL,
            message=(
                f"{len(missing_in_weights)} passport tensor(s) missing from weights, "
                f"{len(missing_in_passport)} weight tensor(s) missing from passport, "
                f"{len(shape_mismatches)} shape mismatch(es), "
                f"{len(dtype_mismatches)} dtype mismatch(es)"
            ),
            details={
                # truncate long lists for readability; full lists kept on disk
                "missing_in_weights": missing_in_weights[:10],
                "missing_in_passport": missing_in_passport[:10],
                "shape_mismatches": shape_mismatches[:10],
                "dtype_mismatches": dtype_mismatches[:10],
                "passport_param_count": len(by_passport),
                "weight_tensor_count": len(by_weight),
            },
            category=CATEGORY,
        )

    return Observation(
        check="internals_vs_weight_files",
        status=Status.PASS,
        message=f"{len(by_passport)} passport tensor(s) match weight headers exactly",
        details={
            "passport_param_count": len(by_passport),
            "weight_tensor_count": len(by_weight),
        },
        category=CATEGORY,
    )


def check_state_dict_completeness(load: PassportLoadResult) -> Observation:
    """`state_dict.missing_keys` and `unexpected_keys` are empty."""
    if not load.has_passport:
        return _not_checked("state_dict_completeness", "no passport loaded")

    sd = load.passport.model_internals.state_dict
    if sd.expected_keys_count is None and sd.found_keys_count is None \
            and not sd.missing_keys and not sd.unexpected_keys:
        return _not_checked(
            "state_dict_completeness",
            "passport.model_internals.state_dict not populated",
        )

    if sd.missing_keys or sd.unexpected_keys:
        return Observation(
            check="state_dict_completeness",
            status=Status.FAIL,
            message=(
                f"{len(sd.missing_keys)} missing key(s), "
                f"{len(sd.unexpected_keys)} unexpected key(s) recorded at load time"
            ),
            details={
                "missing_keys": sd.missing_keys[:20],
                "unexpected_keys": sd.unexpected_keys[:20],
            },
            category=CATEGORY,
        )

    return Observation(
        check="state_dict_completeness",
        status=Status.PASS,
        message=f"state_dict loaded clean ({sd.found_keys_count}/{sd.expected_keys_count} keys)",
        details={
            "expected_keys_count": sd.expected_keys_count,
            "found_keys_count": sd.found_keys_count,
        },
        category=CATEGORY,
    )


def check_no_nan_inf_recorded(load: PassportLoadResult) -> Observation:
    """Did the upstream NaN/Inf scan pass?"""
    if not load.has_passport:
        return _not_checked("no_nan_inf_recorded", "no passport loaded")

    block = load.passport.model_internals.numerical_health.no_nan_inf
    if not block:
        return _not_checked(
            "no_nan_inf_recorded",
            "passport did not record a no_nan_inf result",
        )
    if block.get("passed") is True:
        return Observation(
            check="no_nan_inf_recorded",
            status=Status.PASS,
            message=f"no NaN/Inf in {block.get('params_checked', '?')} params, "
                    f"{block.get('buffers_checked', '?')} buffers",
            details=dict(block),
            category=CATEGORY,
        )
    return Observation(
        check="no_nan_inf_recorded",
        status=Status.FAIL,
        message="upstream recorded NaN or Inf in weights/buffers",
        details=dict(block),
        category=CATEGORY,
    )


def check_determinism_recorded(load: PassportLoadResult) -> Observation:
    """Did two forward passes on the same input produce the same output?"""
    if not load.has_passport:
        return _not_checked("determinism_recorded", "no passport loaded")

    block = load.passport.model_internals.numerical_health.determinism
    if not block:
        return _not_checked(
            "determinism_recorded",
            "passport did not record a determinism result",
        )
    if block.get("passed") is True:
        return Observation(
            check="determinism_recorded",
            status=Status.PASS,
            message=f"forward pass deterministic (max_abs_diff="
                    f"{block.get('max_abs_diff')})",
            details=dict(block),
            category=CATEGORY,
        )
    return Observation(
        check="determinism_recorded",
        status=Status.FAIL,
        message=f"forward pass not deterministic (max_abs_diff="
                f"{block.get('max_abs_diff')})",
        details=dict(block),
        category=CATEGORY,
    )


def check_external_pretrained_assets_pinned(load: PassportLoadResult) -> Observation:
    """Every external pretrained submodule has a non-null `hf_revision`."""
    if not load.has_passport:
        return _not_checked("external_pretrained_assets_pinned", "no passport loaded")

    assets = load.passport.model_internals.pretrained_provenance
    if not assets:
        return Observation(
            check="external_pretrained_assets_pinned",
            status=Status.PASS,
            message="no external pretrained submodules declared",
            details={},
            category=CATEGORY,
        )

    unpinned = [
        {"submodule": a.submodule, "source": a.source}
        for a in assets if not a.hf_revision
    ]
    if unpinned:
        return Observation(
            check="external_pretrained_assets_pinned",
            status=Status.SOFT_SIGNAL,
            message=f"{len(unpinned)} external pretrained submodule(s) without "
                    "hf_revision pin (reproducibility risk)",
            details={"unpinned": unpinned, "total": len(assets)},
            category=CATEGORY,
        )
    return Observation(
        check="external_pretrained_assets_pinned",
        status=Status.PASS,
        message=f"all {len(assets)} external pretrained submodule(s) pinned",
        details={
            "assets": [{"submodule": a.submodule, "hf_revision": a.hf_revision}
                       for a in assets],
        },
        category=CATEGORY,
    )


# ── Helpers ─────────────────────────────────────────────────────────────


def _not_checked(name: str, reason: str) -> Observation:
    return Observation(
        check=name,
        status=Status.NOT_CHECKED,
        message=reason,
        details={},
        category=CATEGORY,
    )


# safetensors uses "F32", "F16", "BF16", "I64" etc.; passport may use
# torch-style "float32", "torch.float32", "bfloat16". This map is the
# minimum needed to compare without dragging torch in.
_DTYPE_ALIASES = {
    "F32": {"FLOAT32", "F32", "FP32"},
    "F16": {"FLOAT16", "F16", "FP16"},
    "BF16": {"BFLOAT16", "BF16"},
    "F64": {"FLOAT64", "F64", "FP64", "DOUBLE"},
    "I64": {"INT64", "I64", "LONG"},
    "I32": {"INT32", "I32"},
    "I8":  {"INT8", "I8"},
    "U8":  {"UINT8", "U8"},
    "BOOL": {"BOOL"},
}


def _dtype_compatible(weight_dtype: str, passport_dtype: str) -> bool:
    w = weight_dtype.upper()
    p = passport_dtype.upper()
    if w == p:
        return True
    aliases = _DTYPE_ALIASES.get(w, {w})
    return p in aliases


PASSPORT_CHECKS: List[Callable[..., Observation]] = [
    check_model_identity_resolvable,
    check_internals_vs_weight_files,
    check_state_dict_completeness,
    check_no_nan_inf_recorded,
    check_determinism_recorded,
    check_external_pretrained_assets_pinned,
]
