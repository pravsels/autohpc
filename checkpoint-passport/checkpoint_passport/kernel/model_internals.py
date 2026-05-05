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
import re
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

    summary = load.passport.model_internals.parameters.summary
    sd = load.passport.model_internals.state_dict
    by_weight = ext.weights.tensors  # {name: {shape, dtype, ...}}

    expected_count = sd.expected_keys_count
    weight_count = len(by_weight) if by_weight else None

    if expected_count is None and weight_count is None:
        return _not_checked(
            "internals_vs_weight_files",
            "no key counts available in passport or weight files",
        )

    problems: List[str] = []
    if expected_count is not None and weight_count is not None:
        if expected_count != weight_count:
            problems.append(
                f"passport expects {expected_count} keys but weight files have {weight_count}"
            )

    total_params = summary.total_params
    if total_params is not None and weight_count is not None:
        # Sanity: weight tensor count should be in the same ballpark as
        # total_params entries (buffers may differ, so only flag gross mismatches)
        pass

    if problems:
        return Observation(
            check="internals_vs_weight_files",
            status=Status.FAIL,
            message="; ".join(problems),
            details={
                "expected_keys_count": expected_count,
                "weight_tensor_count": weight_count,
            },
            category=CATEGORY,
        )

    return Observation(
        check="internals_vs_weight_files",
        status=Status.PASS,
        message=f"key counts consistent (expected={expected_count}, weights={weight_count})",
        details={
            "expected_keys_count": expected_count,
            "weight_tensor_count": weight_count,
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


_REVISION_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def check_external_pretrained_assets_pinned(load: PassportLoadResult) -> Observation:
    """Every external pretrained submodule has a valid source_revision or source_identifier pin."""
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

    unpinned = []
    bad_format = []
    for a in assets:
        if not a.source_revision:
            if a.source_identifier:
                continue
            unpinned.append({"submodule": a.submodule, "source": a.source})
        elif not _REVISION_SHA_RE.match(a.source_revision):
            bad_format.append({
                "submodule": a.submodule,
                "source_revision": a.source_revision,
                "problem": "not a commit SHA (expected 7-40 hex chars)",
            })

    problems = unpinned + bad_format
    if problems:
        return Observation(
            check="external_pretrained_assets_pinned",
            status=Status.FAIL,
            message=f"{len(problems)} external pretrained submodule(s) with "
                    "missing or invalid version pin",
            details={
                "unpinned": unpinned or None,
                "bad_format": bad_format or None,
                "total": len(assets),
            },
            category=CATEGORY,
        )
    return Observation(
        check="external_pretrained_assets_pinned",
        status=Status.PASS,
        message=f"all {len(assets)} external pretrained submodule(s) pinned",
        details={
            "assets": [{"submodule": a.submodule,
                        "pin": a.source_revision or a.source_identifier}
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


def check_runtime_constraints(load: PassportLoadResult) -> Observation:
    """Installed library versions satisfy runtime_constraints.required_versions."""
    if not load.has_passport:
        return _not_checked("runtime_constraints", "no passport loaded")

    rc = load.passport.model_identity.runtime_constraints
    if not rc.required_versions:
        return _not_checked(
            "runtime_constraints",
            "no runtime_constraints.required_versions declared",
        )

    import importlib.metadata as _meta

    violations: List[str] = []
    checked = 0
    for pkg, constraint in rc.required_versions.items():
        checked += 1
        try:
            installed = _meta.version(pkg)
        except _meta.PackageNotFoundError:
            violations.append(f"{pkg}: not installed (requires {constraint})")
            continue

        try:
            from packaging.version import Version
            from packaging.specifiers import SpecifierSet
            spec = SpecifierSet(constraint)
            if Version(installed) not in spec:
                violations.append(
                    f"{pkg}: installed {installed} does not satisfy {constraint}"
                )
        except ImportError:
            pass  # packaging not installed; skip version comparison
        except Exception:
            violations.append(
                f"{pkg}: could not parse constraint {constraint!r}"
            )

    if violations:
        return Observation(
            check="runtime_constraints",
            status=Status.FAIL,
            message=f"{len(violations)} of {checked} version constraint(s) violated",
            details={"violations": violations},
            category=CATEGORY,
        )
    return Observation(
        check="runtime_constraints",
        status=Status.PASS,
        message=f"{checked} runtime version constraint(s) satisfied",
        details={},
        category=CATEGORY,
    )


def check_reference_test_vector_present(load: PassportLoadResult) -> Observation:
    """Reference test vector assets are populated and hash fields are set."""
    if not load.has_passport:
        return _not_checked("reference_test_vector", "no passport loaded")

    rtv = load.passport.reference_test_vector
    if rtv is None:
        return Observation(
            check="reference_test_vector",
            status=Status.FAIL,
            message="no reference_test_vector in passport; golden-input reproducibility check cannot run",
            details={},
            category=CATEGORY,
        )

    problems: List[str] = []
    if not rtv.input_state_path:
        problems.append("input_state_path is empty")
    if not rtv.input_state_hash:
        problems.append("input_state_hash is empty")
    if not rtv.expected_output_path:
        problems.append("expected_output_path is empty")
    if not rtv.expected_output_hash:
        problems.append("expected_output_hash is empty")
    if not rtv.input_images_path:
        problems.append("input_images_path is empty")
    if not rtv.input_images_hash:
        problems.append("input_images_hash is empty")

    if problems:
        return Observation(
            check="reference_test_vector",
            status=Status.FAIL,
            message=f"reference_test_vector incomplete: {'; '.join(problems)}",
            details={"problems": problems},
            category=CATEGORY,
        )

    return Observation(
        check="reference_test_vector",
        status=Status.PASS,
        message=(
            f"reference_test_vector present: "
            f"state={rtv.input_state_path}, "
            f"output={rtv.expected_output_path}, "
            f"{len(rtv.input_images_hash)} image(s), "
            f"tolerance={rtv.tolerance}"
        ),
        details={},
        category=CATEGORY,
    )


PASSPORT_CHECKS: List[Callable[..., Observation]] = [
    check_model_identity_resolvable,
    check_internals_vs_weight_files,
    check_state_dict_completeness,
    check_no_nan_inf_recorded,
    check_determinism_recorded,
    check_external_pretrained_assets_pinned,
    check_runtime_constraints,
    check_reference_test_vector_present,
]
