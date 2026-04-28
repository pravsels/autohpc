"""
Input Expectation: "is the data we'd feed the model what the model expects?"

Two layers of checks here:
  STATIC_CHECKS  -- operate on a CheckpointExtraction alone, no passport.
  PASSPORT_CHECKS -- cross-check passport.input_contract against config,
                     norm stats files, and (optionally) a dataset.

The passport checks catch the wrong-stats-file bug (via stats_fingerprint),
silent renormalisation drift, and dataset key swaps.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..extraction import CheckpointExtraction
from ..observation import Observation, Status
from ..passport import PassportLoadResult
from ..schema import (
    ActionSpec,
    ImageSpec,
    InputContract,
    StateSpec,
    TrainingDatasetSpec,
)


CATEGORY = "input_expectation"


def check_image_keys_declared(ext: CheckpointExtraction) -> Observation:
    """Does the config declare which camera/image keys the model expects?"""
    # Some configs use "image_features", others embed visuals in "input_features"
    input_features = ext.config.get("input_features", {})
    image_features = ext.config.get("image_features", {})

    visual_from_inputs = {
        k: v for k, v in input_features.items()
        if isinstance(v, dict) and v.get("type") == "VISUAL"
    }

    all_image_keys = sorted(set(visual_from_inputs.keys()) | set(image_features.keys()))

    if not all_image_keys:
        return Observation(
            check="image_keys_declared",
            status=Status.FAIL,
            message="no image/visual keys found in config",
            details={
                "input_features_keys": sorted(input_features.keys()),
                "image_features_keys": sorted(image_features.keys()),
            },
            category=CATEGORY,
        )

    return Observation(
        check="image_keys_declared",
        status=Status.PASS,
        message=f"found {len(all_image_keys)} image key(s): {all_image_keys}",
        details={
            "image_keys": all_image_keys,
            "from_input_features": sorted(visual_from_inputs.keys()),
            "from_image_features": sorted(image_features.keys()),
        },
        category=CATEGORY,
    )


def check_state_dim_consistency(ext: CheckpointExtraction) -> Observation:
    """Does the state dim in input_features match the dataset_schema entries?"""
    input_features = ext.config.get("input_features", {})
    state_entry = input_features.get("observation.state", {})
    state_shape = state_entry.get("shape", [])
    input_state_dim = state_shape[0] if state_shape else None

    schema_entries = ext.config.get("dataset_schema", {}).get("state", [])
    schema_dims = {e["key"]: e.get("dim") for e in schema_entries if "key" in e}
    schema_total = sum(d for d in schema_dims.values() if d is not None)

    if input_state_dim is None and not schema_dims:
        return Observation(
            check="state_dim_consistency",
            status=Status.NOT_CHECKED,
            message="no state dim declared in input_features or dataset_schema",
            details={},
            category=CATEGORY,
        )

    # Both sources present -- cross-check
    if input_state_dim is not None and schema_dims:
        if schema_total == input_state_dim:
            return Observation(
                check="state_dim_consistency",
                status=Status.PASS,
                message=f"schema entries sum to {schema_total}, matches input_features state dim",
                details={
                    "input_features_state_dim": input_state_dim,
                    "schema_entries": schema_dims,
                    "schema_total": schema_total,
                },
                category=CATEGORY,
            )
        else:
            return Observation(
                check="state_dim_consistency",
                status=Status.SOFT_SIGNAL,
                message=(
                    f"schema entries sum to {schema_total} but input_features "
                    f"declares state_dim={input_state_dim}"
                ),
                details={
                    "input_features_state_dim": input_state_dim,
                    "schema_entries": schema_dims,
                    "schema_total": schema_total,
                },
                category=CATEGORY,
            )

    return Observation(
        check="state_dim_consistency",
        status=Status.PASS,
        message="only one source of state dim -- no cross-check possible",
        details={"input_features_state_dim": input_state_dim, "schema_entries": schema_dims},
        category=CATEGORY,
    )


def check_norm_stats_vs_config(ext: CheckpointExtraction) -> Observation:
    """Do the norm stats file dimensions match what the config declares?

    The norm stats file is an input-side artifact (it normalizes incoming
    data), so cross-checking its declared dims against the config lives
    here. Output-side norm checks (predicted-action distribution vs
    training stats) live in output_spec.
    """
    if not ext.norm_stats:
        return Observation(
            check="norm_stats_vs_config",
            status=Status.FAIL,
            message="no norm stats files found",
            details={"config_files_found": list(ext.config_files_found.keys())},
            category=CATEGORY,
        )

    mismatches = []
    comparisons = []
    for stats_key, stats_data in ext.norm_stats.items():
        meta = stats_data if not isinstance(stats_data, dict) else stats_data.get("metadata", stats_data)

        stats_action_dim = meta.get("action_dim")
        config_action_dim = ext.declared_action_dim
        if stats_action_dim is not None and config_action_dim is not None:
            comparisons.append({"file": stats_key, "field": "action_dim",
                                "norm_stats": stats_action_dim, "config": config_action_dim})
            if stats_action_dim != config_action_dim:
                mismatches.append(comparisons[-1])

        stats_horizon = meta.get("horizon")
        config_horizon = ext.config.get("horizon") or ext.config.get("chunk_size")
        if stats_horizon is not None and config_horizon is not None:
            comparisons.append({"file": stats_key, "field": "horizon",
                                "norm_stats": stats_horizon, "config": config_horizon})
            if stats_horizon != config_horizon:
                mismatches.append(comparisons[-1])

    if mismatches:
        return Observation(
            check="norm_stats_vs_config",
            status=Status.FAIL,
            message=f"{len(mismatches)} dimension mismatch(es) between norm stats and config",
            details={"mismatches": mismatches, "all_comparisons": comparisons},
            category=CATEGORY,
        )

    return Observation(
        check="norm_stats_vs_config",
        status=Status.PASS,
        message="norm stats dimensions match config",
        details={"comparisons": comparisons},
        category=CATEGORY,
    )


# Static checks that run from a CheckpointExtraction (no passport required).
STATIC_CHECKS: List[Callable[[CheckpointExtraction], Observation]] = [
    check_image_keys_declared,
    check_state_dim_consistency,
    check_norm_stats_vs_config,
]


# ── Passport-driven checks ──────────────────────────────────────────────


def check_input_contract_vs_config(
    load: PassportLoadResult,
    ext: CheckpointExtraction,
) -> Observation:
    """Every claim in input_contract is reflected in config.json.

    We don't check the reverse direction (config has things passport doesn't);
    the passport is the source of truth for what the model expects, the
    config is just one of the artifacts it was assembled from.
    """
    if not load.has_passport:
        return _not_checked("input_contract_vs_config", "no passport loaded")

    ic = load.passport.input_contract
    config = ext.config
    mismatches: List[Dict[str, Any]] = []

    # Image keys: every passport image key must appear in config (in either
    # input_features as VISUAL or top-level image_features).
    config_image_keys = _config_image_keys(config)
    for img in ic.images:
        if img.key not in config_image_keys:
            mismatches.append({
                "kind": "image_key_missing_from_config",
                "passport_key": img.key,
                "config_image_keys": sorted(config_image_keys),
            })
        # Shape, if declared, must match.
        elif img.raw_shape is not None:
            cfg_shape = _config_image_shape(config, img.key)
            if cfg_shape is not None and list(cfg_shape) != list(img.raw_shape):
                mismatches.append({
                    "kind": "image_shape_mismatch",
                    "passport_key": img.key,
                    "passport_shape": list(img.raw_shape),
                    "config_shape": list(cfg_shape),
                })

    # State total_dim must match the config's input_features state dim.
    if ic.state is not None and ic.state.total_dim is not None:
        cfg_state_dim = _config_state_dim(config)
        if cfg_state_dim is not None and cfg_state_dim != ic.state.total_dim:
            mismatches.append({
                "kind": "state_total_dim_mismatch",
                "passport": ic.state.total_dim,
                "config": cfg_state_dim,
            })

    # Action total_dim must match the config's output_features action shape.
    if ic.actions is not None:
        if ic.actions.total_dim is not None and ext.declared_action_dim is not None:
            if ext.declared_action_dim != ic.actions.total_dim:
                mismatches.append({
                    "kind": "action_total_dim_mismatch",
                    "passport": ic.actions.total_dim,
                    "config": ext.declared_action_dim,
                })
        # Horizon must match a horizon-related field in config.
        if ic.actions.horizon is not None:
            cfg_horizon = (
                config.get("horizon")
                or config.get("chunk_size")
                or config.get("n_action_steps")
                or config.get("action_horizon")
            )
            if cfg_horizon is not None and cfg_horizon != ic.actions.horizon:
                mismatches.append({
                    "kind": "action_horizon_mismatch",
                    "passport": ic.actions.horizon,
                    "config": cfg_horizon,
                })

    if mismatches:
        return Observation(
            check="input_contract_vs_config",
            status=Status.FAIL,
            message=f"{len(mismatches)} passport/config mismatch(es)",
            details={"mismatches": mismatches},
            category=CATEGORY,
        )

    return Observation(
        check="input_contract_vs_config",
        status=Status.PASS,
        message="all input_contract claims reflected in config",
        details={
            "checked_image_keys": [i.key for i in ic.images],
            "checked_state_total_dim": ic.state.total_dim if ic.state else None,
            "checked_action_total_dim": ic.actions.total_dim if ic.actions else None,
            "checked_action_horizon": ic.actions.horizon if ic.actions else None,
        },
        category=CATEGORY,
    )


def check_input_contract_vs_norm_stats(
    load: PassportLoadResult,
    ext: CheckpointExtraction,
) -> Observation:
    """Norm sources point at real files; declared dims match file metadata;
    stats fingerprint (file sha256 + per-dim q02/q98 sample) matches.

    Catches the wrong-stats-file bug -- if the upstream skill computed the
    fingerprint from one file and a different file got shipped, this fires.
    """
    if not load.has_passport:
        return _not_checked("input_contract_vs_norm_stats", "no passport loaded")

    ic = load.passport.input_contract
    ckpt_root = ext.path
    findings: List[Dict[str, Any]] = []
    sources_checked: List[str] = []

    for label, spec in (("state", ic.state), ("actions", ic.actions)):
        if spec is None:
            continue
        norm = spec.normalization
        if not norm:
            continue
        source_rel = norm.get("source")
        if not source_rel:
            findings.append({"kind": "missing_source", "side": label})
            continue
        sources_checked.append(source_rel)
        source_path = (ckpt_root / source_rel).resolve()
        if not source_path.is_file():
            findings.append({
                "kind": "source_file_missing",
                "side": label,
                "source": source_rel,
            })
            continue

        try:
            stats_data = json.loads(source_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            findings.append({
                "kind": "source_unreadable",
                "side": label,
                "source": source_rel,
                "error": f"{type(e).__name__}: {e}",
            })
            continue

        # Dim cross-check (look in metadata, then top-level)
        meta = stats_data.get("metadata", stats_data) if isinstance(stats_data, dict) else {}
        declared_dim = norm.get("stats_dim")
        if declared_dim is not None:
            file_dim = meta.get(f"{label[:-1] if label == 'actions' else label}_dim") \
                if label != "actions" else meta.get("action_dim")
            if file_dim is not None and file_dim != declared_dim:
                findings.append({
                    "kind": "dim_mismatch",
                    "side": label,
                    "passport": declared_dim,
                    "stats_file": file_dim,
                })

        # Fingerprint: file sha256
        fp = norm.get("stats_fingerprint") or {}
        declared_sha = fp.get("file_sha256")
        if declared_sha:
            actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if actual_sha != declared_sha:
                findings.append({
                    "kind": "fingerprint_sha_mismatch",
                    "side": label,
                    "source": source_rel,
                    "passport_sha256": declared_sha,
                    "actual_sha256": actual_sha,
                })

        # Fingerprint: per-dim q02 / q98 sample (compare elementwise)
        for q_key in ("per_dim_q02", "per_dim_q98", "per_dim_q02_at_t0", "per_dim_q98_at_t0"):
            declared_q = fp.get(q_key)
            if declared_q is None:
                continue
            # Find matching field in the stats file -- support both flat
            # and per-timestep layouts.
            actual_q = _stats_quantile_sample(stats_data, label, q_key)
            if actual_q is None:
                findings.append({
                    "kind": "fingerprint_quantile_unavailable",
                    "side": label,
                    "key": q_key,
                })
                continue
            if not _quantiles_close(declared_q, actual_q):
                findings.append({
                    "kind": "fingerprint_quantile_mismatch",
                    "side": label,
                    "key": q_key,
                    "passport": declared_q[:5],  # truncated for display
                    "actual": actual_q[:5],
                })

    if findings:
        # Distinguish hard-fail kinds from soft signals.
        hard_kinds = {
            "source_file_missing", "source_unreadable",
            "dim_mismatch", "fingerprint_sha_mismatch",
            "fingerprint_quantile_mismatch",
        }
        any_hard = any(f["kind"] in hard_kinds for f in findings)
        return Observation(
            check="input_contract_vs_norm_stats",
            status=Status.FAIL if any_hard else Status.SOFT_SIGNAL,
            message=f"{len(findings)} stats-file finding(s); "
                    f"{'hard fail' if any_hard else 'soft signals only'}",
            details={"findings": findings, "sources_checked": sources_checked},
            category=CATEGORY,
        )

    return Observation(
        check="input_contract_vs_norm_stats",
        status=Status.PASS,
        message=f"{len(sources_checked)} norm stats source(s) verified",
        details={"sources_checked": sources_checked},
        category=CATEGORY,
    )


def check_input_contract_vs_dataset(
    load: PassportLoadResult,
    ext: CheckpointExtraction,
    dataset_path: Optional[Path] = None,
) -> Observation:
    """Cross-check input_contract image/state keys against a real dataset.

    Auto-resolves to `input_contract.training_datasets[0]` when no dataset_path
    is given AND that entry's `repo` is locally resolvable. The check is
    deliberately format-only: reads `meta/info.json` from the dataset path
    and compares key sets. Anything more invasive (load the dataset, sample
    a frame) belongs to a higher-level test.
    """
    if not load.has_passport:
        return _not_checked("input_contract_vs_dataset", "no passport loaded")

    ic = load.passport.input_contract

    target = dataset_path
    auto_resolved_from = None
    if target is None and ic.training_datasets:
        first = ic.training_datasets[0]
        # We can't fetch from HF here -- if the repo path resolves locally,
        # use it; otherwise return NOT_CHECKED.
        candidate = _local_dataset_candidate(first)
        if candidate is not None:
            target = candidate
            auto_resolved_from = first.repo

    if target is None:
        return Observation(
            check="input_contract_vs_dataset",
            status=Status.NOT_CHECKED,
            message="no dataset path provided and no local copy of "
                    "training_datasets[0] found",
            details={
                "training_datasets": [d.repo for d in ic.training_datasets],
            },
            category=CATEGORY,
        )

    info_path = Path(target) / "meta" / "info.json"
    if not info_path.is_file():
        return Observation(
            check="input_contract_vs_dataset",
            status=Status.SOFT_SIGNAL,
            message=f"dataset path has no meta/info.json at {info_path}",
            details={"dataset_path": str(target)},
            category=CATEGORY,
        )

    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return Observation(
            check="input_contract_vs_dataset",
            status=Status.FAIL,
            message=f"cannot read dataset meta/info.json: {type(e).__name__}",
            details={"dataset_path": str(target), "error": str(e)},
            category=CATEGORY,
        )

    # Apply key_rename_map from the resolved training_datasets entry, so
    # passport keys map back to the dataset's actual keys.
    rename_map = {}
    if ic.training_datasets:
        rename_map = dict(ic.training_datasets[0].key_rename_map)
    inverse_rename = {v: k for k, v in rename_map.items()}

    dataset_features = info.get("features") or {}
    dataset_keys = set(dataset_features.keys())

    expected_keys = set()
    for img in ic.images:
        expected_keys.add(inverse_rename.get(img.key, img.key))
    if ic.state is not None:
        for sk in ic.state.sub_keys:
            k = sk.get("key")
            if k:
                expected_keys.add(inverse_rename.get(k, k))

    missing_in_dataset = sorted(k for k in expected_keys if k not in dataset_keys)

    if missing_in_dataset:
        return Observation(
            check="input_contract_vs_dataset",
            status=Status.FAIL,
            message=f"{len(missing_in_dataset)} passport key(s) absent from dataset",
            details={
                "dataset_path": str(target),
                "auto_resolved_from": auto_resolved_from,
                "missing_in_dataset": missing_in_dataset,
                "dataset_keys": sorted(dataset_keys),
                "rename_map": rename_map,
            },
            category=CATEGORY,
        )

    return Observation(
        check="input_contract_vs_dataset",
        status=Status.PASS,
        message=f"all {len(expected_keys)} passport key(s) present in dataset",
        details={
            "dataset_path": str(target),
            "auto_resolved_from": auto_resolved_from,
            "expected_keys": sorted(expected_keys),
            "rename_map": rename_map,
        },
        category=CATEGORY,
    )


def check_training_datasets_present(load: PassportLoadResult) -> Observation:
    """Soft signal if input_contract.training_datasets is empty/missing."""
    if not load.has_passport:
        return _not_checked("training_datasets_present", "no passport loaded")

    tds = load.passport.input_contract.training_datasets
    if not tds:
        return Observation(
            check="training_datasets_present",
            status=Status.SOFT_SIGNAL,
            message="input_contract.training_datasets is empty",
            details={},
            category=CATEGORY,
        )
    return Observation(
        check="training_datasets_present",
        status=Status.PASS,
        message=f"{len(tds)} training dataset(s) declared",
        details={"datasets": [d.repo for d in tds]},
        category=CATEGORY,
    )


# Loose: HF repo names are "user/name", commits are 7-40 hex chars.
_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


def check_training_datasets_resolvable(load: PassportLoadResult) -> Observation:
    """Format-only: each entry has a plausible repo string and commit sha.

    Best-effort HF API existence is intentionally NOT done here -- it would
    add network calls to the default validation flow. Leave that for an
    opt-in flag in a future iteration.
    """
    if not load.has_passport:
        return _not_checked("training_datasets_resolvable", "no passport loaded")

    tds = load.passport.input_contract.training_datasets
    if not tds:
        return Observation(
            check="training_datasets_resolvable",
            status=Status.NOT_CHECKED,
            message="no training_datasets to format-check",
            details={},
            category=CATEGORY,
        )

    bad: List[Dict[str, Any]] = []
    for i, d in enumerate(tds):
        problems = []
        if not d.repo or not _REPO_RE.match(d.repo):
            problems.append("repo not 'user/name'")
        if not d.commit or not _COMMIT_RE.match(d.commit):
            problems.append("commit not 7-40 hex chars")
        if problems:
            bad.append({"index": i, "repo": d.repo, "commit": d.commit, "problems": problems})

    if bad:
        return Observation(
            check="training_datasets_resolvable",
            status=Status.SOFT_SIGNAL,
            message=f"{len(bad)} training dataset entry(ies) have malformed "
                    "repo/commit fields",
            details={"problems": bad},
            category=CATEGORY,
        )
    return Observation(
        check="training_datasets_resolvable",
        status=Status.PASS,
        message=f"all {len(tds)} training dataset entry(ies) have well-formed pointers",
        details={"datasets": [{"repo": d.repo, "commit": d.commit} for d in tds]},
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


def _config_image_keys(config: Dict[str, Any]) -> set:
    """All keys that the config thinks are visual inputs."""
    keys: set = set()
    for k, v in (config.get("input_features") or {}).items():
        if isinstance(v, dict) and v.get("type") == "VISUAL":
            keys.add(k)
    keys.update((config.get("image_features") or {}).keys())
    return keys


def _config_image_shape(config: Dict[str, Any], key: str) -> Optional[List[int]]:
    inp = (config.get("input_features") or {}).get(key)
    if isinstance(inp, dict):
        s = inp.get("shape")
        if s:
            return s
    img = (config.get("image_features") or {}).get(key)
    if isinstance(img, dict):
        s = img.get("shape")
        if s:
            return s
    return None


def _config_state_dim(config: Dict[str, Any]) -> Optional[int]:
    state = (config.get("input_features") or {}).get("observation.state")
    if isinstance(state, dict):
        shape = state.get("shape") or []
        if shape:
            return shape[0]
    feat = config.get("robot_state_feature") or {}
    if isinstance(feat, dict):
        shape = feat.get("shape") or []
        if shape:
            return shape[0]
    return None


def _stats_quantile_sample(
    stats_data: Any,
    side: str,
    q_key: str,
) -> Optional[List[float]]:
    """Best-effort lookup of a per-dim quantile inside a stats file.

    Stats files vary in layout. We support:
      stats_data[side]["q02"]                   -> flat list
      stats_data[side]["q02"][0]                -> per-timestep, take t=0
    The skill is responsible for keeping its fingerprint format stable;
    here we just try to find a comparable list.
    """
    if not isinstance(stats_data, dict):
        return None
    block = stats_data.get(side[:-1] if side == "actions" else side)
    if block is None:
        # try the singular form too -- "actions" -> "action"
        block = stats_data.get("action") if side == "actions" else None
    if not isinstance(block, dict):
        return None

    # Strip _at_t0 / per_dim_ prefixes when looking up the quantile name.
    base = q_key.replace("per_dim_", "").replace("_at_t0", "")
    raw = block.get(base)
    if raw is None:
        return None
    # Per-timestep layout: list of lists. Take t=0 sample.
    if raw and isinstance(raw[0], list):
        return list(raw[0])
    return list(raw)


def _quantiles_close(a: List[float], b: List[float], rtol: float = 1e-5, atol: float = 1e-7) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(x - y) > atol + rtol * abs(y):
            return False
    return True


def _local_dataset_candidate(d: TrainingDatasetSpec) -> Optional[Path]:
    """Try to locate `d.repo` on the local filesystem (HF cache paths).

    No network calls. Returns None if nothing local is found -- the check
    will then report NOT_CHECKED rather than failing.
    """
    if not d.repo:
        return None
    # HuggingFace standard cache layout:
    #   ~/.cache/huggingface/datasets--<user>--<name>/snapshots/<commit>
    repo_dirname = "datasets--" + d.repo.replace("/", "--")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / repo_dirname / "snapshots"
    if cache_root.is_dir():
        if d.commit:
            commit_dir = cache_root / d.commit
            if commit_dir.is_dir():
                return commit_dir
        # otherwise, latest snapshot if there's exactly one
        snaps = [p for p in cache_root.iterdir() if p.is_dir()]
        if len(snaps) == 1:
            return snaps[0]
    return None


# Passport-driven checks. The runner calls these with a PassportLoadResult
# (and the static extraction). `check_input_contract_vs_dataset` also
# accepts an optional dataset_path, threaded through from the CLI.
def check_camera_identity(load: PassportLoadResult) -> Observation:
    """Camera identity fields populated on image specs (enables swap detection)."""
    if not load.has_passport:
        return _not_checked("camera_identity", "no passport loaded")

    images = load.passport.input_contract.images
    if not images:
        return _not_checked("camera_identity", "no images in input_contract")

    populated = 0
    total = len(images)
    missing_fields: List[str] = []

    for img in images:
        has_any = any([
            img.camera_serial,
            img.camera_usb_path,
            img.reference_frame_hash,
        ])
        if has_any:
            populated += 1
        else:
            missing_fields.append(img.key)

    if populated == 0:
        return Observation(
            check="camera_identity",
            status=Status.SOFT_SIGNAL,
            message=(
                f"no camera identity fields on any of {total} image spec(s); "
                "camera swap detection unavailable"
            ),
            details={"cameras_without_identity": missing_fields},
            category=CATEGORY,
        )

    if populated < total:
        return Observation(
            check="camera_identity",
            status=Status.SOFT_SIGNAL,
            message=f"{populated}/{total} image spec(s) have camera identity fields",
            details={"cameras_without_identity": missing_fields},
            category=CATEGORY,
        )

    return Observation(
        check="camera_identity",
        status=Status.PASS,
        message=f"all {total} image spec(s) have camera identity fields",
        details={},
        category=CATEGORY,
    )


PASSPORT_CHECKS: List[Callable[..., Observation]] = [
    check_input_contract_vs_config,
    check_input_contract_vs_norm_stats,
    check_input_contract_vs_dataset,
    check_training_datasets_present,
    check_training_datasets_resolvable,
    check_camera_identity,
]
