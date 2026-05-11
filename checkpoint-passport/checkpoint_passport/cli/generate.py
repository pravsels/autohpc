"""
generate MODEL_PASSPORT.json from checkpoint files on disk.

Static extraction only — no torch, no GPU, no model import.  Reads
config.json, norm stats, TRAINING_LOG.md, and weight file headers to
populate every field that can be determined deterministically.

Architecture:
    Library function:  generate_passport() — pure data, raises on error.
    CLI wrapper:       main() — parses args, catches exceptions, sys.exit.

    generate_passport() never calls sys.exit(); callers (CLI, tests,
    other tools) handle errors via normal exception flow.

Determinism contract:
    Given the same (config_path, stats_path, generated_at, checkpoint
    contents), generate_passport() returns byte-identical JSON.  All
    sources of non-determinism are controlled:
      - Timestamp:  explicit via generated_at; falls back to utcnow
                    (non-deterministic unless caller pins it).
      - HF network: gated behind resolve_remote_revisions (default off).
                    pretrained source_revision is null when off.
      - File order: sorted() everywhere.

Fields not filled by static generation:
    Runtime extractor:     model_identity (class_name, class_module,
                           library_versions), model_internals (parameters,
                           state_dict, buffers, numerical_health),
                           output_spec.smoke_results.
    Structured metadata:   images[].physical_mounting, camera_serial
                           (user-supplied via deployment config, not guessed).
    Later workflow steps:  transform_pipeline (empty list),
                           known_issues (empty list),
                           reference_test_vector (null),
                           norm_round_trip_results (empty list).

    Validation/signoff decides whether missing values are allowed,
    soft signals, or hard blockers — nothing is left to freeform
    "operator judgment."

Usage:
    generate-passport /path/to/ckpt --config /path/to/ckpt/config.json
    generate-passport /path/to/ckpt --config ... --resolve-remote-revisions

Exit code 0 = passport written.  Exit code 1 = error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from checkpoint_passport.schema import (
    SCHEMA_VERSION,
    ModelPassport,
    InputContract,
    ImageSpec,
    StateSpec,
    ActionSpec,
    DeltaSpec,
    LanguageSpec,
    TemporalSpec,
    TrainingDatasetSpec,
    ModelInternals,
    PretrainedAsset,
    ForwardGraph,
    OutputSpec,
    InferenceParameters,
    WeightIntegrity,
    WeightFileEntry,
    Provenance,
)


# ── Filesystem / git helpers ────────────────────────────────────────────
#
# These are pure side-effect-free reads.  They never write, never call
# network APIs, and never call sys.exit().


def _sha256(path: Path) -> str:
    """Streaming SHA-256 — works for multi-GB safetensors without OOM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(repo: Path) -> Optional[str]:
    """Return the full HEAD commit SHA of a git repo, or None if unavailable."""
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _git_remote_url(repo: Path) -> Optional[str]:
    """Return the 'origin' remote URL for a git repo, or None if unavailable."""
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _git_is_dirty(repo: Path) -> Tuple[bool, str]:
    """Check if tracked files have uncommitted changes (ignores untracked files)."""
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        tracked_changes = "\n".join(
            line for line in r.stdout.strip().splitlines()
            if not line.startswith("??")
        ).strip()
        return bool(tracked_changes), tracked_changes
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, ""


def _find_training_log(ckpt: Path) -> Optional[Path]:
    """Look for TRAINING_LOG.md at the checkpoint root (case-insensitive fallback)."""
    for name in ["TRAINING_LOG.md", "training_log.md"]:
        p = ckpt / name
        if p.exists():
            return p
    return None


def _parse_wandb_url(log_path: Path) -> Optional[str]:
    """Extract W&B URL from TRAINING_LOG.md."""
    try:
        text = log_path.read_text()
        match = re.search(r"https://wandb\.ai/\S+", text)
        return match.group(0).rstrip(")") if match else None
    except OSError:
        return None


DEFAULT_SKIP_FILES = frozenset({
    "MODEL_PASSPORT.json", "PASSPORT_SEED.json", "SIGNOFF.json",
    "README.md", "TRAINING_LOG.md",
})

DEFAULT_SKIP_DIRS: frozenset[str] = frozenset()


def _hashable_files(
    ckpt: Path,
    *,
    extra_skip_files: Optional[List[str]] = None,
    extra_skip_dirs: Optional[List[str]] = None,
) -> List[Path]:
    """Find all files that should be hashed for weight_integrity.

    Includes weight files, config.json, and norm stats — everything
    needed for inference. Excludes hidden files, workflow artifacts
    (passport, signoff, README, training log), and any caller-specified
    files or top-level directories.
    """
    skip_files = DEFAULT_SKIP_FILES | set(extra_skip_files or [])
    skip_dirs = DEFAULT_SKIP_DIRS | set(extra_skip_dirs or [])

    files = []
    for p in sorted(ckpt.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.name in skip_files:
            continue
        if skip_dirs:
            rel = p.relative_to(ckpt)
            if rel.parts[0] in skip_dirs:
                continue
        files.append(p)
    return files


# ── Per-section extractors ──────────────────────────────────────────────
#
# Each _extract_* reads one section of config.json (and optionally norm
# stats) and returns the corresponding schema dataclass.  They are pure
# functions of their arguments — no filesystem access, no network calls.
#
# The config layout follows the LeRobot/RAMEN convention:
#   input_features   — per-modality shape + type declarations
#   output_features  — action shape
#   observation_encoder — vision backbone, text encoder, resize/crop
#   normalization_mapping — which norm type per modality (min_max, etc.)
#   dataset_schema   — per-joint sub-key breakdowns and rotation info
#   objective        — diffusion scheduler config


def _extract_images(config: Dict[str, Any]) -> List[ImageSpec]:
    """Build one ImageSpec per VISUAL input feature (one per camera)."""
    input_features = config.get("input_features", {})
    obs_encoder = config.get("observation_encoder", {})
    vision = obs_encoder.get("vision", {})
    norm_mapping = config.get("normalization_mapping", {})

    resize = vision.get("resize_shape")
    crop = vision.get("crop_shape")
    backbone = vision.get("backbone", "")

    images = []
    for key, feat in sorted(input_features.items()):
        if feat.get("type") != "VISUAL":
            continue
        shape = feat.get("shape", [])
        norm_type = norm_mapping.get("VISUAL")
        norm_info = None
        if norm_type:
            norm_info = {"type": norm_type}
            if backbone:
                norm_info["applied_by"] = f"observation_encoder.vision ({backbone})"

        images.append(ImageSpec(
            key=key,
            raw_shape=shape if shape else None,
            encoder_resize=resize,
            crop={"shape": crop} if crop else None,
            normalization=norm_info,
        ))
    return images


def _expand_dim(entry: Dict[str, Any]) -> int:
    """Compute model-facing dim for a dataset_schema entry."""
    dim = entry.get("dim", 0)
    if entry.get("convert_rotation"):
        # 6D source → 9D rot6d (3 translation + 6 rotation)
        # or 7D source → 10D (3 translation + 6 rotation + 1 gripper)
        return dim + 3
    return dim


def _extract_state(
    config: Dict[str, Any],
    stats: Optional[Dict[str, Any]],
    stats_path: Optional[Path],
    ckpt: Path,
) -> Optional[StateSpec]:
    """State vector breakdown: total dim, per-joint sub-keys, norm info.

    total_dim and sub_keys are model-facing (post rotation-expansion),
    matching input_features.observation.state.shape in config.json.
    """
    input_features = config.get("input_features", {})
    state_feat = input_features.get("observation.state", {})
    if not state_feat:
        return None

    total_dim = state_feat.get("shape", [None])[0]
    dataset_schema = config.get("dataset_schema", {})
    state_entries = dataset_schema.get("state", [])

    sub_keys = []
    for entry in state_entries:
        model_dim = _expand_dim(entry)
        sub_keys.append({
            "key": entry.get("key"),
            "dim": model_dim,
            "convert_rotation": entry.get("convert_rotation", False),
        })

    norm_mapping = config.get("normalization_mapping", {})
    norm_info = None
    if norm_mapping.get("STATE") and stats:
        norm_info = {
            "type": f"RAMEN_{norm_mapping['STATE']}",
            "source": str(stats_path.relative_to(ckpt)) if stats_path else None,
            "stats_dim": total_dim,
        }

    return StateSpec(
        total_dim=total_dim,
        sub_keys=sub_keys,
        normalization=norm_info,
    )


def _extract_actions(
    config: Dict[str, Any],
    stats: Optional[Dict[str, Any]],
    stats_path: Optional[Path],
    ckpt: Path,
) -> Optional[ActionSpec]:
    """Action spec: dim, horizon, sub-keys, norm mask, delta-vs-absolute.

    rot6d_slice from dataset_schema drives the delta_mask — rotation dims
    are absolute (passed through unchanged), everything else is delta.
    This is the info kernel checks use to catch the delta/absolute bug.
    """
    output_features = config.get("output_features", {})
    action_feat = output_features.get("action", {})
    if not action_feat:
        return None

    total_dim = action_feat.get("shape", [None])[0]
    horizon = config.get("horizon")

    dataset_schema = config.get("dataset_schema", {})
    action_entries = dataset_schema.get("action", [])
    rot6d_slice = dataset_schema.get("rot6d_slice")

    sub_keys = []
    for entry in action_entries:
        model_dim = _expand_dim(entry)
        sub_keys.append({
            "key": entry.get("key"),
            "dim": model_dim,
            "convert_rotation": entry.get("convert_rotation", False),
        })

    # norm_mask (only stored when non-uniform) and delta_dims from stats
    norm_mask = None
    delta_dims = None
    if stats:
        nm = stats.get("norm_mask")
        if nm and not all(nm):
            norm_mask = nm

        if rot6d_slice and total_dim:
            delta_mask = [True] * total_dim
            for i in range(rot6d_slice[0], min(rot6d_slice[1], total_dim)):
                delta_mask[i] = False
            delta_dims = DeltaSpec(
                delta_mask=delta_mask,
                absolute_dims_reason=(
                    f"dims {rot6d_slice[0]}-{rot6d_slice[1]-1} are 6D rotation "
                    f"(rot6d) passed through unchanged per "
                    f"dataset_schema.rot6d_slice == {rot6d_slice}"
                ),
            )

    # Normalization info
    norm_mapping = config.get("normalization_mapping", {})
    norm_info = None
    if norm_mapping.get("ACTION") and stats:
        action_stats = stats.get("action", {})
        q02 = action_stats.get("q02")
        layout = None
        if q02 and isinstance(q02[0], list):
            layout = f"(H={len(q02)}, D={len(q02[0])}) per-timestep"

        norm_info = {
            "type": f"RAMEN_{norm_mapping['ACTION']}",
            "source": str(stats_path.relative_to(ckpt)) if stats_path else None,
            "stats_dim": total_dim,
        }
        if layout:
            norm_info["stats_layout"] = layout

    return ActionSpec(
        total_dim=total_dim,
        horizon=horizon,
        sub_keys=sub_keys,
        norm_mask=norm_mask,
        delta_dims=delta_dims,
        normalization=norm_info,
    )


def _extract_language(config: Dict[str, Any]) -> Optional[LanguageSpec]:
    """Extract language conditioning spec (tokenizer, max length) if a text encoder is configured."""
    obs_encoder = config.get("observation_encoder", {})
    text = obs_encoder.get("text")
    if not text:
        return None
    model_name = text.get("model", "")
    return LanguageSpec(
        tokenizer_class="CLIPTokenizer" if "clip" in text.get("type", "").lower() else None,
        tokenizer_version=model_name or None,
        max_sequence_length=77 if "clip" in text.get("type", "").lower() else None,
    )


def _extract_temporal(config: Dict[str, Any]) -> Optional[TemporalSpec]:
    """Extract observation history length, delta indices, and control rate."""
    n_obs = config.get("n_obs_steps")
    if n_obs is None:
        return None
    indices = None
    if n_obs == 2:
        indices = [-1, 0]
    elif n_obs == 1:
        indices = [0]
    return TemporalSpec(
        n_obs_steps=n_obs,
        observation_delta_indices=indices,
        control_rate_hz=config.get("control_rate_hz"),
    )


def _extract_inference_params(config: Dict[str, Any]) -> Optional[InferenceParameters]:
    """Extract diffusion scheduler / inference config from the 'objective' block."""
    obj = config.get("objective", {})
    if not obj:
        return None

    return InferenceParameters(
        type=obj.get("type"),
        num_inference_steps=obj.get("num_inference_steps"),
        scheduler=obj.get("noise_scheduler_type"),
        prediction_type=obj.get("prediction_type"),
        clip_sample=obj.get("clip_sample"),
        clip_sample_range=obj.get("clip_sample_range"),
        chunk_aggregation="first_n_action_steps",
        chunks_executed_per_inference=config.get("n_action_steps"),
    )


# ── Pretrained asset + HF resolution ────────────────────────────────────
#
# Pretrained assets (vision backbone, text encoder) are embedded sub-
# models whose exact version matters for reproducibility.  Pinning their
# source_revision requires a HF API call, which is gated behind
# resolve_remote so that default generation is fully offline.


def _extract_pretrained(
    config: Dict[str, Any],
    *,
    resolve_remote: bool = False,
) -> List[PretrainedAsset]:
    """Catalog pretrained sub-models (vision backbone, text encoder) from config."""
    obs_encoder = config.get("observation_encoder", {})
    assets = []
    vision = obs_encoder.get("vision", {})
    if vision:
        backbone = vision.get("backbone")
        hf_repo = f"timm/{backbone}" if backbone else None
        revision = None
        if resolve_remote and hf_repo:
            revision = _resolve_hf_revision(hf_repo)
        assets.append(PretrainedAsset(
            submodule="observation_encoder.vision",
            source="timm" if backbone else None,
            source_identifier=backbone,
            source_revision=revision,
            frozen_in_training=False,
            lr_multiplier=vision.get("lr_multiplier"),
        ))
    text = obs_encoder.get("text", {})
    if text and text.get("model"):
        model_id = text.get("model")
        revision = _resolve_hf_revision(model_id) if resolve_remote else None
        assets.append(PretrainedAsset(
            submodule="observation_encoder.text",
            source="huggingface",
            source_identifier=model_id,
            source_revision=revision,
        ))
    return assets


def _resolve_hf_revision(repo_id: str) -> Optional[str]:
    """Best-effort resolve of a HuggingFace model repo's current commit SHA."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id)
        return info.sha
    except Exception:
        return None


def _parse_dataset_spec(spec: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse 'repo[@commit][:loader_class]' into (repo, commit, loader_class).

    Examples:
        'org/dataset'                           -> ('org/dataset', None, None)
        'org/dataset@abc123'                    -> ('org/dataset', 'abc123', None)
        'org/dataset:loader.Class'              -> ('org/dataset', None, 'loader.Class')
        'org/dataset@abc123:loader.Class'       -> ('org/dataset', 'abc123', 'loader.Class')
    """
    loader_class = None
    commit = None

    # Split loader_class off the end first — but only if '@' precedes ':'
    # or there's no '@' at all, so we don't confuse '@' inside a loader name.
    if "@" in spec:
        repo_commit, _, rest = spec.partition("@")
        if ":" in rest:
            commit_part, _, loader_class = rest.partition(":")
        else:
            commit_part = rest
        commit = commit_part or None
        repo = repo_commit
    elif ":" in spec:
        repo, _, loader_class = spec.partition(":")
    else:
        repo = spec

    return repo, commit, loader_class or None


# ── Training dataset resolution ─────────────────────────────────────────
#
# Datasets can be specified three ways, in priority order:
#   1. CLI --dataset flags (may include pinned @commit and :loader)
#   2. config.json "datasets" / "dataset" / "dataset_repo_id" keys
#   3. HF API resolution (only when resolve_remote=True)
#
# Pinned specs like 'org/data@abc123:loader.Class' are parsed by
# _parse_dataset_spec and bypass HF resolution entirely — the commit
# is taken from the spec itself.


def _extract_training_datasets(
    config: Dict[str, Any],
    dataset_repos: Optional[List[str]],
    loader_classes: Optional[List[str]],
    *,
    resolve_remote: bool = False,
) -> List[TrainingDatasetSpec]:
    """Build training_datasets from CLI args, falling back to config.json."""
    repos = list(dataset_repos or [])
    loaders = list(loader_classes or [])

    # Parse pinned specs from repos list (may contain @commit and :loader)
    parsed: List[Tuple[str, Optional[str], Optional[str]]] = []
    for spec in repos:
        parsed.append(_parse_dataset_spec(spec))

    if not parsed:
        cfg_datasets = config.get("datasets") or config.get("dataset")
        if isinstance(cfg_datasets, list):
            for entry in cfg_datasets:
                if isinstance(entry, dict):
                    parsed.append((
                        entry.get("repo") or entry.get("repo_id") or "",
                        None,
                        entry.get("loader_class") or None,
                    ))
                elif isinstance(entry, str):
                    parsed.append(_parse_dataset_spec(entry))
        elif isinstance(cfg_datasets, str):
            parsed.append(_parse_dataset_spec(cfg_datasets))
        if not parsed:
            single = config.get("dataset_repo_id")
            if single:
                parsed.append(_parse_dataset_spec(single))

    results = []
    for i, (repo, pinned_commit, spec_loader) in enumerate(parsed):
        if not repo:
            continue
        loader = spec_loader or (loaders[i] if i < len(loaders) else None)
        commit = pinned_commit
        if commit is None and resolve_remote:
            commit = _resolve_hf_dataset_revision(repo)
        results.append(TrainingDatasetSpec(
            repo=repo,
            commit=commit,
            loader_class=loader,
        ))
    return results


def _resolve_hf_dataset_revision(repo_id: str) -> Optional[str]:
    """Best-effort resolve of a HuggingFace dataset's current commit SHA."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(repo_id)
        return info.sha
    except Exception:
        return None


# ── Library API ─────────────────────────────────────────────────────────
#
# generate_passport() is the library entry point.  It:
#   1. Loads config and (optional) stats from explicit paths
#   2. Extracts six passport sections from config + stats + filesystem
#   3. Returns a ModelPassport dataclass — caller writes JSON
#
# It never calls sys.exit(); errors are raised as ValueError or
# FileNotFoundError so callers (CLI, tests, other tools) can handle them.


def generate_passport(
    checkpoint_dir: Path,
    *,
    config_path: Path,
    stats_path: Optional[Path] = None,
    generated_at: Optional[str] = None,
    resolve_remote_revisions: bool = False,
    target_repo: Optional[Path] = None,
    training_repo: Optional[Path] = None,
    dataset_repos: Optional[List[str]] = None,
    loader_classes: Optional[List[str]] = None,
    extra_skip_files: Optional[List[str]] = None,
    extra_skip_dirs: Optional[List[str]] = None,
) -> ModelPassport:
    """Build a ModelPassport from checkpoint files on disk.

    Args:
        checkpoint_dir: root of the checkpoint tree (weight files live here).
        config_path:    path to config.json (required).
        stats_path:     path to norm stats JSON (optional).
        generated_at:   ISO 8601 timestamp; None = use current UTC time.
        resolve_remote_revisions: if True, call HF APIs to pin pretrained
                        asset and dataset commit SHAs.  Default is offline.
        target_repo:    deployment repo — dirty tree is a hard error.
        training_repo:  training repo — populates provenance commits.
        dataset_repos:  list of 'repo[@commit][:loader]' specs.
        loader_classes: parallel list of loader classes (prefer the
                        ':loader' syntax in dataset_repos).
        extra_skip_files: filenames to exclude from weight_integrity hashing.
        extra_skip_dirs:  top-level directory names to exclude from hashing.

    Returns:
        A fully populated ModelPassport (Phase 1 fields filled, Phase 2
        fields left null).

    Raises:
        ValueError: dirty target repo, etc.
        FileNotFoundError: config or stats path doesn't exist on disk.
    """
    ckpt = checkpoint_dir.resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    config = json.loads(config_path.read_text())

    stats: Optional[Dict[str, Any]] = None
    if stats_path is not None:
        if not stats_path.exists():
            raise FileNotFoundError(f"stats not found: {stats_path}")
        stats = json.loads(stats_path.read_text())

    training_log = _find_training_log(ckpt)

    # -- provenance --
    prov = Provenance(
        run_log_path=training_log.name if training_log else None,
    )
    if training_log:
        wandb_url = _parse_wandb_url(training_log)
        if wandb_url:
            prov.run_log_path = training_log.name

    if training_repo:
        prov.training_repo = _git_remote_url(training_repo) or str(training_repo)
        prov.training_repo_commit = _git_head(training_repo)

    if target_repo:
        dirty, dirty_files = _git_is_dirty(target_repo)
        if dirty:
            raise ValueError(
                f"target repo {target_repo} has uncommitted changes:\n"
                f"{dirty_files}\n"
                "Refusing to generate passport against dirty deployment code."
            )
        prov.deployment_repo = _git_remote_url(target_repo) or str(target_repo)
        prov.deployment_repo_commit = _git_head(target_repo)

    # -- input_contract --
    images = _extract_images(config)
    state = _extract_state(config, stats, stats_path, ckpt)
    actions = _extract_actions(config, stats, stats_path, ckpt)
    language = _extract_language(config)
    temporal = _extract_temporal(config)

    training_datasets = _extract_training_datasets(
        config, dataset_repos, loader_classes,
        resolve_remote=resolve_remote_revisions,
    )

    input_contract = InputContract(
        images=images,
        state=state,
        actions=actions,
        language=language,
        temporal=temporal,
        training_datasets=training_datasets,
    )

    # -- output_spec --
    output_spec = OutputSpec(
        inference_parameters=_extract_inference_params(config),
    )

    # -- weight_integrity --
    files_to_hash = _hashable_files(
        ckpt,
        extra_skip_files=extra_skip_files,
        extra_skip_dirs=extra_skip_dirs,
    )
    weight_files = []
    for f in files_to_hash:
        weight_files.append(WeightFileEntry(
            path=str(f.relative_to(ckpt)),
            sha256=_sha256(f),
            size_bytes=f.stat().st_size,
        ))
    weight_integrity = WeightIntegrity(weight_files=weight_files)

    # -- model_internals (partial: pretrained provenance from config) --
    pretrained = _extract_pretrained(config, resolve_remote=resolve_remote_revisions)
    forward_keys = [k for k in sorted(config.get("input_features", {}).keys())]
    if language:
        forward_keys.append("task")
    sample_input_shapes = {}
    for key, feat in sorted(config.get("input_features", {}).items()):
        shape = feat.get("shape", [])
        if shape:
            n_obs = config.get("n_obs_steps", 1)
            sample_input_shapes[key] = [1, n_obs] + shape

    action_dim = config.get("output_features", {}).get("action", {}).get("shape", [None])[0]
    horizon = config.get("horizon")
    sample_output_shapes = {}
    if action_dim and horizon:
        sample_output_shapes["action"] = [1, horizon, action_dim]

    model_internals = ModelInternals(
        pretrained_provenance=pretrained,
        forward_graph=ForwardGraph(
            expected_input_keys=forward_keys,
            sample_input_shapes=sample_input_shapes,
            sample_output_shapes=sample_output_shapes,
        ),
    )

    passport = ModelPassport(
        generated_by="generate-passport",
        generated_at=generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        stack=config.get("policy_type", "unknown"),
        input_contract=input_contract,
        model_internals=model_internals,
        output_spec=output_spec,
        weight_integrity=weight_integrity,
        provenance=prov,
    )

    return passport


# ── CLI wrapper ─────────────────────────────────────────────────────────
#
# main() is the thin CLI shell around generate_passport().  It owns:
#   - argparse setup (--config is required)
#   - exception → exit-code translation
#   - JSON serialization and summary output
#
# All business logic lives in generate_passport() so it can be called
# from tests, other CLIs (materialize, merge), and agent toolchains
# without subprocess overhead.


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate-passport",
        description="Generate MODEL_PASSPORT.json from checkpoint files (Phase 1 static extraction)",
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument(
        "--config", type=Path, required=True, dest="config_path",
        help="path to config.json",
    )
    parser.add_argument(
        "--stats", type=Path, default=None, dest="stats_path",
        help="path to norm stats JSON",
    )
    parser.add_argument(
        "--generated-at", type=str, default=None,
        help="ISO 8601 timestamp; omit for current UTC time",
    )
    parser.add_argument(
        "--resolve-remote-revisions", action="store_true", default=False,
        help="call HuggingFace APIs to resolve pretrained asset / dataset commits",
    )
    parser.add_argument(
        "--target-repo", type=Path, default=None,
        help="deployment target repo; populates provenance.deployment_repo_commit",
    )
    parser.add_argument(
        "--training-repo", type=Path, default=None,
        help="training model repo; populates provenance.training_repo_commit",
    )
    parser.add_argument(
        "--dataset", type=str, action="append", default=None, dest="datasets",
        help="dataset as repo[@commit]:loader_class (repeatable), e.g. "
             "user/dataset@abc123:lerobot.datasets.LeRobotDataset",
    )
    parser.add_argument(
        "--skip-dir", type=str, action="append", default=None, dest="skip_dirs",
        help="top-level directory to exclude from weight_integrity hashing "
             "(repeatable, e.g. --skip-dir retain)",
    )
    parser.add_argument(
        "--skip-file", type=str, action="append", default=None, dest="skip_files",
        help="filename to exclude from weight_integrity hashing "
             "(repeatable, e.g. --skip-file wandb_run.json)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output path (default: <checkpoint_dir>/MODEL_PASSPORT.json)",
    )
    args = parser.parse_args()

    dataset_repos: List[str] = []
    loader_classes: List[str] = []
    for ds in (args.datasets or []):
        if ":" in ds:
            repo, loader = ds.split(":", 1)
            dataset_repos.append(repo)
            loader_classes.append(loader)
        else:
            dataset_repos.append(ds)

    try:
        passport = generate_passport(
            args.checkpoint_dir,
            config_path=args.config_path,
            stats_path=args.stats_path,
            generated_at=args.generated_at,
            resolve_remote_revisions=args.resolve_remote_revisions,
            target_repo=args.target_repo,
            training_repo=args.training_repo,
            dataset_repos=dataset_repos or None,
            loader_classes=loader_classes or None,
            extra_skip_files=args.skip_files,
            extra_skip_dirs=args.skip_dirs,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or (args.checkpoint_dir / "MODEL_PASSPORT.json")
    out_path.write_text(json.dumps(passport.to_dict(), indent=2) + "\n")
    print(f"Wrote {out_path}")

    # Summary
    d = passport.to_dict()
    n_images = len(d.get("input_contract", {}).get("images", []))
    n_weight = len(d.get("weight_integrity", {}).get("weight_files", []))
    n_datasets = len(d.get("input_contract", {}).get("training_datasets", []))
    prov = d.get("provenance", {})
    print(f"  images: {n_images}")
    print(f"  weight files hashed: {n_weight}")
    print(f"  training datasets: {n_datasets}")
    print(f"  deployment_repo_commit: {prov.get('deployment_repo_commit', 'not set')}")
    print(f"  training_repo_commit: {prov.get('training_repo_commit', 'not set')}")
    print()
    print("Not filled by static generation:")
    print("  runtime extractor:   model_identity, model_internals, smoke_results")
    print("  structured metadata: images[].physical_mounting, camera_serial")
    print("  later workflow:      transform_pipeline, known_issues, reference_test_vector")


if __name__ == "__main__":
    main()
