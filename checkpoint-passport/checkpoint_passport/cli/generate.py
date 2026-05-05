"""
CLI entry point: generate MODEL_PASSPORT.json from checkpoint files on disk.

Phase 1 (static) extraction only — no torch, no GPU, no model import.
Reads config.json, norm stats, TRAINING_LOG.md, and weight file headers
to populate every field that can be determined deterministically.

Fields that require model loading (Phase 2) or human judgment are left
null so an agent or a future --phase2 script can fill them in.

Usage:
    generate-passport /path/to/checkpoint
    generate-passport /path/to/checkpoint --target-repo /path/to/deploy/repo
    generate-passport /path/to/checkpoint --training-repo /path/to/model/repo
    generate-passport /path/to/checkpoint --dataset-repo user/dataset --loader-class lerobot.datasets.LeRobotDataset

Exit code 0 = passport written successfully.
Exit code 1 = error (missing config, dirty target repo, etc.).
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
    GeneratedBy,
    InputContract,
    ImageSpec,
    StateSpec,
    ActionSpec,
    DeltaSpec,
    LanguageSpec,
    TemporalSpec,
    TrainingDatasetSpec,
    ModelIdentity,
    RuntimeConstraints,
    ModelInternals,
    PretrainedAsset,
    ForwardGraph,
    OutputSpec,
    OutputActions,
    InferenceParameters,
    PostProcessing,
    WeightIntegrity,
    WeightFileEntry,
    Provenance,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(repo: Path) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _git_remote_url(repo: Path) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _git_is_dirty(repo: Path) -> Tuple[bool, str]:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        output = r.stdout.strip()
        return bool(output), output
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, ""


def _find_config(ckpt: Path) -> Optional[Path]:
    for p in sorted(ckpt.rglob("config.json")):
        if ".cache" not in p.parts:
            return p
    return None


def _find_stats(ckpt: Path) -> Optional[Path]:
    for p in sorted(ckpt.rglob("*stats*.json")):
        if ".cache" not in p.parts:
            return p
    return None


def _find_training_log(ckpt: Path) -> Optional[Path]:
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


def _hashable_files(ckpt: Path) -> List[Path]:
    """Find all files that should be hashed for weight_integrity.

    Includes weight files, config.json, and norm stats — everything
    needed for inference. Excludes README, TRAINING_LOG, passport/signoff,
    and hidden files.
    """
    skip = {"MODEL_PASSPORT.json", "SIGNOFF.json", "README.md", "TRAINING_LOG.md"}
    files = []
    for p in sorted(ckpt.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.name in skip:
            continue
        files.append(p)
    return files


def _extract_images(config: Dict[str, Any]) -> List[ImageSpec]:
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
            norm_info = {
                "type": norm_type,
                "scope": "VISUAL",
            }
            if backbone:
                norm_info["applied_by"] = f"observation_encoder.vision ({backbone})"

        images.append(ImageSpec(
            key=key,
            raw_shape=shape if shape else None,
            encoder_resize=resize,
            crop={"shape": crop} if crop else None,
            color_order="RGB",
            channel_layout="CHW" if shape and len(shape) == 3 and shape[0] <= 4 else None,
            dtype="float32",
            value_range=[0.0, 1.0],
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
        # Add fingerprint
        if stats_path:
            fingerprint: Dict[str, Any] = {"file_sha256": _sha256(stats_path)}
            state_stats = stats.get("state", {})
            q02 = state_stats.get("q02")
            q98 = state_stats.get("q98")
            if q02:
                fingerprint["per_dim_q02_at_t0"] = q02[0] if isinstance(q02[0], list) else q02
            if q98:
                fingerprint["per_dim_q98_at_t0"] = q98[0] if isinstance(q98[0], list) else q98
            norm_info["stats_fingerprint"] = fingerprint

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

    # norm_mask and delta_dims from stats
    norm_mask = None
    delta_dims = None
    if stats:
        nm = stats.get("norm_mask")
        if nm:
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
        if stats_path:
            fingerprint: Dict[str, Any] = {"file_sha256": _sha256(stats_path)}
            if q02:
                fingerprint["per_dim_q02_at_t0"] = q02[0] if isinstance(q02[0], list) else q02
            q98 = action_stats.get("q98")
            if q98:
                fingerprint["per_dim_q98_at_t0"] = q98[0] if isinstance(q98[0], list) else q98
            norm_info["stats_fingerprint"] = fingerprint

    return ActionSpec(
        total_dim=total_dim,
        horizon=horizon,
        sub_keys=sub_keys,
        norm_mask=norm_mask,
        delta_dims=delta_dims,
        normalization=norm_info,
    )


def _extract_language(config: Dict[str, Any]) -> Optional[LanguageSpec]:
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
    obj = config.get("objective", {})
    if not obj:
        return None
    extra = {}
    for k in ["num_train_timesteps", "beta_schedule"]:
        if k in obj:
            extra[k] = obj[k]
    clip_val = config.get("ramen_clip_value")
    if clip_val is not None:
        extra["ramen_clip_value"] = clip_val

    return InferenceParameters(
        type=obj.get("type"),
        num_inference_steps=obj.get("num_inference_steps"),
        scheduler=obj.get("noise_scheduler_type"),
        prediction_type=obj.get("prediction_type"),
        clip_sample=obj.get("clip_sample"),
        clip_sample_range=obj.get("clip_sample_range"),
        chunk_aggregation="first_n_action_steps",
        chunks_executed_per_inference=config.get("n_action_steps"),
        extra=extra,
    )


def _extract_pretrained(config: Dict[str, Any]) -> List[PretrainedAsset]:
    obs_encoder = config.get("observation_encoder", {})
    assets = []
    vision = obs_encoder.get("vision", {})
    if vision:
        backbone = vision.get("backbone")
        assets.append(PretrainedAsset(
            submodule="observation_encoder.vision",
            source="timm" if backbone else None,
            timm_string=backbone,
            frozen_in_training=False,
            lr_multiplier=vision.get("lr_multiplier"),
        ))
    text = obs_encoder.get("text", {})
    if text and text.get("model"):
        assets.append(PretrainedAsset(
            submodule="observation_encoder.text",
            source="huggingface",
            hf_revision=None,
        ))
    return assets


def generate_passport(
    checkpoint_dir: Path,
    *,
    target_repo: Optional[Path] = None,
    training_repo: Optional[Path] = None,
    dataset_repo: Optional[str] = None,
    loader_class: Optional[str] = None,
) -> ModelPassport:
    ckpt = checkpoint_dir.resolve()

    config_path = _find_config(ckpt)
    if config_path is None:
        print("error: no config.json found in checkpoint", file=sys.stderr)
        sys.exit(1)
    config = json.loads(config_path.read_text())

    stats_path = _find_stats(ckpt)
    stats = json.loads(stats_path.read_text()) if stats_path else None

    training_log = _find_training_log(ckpt)

    # -- provenance --
    prov = Provenance(
        run_log_path=training_log.name if training_log else None,
        merged_config_sha256=_sha256(config_path),
        config_snapshot_path=str(config_path.relative_to(ckpt)),
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
            print(
                f"error: target repo {target_repo} has uncommitted changes:\n"
                f"{dirty_files}\n"
                "Refusing to generate passport against dirty deployment code.",
                file=sys.stderr,
            )
            sys.exit(1)
        prov.deployment_repo = _git_remote_url(target_repo) or str(target_repo)
        prov.deployment_repo_commit = _git_head(target_repo)

    # -- input_contract --
    images = _extract_images(config)
    state = _extract_state(config, stats, stats_path, ckpt)
    actions = _extract_actions(config, stats, stats_path, ckpt)
    language = _extract_language(config)
    temporal = _extract_temporal(config)

    training_datasets = []
    if dataset_repo:
        training_datasets.append(TrainingDatasetSpec(
            repo=dataset_repo,
            loader_class=loader_class,
            contributes_to_norm_stats=True,
        ))

    input_contract = InputContract(
        images=images,
        state=state,
        actions=actions,
        language=language,
        temporal=temporal,
        training_datasets=training_datasets,
    )

    # -- output_spec --
    horizon = config.get("horizon")
    output_spec = OutputSpec(
        actions=OutputActions(
            layout="mirrors input_contract.actions",
            sub_keys="see input_contract.actions.sub_keys",
            horizon=horizon,
            control_rate_hz=config.get("control_rate_hz"),
        ),
        inference_parameters=_extract_inference_params(config),
        post_processing=PostProcessing(unnormalize=True),
    )

    # -- weight_integrity --
    files_to_hash = _hashable_files(ckpt)
    weight_files = []
    for f in files_to_hash:
        weight_files.append(WeightFileEntry(
            path=str(f.relative_to(ckpt)),
            sha256=_sha256(f),
            size_bytes=f.stat().st_size,
        ))
    weight_integrity = WeightIntegrity(weight_files=weight_files)

    # -- model_internals (partial: pretrained provenance from config) --
    pretrained = _extract_pretrained(config)
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

    # -- runtime_constraints from config --
    runtime = RuntimeConstraints()

    passport = ModelPassport(
        generated_by=GeneratedBy(
            tool="generate-passport",
            version=SCHEMA_VERSION,
        ),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        stack=config.get("policy_type", "unknown"),
        input_contract=input_contract,
        model_identity=ModelIdentity(runtime_constraints=runtime),
        model_internals=model_internals,
        output_spec=output_spec,
        weight_integrity=weight_integrity,
        provenance=prov,
    )

    return passport


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate-passport",
        description="Generate MODEL_PASSPORT.json from checkpoint files (Phase 1 static extraction)",
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument(
        "--target-repo", type=Path, default=None,
        help="deployment target repo; populates provenance.deployment_repo_commit",
    )
    parser.add_argument(
        "--training-repo", type=Path, default=None,
        help="training model repo; populates provenance.training_repo_commit",
    )
    parser.add_argument(
        "--dataset-repo", type=str, default=None,
        help="HuggingFace dataset repo id (e.g. user/dataset_name)",
    )
    parser.add_argument(
        "--loader-class", type=str, default=None,
        help="dataset loader class (e.g. lerobot.datasets.LeRobotDataset)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output path (default: <checkpoint_dir>/MODEL_PASSPORT.json)",
    )
    args = parser.parse_args()

    passport = generate_passport(
        args.checkpoint_dir,
        target_repo=args.target_repo,
        training_repo=args.training_repo,
        dataset_repo=args.dataset_repo,
        loader_class=args.loader_class,
    )

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
    print("Phase 2 fields left null (needs model load):")
    print("  model_identity.class_name, class_module, library_versions")
    print("  model_internals (parameters, state_dict, numerical_health)")
    print("  smoke_results, reference_test_vector, norm_round_trip_results")
    print()
    print("Judgment fields left null (needs operator):")
    print("  images[].physical_mounting, camera_serial")
    print("  transform_pipeline[], known_issues[]")


if __name__ == "__main__":
    main()
