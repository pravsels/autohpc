"""
Assemble a MODEL_PASSPORT.json from a passport seed + checkpoint files.

This is the second step for runtime-extracted checkpoints (e.g. OpenPI):
  1. extract-passport-seed openpi → PASSPORT_SEED.json
  2. assemble-passport → MODEL_PASSPORT.json

The assembler owns: schema_version, generated_at, generated_by,
weight_integrity, provenance.  The seed owns: stack, input_contract,
output_spec, model_identity, model_internals, transform_pipeline.

For config-bearing checkpoints (LeRobot/RAMEN), generate-passport writes
MODEL_PASSPORT.json directly — it is NOT refactored through this path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from checkpoint_passport.passport_seed import validate_seed, ALLOWED_SEED_SECTIONS
from checkpoint_passport.schema import SCHEMA_VERSION, _dataclass_to_dict, _prune

from checkpoint_passport.cli.generate import (
    _sha256,
    _hashable_files,
    _find_training_log,
    _parse_wandb_url,
    _git_head,
    _git_remote_url,
    _owning_git_repo,
)


def assemble_passport(
    checkpoint_dir: Path,
    seed: Dict[str, Any],
    *,
    generated_at: Optional[str] = None,
    target_repo: Optional[Path] = None,
    training_repo: Optional[Path] = None,
    extra_skip_files: Optional[list[str]] = None,
    extra_skip_dirs: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Build a complete passport dict from a validated seed + checkpoint files.

    Args:
        checkpoint_dir: root of the checkpoint tree.
        seed:           validated passport seed dict.
        generated_at:   ISO 8601 timestamp; None = use current UTC time.
        target_repo:    deployment repo — optional debug provenance.
        training_repo:  training repo — populates provenance commits.
        extra_skip_files: filenames to exclude from weight_integrity hashing.
        extra_skip_dirs:  top-level directory names to exclude from hashing.

    Returns:
        Complete passport dict ready for JSON serialization.

    Raises:
        ValueError: invalid seed, etc.
        FileNotFoundError: checkpoint_dir doesn't exist.
    """
    ckpt = checkpoint_dir.resolve()
    if not ckpt.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    validate_seed(seed)

    passport: Dict[str, Any] = {}

    passport["schema_version"] = SCHEMA_VERSION
    passport["generated_by"] = "assemble-passport"
    passport["generated_at"] = generated_at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    for section in ALLOWED_SEED_SECTIONS:
        if section in seed:
            passport[section] = seed[section]

    files_to_hash = _hashable_files(
        ckpt,
        extra_skip_files=extra_skip_files,
        extra_skip_dirs=extra_skip_dirs,
    )
    weight_files = []
    for f in files_to_hash:
        weight_files.append({
            "path": str(f.relative_to(ckpt)),
            "sha256": _sha256(f),
            "size_bytes": f.stat().st_size,
        })
    passport["weight_integrity"] = {"weight_files": weight_files}

    provenance: Dict[str, Any] = {}
    training_log = _find_training_log(ckpt)
    if training_log:
        provenance["run_log_path"] = training_log.name

    if training_repo:
        provenance["training_repo"] = _git_remote_url(training_repo) or str(
            training_repo
        )
        commit = _git_head(training_repo)
        if commit:
            provenance["training_repo_commit"] = commit

    passport_repo = _owning_git_repo(Path(__file__))
    if passport_repo:
        provenance["passport_creation_repo"] = (
            _git_remote_url(passport_repo) or str(passport_repo)
        )
        commit = _git_head(passport_repo)
        if commit:
            provenance["passport_creation_repo_commit"] = commit

    if target_repo:
        provenance["deployment_repo"] = _git_remote_url(target_repo) or str(
            target_repo
        )
        commit = _git_head(target_repo)
        if commit:
            provenance["deployment_repo_commit"] = commit

    if provenance:
        passport["provenance"] = provenance

    return _prune(passport) or {}
