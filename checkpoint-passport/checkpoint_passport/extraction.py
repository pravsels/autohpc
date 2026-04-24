"""
What a checkpoint declares about itself, extracted from files on disk.

Framework-agnostic: reads safetensors headers, JSON configs, and norm stats
files directly. No model loading involved.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


# ── File discovery ──────────────────────────────────────────────────────


def read_safetensors_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read tensor names, shapes, and dtypes from a safetensors file header.

    Safetensors stores a JSON header in the first N bytes: 8-byte little-endian
    size prefix, then the header JSON.  Each key is a tensor name mapping to
    {"dtype": ..., "shape": [...], "data_offsets": [start, end]}.
    """
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    # Drop the __metadata__ key (arbitrary string k/v, not a tensor)
    return {
        k: v for k, v in header.items()
        if k != "__metadata__" and isinstance(v, dict)
    }


def find_weight_files(checkpoint_dir: Path) -> List[Path]:
    """Recursively find .safetensors and .bin weight files."""
    patterns = ["**/*.safetensors", "**/*.bin"]
    files = []
    for p in patterns:
        files.extend(checkpoint_dir.glob(p))
    return sorted(files)


def find_config_files(checkpoint_dir: Path) -> Dict[str, Path]:
    """Recursively find config JSONs, YAMLs, and any *_stats*.json files."""
    found = {}
    config_names = {
        "config.json", "config.yaml",
        "generation_config.json", "preprocessor_config.json",
        "tokenizer_config.json",
    }
    for p in checkpoint_dir.rglob("*"):
        if not p.is_file():
            continue
        if ".cache" in p.parts:
            continue
        if p.name in config_names:
            key = str(p.relative_to(checkpoint_dir))
            found[key] = p
        # Catch norm_stats.json, ramen_stats.json, delta_norm_stats.json, etc.
        elif "_stats" in p.name and p.suffix == ".json":
            key = f"stats:{p.relative_to(checkpoint_dir)}"
            found[key] = p
    return found


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


# ── Weight manifest ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class WeightManifest:
    """Tensor-level metadata extracted from weight files (no actual weights loaded)."""

    # {tensor_name: {"dtype": str, "shape": list[int], "data_offsets": [int,int]}}
    tensors: Dict[str, Dict[str, Any]]
    # Filenames that were found (e.g. ["model.safetensors"])
    files: List[str]

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: Path) -> "WeightManifest":
        weight_files = find_weight_files(checkpoint_dir)
        all_tensors: Dict[str, Dict[str, Any]] = {}
        for wf in weight_files:
            # .bin files (pickle) don't have a readable header — skip
            if wf.suffix == ".safetensors":
                all_tensors.update(read_safetensors_metadata(wf))
        return cls(
            tensors=all_tensors,
            files=[f.name for f in weight_files],
        )

    @property
    def parameter_names(self) -> List[str]:
        return sorted(self.tensors.keys())

    def shape_of(self, name: str) -> Optional[List[int]]:
        entry = self.tensors.get(name)
        return entry.get("shape") if entry else None


# ── Checkpoint extraction ────────────────────────────────────────────────


@dataclass(frozen=True)
class CheckpointExtraction:
    """Everything found in a checkpoint directory, structured for cross-checking."""

    path: Path
    config: Dict[str, Any]
    weights: WeightManifest
    norm_stats: Dict[str, Any]
    config_files_found: Dict[str, str]

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: str | Path) -> "CheckpointExtraction":
        path = Path(checkpoint_dir)

        config_files = find_config_files(path)

        # Load the first config.json or config.yaml found (deterministic order)
        config: Dict[str, Any] = {}
        config_candidates = [
            k for k in sorted(config_files.keys())
            if k.endswith("config.json") or k.endswith("config.yaml")
        ]
        for candidate in config_candidates:
            cp = config_files[candidate]
            if cp.suffix == ".json":
                config = load_json(cp)
            else:
                import yaml
                with open(cp) as f:
                    config = yaml.safe_load(f) or {}
            break

        weights = WeightManifest.from_checkpoint(path)

        norm_stats: Dict[str, Any] = {}
        for key, ns_path in config_files.items():
            if key.startswith("stats:"):
                norm_stats[key] = load_json(ns_path)

        return cls(
            path=path,
            config=config,
            weights=weights,
            norm_stats=norm_stats,
            config_files_found={k: str(v) for k, v in config_files.items()},
        )

    # ── Accessors: interpret raw config fields ──────────────────────────
    # Return None when the field isn't present -- itself a reportable
    # finding (the checks decide what to make of it).

    @property
    def declared_architecture(self) -> Optional[str]:
        return (
            self.config.get("_target_")
            or self.config.get("policy_type")
            or self.config.get("model_type")
            or (self.config.get("architectures", [None])[0])
        )

    @property
    def declared_action_dim(self) -> Optional[int]:
        # LeRobot convention: output_features.action.shape = [dim]
        shape = (
            self.config
            .get("output_features", {})
            .get("action", {})
            .get("shape", [])
        )
        return shape[0] if shape else None

    @property
    def declared_action_horizon(self) -> Optional[int]:
        # "chunk_size" (LeRobot ACT), "action_horizon" (OpenPI)
        return self.config.get("chunk_size") or self.config.get("action_horizon")

    @property
    def declared_action_mode(self) -> Literal["delta", "absolute", "unknown"]:
        if self.config.get("use_delta_actions"):
            return "delta"
        if "use_delta_actions" in self.config:
            return "absolute"
        return "unknown"

    @property
    def declared_image_keys(self) -> List[str]:
        return sorted(self.config.get("image_features", {}).keys())

    @property
    def declared_state_dim(self) -> Optional[int]:
        feat = self.config.get("robot_state_feature", {})
        if isinstance(feat, dict):
            shape = feat.get("shape", [])
            return shape[0] if shape else None
        return None

    @property
    def has_reward_head(self) -> bool:
        out = self.config.get("output_features", {})
        return "reward" in out or "expected_value" in out
