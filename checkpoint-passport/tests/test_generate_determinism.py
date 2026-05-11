"""
Tests for deterministic passport generation.

Verifies that generate_passport() with explicit inputs produces identical
output on repeated calls, that HF remote resolution is gated behind an
opt-in flag, and that pinned dataset specs populate commit hashes without
network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from checkpoint_passport.cli.generate import generate_passport


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def minimal_checkpoint(tmp_path: Path) -> Path:
    """A checkpoint dir with a config.json and a tiny placeholder weight file.

    This is the smallest valid input for generate_passport with explicit
    paths — no filesystem discovery required.
    """
    config = {
        "policy_type": "diffusion",
        "input_features": {
            "observation.images.front": {"type": "VISUAL", "shape": [3, 96, 96]},
            "observation.state": {"type": "STATE", "shape": [7]},
        },
        "output_features": {
            "action": {"shape": [7]},
        },
        "observation_encoder": {
            "vision": {"resize_shape": [96, 96], "backbone": "resnet18"},
        },
        "normalization_mapping": {"VISUAL": "imagenet", "STATE": "min_max", "ACTION": "min_max"},
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"\x00" * 64)

    return tmp_path


# ── 1. Determinism: same explicit inputs → identical dicts ──────────────


def test_repeated_calls_produce_identical_output(minimal_checkpoint: Path):
    """Two calls with the same explicit inputs must return byte-identical
    JSON, proving no hidden non-determinism (timestamps, random UUIDs, etc.)."""
    fixed_time = "2025-01-01T00:00:00Z"

    result_a = generate_passport(
        minimal_checkpoint,
        config_path=minimal_checkpoint / "config.json",
        generated_at=fixed_time,
        resolve_remote_revisions=False,
    )
    result_b = generate_passport(
        minimal_checkpoint,
        config_path=minimal_checkpoint / "config.json",
        generated_at=fixed_time,
        resolve_remote_revisions=False,
    )

    assert result_a.to_dict() == result_b.to_dict()


def test_generated_at_is_honoured(minimal_checkpoint: Path):
    """When generated_at is provided, the passport uses it verbatim
    instead of calling datetime.now()."""
    fixed_time = "2025-06-15T12:00:00Z"

    result = generate_passport(
        minimal_checkpoint,
        config_path=minimal_checkpoint / "config.json",
        generated_at=fixed_time,
        resolve_remote_revisions=False,
    )

    assert result.generated_at == fixed_time


# ── 2. HF resolution gated behind flag ─────────────────────────────────


def test_hf_resolvers_not_called_when_disabled(minimal_checkpoint: Path):
    """With resolve_remote_revisions=False, no HuggingFace API is touched.
    Monkeypatch the resolvers to explode if called."""
    with patch(
        "checkpoint_passport.cli.generate._resolve_hf_revision",
        side_effect=AssertionError("_resolve_hf_revision must not be called"),
    ), patch(
        "checkpoint_passport.cli.generate._resolve_hf_dataset_revision",
        side_effect=AssertionError("_resolve_hf_dataset_revision must not be called"),
    ):
        generate_passport(
            minimal_checkpoint,
            config_path=minimal_checkpoint / "config.json",
            generated_at="2025-01-01T00:00:00Z",
            resolve_remote_revisions=False,
        )


# ── 3. Pinned dataset spec populates commit without network ─────────────


def test_pinned_dataset_spec_populates_commit(minimal_checkpoint: Path):
    """A dataset spec like 'org/dataset@abc123:loader.Class' should parse
    the commit from the '@' syntax and not call the HF resolver."""
    with patch(
        "checkpoint_passport.cli.generate._resolve_hf_dataset_revision",
        side_effect=AssertionError("must not call HF for pinned dataset"),
    ):
        result = generate_passport(
            minimal_checkpoint,
            config_path=minimal_checkpoint / "config.json",
            generated_at="2025-01-01T00:00:00Z",
            resolve_remote_revisions=False,
            dataset_repos=["example-org/example-dataset@abc123:lerobot.datasets.LeRobotDataset"],
        )

    datasets = result.input_contract.training_datasets
    assert len(datasets) == 1
    assert datasets[0].repo == "example-org/example-dataset"
    assert datasets[0].commit == "abc123"
    assert datasets[0].loader_class == "lerobot.datasets.LeRobotDataset"


# ── 4. Library API raises, never calls sys.exit ─────────────────────────


def test_missing_config_raises_not_exits(tmp_path: Path):
    """When config_path points to a nonexistent file, the library
    function should raise FileNotFoundError, not call sys.exit()."""
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"\x00" * 64)

    with pytest.raises(FileNotFoundError):
        generate_passport(
            tmp_path,
            config_path=tmp_path / "config.json",
            generated_at="2025-01-01T00:00:00Z",
            resolve_remote_revisions=False,
        )
