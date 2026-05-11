"""
Tests for passport assembly from seed + checkpoint files.

Verifies that assemble_passport correctly:
  - merges seed sections into the final passport
  - computes weight_integrity from checkpoint files
  - populates assembler-owned metadata
  - rejects invalid seeds
  - preserves the static generator path (generate-passport is untouched)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from checkpoint_passport.assemble_passport import assemble_passport
from checkpoint_passport.passport_seed import InvalidSeedError
from checkpoint_passport.schema import SCHEMA_VERSION


# ── Fixtures ─────────────────────────────────────────────────────────────


def _minimal_seed(**overrides) -> dict:
    """Minimal valid seed with extractor metadata."""
    seed = {
        "extractor": {
            "extractor_name": "openpi",
            "extractor_version": "0.1.0",
        },
        "stack": "openpi",
    }
    seed.update(overrides)
    return seed


def _make_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal checkpoint directory with a dummy weight file."""
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "weights.bin").write_bytes(b"fake-weight-data-1234")
    return ckpt


# ── 1. Basic assembly ────────────────────────────────────────────────────


def test_assembles_minimal_seed(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    seed = _minimal_seed()

    passport = assemble_passport(
        ckpt, seed, generated_at="2026-05-11T10:00:00Z",
    )

    assert passport["schema_version"] == SCHEMA_VERSION
    assert passport["generated_by"] == "assemble-passport"
    assert passport["generated_at"] == "2026-05-11T10:00:00Z"
    assert passport["stack"] == "openpi"
    assert "weight_integrity" in passport


def test_default_generated_at_is_utc(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    seed = _minimal_seed()

    passport = assemble_passport(ckpt, seed)

    assert passport["generated_at"].endswith("Z")
    assert "T" in passport["generated_at"]


# ── 2. Seed sections are merged ──────────────────────────────────────────


def test_seed_sections_carried_through(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    seed = _minimal_seed(
        input_contract={"actions": {"total_dim": 7, "horizon": 50}},
        output_spec={"actions": {"horizon": 50}},
        model_identity={"class_name": "pi05"},
        model_internals={"forward_graph": {"expected_input_keys": ["obs"]}},
        transform_pipeline=[{"order": 0, "name": "resize"}],
    )

    passport = assemble_passport(
        ckpt, seed, generated_at="2026-05-11T10:00:00Z",
    )

    assert passport["input_contract"]["actions"]["total_dim"] == 7
    assert passport["output_spec"]["actions"]["horizon"] == 50
    assert passport["model_identity"]["class_name"] == "pi05"
    assert passport["model_internals"]["forward_graph"]["expected_input_keys"] == ["obs"]
    assert passport["transform_pipeline"][0]["name"] == "resize"


def test_extractor_metadata_not_in_passport(tmp_path):
    """The 'extractor' block is seed metadata, not a passport section."""
    ckpt = _make_checkpoint(tmp_path)
    seed = _minimal_seed()

    passport = assemble_passport(
        ckpt, seed, generated_at="2026-05-11T10:00:00Z",
    )

    assert "extractor" not in passport


# ── 3. Weight integrity ──────────────────────────────────────────────────


def test_weight_integrity_hashes_checkpoint_files(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    (ckpt / "config.json").write_text('{"key": "value"}')
    seed = _minimal_seed()

    passport = assemble_passport(
        ckpt, seed, generated_at="2026-05-11T10:00:00Z",
    )

    wi = passport["weight_integrity"]
    files = wi["weight_files"]
    paths = [f["path"] for f in files]
    assert "weights.bin" in paths
    assert "config.json" in paths

    for f in files:
        assert "sha256" in f
        assert "size_bytes" in f
        assert len(f["sha256"]) == 64


def test_weight_integrity_excludes_passport_and_signoff(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    (ckpt / "MODEL_PASSPORT.json").write_text("{}")
    (ckpt / "SIGNOFF.json").write_text("{}")
    (ckpt / "README.md").write_text("# Readme")
    (ckpt / "TRAINING_LOG.md").write_text("# Log")
    seed = _minimal_seed()

    passport = assemble_passport(
        ckpt, seed, generated_at="2026-05-11T10:00:00Z",
    )

    paths = [f["path"] for f in passport["weight_integrity"]["weight_files"]]
    assert "MODEL_PASSPORT.json" not in paths
    assert "SIGNOFF.json" not in paths
    assert "README.md" not in paths
    assert "TRAINING_LOG.md" not in paths


# ── 4. Provenance ────────────────────────────────────────────────────────


def test_training_log_populates_provenance(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    (ckpt / "TRAINING_LOG.md").write_text("# Training log\nhttps://wandb.ai/run/123")
    seed = _minimal_seed()

    passport = assemble_passport(
        ckpt, seed, generated_at="2026-05-11T10:00:00Z",
    )

    assert passport["provenance"]["run_log_path"] == "TRAINING_LOG.md"


def test_no_training_log_no_provenance_crash(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    seed = _minimal_seed()

    passport = assemble_passport(
        ckpt, seed, generated_at="2026-05-11T10:00:00Z",
    )

    # No provenance section when there's nothing to populate
    assert "provenance" not in passport or passport.get("provenance") is None or passport.get("provenance") == {}


# ── 5. Determinism ───────────────────────────────────────────────────────


def test_deterministic_with_pinned_timestamp(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    seed = _minimal_seed(
        input_contract={"actions": {"total_dim": 7}},
    )

    p1 = assemble_passport(ckpt, seed, generated_at="2026-05-11T10:00:00Z")
    p2 = assemble_passport(ckpt, seed, generated_at="2026-05-11T10:00:00Z")

    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)


# ── 6. Invalid seed rejection ────────────────────────────────────────────


def test_rejects_seed_with_assembler_owned_keys(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    seed = _minimal_seed()
    seed["schema_version"] = "0.2"

    with pytest.raises(InvalidSeedError, match="assembler-owned"):
        assemble_passport(ckpt, seed)


def test_rejects_seed_with_unknown_keys(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    seed = _minimal_seed()
    seed["bogus_section"] = {"data": True}

    with pytest.raises(InvalidSeedError, match="unknown top-level keys"):
        assemble_passport(ckpt, seed)


def test_rejects_seed_missing_extractor_metadata(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    seed = {"stack": "openpi"}

    with pytest.raises(InvalidSeedError, match="extractor"):
        assemble_passport(ckpt, seed)


# ── 7. Checkpoint dir validation ─────────────────────────────────────────


def test_missing_checkpoint_dir_raises(tmp_path):
    seed = _minimal_seed()
    with pytest.raises(FileNotFoundError, match="not found"):
        assemble_passport(tmp_path / "nonexistent", seed)


# ── 8. Pruning ───────────────────────────────────────────────────────────


def test_null_and_empty_fields_are_pruned(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    seed = _minimal_seed(
        input_contract={
            "actions": {"total_dim": 7},
            "images": [],
            "state": None,
        },
    )

    passport = assemble_passport(
        ckpt, seed, generated_at="2026-05-11T10:00:00Z",
    )

    ic = passport.get("input_contract", {})
    assert "images" not in ic
    assert "state" not in ic
