"""
Tests for passport seed validation.

Verifies that validate_seed enforces the boundary between extractor-owned
and assembler-owned sections, rejects unknown keys, and requires extractor
metadata.
"""

from __future__ import annotations

import pytest

from checkpoint_passport.passport_seed import (
    validate_seed,
    InvalidSeedError,
    ALLOWED_SEED_SECTIONS,
    ASSEMBLER_OWNED_SECTIONS,
)


def _valid_seed(**overrides) -> dict:
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


# ── 1. Valid seeds pass ──────────────────────────────────────────────────


def test_minimal_valid_seed_passes():
    validate_seed(_valid_seed())


def test_seed_with_all_allowed_sections_passes():
    seed = _valid_seed(
        input_contract={"actions": {"horizon": 50}},
        output_spec={"actions": {"horizon": 50}},
        model_identity={"class_name": "pi05"},
        model_internals={"forward_graph": {}},
        transform_pipeline=[{"order": 0, "name": "resize"}],
    )
    validate_seed(seed)


# ── 2. Assembler-owned sections are rejected ─────────────────────────────


@pytest.mark.parametrize("forbidden_key", sorted(ASSEMBLER_OWNED_SECTIONS))
def test_assembler_owned_sections_rejected(forbidden_key: str):
    seed = _valid_seed()
    seed[forbidden_key] = "some_value"
    with pytest.raises(InvalidSeedError, match="assembler-owned"):
        validate_seed(seed)


# ── 3. Unknown top-level keys are rejected ───────────────────────────────


def test_unknown_top_level_key_rejected():
    seed = _valid_seed()
    seed["bogus_section"] = {"data": "stuff"}
    with pytest.raises(InvalidSeedError, match="unknown top-level keys"):
        validate_seed(seed)


def test_multiple_unknown_keys_all_reported():
    seed = _valid_seed()
    seed["foo"] = 1
    seed["bar"] = 2
    with pytest.raises(InvalidSeedError) as exc_info:
        validate_seed(seed)
    assert "bar" in str(exc_info.value)
    assert "foo" in str(exc_info.value)


# ── 4. Extractor metadata is required ───────────────────────────────────


def test_missing_extractor_block_rejected():
    seed = {"stack": "openpi"}
    with pytest.raises(InvalidSeedError, match="extractor"):
        validate_seed(seed)


def test_extractor_missing_required_keys_rejected():
    seed = {
        "extractor": {"extractor_name": "openpi"},
        "stack": "openpi",
    }
    with pytest.raises(InvalidSeedError, match="extractor_version"):
        validate_seed(seed)


# ── 5. Multiple errors reported together ─────────────────────────────────


def test_multiple_errors_all_reported():
    seed = {
        "schema_version": "0.2",
        "bogus": "data",
    }
    with pytest.raises(InvalidSeedError) as exc_info:
        validate_seed(seed)
    msg = str(exc_info.value)
    assert "assembler-owned" in msg
    assert "unknown top-level keys" in msg
    assert "extractor" in msg
