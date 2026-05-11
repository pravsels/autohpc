"""
Tests for check-publish-ready and publish-checkpoint CLIs.

Verifies:
  - missing required files (README.md, TRAINING_LOG.md, MODEL_PASSPORT.json, SIGNOFF.json)
  - empty README/TRAINING_LOG
  - invalid passport JSON
  - missing/invalid signoff
  - --json output shape
  - multiple errors reported together
  - validation errors reported separately from packaging errors
  - upload refuses when check-publish-ready fails
  - download refuses success when validate-checkpoint --require-signoff fails
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from checkpoint_passport.cli.check_publish_ready import (
    check_publish_ready,
    PublishReadyResult,
)
from checkpoint_passport.cli.publish_checkpoint import (
    upload_checkpoint,
    download_checkpoint,
    PublishGateError,
    ValidationGateError,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_ready_checkpoint(tmp_path: Path) -> Path:
    """Create a checkpoint dir that passes all publish-ready checks."""
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "weights.bin").write_bytes(b"fake-weights")
    (ckpt / "README.md").write_text("# My Model\nSome description.")
    (ckpt / "TRAINING_LOG.md").write_text("# Training Log\nStep 1000: loss=0.1")
    (ckpt / "MODEL_PASSPORT.json").write_text(json.dumps({
        "schema_version": "0.2",
        "generated_by": "test",
        "generated_at": "2026-05-11T10:00:00Z",
    }))
    (ckpt / "SIGNOFF.json").write_text(json.dumps({
        "schema_version": "0.1",
        "signed_at": "2026-05-11T10:01:00Z",
        "verdict": "pass",
    }))
    return ckpt


# ── 1. Fully ready checkpoint ────────────────────────────────────────────


def test_ready_checkpoint_passes(tmp_path):
    ckpt = _make_ready_checkpoint(tmp_path)

    mock_result = MagicMock()
    mock_result.has_failures = False
    mock_result.observations = []

    with patch(
        "checkpoint_passport.cli.check_publish_ready.self_validate_passport",
        return_value=mock_result,
    ):
        result = check_publish_ready(ckpt)

    assert result.ready is True
    assert result.packaging_errors == []
    assert result.validation_errors == []


# ── 2. Missing required files ────────────────────────────────────────────


@pytest.mark.parametrize("filename", [
    "README.md",
    "TRAINING_LOG.md",
    "MODEL_PASSPORT.json",
    "SIGNOFF.json",
])
def test_missing_required_file(tmp_path, filename):
    ckpt = _make_ready_checkpoint(tmp_path)
    (ckpt / filename).unlink()

    result = check_publish_ready(ckpt)

    assert result.ready is False
    assert any(filename in e for e in result.packaging_errors)


def test_all_required_files_missing(tmp_path):
    ckpt = tmp_path / "empty_checkpoint"
    ckpt.mkdir()

    result = check_publish_ready(ckpt)

    assert result.ready is False
    assert len(result.packaging_errors) >= 4


# ── 3. Empty README / TRAINING_LOG ───────────────────────────────────────


def test_empty_readme_fails(tmp_path):
    ckpt = _make_ready_checkpoint(tmp_path)
    (ckpt / "README.md").write_text("")

    result = check_publish_ready(ckpt)

    assert result.ready is False
    assert any("README.md" in e and "empty" in e.lower() for e in result.packaging_errors)


def test_empty_training_log_fails(tmp_path):
    ckpt = _make_ready_checkpoint(tmp_path)
    (ckpt / "TRAINING_LOG.md").write_text("")

    result = check_publish_ready(ckpt)

    assert result.ready is False
    assert any("TRAINING_LOG.md" in e and "empty" in e.lower() for e in result.packaging_errors)


def test_whitespace_only_readme_fails(tmp_path):
    ckpt = _make_ready_checkpoint(tmp_path)
    (ckpt / "README.md").write_text("   \n  \n  ")

    result = check_publish_ready(ckpt)

    assert result.ready is False
    assert any("README.md" in e for e in result.packaging_errors)


# ── 4. Invalid passport JSON ────────────────────────────────────────────


def test_invalid_passport_json(tmp_path):
    ckpt = _make_ready_checkpoint(tmp_path)
    (ckpt / "MODEL_PASSPORT.json").write_text("not valid json {{{")

    result = check_publish_ready(ckpt)

    assert result.ready is False
    assert any("MODEL_PASSPORT.json" in e for e in result.packaging_errors)


def test_passport_not_a_dict(tmp_path):
    ckpt = _make_ready_checkpoint(tmp_path)
    (ckpt / "MODEL_PASSPORT.json").write_text(json.dumps([1, 2, 3]))

    result = check_publish_ready(ckpt)

    assert result.ready is False
    assert any("MODEL_PASSPORT.json" in e for e in result.packaging_errors)


# ── 5. Invalid signoff ──────────────────────────────────────────────────


def test_invalid_signoff_json(tmp_path):
    ckpt = _make_ready_checkpoint(tmp_path)
    (ckpt / "SIGNOFF.json").write_text("broken json!!!")

    result = check_publish_ready(ckpt)

    assert result.ready is False
    assert any("SIGNOFF.json" in e for e in result.packaging_errors)


def test_signoff_not_a_dict(tmp_path):
    ckpt = _make_ready_checkpoint(tmp_path)
    (ckpt / "SIGNOFF.json").write_text('"just a string"')

    result = check_publish_ready(ckpt)

    assert result.ready is False
    assert any("SIGNOFF.json" in e for e in result.packaging_errors)


# ── 6. --json output shape ──────────────────────────────────────────────


def test_result_to_json_ready(tmp_path):
    ckpt = _make_ready_checkpoint(tmp_path)

    mock_result = MagicMock()
    mock_result.has_failures = False
    mock_result.observations = []

    with patch(
        "checkpoint_passport.cli.check_publish_ready.self_validate_passport",
        return_value=mock_result,
    ):
        result = check_publish_ready(ckpt)

    payload = result.to_dict()

    assert payload["ready"] is True
    assert payload["packaging_errors"] == []
    assert payload["validation_errors"] == []
    assert payload["checkpoint_dir"] == str(ckpt)


def test_result_to_json_not_ready(tmp_path):
    ckpt = _make_ready_checkpoint(tmp_path)
    (ckpt / "README.md").unlink()
    (ckpt / "SIGNOFF.json").unlink()

    result = check_publish_ready(ckpt)
    payload = result.to_dict()

    assert payload["ready"] is False
    assert len(payload["packaging_errors"]) >= 2
    assert isinstance(payload["packaging_errors"], list)
    assert all(isinstance(e, str) for e in payload["packaging_errors"])


# ── 7. Multiple errors reported together ────────────────────────────────


def test_multiple_errors_all_reported(tmp_path):
    ckpt = _make_ready_checkpoint(tmp_path)
    (ckpt / "README.md").write_text("")
    (ckpt / "SIGNOFF.json").unlink()
    (ckpt / "MODEL_PASSPORT.json").write_text("bad json")

    result = check_publish_ready(ckpt)

    assert result.ready is False
    assert len(result.errors) >= 3


# ── 8. Checkpoint dir does not exist ────────────────────────────────────


def test_nonexistent_checkpoint_dir(tmp_path):
    result = check_publish_ready(tmp_path / "no_such_dir")

    assert result.ready is False
    assert any("not found" in e.lower() or "does not exist" in e.lower() for e in result.errors)


# ── 9. Packaging vs validation errors reported separately ───────────────


def test_packaging_and_validation_errors_separate(tmp_path):
    ckpt = _make_ready_checkpoint(tmp_path)
    (ckpt / "README.md").write_text("")

    result = check_publish_ready(ckpt)

    assert result.ready is False
    assert any("README.md" in e for e in result.packaging_errors)
    assert isinstance(result.validation_errors, list)


# ── 10. Validation errors surface from internal validator ────────────────


def test_validation_errors_from_internal_validator(tmp_path):
    """When packaging is clean, the internal validator runs and its hard
    failures show up as validation_errors."""
    ckpt = _make_ready_checkpoint(tmp_path)

    result = check_publish_ready(ckpt)

    assert result.packaging_errors == []
    assert len(result.validation_errors) > 0
    assert result.ready is False


# ── 11. Upload refuses when check-publish-ready fails ───────────────────


def test_upload_refuses_when_not_publish_ready(tmp_path):
    ckpt = _make_ready_checkpoint(tmp_path)
    (ckpt / "README.md").unlink()

    with pytest.raises(PublishGateError, match="not publish-ready"):
        upload_checkpoint(
            checkpoint_dir=ckpt,
            repo_id="test-org/test-repo",
            revision="main",
        )


def test_upload_calls_hf_when_ready(tmp_path):
    ckpt = _make_ready_checkpoint(tmp_path)

    mock_val_result = MagicMock()
    mock_val_result.has_failures = False
    mock_val_result.observations = []

    with patch(
        "checkpoint_passport.cli.check_publish_ready.self_validate_passport",
        return_value=mock_val_result,
    ), patch(
        "checkpoint_passport.cli.publish_checkpoint._hf_upload"
    ) as mock_upload:
        upload_checkpoint(
            checkpoint_dir=ckpt,
            repo_id="test-org/test-repo",
            revision="main",
        )

    mock_upload.assert_called_once_with(ckpt, "test-org/test-repo", "main")


# ── 11. Download refuses when validation fails ──────────────────────────


def test_download_refuses_when_validation_fails(tmp_path):
    out_dir = tmp_path / "downloaded"
    out_dir.mkdir()
    (out_dir / "weights.bin").write_bytes(b"fake")

    with patch(
        "checkpoint_passport.cli.publish_checkpoint._hf_download",
        return_value=out_dir,
    ):
        with pytest.raises(ValidationGateError, match="post-download validation"):
            download_checkpoint(
                repo_id="test-org/test-repo",
                revision="abc123",
                out=out_dir,
            )


def test_download_succeeds_when_validation_passes(tmp_path):
    out_dir = _make_ready_checkpoint(tmp_path)

    with patch(
        "checkpoint_passport.cli.publish_checkpoint._hf_download",
        return_value=out_dir,
    ), patch(
        "checkpoint_passport.cli.publish_checkpoint.self_validate_passport",
    ) as mock_val:
        mock_result = MagicMock()
        mock_result.has_failures = False
        mock_val.return_value = mock_result

        result_path = download_checkpoint(
            repo_id="test-org/test-repo",
            revision="abc123",
            out=out_dir,
        )

    assert result_path == out_dir
    mock_val.assert_called_once()
