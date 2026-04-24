"""
End-to-end tests for `self_validate_passport`.

Covers three regimes:
  1. Passportless checkpoint (existing dit_block_tower_norm_fix fixture).
     Static checks pass / soft-signal as before; passport-driven checks
     emit NOT_CHECKED; signoff_present soft-signals (or hard-fails when
     --require-signoff is set).
  2. Minimal hand-written passport (no signoff). Passport-driven checks
     start firing; signoff_present still soft-signals.
  3. Passport + signoff with matching sha. Signoff checks all pass; the
     verdict drops back to soft-signal (only the state-dim soft signal
     remains).

The fixtures live in tests/fixtures/ and are constructed from the existing
dit_block_tower_norm_fix checkpoint files via symlinks where possible to
keep the test data small.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from checkpoint_passport import (
    Status,
    self_validate_passport,
    SCHEMA_VERSION,
    SIGNOFF_SCHEMA_VERSION,
    PASSPORT_FILENAME,
    SIGNOFF_FILENAME,
)


# These tests need a real LeRobot-style checkpoint on disk.  Point at one
# via $CHECKPOINT_PASSPORT_TEST_CHECKPOINT (absolute path).  If unset, the
# entire module is skipped -- this lets `pytest` be safe to run in any
# clone of autohpc, regardless of whether the developer has a checkpoint
# tree available.
_CHECKPOINT_ENV = "CHECKPOINT_PASSPORT_TEST_CHECKPOINT"
_checkpoint_env_value = os.environ.get(_CHECKPOINT_ENV)
if not _checkpoint_env_value:
    pytest.skip(
        f"set ${_CHECKPOINT_ENV} to a checkpoint dir to run these "
        "integration tests (e.g. dit_block_tower_norm_fix)",
        allow_module_level=True,
    )

CHECKPOINT_ROOT = Path(_checkpoint_env_value).resolve()
if not CHECKPOINT_ROOT.is_dir():
    pytest.skip(
        f"${_CHECKPOINT_ENV}={CHECKPOINT_ROOT} is not a directory",
        allow_module_level=True,
    )


# ── Helpers ─────────────────────────────────────────────────────────────


def _by_check(observations) -> dict:
    """Index observations by check name for easy assertion."""
    return {o.check: o for o in observations}


def _build_minimal_passport(stack: str = "diffusion") -> dict:
    """A hand-written v1.0 passport that's just rich enough to exercise
    the passport-driven checks against the dit_block_tower fixture."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": {"tool": "self_validate test", "version": "0"},
        "stack": stack,
        "input_contract": {
            "images": [
                {"key": "observation.images.front"},
                {"key": "observation.images.wrist"},
            ],
            "state": {"total_dim": 16},
            "actions": {
                "total_dim": 17,
                "horizon": 32,
                "normalization": {
                    # dit_block_tower stores its stats in assets/, not at
                    # the checkpoint root, so this source path is the
                    # tilde here -- the check should report
                    # "source_file_missing" not crash.
                    "source": "assets/ramen_stats.json",
                    "stats_dim": 17,
                },
            },
            "training_datasets": [
                {"repo": "lerobot/aloha-fake", "commit": "abc1234"},
            ],
        },
        "model_identity": {
            # Pick a class that always imports under stdlib so the test
            # doesn't depend on transformers being installed.
            "class_module": "collections",
            "class_name": "OrderedDict",
        },
        "model_internals": {
            "parameters": {
                "summary": {"total_params": 1, "trainable_params": 1},
                "by_name": [],
            },
            "state_dict": {"expected_keys_count": 569, "found_keys_count": 569},
            "numerical_health": {
                "no_nan_inf": {"passed": True, "params_checked": 569, "buffers_checked": 0},
                "determinism": {"passed": True, "max_abs_diff": 0.0},
            },
        },
        "output_spec": {
            "actions": {
                "horizon": 32,
                "sub_keys": "see input_contract.actions.sub_keys",
            },
            "smoke_results": {
                "calibration_batch_source": "synthetic",
                "calibration_batch_size": 1,
                "determinism": {"status": "pass"},
                "nan_inf": {"status": "pass"},
                "liveness": {"status": "pass"},
                "distribution": {"ratio_in_acceptable_range": True},
                "range_check": {"in_expected_range": True},
            },
        },
        "weight_integrity": {"weight_files": []},
        "provenance": {
            "training_repo": "alpha-robotics",
            "training_repo_commit": "abc1234567",
            "run_log_path": "docs/run-logs/dit_block_tower_norm_fix.md",
        },
    }


def _build_signoff(passport_path: Path) -> dict:
    """Build a signoff that names only the passport.

    We deliberately don't list weight files here because the test fixture
    mirrors them via symlinks pointing outside the fixture root, which
    the signoff path-escape security check (correctly) rejects. Listing
    only the passport exercises the checksum path; the absent weight
    list separately surfaces as a SOFT_SIGNAL, not a FAIL.
    """
    return {
        "schema_version": SIGNOFF_SCHEMA_VERSION,
        "signed_by": {"tool": "self_validate test", "version": "0"},
        "verdict": "pass",
        "artifacts": [
            {
                "path": PASSPORT_FILENAME,
                "sha256": hashlib.sha256(passport_path.read_bytes()).hexdigest(),
            },
        ],
    }


def _make_fixture_with_passport(
    tmp_root: Path,
    *,
    add_signoff: bool,
) -> Path:
    """Mirror the real checkpoint into tmp via symlinks, then write a
    fresh passport (and optionally a signoff) at its root."""
    fixture = tmp_root / "ckpt"
    fixture.mkdir(parents=True, exist_ok=True)
    # Mirror every file under CHECKPOINT_ROOT (excluding integrity_report.*)
    # using symlinks so we don't actually duplicate the safetensors blobs.
    for p in CHECKPOINT_ROOT.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("integrity_report."):
            continue
        if p.name == PASSPORT_FILENAME or p.name == SIGNOFF_FILENAME:
            continue  # we'll write our own
        dst = fixture / p.relative_to(CHECKPOINT_ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        dst.symlink_to(p.resolve())

    passport = _build_minimal_passport()
    passport_path = fixture / PASSPORT_FILENAME
    passport_path.write_text(json.dumps(passport, indent=2))

    if add_signoff:
        signoff = _build_signoff(passport_path)
        (fixture / SIGNOFF_FILENAME).write_text(json.dumps(signoff, indent=2))

    return fixture


# ── 1. Passportless ──────────────────────────────────────────────────────


def test_passportless_checkpoint():
    """The legacy fixture has no passport: signoff soft-signals; static
    checks behave as before; every passport-driven check is NOT_CHECKED."""
    result = self_validate_passport(CHECKPOINT_ROOT)
    by_check = _by_check(result.observations)

    # Static checks unchanged
    assert by_check["weight_file_integrity"].status is Status.PASS
    assert by_check["action_dim_config_vs_weights"].status is Status.PASS
    assert by_check["horizon_consistency"].status is Status.PASS
    assert by_check["norm_stats_vs_config"].status is Status.PASS
    assert by_check["image_keys_declared"].status is Status.PASS
    assert by_check["state_dim_consistency"].status is Status.SOFT_SIGNAL

    # No-passport degradations
    assert by_check["signoff_present"].status is Status.SOFT_SIGNAL
    assert by_check["model_identity_resolvable"].status is Status.NOT_CHECKED
    assert by_check["input_contract_vs_config"].status is Status.NOT_CHECKED
    assert by_check["output_actions_vs_input_actions"].status is Status.NOT_CHECKED
    assert by_check["training_repo_commit_format"].status is Status.NOT_CHECKED

    assert result.verdict == "human_look_here"
    assert not result.has_failures


def test_require_signoff_promotes_to_failure():
    """Same fixture; --require-signoff turns the soft signal into a fail."""
    result = self_validate_passport(CHECKPOINT_ROOT, require_signoff=True)
    by_check = _by_check(result.observations)
    assert by_check["signoff_present"].status is Status.FAIL
    assert result.verdict == "dont_ship"
    assert result.has_failures


# ── 2. Passport, no signoff ─────────────────────────────────────────────


def test_minimal_passport_without_signoff(tmp_path):
    """Hand-written passport; no SIGNOFF.json. Passport-driven checks
    fire (some pass, some soft-signal); signoff_present soft-signals."""
    fixture = _make_fixture_with_passport(tmp_path, add_signoff=False)

    result = self_validate_passport(fixture)
    by_check = _by_check(result.observations)

    # signoff: no file -> soft signal, downstream signoff checks NOT_CHECKED
    assert by_check["signoff_present"].status is Status.SOFT_SIGNAL
    assert by_check["passport_checksum_valid"].status is Status.NOT_CHECKED

    # model_identity resolves cleanly to a stdlib class
    assert by_check["model_identity_resolvable"].status is Status.PASS

    # input_contract checks fire and pass against the real config
    assert by_check["input_contract_vs_config"].status is Status.PASS
    assert by_check["training_datasets_present"].status is Status.PASS

    # output_spec is populated -> these pass
    assert by_check["output_actions_vs_input_actions"].status is Status.PASS
    assert by_check["smoke_results_pass"].status is Status.PASS

    # provenance soft-signals on the short commit (10 chars is fine, full
    # 7-40 hex match passes)
    assert by_check["training_repo_commit_format"].status is Status.PASS
    assert by_check["run_log_path_format"].status is Status.PASS

    # Determinism / no-NaN-inf were recorded as passed
    assert by_check["no_nan_inf_recorded"].status is Status.PASS
    assert by_check["determinism_recorded"].status is Status.PASS

    # internals_vs_weight_files is NOT_CHECKED because by_name is empty
    assert by_check["internals_vs_weight_files"].status is Status.NOT_CHECKED

    assert not result.has_failures
    # Soft signals: signoff missing + state dim
    assert result.verdict == "human_look_here"


# ── 3. Passport + signoff ───────────────────────────────────────────────


def test_passport_and_signoff_clean(tmp_path):
    """With matching signoff, signoff checks all pass and the only
    remaining soft signal is the legitimate state_dim observation."""
    fixture = _make_fixture_with_passport(tmp_path, add_signoff=True)

    result = self_validate_passport(fixture)
    by_check = _by_check(result.observations)

    assert by_check["signoff_present"].status is Status.PASS
    assert by_check["signoff_schema_version_supported"].status is Status.PASS
    assert by_check["passport_checksum_valid"].status is Status.PASS
    # Signoff intentionally lists no weights -> soft signal, not fail.
    assert by_check["weight_files_match_signoff"].status is Status.SOFT_SIGNAL

    # Original signal still surfaces
    assert by_check["state_dim_consistency"].status is Status.SOFT_SIGNAL

    assert result.verdict == "human_look_here"
    assert not result.has_failures


def test_passport_signoff_corruption_detected(tmp_path):
    """Tampering with the passport after sign-off is a hard fail."""
    fixture = _make_fixture_with_passport(tmp_path, add_signoff=True)

    # Tamper with the passport (rewrite a single field)
    passport_path = fixture / PASSPORT_FILENAME
    p = json.loads(passport_path.read_text())
    p["stack"] = "tampered"
    passport_path.write_text(json.dumps(p, indent=2))

    result = self_validate_passport(fixture)
    by_check = _by_check(result.observations)
    assert by_check["passport_checksum_valid"].status is Status.FAIL
    assert result.has_failures


# ── 4. --skip-section semantics ─────────────────────────────────────────


def test_skip_section_drops_observations_and_changes_verdict():
    """Skipping `signoff` removes the missing-signoff soft signal, so the
    passportless fixture's verdict drops from human_look_here to ship_it.

    The dropped category must not appear in the report and must not
    contribute to has_failures.
    """
    baseline = self_validate_passport(CHECKPOINT_ROOT)
    assert baseline.verdict == "human_look_here"
    assert baseline.skipped_sections == []

    # Skip both the signoff soft signal and the input-expectation
    # state-dim soft signal -- nothing left to flag.
    result = self_validate_passport(
        CHECKPOINT_ROOT,
        skip_sections=["signoff", "input_expectation"],
    )
    by_check = _by_check(result.observations)

    assert "signoff_present" not in by_check
    assert "state_dim_consistency" not in by_check
    assert "weight_file_integrity" in by_check  # unaffected categories stay
    assert result.skipped_sections == ["input_expectation", "signoff"]
    assert result.verdict == "ship_it"
    assert not result.has_failures


def test_skip_section_unknown_category_is_ignored_with_note():
    """Unknown category names don't error; they're noted and ignored."""
    result = self_validate_passport(
        CHECKPOINT_ROOT,
        skip_sections=["bogus_section", "signoff"],
    )
    # signoff actually got filtered
    by_check = _by_check(result.observations)
    assert "signoff_present" not in by_check
    # bogus appears in notes
    assert any("bogus_section" in n for n in result.notes)


def test_report_header_present_for_passportless_fixture():
    """The header should always render once a passport-load result is in
    scope, even when no passport file exists. Operators read the header
    first to know what's actually being checked."""
    result = self_validate_passport(CHECKPOINT_ROOT)
    rendered = result.render()
    assert "Load status:" in rendered
    assert "no passport on disk" in rendered
    assert "Passport:" in rendered
    assert "Signoff:" in rendered
    assert "Sections:" in rendered


def test_report_header_lists_sections_when_passport_present(tmp_path):
    """A populated passport renders its section list in the header so
    operators can see at a glance which checks were exercisable."""
    fixture = _make_fixture_with_passport(tmp_path, add_signoff=True)
    result = self_validate_passport(fixture)
    rendered = result.render()
    # all six known sections are populated by the minimal passport
    for section in (
        "input_contract", "model_identity", "model_internals",
        "output_spec", "weight_integrity", "provenance",
    ):
        assert section in rendered, f"missing section {section} in header"
    assert "schema_version=0.1" in rendered


# ── 5. Security guard ──────────────────────────────────────────────────


def test_require_signoff_with_skip_signoff_is_rejected():
    """Combining `--require-signoff` and `--skip-section signoff` would
    silently let a tampered passport pass with exit 0. The API rejects
    this combination at call time so it cannot reach production."""
    import pytest

    with pytest.raises(ValueError, match="incompatible"):
        self_validate_passport(
            CHECKPOINT_ROOT,
            require_signoff=True,
            skip_sections=["signoff"],
        )


def test_skip_section_signoff_alone_is_allowed():
    """Skipping signoff WITHOUT requiring it is a valid (if dangerous)
    development-time configuration and must not raise."""
    # Should not raise; just returns a result with signoff observations
    # filtered out.
    result = self_validate_passport(
        CHECKPOINT_ROOT,
        skip_sections=["signoff"],
    )
    by_check = _by_check(result.observations)
    assert "signoff_present" not in by_check
    assert "passport_checksum_valid" not in by_check


# ── 6. Header for malformed passport on disk ───────────────────────────


def test_report_header_marks_unparseable_passport(tmp_path):
    """When MODEL_PASSPORT.json exists on disk but isn't valid JSON, the
    header must say 'unparseable' rather than 'no passport' so the
    operator knows the file is there and broken (different action than
    'file is missing')."""
    fixture = tmp_path / "ckpt"
    fixture.mkdir()
    # Mirror only the minimum needed for extraction to not crash; we don't
    # actually run any model-internals checks here.
    for src_name in ("config.json",):
        src = CHECKPOINT_ROOT / src_name
        if src.exists():
            (fixture / src_name).symlink_to(src.resolve())
    # Drop a malformed passport
    (fixture / PASSPORT_FILENAME).write_text("not valid json {")

    result = self_validate_passport(fixture)
    rendered = result.render()
    assert "unparseable" in rendered
    assert "no passport on disk" not in rendered


if __name__ == "__main__":
    import tempfile

    test_passportless_checkpoint()
    test_require_signoff_promotes_to_failure()
    with tempfile.TemporaryDirectory() as td:
        test_minimal_passport_without_signoff(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_passport_and_signoff_clean(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_passport_signoff_corruption_detected(Path(td))
    test_skip_section_drops_observations_and_changes_verdict()
    test_skip_section_unknown_category_is_ignored_with_note()
    test_report_header_present_for_passportless_fixture()
    with tempfile.TemporaryDirectory() as td:
        test_report_header_lists_sections_when_passport_present(Path(td))
    test_require_signoff_with_skip_signoff_is_rejected()
    test_skip_section_signoff_alone_is_allowed()
    with tempfile.TemporaryDirectory() as td:
        test_report_header_marks_unparseable_passport(Path(td))
    print("OK")
