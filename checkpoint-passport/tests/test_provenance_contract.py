from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
import os

from checkpoint_passport.kernel.provenance import (
    check_deployment_repo_commit,
    check_training_repo_commit_format,
    check_passport_creation_repo_commit_format,
)
from checkpoint_passport.observation import Status
from checkpoint_passport.schema import ModelPassport, Provenance


def _load_with_provenance(provenance: Provenance):
    return SimpleNamespace(
        has_passport=True,
        passport=ModelPassport(provenance=provenance),
    )


def test_training_commit_missing_is_hard_failure():
    load = _load_with_provenance(Provenance())

    obs = check_training_repo_commit_format(load)

    assert obs.status is Status.FAIL
    assert obs.check == "training_repo_commit_format"


def test_passport_creation_commit_missing_is_hard_failure():
    load = _load_with_provenance(Provenance())

    obs = check_passport_creation_repo_commit_format(load)

    assert obs.status is Status.FAIL
    assert obs.check == "passport_creation_repo_commit_format"


def test_passport_creation_commit_format_passes_when_present():
    load = _load_with_provenance(
        Provenance(
            passport_creation_repo="https://example.com/autohpc.git",
            passport_creation_repo_commit="abcdef1234567890",
        )
    )

    obs = check_passport_creation_repo_commit_format(load)

    assert obs.status is Status.PASS


def test_deployment_repo_commit_mismatch_is_not_a_hard_gate(
    tmp_path: Path,
):
    target_repo = _make_clean_git_repo(tmp_path / "target")

    load = _load_with_provenance(
        Provenance(
            deployment_repo=str(target_repo),
            deployment_repo_commit="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        )
    )

    obs = check_deployment_repo_commit(
        load,
        target_repo=target_repo,
        require_signoff=True,
    )

    assert obs.status is Status.SOFT_SIGNAL


def test_dirty_deployment_repo_is_not_a_hard_gate(tmp_path: Path):
    target_repo = _make_clean_git_repo(tmp_path / "target")
    expected = subprocess.run(
        ["git", "-C", str(target_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (target_repo / "file.txt").write_text("dirty\n")

    load = _load_with_provenance(
        Provenance(
            deployment_repo=str(target_repo),
            deployment_repo_commit=expected,
        )
    )

    obs = check_deployment_repo_commit(
        load,
        target_repo=target_repo,
        require_signoff=True,
    )

    assert obs.status is Status.SOFT_SIGNAL


def _make_clean_git_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    (path / "file.txt").write_text("current\n")
    subprocess.run(["git", "add", "file.txt"], cwd=path, check=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test User",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test User",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=path,
        check=True,
        capture_output=True,
        env=env,
    )
    return path
