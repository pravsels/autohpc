"""
Provenance: format checks for debug/reproduction pointers.

Format checks: pointers only, no deep validation. We check that strings
look like valid commit shas and run-log paths but never network-resolve
them or require the run log to exist on the machine doing the validation.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable, List, Optional
from urllib.parse import urlparse

from ..observation import Observation, Status
from ..passport import PassportLoadResult


CATEGORY = "provenance"


# Git short shas are 7+ hex; long shas are 40. Be permissive.
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
# Run log path: relative .md file with no .. components.
_RUN_LOG_PATH_RE = re.compile(r"^[A-Za-z0-9._/\-]+\.md$")
# URI schemes accepted as a run-log pointer (W&B, MLflow, etc. all serve over https).
_RUN_LOG_URI_SCHEMES = frozenset({"http", "https"})


def check_training_repo_commit_format(load: PassportLoadResult) -> Observation:
    """`provenance.training_repo_commit` looks like a git sha."""
    if not load.has_passport:
        return _not_checked("training_repo_commit_format", "no passport loaded")

    pv = load.passport.provenance
    commit = pv.training_repo_commit
    if not commit:
        return Observation(
            check="training_repo_commit_format",
            status=Status.FAIL,
            message="provenance.training_repo_commit not declared",
            details={"training_repo": pv.training_repo},
            category=CATEGORY,
        )

    if not _COMMIT_RE.match(commit):
        return Observation(
            check="training_repo_commit_format",
            status=Status.FAIL,
            message=f"training_repo_commit {commit!r} does not look like a "
                    "git sha (7-40 hex chars)",
            details={"commit": commit, "training_repo": pv.training_repo},
            category=CATEGORY,
        )
    return Observation(
        check="training_repo_commit_format",
        status=Status.PASS,
        message=f"training_repo_commit format ok ({commit[:8]}...)",
        details={"commit": commit, "training_repo": pv.training_repo},
        category=CATEGORY,
    )


def check_passport_creation_repo_commit_format(load: PassportLoadResult) -> Observation:
    """`provenance.passport_creation_repo_commit` looks like a git sha.

    This records the autohpc/tooling snapshot that produced the passport.
    It is required provenance for debugging passport/schema/extractor bugs,
    but it is not compared against the current checkout.
    """
    if not load.has_passport:
        return _not_checked(
            "passport_creation_repo_commit_format",
            "no passport loaded",
        )

    pv = load.passport.provenance
    commit = pv.passport_creation_repo_commit
    if not commit:
        return Observation(
            check="passport_creation_repo_commit_format",
            status=Status.FAIL,
            message="provenance.passport_creation_repo_commit not declared",
            details={"passport_creation_repo": pv.passport_creation_repo},
            category=CATEGORY,
        )

    if not _COMMIT_RE.match(commit):
        return Observation(
            check="passport_creation_repo_commit_format",
            status=Status.FAIL,
            message=(
                f"passport_creation_repo_commit {commit!r} does not look "
                "like a git sha (7-40 hex chars)"
            ),
            details={
                "commit": commit,
                "passport_creation_repo": pv.passport_creation_repo,
            },
            category=CATEGORY,
        )

    return Observation(
        check="passport_creation_repo_commit_format",
        status=Status.PASS,
        message=f"passport_creation_repo_commit format ok ({commit[:8]}...)",
        details={"commit": commit, "passport_creation_repo": pv.passport_creation_repo},
        category=CATEGORY,
    )


def check_run_log_path_format(load: PassportLoadResult) -> Observation:
    """`provenance.run_log_path` looks like a sane pointer to the training run log.

    Two acceptable forms:
      - relative path to a .md file under the checkpoint repo (no `..`)
      - http / https URI to an external dashboard (W&B, MLflow, …)

    We don't require the path to resolve / the URL to be reachable -- the
    run log lives off-checkpoint and may not be accessible at validation
    time. Format-only.
    """
    if not load.has_passport:
        return _not_checked("run_log_path_format", "no passport loaded")

    pv = load.passport.provenance
    path = pv.run_log_path
    if not path:
        return Observation(
            check="run_log_path_format",
            status=Status.SOFT_SIGNAL,
            message="provenance.run_log_path not declared",
            details={},
            category=CATEGORY,
        )

    # URI form -- any acceptable scheme + non-empty netloc is fine.
    if "://" in path:
        parsed = urlparse(path)
        if parsed.scheme not in _RUN_LOG_URI_SCHEMES:
            return Observation(
                check="run_log_path_format",
                status=Status.SOFT_SIGNAL,
                message=(
                    f"run_log_path URI scheme {parsed.scheme!r} not in "
                    f"accepted set {sorted(_RUN_LOG_URI_SCHEMES)}"
                ),
                details={"run_log_path": path},
                category=CATEGORY,
            )
        if not parsed.netloc:
            return Observation(
                check="run_log_path_format",
                status=Status.SOFT_SIGNAL,
                message=f"run_log_path URI {path!r} has no host component",
                details={"run_log_path": path},
                category=CATEGORY,
            )
        return Observation(
            check="run_log_path_format",
            status=Status.PASS,
            message=f"run_log_path format ok ({parsed.scheme}://{parsed.netloc}…)",
            details={"run_log_path": path, "form": "uri"},
            category=CATEGORY,
        )

    # Path form -- relative .md, no parent traversal.
    if ".." in path.split("/"):
        return Observation(
            check="run_log_path_format",
            status=Status.SOFT_SIGNAL,
            message="run_log_path contains parent traversal '..' (fragile pointer)",
            details={"run_log_path": path},
            category=CATEGORY,
        )

    if not _RUN_LOG_PATH_RE.match(path):
        return Observation(
            check="run_log_path_format",
            status=Status.SOFT_SIGNAL,
            message=(
                f"run_log_path {path!r} is neither a relative .md path nor "
                "an http(s) URI"
            ),
            details={"run_log_path": path},
            category=CATEGORY,
        )

    return Observation(
        check="run_log_path_format",
        status=Status.PASS,
        message=f"run_log_path format ok ({path})",
        details={"run_log_path": path, "form": "path"},
        category=CATEGORY,
    )


def check_deployment_repo_commit(
    load: PassportLoadResult,
    target_repo: Optional[Path] = None,
    require_signoff: bool = False,
) -> Observation:
    """Report deployment repo commit drift as debug context, not a load gate."""
    if not load.has_passport:
        return _not_checked("deployment_repo_commit", "no passport loaded")

    pv = load.passport.provenance
    expected = pv.deployment_repo_commit

    if not expected:
        return Observation(
            check="deployment_repo_commit",
            status=Status.SOFT_SIGNAL,
            message="provenance.deployment_repo_commit not declared",
            details={"deployment_repo": pv.deployment_repo},
            category=CATEGORY,
        )

    if not expected or not _COMMIT_RE.match(expected):
        return Observation(
            check="deployment_repo_commit",
            status=Status.SOFT_SIGNAL,
            message=f"deployment_repo_commit {expected!r} is not a valid git sha",
            details={"deployment_repo_commit": expected},
            category=CATEGORY,
        )

    if target_repo is None:
        return _not_checked(
            "deployment_repo_commit",
            "no --target-repo provided; cannot verify deployment_repo_commit",
        )

    if not target_repo.is_dir():
        return Observation(
            check="deployment_repo_commit",
            status=Status.SOFT_SIGNAL,
            message=f"target repo path does not exist: {target_repo}",
            details={"target_repo": str(target_repo)},
            category=CATEGORY,
        )

    try:
        head = subprocess.run(
            ["git", "-C", str(target_repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return Observation(
            check="deployment_repo_commit",
            status=Status.SOFT_SIGNAL,
            message=f"could not run git in target repo: {exc}",
            details={"target_repo": str(target_repo)},
            category=CATEGORY,
        )

    if head.returncode != 0:
        return Observation(
            check="deployment_repo_commit",
            status=Status.SOFT_SIGNAL,
            message=f"git rev-parse HEAD failed in {target_repo}: {head.stderr.strip()}",
            details={"target_repo": str(target_repo)},
            category=CATEGORY,
        )

    actual_commit = head.stdout.strip()

    try:
        dirty = subprocess.run(
            ["git", "-C", str(target_repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return Observation(
            check="deployment_repo_commit",
            status=Status.SOFT_SIGNAL,
            message=f"could not inspect target repo dirty state: {exc}",
            details={
                "expected_commit": expected,
                "actual_commit": actual_commit,
                "target_repo": str(target_repo),
            },
            category=CATEGORY,
        )

    if dirty.returncode != 0:
        return Observation(
            check="deployment_repo_commit",
            status=Status.SOFT_SIGNAL,
            message=f"git status failed in {target_repo}: {dirty.stderr.strip()}",
            details={
                "expected_commit": expected,
                "actual_commit": actual_commit,
                "target_repo": str(target_repo),
            },
            category=CATEGORY,
        )

    dirty_files = "\n".join(
        line for line in dirty.stdout.strip().splitlines()
        if not line.startswith("??")
    ).strip()

    failures = []
    if actual_commit != expected:
        failures.append(
            f"HEAD is {actual_commit[:8]}… but passport expects {expected[:8]}…"
        )
    if dirty_files:
        failures.append(
            f"working tree is dirty ({len(dirty_files.splitlines())} file(s) changed)"
        )

    if failures:
        return Observation(
            check="deployment_repo_commit",
            status=Status.SOFT_SIGNAL,
            message="deployment repo mismatch: " + "; ".join(failures),
            details={
                "expected_commit": expected,
                "actual_commit": actual_commit,
                "dirty_files": dirty_files or None,
                "target_repo": str(target_repo),
            },
            category=CATEGORY,
        )

    return Observation(
        check="deployment_repo_commit",
        status=Status.PASS,
        message=f"deployment repo matches passport ({actual_commit[:8]}…, clean tree)",
        details={
            "commit": actual_commit,
            "target_repo": str(target_repo),
        },
        category=CATEGORY,
    )


def _not_checked(name: str, reason: str) -> Observation:
    return Observation(
        check=name,
        status=Status.NOT_CHECKED,
        message=reason,
        details={},
        category=CATEGORY,
    )


PASSPORT_CHECKS: List[Callable[..., Observation]] = [
    check_training_repo_commit_format,
    check_passport_creation_repo_commit_format,
    check_run_log_path_format,
    check_deployment_repo_commit,
]
