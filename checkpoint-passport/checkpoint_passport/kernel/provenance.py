"""
Provenance: lightweight format checks on the pointers back to source.

Per the plan: pointers only, no deep validation. We check that strings look
like valid commit shas and run-log paths but never network-resolve them or
require the run log to exist on the machine doing the validation.
"""

from __future__ import annotations

import re
from typing import Callable, List
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
            status=Status.SOFT_SIGNAL,
            message="provenance.training_repo_commit not declared",
            details={"training_repo": pv.training_repo},
            category=CATEGORY,
        )

    if not _COMMIT_RE.match(commit):
        return Observation(
            check="training_repo_commit_format",
            status=Status.SOFT_SIGNAL,
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
    check_run_log_path_format,
]
