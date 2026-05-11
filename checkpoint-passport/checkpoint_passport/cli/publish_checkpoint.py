"""
CLI: publish-checkpoint

Bounded HF upload/download helpers that enforce validation gates so agents
do not author ad hoc upload/download scripts.

Upload runs check-publish-ready first and refuses on failure.
Download runs validate-checkpoint --require-signoff after fetching
and refuses to report success on failure.

Usage:
    publish-checkpoint upload \
      --checkpoint-dir <ckpt> \
      --repo-id <user-or-org>/<repo> \
      --revision main \
      [--dataset-path /path/to/local/dataset] \
      [--target-repo /path/to/deploy/repo]

    publish-checkpoint download \
      --repo-id <user-or-org>/<repo> \
      --revision <sha-or-branch> \
      --out <local_dir> \
      [--dataset-path /path/to/local/dataset] \
      [--target-repo /path/to/deploy/repo]

Exit code 0 = success.  Exit code 1 = gate failure or runtime error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from checkpoint_passport import self_validate_passport
from checkpoint_passport.cli.check_publish_ready import check_publish_ready


class PublishGateError(Exception):
    """Upload blocked because the checkpoint is not publish-ready."""


class ValidationGateError(Exception):
    """Download blocked because post-download validation failed."""


def _hf_upload(
    checkpoint_dir: Path,
    repo_id: str,
    revision: str,
    ignore_patterns: list[str] | None = None,
) -> None:
    """Call the HF Hub API to upload a checkpoint directory.

    Creates the repo if it doesn't already exist.
    Separated so tests can mock this without touching the real HF API.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed. "
            "Install it with: pip install huggingface_hub"
        )
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    kwargs: dict = {
        "folder_path": str(checkpoint_dir),
        "repo_id": repo_id,
        "revision": revision,
        "repo_type": "model",
    }
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns
    api.upload_folder(**kwargs)


def _hf_download(repo_id: str, revision: str, out: Path) -> Path:
    """Call the HF Hub API to download a checkpoint.

    Separated so tests can mock this without touching the real HF API.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed. "
            "Install it with: pip install huggingface_hub"
        )
    return Path(snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(out),
    ))


def upload_checkpoint(
    checkpoint_dir: str | Path,
    repo_id: str,
    revision: str,
    ignore_patterns: list[str] | None = None,
    *,
    dataset_path: Optional[str | Path] = None,
    target_repo: Optional[str | Path] = None,
) -> None:
    """Upload a checkpoint to HF Hub, gated by check-publish-ready.

    Raises PublishGateError if the checkpoint is not publish-ready.
    """
    ckpt = Path(checkpoint_dir)
    result = check_publish_ready(
        ckpt,
        dataset_path=dataset_path,
        target_repo=target_repo,
    )
    if not result.ready:
        msg = "Checkpoint is not publish-ready:\n"
        for err in result.errors:
            msg += f"  - {err}\n"
        raise PublishGateError(msg)

    _hf_upload(ckpt, repo_id, revision, ignore_patterns=ignore_patterns)


def download_checkpoint(
    repo_id: str,
    revision: str,
    out: str | Path,
    *,
    dataset_path: Optional[str | Path] = None,
    target_repo: Optional[str | Path] = None,
) -> Path:
    """Download a checkpoint from HF Hub, gated by post-download validation.

    Raises ValidationGateError if post-download validation with
    require_signoff=True reports hard failures.
    """
    out_path = Path(out)
    downloaded = _hf_download(repo_id, revision, out_path)

    result = self_validate_passport(
        downloaded,
        dataset_path=dataset_path,
        target_repo=target_repo,
        require_signoff=True,
    )
    if result.has_failures:
        failures = [
            f"{o.check}: {o.message}"
            for o in result.observations
            if o.status.value == "fail"
        ]
        msg = "Post-download validation failed:\n"
        for f in failures:
            msg += f"  - {f}\n"
        raise ValidationGateError(f"post-download validation failed:\n{msg}")

    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="publish-checkpoint",
        description="Upload or download checkpoints with validation gates.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── upload ──
    up = subparsers.add_parser("upload", help="Upload checkpoint to HF Hub")
    up.add_argument(
        "--checkpoint-dir", type=Path, required=True,
        help="path to the checkpoint directory",
    )
    up.add_argument(
        "--repo-id", type=str, required=True,
        help="HF Hub repo ID (e.g. user-or-org/repo)",
    )
    up.add_argument(
        "--revision", type=str, default="main",
        help="branch or revision to upload to (default: main)",
    )
    up.add_argument(
        "--ignore-patterns", type=str, nargs="+", default=None,
        help="glob patterns to exclude from upload (e.g. 'retain/**')",
    )
    up.add_argument(
        "--dataset-path", type=Path, default=None,
        help="local LeRobot dataset directory; enables input_contract_vs_dataset",
    )
    up.add_argument(
        "--target-repo", type=Path, default=None,
        help="deployment target repo; enables deployment_repo_commit check",
    )

    # ── download ──
    down = subparsers.add_parser("download", help="Download checkpoint from HF Hub")
    down.add_argument(
        "--repo-id", type=str, required=True,
        help="HF Hub repo ID (e.g. user-or-org/repo)",
    )
    down.add_argument(
        "--revision", type=str, required=True,
        help="commit SHA or branch to download",
    )
    down.add_argument(
        "--out", type=Path, required=True,
        help="local directory to download into",
    )
    down.add_argument(
        "--dataset-path", type=Path, default=None,
        help="local LeRobot dataset directory; enables input_contract_vs_dataset",
    )
    down.add_argument(
        "--target-repo", type=Path, default=None,
        help="deployment target repo; enables deployment_repo_commit check",
    )

    args = parser.parse_args()

    try:
        if args.command == "upload":
            upload_checkpoint(
                checkpoint_dir=args.checkpoint_dir,
                repo_id=args.repo_id,
                revision=args.revision,
                ignore_patterns=args.ignore_patterns,
                dataset_path=args.dataset_path,
                target_repo=args.target_repo,
            )
            print(
                f"Uploaded {args.checkpoint_dir} to "
                f"{args.repo_id}@{args.revision}"
            )

        elif args.command == "download":
            downloaded = download_checkpoint(
                repo_id=args.repo_id,
                revision=args.revision,
                out=args.out,
                dataset_path=args.dataset_path,
                target_repo=args.target_repo,
            )
            print(
                f"Downloaded {args.repo_id}@{args.revision} to "
                f"{downloaded}"
            )

    except (PublishGateError, ValidationGateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
