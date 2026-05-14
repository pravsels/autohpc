"""
CLI: publish-checkpoint

Bounded HF upload/download helpers that enforce validation gates so agents
do not author ad hoc upload/download scripts.

Upload runs check-publish-ready first and refuses on failure. Build an explicit
publish package before upload; do not upload a training checkpoint root directly.
Download runs validate-checkpoint --require-signoff after fetching
and refuses to report success on failure.

Usage:
    publish-checkpoint stage \
      --out <publish_dir> \
      --file <src[:dest]> \
      --dir <src[:dest]>

    publish-checkpoint upload \
      --publish-dir <publish_dir> \
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
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

from checkpoint_passport import self_validate_passport
from checkpoint_passport.cli.check_publish_ready import check_publish_ready


class PublishGateError(Exception):
    """Upload blocked because the checkpoint is not publish-ready."""


class StagePackageError(Exception):
    """Staging blocked because the requested package is unsafe or invalid."""


class ValidationGateError(Exception):
    """Download blocked because post-download validation failed."""


_TRAINING_ARTIFACT_MARKERS = {
    "train_state",
    "optimizer",
    "opt_state",
    "ema_state",
}


def _parse_stage_spec(spec: str) -> tuple[Path, Path]:
    src_text, sep, dest_text = spec.partition(":")
    src = Path(src_text)
    dest = Path(dest_text) if sep else Path(src.name)
    if dest.is_absolute() or ".." in dest.parts:
        raise StagePackageError(f"unsafe destination path: {dest}")
    return src, dest


def _is_training_artifact(path: Path) -> bool:
    return any(part in _TRAINING_ARTIFACT_MARKERS for part in path.parts)


def _copy_file(src: Path, dest: Path) -> int:
    if not src.is_file():
        raise StagePackageError(f"file not found: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest.stat().st_size


def _copy_dir(src: Path, dest: Path) -> int:
    if not src.is_dir():
        raise StagePackageError(f"directory not found: {src}")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, symlinks=True)
    return sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())


def stage_package(
    out: str | Path,
    *,
    files: list[str],
    dirs: list[str],
    include_train_state: bool = False,
) -> dict:
    """Create an explicit publish package from selected files and directories."""
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    entries = []
    total_bytes = 0

    for spec in files:
        src, dest = _parse_stage_spec(spec)
        if not include_train_state and _is_training_artifact(src):
            raise StagePackageError(f"refusing training artifact by default: {src}")
        size = _copy_file(src, out_path / dest)
        total_bytes += size
        entries.append({
            "type": "file",
            "source": str(src),
            "dest": str(dest),
            "bytes": size,
        })

    for spec in dirs:
        src, dest = _parse_stage_spec(spec)
        if not include_train_state and _is_training_artifact(src):
            raise StagePackageError(f"refusing training artifact by default: {src}")
        size = _copy_dir(src, out_path / dest)
        total_bytes += size
        entries.append({
            "type": "dir",
            "source": str(src),
            "dest": str(dest),
            "bytes": size,
        })

    return {
        "publish_dir": str(out_path),
        "total_bytes": total_bytes,
        "entries": entries,
    }


def _hf_upload(
    publish_dir: Path,
    repo_id: str,
    revision: str,
    ignore_patterns: list[str] | None = None,
) -> None:
    """Call the HF Hub API to upload a publish package directory.

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
        "folder_path": str(publish_dir),
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
    *,
    publish_dir: str | Path | None = None,
    repo_id: str,
    revision: str,
    ignore_patterns: list[str] | None = None,
    checkpoint_dir: str | Path | None = None,
    dataset_path: Optional[str | Path] = None,
    target_repo: Optional[str | Path] = None,
) -> None:
    """Upload a staged publish package to HF Hub, gated by check-publish-ready.

    Raises PublishGateError if the checkpoint is not publish-ready.
    """
    if publish_dir is None:
        publish_dir = checkpoint_dir
    if publish_dir is None:
        raise ValueError("publish_dir is required")
    ckpt = Path(publish_dir)
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

    # ── stage ──
    stage = subparsers.add_parser(
        "stage",
        help="Create an explicit publish package from selected files/directories",
    )
    stage.add_argument(
        "--out", type=Path, required=True,
        help="publish package directory to create",
    )
    stage.add_argument(
        "--file", action="append", default=[],
        help="file to include, as SRC or SRC:DEST; repeatable",
    )
    stage.add_argument(
        "--dir", action="append", default=[],
        help="directory to include, as SRC or SRC:DEST; repeatable",
    )
    stage.add_argument(
        "--include-train-state", action="store_true", default=False,
        help="allow train_state/optimizer artifacts in the publish package",
    )

    # ── upload ──
    up = subparsers.add_parser("upload", help="Upload publish package to HF Hub")
    upload_source = up.add_mutually_exclusive_group(required=True)
    upload_source.add_argument(
        "--publish-dir", type=Path, default=None,
        help="path to the staged publish package directory",
    )
    upload_source.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="deprecated alias for --publish-dir",
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
        if args.command == "stage":
            manifest = stage_package(
                args.out,
                files=args.file,
                dirs=args.dir,
                include_train_state=args.include_train_state,
            )
            print(json.dumps(manifest, indent=2))

        elif args.command == "upload":
            upload_checkpoint(
                publish_dir=args.publish_dir,
                checkpoint_dir=args.checkpoint_dir,
                repo_id=args.repo_id,
                revision=args.revision,
                ignore_patterns=args.ignore_patterns,
                dataset_path=args.dataset_path,
                target_repo=args.target_repo,
            )
            print(
                f"Uploaded {args.publish_dir or args.checkpoint_dir} to "
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

    except (StagePackageError, PublishGateError, ValidationGateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
