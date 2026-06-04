from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autohpc_wandb_sync.sync import (
    WandbSyncConfig,
    WandbSyncError,
    build_wandb_sync_command,
    coerce_paths,
    find_token_file,
    render_command,
    sync_wandb,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autohpc-wandb-sync",
        description="Sync an offline W&B run inside an Apptainer container.",
    )
    parser.add_argument(
        "--offline-run-dir",
        type=Path,
        required=True,
        help="path to wandb/offline-run-* directory",
    )
    parser.add_argument(
        "--container",
        type=Path,
        required=True,
        help="Apptainer/Singularity image containing wandb",
    )
    parser.add_argument("--entity", required=True, help="W&B entity/team")
    parser.add_argument(
        "--project",
        "--to-project",
        dest="project",
        required=True,
        help="W&B project; required to avoid stale offline defaults",
    )
    parser.add_argument(
        "--wandb-token-file",
        type=Path,
        default=None,
        help="token file (default: ~/.wandb_token or ~/.wandb_key)",
    )
    parser.add_argument(
        "--bind",
        action="append",
        default=[],
        help="path to bind into the container; repeatable",
    )
    parser.add_argument(
        "--no-srun",
        action="store_true",
        help="run apptainer directly instead of wrapping it in srun",
    )
    parser.add_argument(
        "--srun-arg",
        action="append",
        default=None,
        help="override/add srun argument; repeatable",
    )
    parser.add_argument(
        "--apptainer-arg",
        action="append",
        default=None,
        help="extra argument passed to apptainer exec; repeatable",
    )
    parser.add_argument(
        "--wandb-arg",
        action="append",
        default=None,
        help="extra argument passed to wandb sync; repeatable",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the command with token path redacted and do not run it",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="run without interactive confirmation",
    )
    args = parser.parse_args()

    token_file = args.wandb_token_file or find_token_file()
    cfg = WandbSyncConfig(
        offline_run_dir=args.offline_run_dir,
        container=args.container,
        entity=args.entity,
        project=args.project,
        token_file=token_file,
        binds=coerce_paths(args.bind),
        use_srun=not args.no_srun,
        srun_args=args.srun_arg,
        apptainer_args=args.apptainer_arg,
        wandb_args=args.wandb_arg,
    )

    try:
        command = build_wandb_sync_command(cfg)
    except WandbSyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"W&B sync target: {args.entity}/{args.project}")
    print(f"Offline run: {args.offline_run_dir}")
    print("Command:")
    print(render_command(command, token_file))

    if args.dry_run:
        return

    if not args.yes and not _confirm():
        print("aborted")
        sys.exit(2)

    try:
        result = sync_wandb(cfg)
    except WandbSyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.url:
        print(f"Synced URL: {result.url}")
    else:
        print("warning: sync completed but no W&B URL was detected", file=sys.stderr)


def _confirm() -> bool:
    answer = input("Proceed with W&B sync? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


if __name__ == "__main__":
    main()
