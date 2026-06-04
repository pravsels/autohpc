from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autohpc_wandb_sync.sync import (
    InnerSyncConfig,
    LaunchSyncConfig,
    WandbSyncError,
    build_inner_sync_command,
    build_launch_command,
    coerce_paths,
    find_token_file,
    render_command,
    run_command,
    run_inner_sync,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autohpc-wandb-sync",
        description="Sync an offline W&B run with explicit entity/project.",
    )
    subparsers = parser.add_subparsers(dest="command")

    _add_sync_parser(subparsers.add_parser(
        "sync",
        help="run wandb sync in the current container/environment",
    ))
    _add_launch_parser(subparsers.add_parser(
        "launch",
        help="launch autohpc-wandb-sync sync inside Apptainer",
    ))

    args = _parse_args(parser)
    if args.command == "launch":
        _main_launch(args)
    else:
        _main_sync(args)


def _parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    raw_args = sys.argv[1:]
    if raw_args and raw_args[0] not in {"sync", "launch", "-h", "--help"}:
        raw_args = ["sync", *raw_args]
    return parser.parse_args(raw_args)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
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


def _add_sync_parser(parser: argparse.ArgumentParser) -> None:
    _add_common_args(parser)
    parser.add_argument(
        "offline_run_dir",
        type=Path,
        help="path to wandb/offline-run-* directory",
    )


def _add_launch_parser(parser: argparse.ArgumentParser) -> None:
    _add_common_args(parser)
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
        help="Apptainer/Singularity image containing autohpc-wandb-sync and wandb",
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
        "--ssh-host",
        default=None,
        help="run the Slurm/Apptainer launch command on this host via ssh",
    )
    parser.add_argument(
        "--module",
        action="append",
        default=None,
        help="remote environment module to load before srun; repeatable",
    )


def _main_sync(args: argparse.Namespace) -> None:
    token_file = args.wandb_token_file or find_token_file()
    cfg = InnerSyncConfig(
        offline_run_dir=args.offline_run_dir,
        entity=args.entity,
        project=args.project,
        token_file=token_file,
        wandb_args=args.wandb_arg,
    )

    try:
        command = build_inner_sync_command(cfg)
    except WandbSyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_preflight(args.entity, args.project, args.offline_run_dir, command, token_file)
    if args.dry_run:
        return
    if not args.yes and not _confirm():
        print("aborted")
        sys.exit(2)

    try:
        result = run_inner_sync(cfg)
    except WandbSyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    _print_result(result)


def _main_launch(args: argparse.Namespace) -> None:
    token_file = args.wandb_token_file or find_token_file()
    cfg = LaunchSyncConfig(
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
        ssh_host=args.ssh_host,
        modules=args.module,
    )

    try:
        command = build_launch_command(cfg)
    except WandbSyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_preflight(args.entity, args.project, args.offline_run_dir, command, token_file)
    if args.dry_run:
        return
    if not args.yes and not _confirm():
        print("aborted")
        sys.exit(2)

    try:
        result = run_command(command)
    except WandbSyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    _print_result(result)


def _print_preflight(
    entity: str,
    project: str,
    offline_run_dir: Path,
    command: list[str],
    token_file: Path | None,
) -> None:
    print(f"W&B sync target: {entity}/{project}")
    print(f"Offline run: {offline_run_dir}")
    print("Command:")
    print(render_command(command, token_file))


def _print_result(result) -> None:
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
