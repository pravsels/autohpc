from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manifest import MANIFEST_FILENAME, verify_manifest, write_manifest


def manifest_main() -> None:
    parser = argparse.ArgumentParser(
        prog="manifest-checkpoint",
        description="Write a SHA-256 manifest for a checkpoint bundle.",
    )
    parser.add_argument("checkpoint_dir", type=Path)
    args = parser.parse_args()

    try:
        output = write_manifest(args.checkpoint_dir)
        payload = json.loads(output.read_text())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(
        f"wrote {output} "
        f"({len(payload['artifacts'])} files, "
        f"{len(payload['skipped_symlinks'])} symlinks skipped)"
    )


def verify_main() -> None:
    parser = argparse.ArgumentParser(
        prog="verify-checkpoint",
        description=f"Verify files declared in {MANIFEST_FILENAME}.",
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail when unlisted files or symlinks are present",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="print structured JSON",
    )
    args = parser.parse_args()

    result = verify_manifest(args.checkpoint_dir, strict=args.strict)
    if args.json_output:
        print(json.dumps({
            "valid": result.valid,
            "checked": result.checked,
            "errors": result.errors,
            "extra_files": result.extra_files,
        }, indent=2))
    elif result.valid:
        print(f"verified {result.checked} files")
        if result.extra_files:
            print(
                "warning: unlisted files present: "
                + ", ".join(result.extra_files)
            )
    else:
        print("verification failed", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)

    raise SystemExit(0 if result.valid else 1)
