"""
CLI: assemble-passport

Takes a PASSPORT_SEED.json (from extract-passport-seed) and a checkpoint
directory, computes weight_integrity and provenance, and writes the final
MODEL_PASSPORT.json.

Usage:
    assemble-passport \
      --checkpoint-dir <ckpt> \
      --seed <ckpt>/PASSPORT_SEED.json \
      --out <ckpt>/MODEL_PASSPORT.json

Exit code 0 = passport written.  Exit code 1 = error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from checkpoint_passport.passport_seed import InvalidSeedError
from checkpoint_passport.assemble_passport import assemble_passport


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="assemble-passport",
        description=(
            "Assemble MODEL_PASSPORT.json from a passport seed and checkpoint files."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, required=True,
        help="path to the checkpoint directory",
    )
    parser.add_argument(
        "--seed", type=Path, required=True,
        help="path to PASSPORT_SEED.json",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output path (default: <checkpoint-dir>/MODEL_PASSPORT.json)",
    )
    parser.add_argument(
        "--generated-at", type=str, default=None,
        help="ISO 8601 timestamp; omit for current UTC time",
    )
    parser.add_argument(
        "--target-repo", type=Path, default=None,
        help="deployment target repo; populates provenance.deployment_repo_commit",
    )
    parser.add_argument(
        "--training-repo", type=Path, default=None,
        help="training model repo; populates provenance.training_repo_commit",
    )
    parser.add_argument(
        "--skip-dir", type=str, action="append", default=None, dest="skip_dirs",
        help="top-level directory to exclude from weight_integrity hashing "
             "(repeatable, e.g. --skip-dir retain)",
    )
    parser.add_argument(
        "--skip-file", type=str, action="append", default=None, dest="skip_files",
        help="filename to exclude from weight_integrity hashing "
             "(repeatable, e.g. --skip-file wandb_run.json)",
    )
    args = parser.parse_args()

    if not args.seed.exists():
        print(f"error: seed file not found: {args.seed}", file=sys.stderr)
        sys.exit(1)

    try:
        seed = json.loads(args.seed.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"error: cannot read seed file: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        passport = assemble_passport(
            args.checkpoint_dir,
            seed,
            generated_at=args.generated_at,
            target_repo=args.target_repo,
            training_repo=args.training_repo,
            extra_skip_files=args.skip_files,
            extra_skip_dirs=args.skip_dirs,
        )
    except InvalidSeedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or (args.checkpoint_dir / "MODEL_PASSPORT.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(passport, indent=2) + "\n")
    print(f"Wrote {out_path}")

    n_weight = len(passport.get("weight_integrity", {}).get("weight_files", []))
    has_prov = bool(passport.get("provenance"))
    seed_sections = [k for k in passport if k not in {
        "schema_version", "generated_by", "generated_at",
        "weight_integrity", "provenance",
    }]
    print(f"  seed sections: {', '.join(seed_sections) if seed_sections else 'none'}")
    print(f"  weight files hashed: {n_weight}")
    print(f"  provenance: {'yes' if has_prov else 'none'}")


if __name__ == "__main__":
    main()
