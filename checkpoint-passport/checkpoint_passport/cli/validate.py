"""
CLI entry point: run all Layer 1 validation checks on a checkpoint directory.

Loads MODEL_PASSPORT.json + SIGNOFF.json when present and runs every
deterministic check defined in `checkpoint_passport`. When no passport is
present, only the static checks have meaningful output; passport-driven
checks emit NOT_CHECKED (hidden by default).

Usage:
    validate-checkpoint /path/to/checkpoint
    validate-checkpoint /path/to/checkpoint --json
    validate-checkpoint /path/to/checkpoint --out report.json
    validate-checkpoint /path/to/checkpoint --norm-stats /extra/stats.json
    validate-checkpoint /path/to/checkpoint --dataset-path /local/dataset
    validate-checkpoint /path/to/checkpoint --target-repo /path/to/deploy/repo
    validate-checkpoint /path/to/checkpoint --require-signoff
    validate-checkpoint /path/to/checkpoint --show-not-checked
    validate-checkpoint /path/to/checkpoint --skip-section signoff

Exit code 0 = no hard failures (pass or soft signals).
Exit code 1 = at least one hard failure.
Exit code 2 = CLI usage error (e.g. mutually exclusive flags).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from checkpoint_passport import self_validate_passport
from checkpoint_passport.runner import KNOWN_CATEGORIES


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="validate-checkpoint",
        description="Validate a policy checkpoint directory",
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument(
        "--json", action="store_true",
        help="output structured JSON instead of human report",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="write report to file (default: stdout)",
    )
    parser.add_argument(
        "--norm-stats", type=Path, action="append", default=[],
        help="additional norm stats files to include (can be repeated)",
    )
    parser.add_argument(
        "--dataset-path", type=Path, default=None,
        help="local LeRobot dataset directory; enables input_contract_vs_dataset",
    )
    parser.add_argument(
        "--target-repo", type=Path, default=None,
        help="deployment target repo; enables deployment_repo_commit check",
    )
    parser.add_argument(
        "--require-signoff", action="store_true",
        help="missing SIGNOFF.json becomes hard fail (use on prod deploys)",
    )
    parser.add_argument(
        "--show-not-checked", action="store_true",
        help="include NOT_CHECKED rows in the report (verbose forensic view)",
    )
    parser.add_argument(
        "--skip-section", action="append", default=[],
        choices=list(KNOWN_CATEGORIES),
        help=(
            "drop every check whose category matches; useful while a "
            "section's checks are still being shaken out. Repeat to skip "
            "multiple. WARNING: --skip-section signoff disables all "
            "tamper-detection checks (passport + weight-file checksums) "
            "and must NOT be used on production deploys; it is rejected "
            "when combined with --require-signoff."
        ),
    )
    args = parser.parse_args()

    try:
        result = self_validate_passport(
            args.checkpoint_dir,
            dataset_path=args.dataset_path,
            target_repo=args.target_repo,
            require_signoff=args.require_signoff,
            extra_norm_stats=args.norm_stats or None,
            skip_sections=args.skip_section or None,
        )
    except ValueError as e:
        # Currently this only fires for --require-signoff +
        # --skip-section signoff. Surface it as a CLI usage error
        # (exit 2) rather than a check failure (exit 1) so CI
        # configuration mistakes are obviously distinct from real
        # checkpoint failures.
        parser.error(str(e))

    if args.json:
        payload = {
            "checkpoint_dir": str(result.checkpoint_dir),
            "verdict": result.verdict,
            "observations": [o.to_dict() for o in result.observations],
            "passport_load_status": result.passport_load.status.value,
            "passport_load_notes": result.passport_load.notes,
            "skipped_sections": result.skipped_sections,
        }
        output = json.dumps(payload, indent=2)
    else:
        output = result.render(show_not_checked=args.show_not_checked)

    if args.out:
        args.out.write_text(output)
        print(f"Report written to {args.out}")
    else:
        print(output)

    sys.exit(1 if result.has_failures else 0)


if __name__ == "__main__":
    main()
