"""
CLI: check-publish-ready

Pre-upload gate that verifies a checkpoint directory contains the required
packaging artifacts (README.md, TRAINING_LOG.md, MODEL_PASSPORT.json,
SIGNOFF.json) and that they are structurally valid.  Also runs the internal
validator with --require-signoff semantics to catch passport/signoff
integrity issues.

Packaging errors (missing files, empty docs, bad JSON) are reported
separately from validation errors (passport/signoff integrity failures).

Usage:
    check-publish-ready <checkpoint_dir>
    check-publish-ready <checkpoint_dir> --target-repo /path/to/deploy/repo
    check-publish-ready <checkpoint_dir> --json

Exit code 0 = ready to publish.  Exit code 1 = not ready.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from checkpoint_passport import self_validate_passport, Status


REQUIRED_FILES = [
    "README.md",
    "TRAINING_LOG.md",
    "MODEL_PASSPORT.json",
    "SIGNOFF.json",
]

NON_EMPTY_FILES = [
    "README.md",
    "TRAINING_LOG.md",
]


@dataclass
class PublishReadyResult:
    checkpoint_dir: Path
    packaging_errors: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)

    @property
    def errors(self) -> List[str]:
        return self.packaging_errors + self.validation_errors

    @property
    def ready(self) -> bool:
        return len(self.packaging_errors) == 0 and len(self.validation_errors) == 0

    def to_dict(self) -> dict:
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "ready": self.ready,
            "packaging_errors": list(self.packaging_errors),
            "validation_errors": list(self.validation_errors),
        }


def check_publish_ready(
    checkpoint_dir: str | Path,
    *,
    target_repo: str | Path | None = None,
) -> PublishReadyResult:
    """Check whether a checkpoint directory is ready for HF upload.

    Returns a PublishReadyResult with all discovered errors, split into
    packaging errors (missing/empty/malformed files) and validation errors
    (passport/signoff integrity failures from the internal validator).

    Does not short-circuit — reports every problem it finds.
    """
    ckpt = Path(checkpoint_dir)
    packaging: List[str] = []
    validation: List[str] = []

    if not ckpt.is_dir():
        return PublishReadyResult(
            checkpoint_dir=ckpt,
            packaging_errors=[f"Checkpoint directory does not exist: {ckpt}"],
        )

    present_files: set[str] = set()

    for filename in REQUIRED_FILES:
        path = ckpt / filename
        if not path.is_file():
            packaging.append(f"Missing required file: {filename}")
        else:
            present_files.add(filename)

    for filename in NON_EMPTY_FILES:
        if filename not in present_files:
            continue
        content = (ckpt / filename).read_text()
        if not content.strip():
            packaging.append(f"{filename} is empty")

    if "MODEL_PASSPORT.json" in present_files:
        try:
            passport = json.loads((ckpt / "MODEL_PASSPORT.json").read_text())
            if not isinstance(passport, dict):
                packaging.append(
                    "MODEL_PASSPORT.json must contain a JSON object, "
                    f"got {type(passport).__name__}"
                )
        except (json.JSONDecodeError, OSError) as exc:
            packaging.append(f"MODEL_PASSPORT.json is not valid JSON: {exc}")

    if "SIGNOFF.json" in present_files:
        try:
            signoff = json.loads((ckpt / "SIGNOFF.json").read_text())
            if not isinstance(signoff, dict):
                packaging.append(
                    "SIGNOFF.json must contain a JSON object, "
                    f"got {type(signoff).__name__}"
                )
        except (json.JSONDecodeError, OSError) as exc:
            packaging.append(f"SIGNOFF.json is not valid JSON: {exc}")

    # Run internal validator with require_signoff if packaging looks viable.
    # Skip if passport or signoff are already known-broken at the file level
    # — the validator would just re-report the same parse errors.
    can_validate = (
        "MODEL_PASSPORT.json" in present_files
        and "SIGNOFF.json" in present_files
        and not packaging  # no packaging-level errors yet
    )
    if can_validate:
        try:
            result = self_validate_passport(
                ckpt,
                require_signoff=True,
                target_repo=target_repo,
            )
            for obs in result.observations:
                if obs.status is Status.FAIL:
                    validation.append(f"{obs.check}: {obs.message}")
        except Exception as exc:
            validation.append(f"Validation error: {exc}")

    return PublishReadyResult(
        checkpoint_dir=ckpt,
        packaging_errors=packaging,
        validation_errors=validation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="check-publish-ready",
        description=(
            "Verify a checkpoint directory has the required packaging "
            "artifacts before HF upload."
        ),
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="output structured JSON instead of human-readable report",
    )
    parser.add_argument(
        "--target-repo", type=Path, default=None,
        help="deployment target repo; enables deployment_repo_commit check",
    )
    args = parser.parse_args()

    result = check_publish_ready(
        args.checkpoint_dir,
        target_repo=args.target_repo,
    )

    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        if result.ready:
            print(f"ready: {args.checkpoint_dir}")
        else:
            print(f"NOT ready: {args.checkpoint_dir}")
            if result.packaging_errors:
                print("  Packaging:")
                for err in result.packaging_errors:
                    print(f"    - {err}")
            if result.validation_errors:
                print("  Validation:")
                for err in result.validation_errors:
                    print(f"    - {err}")

    sys.exit(0 if result.ready else 1)


if __name__ == "__main__":
    main()
