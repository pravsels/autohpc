"""
CLI entry point: hash the passport + weight files and write SIGNOFF.json.

Sign only after `validate-checkpoint` passes (no hard fails).  The
verdict in the signoff is derived from the validator's own verdict:

    ship_it          -> "pass"
    human_look_here  -> "soft_signal"   (must supply --reason)
    dont_ship        -> refuses to sign  (exit 1, prints failures)

The signed file lists every weight file the passport already declares
under `weight_integrity.weight_files`, recomputed live from disk.  The
passport itself is hashed and added to the artifact list so any later
edit to the passport invalidates the signoff (which is the whole point).

Usage:
    sign-checkpoint /path/to/checkpoint
    sign-checkpoint /path/to/checkpoint --reason "documented soft signal X"
    sign-checkpoint /path/to/checkpoint --tool my-passport-agent --version 1.2
    sign-checkpoint /path/to/checkpoint --dry-run   # print, don't write

Exit codes:
    0 = signoff written (or would be, with --dry-run)
    1 = validator reports hard failures; nothing written
    2 = CLI usage error (missing --reason for soft-signal verdict, etc.)
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

from checkpoint_passport import (
    PASSPORT_FILENAME,
    SIGNOFF_FILENAME,
    SIGNOFF_SCHEMA_VERSION,
    Status,
    self_validate_passport,
)
from checkpoint_passport.schema import Signoff, SignoffArtifact, SignoffSigner


_TOOL_NAME = "sign-checkpoint"
_TOOL_VERSION = "0.1.0"


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _isoformat_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _derive_verdict(validator_verdict: str) -> str:
    """Map validator verdict (ship_it / human_look_here / dont_ship) to
    the SIGNOFF.json verdict vocabulary (pass / soft_signal / fail)."""
    return {
        "ship_it": "pass",
        "human_look_here": "soft_signal",
        "dont_ship": "fail",
    }[validator_verdict]


def _summarise_soft_signals(observations) -> str:
    """One-line auto-reason listing the soft-signal check names.  Used
    only when the operator forgot --reason; we still want the signoff to
    record *something* about why a soft signal was accepted."""
    soft = [o.check for o in observations if o.status is Status.SOFT_SIGNAL]
    if not soft:
        return ""
    return f"auto: {len(soft)} soft signal(s) accepted -- " + ", ".join(soft)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sign-checkpoint",
        description=(
            "Compute file hashes and write SIGNOFF.json after a passing "
            "validate-checkpoint run."
        ),
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument(
        "--reason", type=str, default=None,
        help=(
            "one-line justification stored in signoff.verdict_reason. "
            "Required when the validator reports soft signals; ignored "
            "otherwise."
        ),
    )
    parser.add_argument(
        "--tool", type=str, default=_TOOL_NAME,
        help="signed_by.tool field (default: %(default)s)",
    )
    parser.add_argument(
        "--version", type=str, default=_TOOL_VERSION,
        help="signed_by.version field (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the would-be signoff to stdout; do not write SIGNOFF.json",
    )
    parser.add_argument(
        "--norm-stats", type=Path, action="append", default=[],
        help="forwarded to validate-checkpoint (extra norm stats files)",
    )
    parser.add_argument(
        "--dataset-path", type=Path, default=None,
        help="forwarded to validate-checkpoint (local dataset path)",
    )
    args = parser.parse_args()

    ckpt_dir: Path = args.checkpoint_dir.resolve()
    passport_path = ckpt_dir / PASSPORT_FILENAME

    if not passport_path.is_file():
        parser.error(
            f"no {PASSPORT_FILENAME} at {ckpt_dir}; cannot sign without a passport"
        )

    extra_norm_stats = args.norm_stats or None
    dataset_path = args.dataset_path

    result = self_validate_passport(
        ckpt_dir,
        dataset_path=dataset_path,
        extra_norm_stats=extra_norm_stats,
    )

    if result.has_failures:
        print(
            f"refusing to sign: validator verdict = {result.verdict!r}",
            file=sys.stderr,
        )
        print(
            "the following hard checks failed -- fix the passport (or the "
            "checkpoint) and re-run:",
            file=sys.stderr,
        )
        for o in result.observations:
            if o.status is Status.FAIL:
                print(f"  {o.check}: {o.message}", file=sys.stderr)
        sys.exit(1)

    sign_verdict = _derive_verdict(result.verdict)

    # Verdict reason policy:
    #   pass         -> reason optional (defaults to a short auto-string)
    #   soft_signal  -> reason required (operator must acknowledge)
    if sign_verdict == "soft_signal" and not args.reason:
        auto = _summarise_soft_signals(result.observations)
        parser.error(
            "validator reports soft signals; --reason is required so the "
            "signoff records why each was accepted.\n  "
            f"auto-summary you can paste / extend: {auto!r}"
        )

    verdict_reason = args.reason
    if verdict_reason is None and sign_verdict == "pass":
        verdict_reason = "all hard checks pass"

    # Build the artifact list: passport + every weight file the passport
    # already declared.  We do NOT re-discover weight files on disk --
    # the passport's weight_integrity is the single source of truth, and
    # any drift between passport-declared and on-disk files would have
    # been caught by the validator's `internals_vs_weight_files` check.
    artifacts: List[SignoffArtifact] = []

    artifacts.append(SignoffArtifact(
        path=PASSPORT_FILENAME,
        sha256=_sha256_file(passport_path),
    ))

    passport = result.passport_load.passport
    if passport is not None:
        for wf in passport.weight_integrity.weight_files:
            file_on_disk = (ckpt_dir / wf.path).resolve()
            if not file_on_disk.is_file():
                # Validator should have caught this but be defensive --
                # signing a manifest that points at missing files is worse
                # than failing loudly here.
                print(
                    f"refusing to sign: weight file declared in passport but "
                    f"missing on disk: {wf.path}",
                    file=sys.stderr,
                )
                sys.exit(1)
            artifacts.append(SignoffArtifact(
                path=wf.path,
                sha256=_sha256_file(file_on_disk),
            ))

    signoff = Signoff(
        schema_version=SIGNOFF_SCHEMA_VERSION,
        signed_at=_isoformat_now(),
        signed_by=SignoffSigner(tool=args.tool, version=args.version),
        artifacts=artifacts,
        verdict=sign_verdict,
        verdict_reason=verdict_reason,
    )

    payload = json.dumps(asdict(signoff), indent=2)

    if args.dry_run:
        print(payload)
        print(
            f"\n[dry-run] would write to {ckpt_dir / SIGNOFF_FILENAME}",
            file=sys.stderr,
        )
        sys.exit(0)

    out_path = ckpt_dir / SIGNOFF_FILENAME
    out_path.write_text(payload + "\n")
    print(
        f"wrote {out_path}  verdict={sign_verdict}  "
        f"artifacts={len(artifacts)}",
    )

    # Re-validate with --require-signoff semantics to confirm the signoff
    # we just wrote round-trips clean.  Mirrors what a CI gate would do.
    # If the round-trip fails we delete the freshly written signoff so the
    # checkpoint is left in its pre-sign state -- a half-good signoff on
    # disk is worse than no signoff (downstream gates would treat it as
    # authoritative).
    confirm = self_validate_passport(
        ckpt_dir,
        dataset_path=dataset_path,
        extra_norm_stats=extra_norm_stats,
        require_signoff=True,
    )
    if confirm.has_failures:
        try:
            out_path.unlink()
        except OSError:
            pass
        print(
            "ERROR: post-sign re-validation reports hard failures. "
            f"Deleted the newly written {out_path} -- the checkpoint is "
            "in its pre-sign state. Failing checks:",
            file=sys.stderr,
        )
        for o in confirm.observations:
            if o.status is Status.FAIL:
                print(f"  {o.check}: {o.message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
