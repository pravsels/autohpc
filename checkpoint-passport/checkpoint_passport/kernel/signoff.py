"""
Signoff verification: did the upstream skill sign off on this passport, and
do the artifact hashes still match?

Cheap, deterministic, and unambiguous. Runs first because everything
downstream loses meaning if checksums don't match.

Soft-vs-hard rules (per plan B4):
  signoff missing                       -> SOFT_SIGNAL (default) /
                                            FAIL when --require-signoff
  signoff schema version unsupported    -> SOFT_SIGNAL
  passport checksum mismatch            -> FAIL  (tampering / corruption)
  weight file checksum mismatch         -> FAIL  (download corruption)

The `--require-signoff` flag is honoured by `check_signoff_present` via
the `require_signoff` argument.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from ..observation import Observation, Status
from ..passport import PassportLoadResult, PASSPORT_FILENAME
from ..schema import is_signoff_version_supported


CATEGORY = "signoff"


# ── Hash helpers ────────────────────────────────────────────────────────


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute hex sha256 of a file by streaming in 1MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ── Checks ──────────────────────────────────────────────────────────────


def check_signoff_present(
    load: PassportLoadResult,
    require_signoff: bool = False,
) -> Observation:
    """Is SIGNOFF.json present and parseable alongside the passport?

    Three states matter to operators:
      - file present and parsed cleanly         -> PASS
      - file present on disk but unparseable    -> FAIL (always; bad data
        is worse than no data, and the signoff is the security boundary)
      - file not on disk                        -> SOFT_SIGNAL by default,
        FAIL when require_signoff=True
    """
    if load.has_signoff:
        return Observation(
            check="signoff_present",
            status=Status.PASS,
            message=f"SIGNOFF.json found (verdict={load.signoff.verdict!r})",
            details={
                "signoff_path": str(load.signoff_path),
                "verdict": load.signoff.verdict,
                "signed_at": load.signoff.signed_at,
            },
            category=CATEGORY,
        )

    if load.signoff_present_on_disk:
        # File exists but loader couldn't produce a usable Signoff dataclass.
        return Observation(
            check="signoff_present",
            status=Status.FAIL,
            message=(
                f"SIGNOFF.json found at {load.signoff_path!s} but could not "
                "be parsed (see loader notes)"
            ),
            details={
                "signoff_path": str(load.signoff_path),
                "loader_notes": [
                    n for n in load.notes if str(load.signoff_path) in n
                ] or load.notes,
            },
            category=CATEGORY,
        )

    severity = Status.FAIL if require_signoff else Status.SOFT_SIGNAL
    return Observation(
        check="signoff_present",
        status=severity,
        message=(
            "SIGNOFF.json not found on disk"
            + (" (--require-signoff)" if require_signoff else "")
        ),
        details={"checkpoint_dir": str(load.checkpoint_dir)},
        category=CATEGORY,
    )


def check_signoff_schema_version_supported(load: PassportLoadResult) -> Observation:
    """Is SIGNOFF.schema_version one this validator knows how to read?"""
    if not load.has_signoff:
        return Observation(
            check="signoff_schema_version_supported",
            status=Status.NOT_CHECKED,
            message="no signoff to version-check",
            details={},
            category=CATEGORY,
        )

    version = load.signoff.schema_version
    if is_signoff_version_supported(version):
        return Observation(
            check="signoff_schema_version_supported",
            status=Status.PASS,
            message=f"signoff schema_version={version!r} supported",
            details={"version": version},
            category=CATEGORY,
        )

    return Observation(
        check="signoff_schema_version_supported",
        status=Status.SOFT_SIGNAL,
        message=f"signoff schema_version={version!r} not in compatibility table",
        details={"version": version},
        category=CATEGORY,
    )


def check_passport_checksum_valid(load: PassportLoadResult) -> Observation:
    """Recompute sha256 of MODEL_PASSPORT.json on disk and compare to signoff."""
    if not load.has_signoff:
        return Observation(
            check="passport_checksum_valid",
            status=Status.NOT_CHECKED,
            message="no signoff -- nothing to compare against",
            details={},
            category=CATEGORY,
        )

    if load.passport_path is None or not load.passport_path.is_file():
        return Observation(
            check="passport_checksum_valid",
            status=Status.FAIL,
            message=f"signoff present but {PASSPORT_FILENAME} missing on disk",
            details={"checkpoint_dir": str(load.checkpoint_dir)},
            category=CATEGORY,
        )

    declared = _signoff_lookup(load, PASSPORT_FILENAME)
    if declared is None:
        return Observation(
            check="passport_checksum_valid",
            status=Status.SOFT_SIGNAL,
            message=f"signoff lists no entry for {PASSPORT_FILENAME}",
            details={"signoff_paths": _signoff_paths(load)},
            category=CATEGORY,
        )

    actual = _sha256_file(load.passport_path)
    if actual == declared:
        return Observation(
            check="passport_checksum_valid",
            status=Status.PASS,
            message="passport sha256 matches signoff",
            details={"sha256": actual},
            category=CATEGORY,
        )

    return Observation(
        check="passport_checksum_valid",
        status=Status.FAIL,
        message="passport sha256 does not match signoff (tampering or corruption)",
        details={"signoff_sha256": declared, "actual_sha256": actual},
        category=CATEGORY,
    )


def check_weight_files_match_signoff(load: PassportLoadResult) -> Observation:
    """Bidirectional weight-file integrity check against the signoff.

    For every weight file the signoff lists:
      - artifact path must resolve to a real file UNDER the checkpoint root
        (rejects "../../../etc/passwd" and any other path that escapes the
        checkpoint dir, even if readable; this is a security gate, not a
        consistency one)
      - file must exist on disk
      - sha256 of the on-disk bytes must equal what the signoff records

    AND, for every weight file on disk:
      - it must be listed in the signoff (catches a corrupted shard that
        wasn't recorded, or weights that drifted in after sign-time)
    """
    if not load.has_signoff:
        return Observation(
            check="weight_files_match_signoff",
            status=Status.NOT_CHECKED,
            message="no signoff -- nothing to compare against",
            details={},
            category=CATEGORY,
        )

    weight_artifacts = [
        a for a in load.signoff.artifacts
        if a.path != PASSPORT_FILENAME
    ]
    if not weight_artifacts:
        return Observation(
            check="weight_files_match_signoff",
            status=Status.SOFT_SIGNAL,
            message="signoff lists no weight files",
            details={"signoff_paths": _signoff_paths(load)},
            category=CATEGORY,
        )

    # Resolve checkpoint root once for traversal containment.
    root = load.checkpoint_dir.resolve()

    mismatches: List[dict] = []
    missing: List[str] = []
    matches: List[str] = []
    escapes: List[dict] = []
    listed_resolved: set[Path] = set()

    for art in weight_artifacts:
        # Reject paths that escape the checkpoint root (resolve() collapses
        # any "..", so we compare resolved paths). Symlinks are followed by
        # resolve(); if the symlink target is outside root, we still flag it.
        try:
            resolved = (load.checkpoint_dir / art.path).resolve()
        except (OSError, RuntimeError) as e:
            escapes.append({"path": art.path, "error": f"{type(e).__name__}: {e}"})
            continue

        try:
            resolved.relative_to(root)
        except ValueError:
            escapes.append({
                "path": art.path,
                "resolved_to": str(resolved),
                "reason": "escapes checkpoint root",
            })
            continue

        listed_resolved.add(resolved)

        if not resolved.is_file():
            missing.append(art.path)
            continue
        actual = _sha256_file(resolved)
        if actual != art.sha256:
            mismatches.append({
                "path": art.path,
                "signoff_sha256": art.sha256,
                "actual_sha256": actual,
            })
        else:
            matches.append(art.path)

    # Reverse direction: any weight files on disk that the signoff didn't
    # list. We define "weight file" as anything matching common weight
    # extensions; non-weight files (configs, READMEs, tokenizers) are not
    # the signoff's responsibility and aren't surfaced here.
    extras = _disk_weight_files_not_in_signoff(root, listed_resolved)

    problems = bool(mismatches or missing or escapes or extras)
    if problems:
        msg_parts = []
        if mismatches:
            msg_parts.append(f"{len(mismatches)} sha mismatch(es)")
        if missing:
            msg_parts.append(f"{len(missing)} missing")
        if escapes:
            msg_parts.append(f"{len(escapes)} escape(s) checkpoint root")
        if extras:
            msg_parts.append(f"{len(extras)} on-disk weight(s) not in signoff")
        return Observation(
            check="weight_files_match_signoff",
            status=Status.FAIL,
            message=f"weight integrity broken: {', '.join(msg_parts)}",
            details={
                "mismatches": mismatches,
                "missing": missing,
                "path_escapes": escapes,
                "unlisted_on_disk": extras,
                "matches": matches,
            },
            category=CATEGORY,
        )

    return Observation(
        check="weight_files_match_signoff",
        status=Status.PASS,
        message=f"all {len(matches)} weight file(s) match signoff "
                "(no extras on disk)",
        details={"files": matches},
        category=CATEGORY,
    )


# Common weight file extensions for the reverse-direction extras check.
# Conservative -- we don't want to flag a tokenizer as a missing weight.
_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}


def _disk_weight_files_not_in_signoff(
    root: Path,
    listed_resolved: set[Path],
) -> List[str]:
    """Recursively scan checkpoint root for weight files; return any whose
    resolved path is not in `listed_resolved` (i.e. signoff missed them).
    Paths returned are relative to root for readability.
    """
    extras: List[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _WEIGHT_SUFFIXES:
            continue
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved in listed_resolved:
            continue
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            rel = resolved
        extras.append(str(rel))
    return sorted(extras)


# ── Helpers ─────────────────────────────────────────────────────────────


def _signoff_lookup(load: PassportLoadResult, name: str):
    """Return the sha256 the signoff records for `name`, or None."""
    if not load.has_signoff:
        return None
    for art in load.signoff.artifacts:
        if art.path == name:
            return art.sha256
    return None


def _signoff_paths(load: PassportLoadResult) -> List[str]:
    if not load.has_signoff:
        return []
    return [a.path for a in load.signoff.artifacts]


# ── Registration ────────────────────────────────────────────────────────


def run_signoff_checks(
    load: PassportLoadResult,
    require_signoff: bool = False,
) -> List[Observation]:
    """Run every signoff check in order. Honors --require-signoff for the
    presence check; the rest are NOT_CHECKED when no signoff is present."""
    return [
        check_signoff_present(load, require_signoff=require_signoff),
        check_signoff_schema_version_supported(load),
        check_passport_checksum_valid(load),
        check_weight_files_match_signoff(load),
    ]
