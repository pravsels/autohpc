from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_FILENAME = "CHECKPOINT_MANIFEST.json"
SCHEMA_VERSION = "1"
_IGNORED_PARTS = {".git", "__pycache__"}


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    checked: int
    errors: list[str] = field(default_factory=list)
    extra_files: list[str] = field(default_factory=list)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_paths(root: Path) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    skipped_symlinks: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        if path.name == MANIFEST_FILENAME:
            continue
        if path.is_symlink():
            skipped_symlinks.append(relative.as_posix())
            continue
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix()), skipped_symlinks


def build_manifest(checkpoint_dir: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {root}")

    paths, skipped_symlinks = _artifact_paths(root)
    artifacts = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "algorithm": "sha256",
        "artifacts": artifacts,
        "skipped_symlinks": skipped_symlinks,
    }


def write_manifest(checkpoint_dir: str | Path) -> Path:
    root = Path(checkpoint_dir).resolve()
    manifest = build_manifest(root)
    output = root / MANIFEST_FILENAME

    fd, temp_name = tempfile.mkstemp(prefix=".checkpoint-manifest-", dir=root)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        os.replace(temp_name, output)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return output


def _safe_artifact_path(root: Path, raw_path: Any) -> tuple[Path | None, str | None]:
    if not isinstance(raw_path, str) or not raw_path:
        return None, "artifact path must be a non-empty string"
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None, f"unsafe artifact path: {raw_path!r}"
    candidate = root / relative
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None, f"artifact escapes checkpoint root: {raw_path!r}"
    return candidate, None


def verify_manifest(
    checkpoint_dir: str | Path,
    *,
    strict: bool = False,
) -> VerificationResult:
    root = Path(checkpoint_dir).resolve()
    manifest_path = root / MANIFEST_FILENAME
    errors: list[str] = []

    if not manifest_path.is_file():
        return VerificationResult(
            valid=False,
            checked=0,
            errors=[f"missing {MANIFEST_FILENAME}"],
        )

    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return VerificationResult(
            valid=False,
            checked=0,
            errors=[f"invalid {MANIFEST_FILENAME}: {exc}"],
        )

    if not isinstance(payload, dict):
        return VerificationResult(
            valid=False,
            checked=0,
            errors=[f"{MANIFEST_FILENAME} must contain a JSON object"],
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {payload.get('schema_version')!r}")
    if payload.get("algorithm") != "sha256":
        errors.append(f"unsupported algorithm: {payload.get('algorithm')!r}")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return VerificationResult(
            valid=False,
            checked=0,
            errors=errors + ["artifacts must be a list"],
        )

    declared: set[str] = set()
    checked = 0
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            errors.append("artifact entry must be a JSON object")
            continue
        raw_path = artifact.get("path")
        path, path_error = _safe_artifact_path(root, raw_path)
        if path_error:
            errors.append(path_error)
            continue
        assert path is not None
        normalized = Path(raw_path).as_posix()
        if normalized in declared:
            errors.append(f"duplicate artifact path: {normalized}")
            continue
        declared.add(normalized)
        if path.is_symlink():
            errors.append(f"artifact is a symlink: {normalized}")
            continue
        if not path.is_file():
            errors.append(f"missing artifact: {normalized}")
            continue

        expected_size = artifact.get("size_bytes")
        actual_size = path.stat().st_size
        if not isinstance(expected_size, int) or expected_size != actual_size:
            errors.append(
                f"size mismatch for {normalized}: expected {expected_size!r}, got {actual_size}"
            )
            continue

        expected_hash = artifact.get("sha256")
        actual_hash = _sha256_file(path)
        if not isinstance(expected_hash, str) or expected_hash != actual_hash:
            errors.append(f"sha256 mismatch for {normalized}")
            continue
        checked += 1

    disk_paths, skipped_symlinks = _artifact_paths(root)
    extra_files = sorted(
        path.relative_to(root).as_posix()
        for path in disk_paths
        if path.relative_to(root).as_posix() not in declared
    )
    extra_files.extend(sorted(skipped_symlinks))
    if strict and extra_files:
        errors.append(f"unlisted files present: {', '.join(extra_files)}")

    return VerificationResult(
        valid=not errors,
        checked=checked,
        errors=errors,
        extra_files=extra_files,
    )
