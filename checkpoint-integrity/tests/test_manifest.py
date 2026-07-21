import json
from pathlib import Path

from checkpoint_integrity import MANIFEST_FILENAME, verify_manifest, write_manifest


def test_write_and_verify_manifest(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model": "act"}\n')
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.bin").write_bytes(b"weights")

    manifest_path = write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text())

    assert manifest_path.name == MANIFEST_FILENAME
    assert [entry["path"] for entry in payload["artifacts"]] == [
        "config.json",
        "weights/model.bin",
    ]
    result = verify_manifest(tmp_path)
    assert result.valid
    assert result.checked == 2


def test_verify_detects_modified_file(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"before")
    write_manifest(tmp_path)

    artifact.write_bytes(b"after")

    result = verify_manifest(tmp_path)
    assert not result.valid
    assert any(
        "mismatch for model.bin" in error
        for error in result.errors
    )


def test_extra_files_warn_by_default_and_fail_strict(tmp_path: Path) -> None:
    (tmp_path / "model.bin").write_bytes(b"weights")
    write_manifest(tmp_path)
    (tmp_path / ".gitattributes").write_text("*.bin filter=lfs\n")

    default_result = verify_manifest(tmp_path)
    strict_result = verify_manifest(tmp_path, strict=True)

    assert default_result.valid
    assert default_result.extra_files == [".gitattributes"]
    assert not strict_result.valid
    assert any("unlisted files present" in error for error in strict_result.errors)


def test_symlink_is_not_manifested(tmp_path: Path) -> None:
    target = tmp_path / "step_10000"
    target.mkdir()
    (target / "model.bin").write_bytes(b"weights")
    (tmp_path / "last").symlink_to(target, target_is_directory=True)

    manifest_path = write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text())

    assert payload["skipped_symlinks"] == ["last"]
    assert [entry["path"] for entry in payload["artifacts"]] == [
        "step_10000/model.bin",
    ]
