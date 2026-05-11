"""
Tests for reference test vector extraction and dataset commit validation.

Verifies:
  - OpenPI extraction with reference data writes reference_test_vector to seed
  - State array saved as .npy, hashed, referenced by relative path
  - Image frames saved, hashed per camera/frame, referenced by relative path
  - Missing dataset path fails with clear error
  - Missing required keys in reference sample fails
  - reference_test_vector is an allowed passport seed section
  - training_datasets_resolvable treats missing commits as soft signal
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from checkpoint_passport.passport_seed import validate_seed, ALLOWED_SEED_SECTIONS
from checkpoint_passport.runtime_extractors.openpi import OpenPIExtractor
from checkpoint_passport.runtime_adapters.base import RuntimeAdapter


# ── Fake OpenPI module tree ─────────────────────────────────────────────

FAKE_OPENPI_MODULES = [
    "openpi",
    "openpi.models",
    "openpi.models.model",
    "openpi.shared",
    "openpi.shared.download",
    "openpi.training",
    "openpi.training.config",
]


def _make_fake_output_transform(action_dim: int = 7):
    t = types.SimpleNamespace()
    t.action_dim = action_dim
    return t


def _make_fake_data_config(*, action_dim: int = 7):
    output_transform = _make_fake_output_transform(action_dim)
    data_transforms = types.SimpleNamespace()
    data_transforms.inputs = []
    data_transforms.outputs = [output_transform]
    dc = types.SimpleNamespace()
    dc.data_transforms = data_transforms
    return dc


def _make_fake_train_config(
    *,
    model_type: str = "pi05",
    action_dim: int = 32,
    action_horizon: int = 50,
    default_prompt: str = "build a block tower",
    repo_id: str = "villekuosmanen/build_block_tower",
    robot_action_dim: int = 7,
):
    model_type_enum = types.SimpleNamespace()
    model_type_enum.value = model_type

    model = types.SimpleNamespace()
    model.model_type = model_type_enum
    model.action_dim = action_dim
    model.action_horizon = action_horizon

    data_factory = types.SimpleNamespace()
    data_factory.use_delta_actions = True
    data_factory.default_prompt = default_prompt
    data_factory.repo_id = repo_id
    data_factory.joints_only = True
    data_factory.delta_action_mask = None
    data_factory.create = lambda assets_dirs, model_config: _make_fake_data_config(
        action_dim=robot_action_dim,
    )

    cfg = types.SimpleNamespace()
    cfg.model = model
    cfg.data = data_factory
    cfg.assets_dirs = Path("/fake/assets")
    return cfg


# ── Fake RuntimeAdapter with reference sample support ───────────────────


class FakeRuntimeAdapterWithReference(RuntimeAdapter):
    """Fake adapter that returns a reference sample from a dataset."""

    def __init__(
        self,
        *,
        state_dim: int = 7,
        n_frames: int = 10,
        image_size: int = 224,
        camera_keys: Optional[List[str]] = None,
        prompt: str = "build a block tower",
    ):
        self._state_dim = state_dim
        self._n_frames = n_frames
        self._image_size = image_size
        self._camera_keys = camera_keys or ["front", "wrist"]
        self._prompt = prompt

    def load(self, checkpoint_dir: Path, *, device: str, **kwargs: Any) -> None:
        pass

    def count_parameters(self) -> Optional[Dict[str, int]]:
        return {"total_params": 1_500_000}

    def smoke_inference(self) -> Optional[Dict[str, Any]]:
        return {"status": "pass", "elapsed_ms": 42.0}

    def library_versions(self) -> Dict[str, str]:
        return {"python": "3.11.14"}

    def extract_reference_sample(
        self,
        dataset_path: Path,
        *,
        episode_index: int = 0,
        start_frame: int = 0,
        num_frames: int = 10,
    ) -> Dict[str, Any]:
        states = np.random.randn(num_frames, self._state_dim).astype(np.float32)
        images: Dict[str, List[np.ndarray]] = {}
        for cam in self._camera_keys:
            images[cam] = [
                np.random.randint(
                    0, 255,
                    (self._image_size, self._image_size, 3),
                    dtype=np.uint8,
                )
                for _ in range(num_frames)
            ]
        return {
            "states": states,
            "images": images,
            "prompt": self._prompt,
            "episode_index": episode_index,
            "start_frame": start_frame,
            "num_frames": num_frames,
            "dataset_path": str(dataset_path),
        }


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def _fake_openpi(monkeypatch):
    """Install fake 'openpi' package into sys.modules."""
    for mod_name in FAKE_OPENPI_MODULES:
        monkeypatch.setitem(sys.modules, mod_name, types.ModuleType(mod_name))

    fake_config_mod = sys.modules["openpi.training.config"]
    fake_config_mod.get_config = lambda name: _make_fake_train_config()

    openpi_mod = sys.modules["openpi"]
    openpi_mod.__version__ = "0.3.1"

    yield

    for mod_name in FAKE_OPENPI_MODULES:
        sys.modules.pop(mod_name, None)


def _patch_adapter(monkeypatch, adapter=None):
    fake = adapter or FakeRuntimeAdapterWithReference()
    monkeypatch.setattr(
        "checkpoint_passport.runtime_adapters.openpi.OpenPIRuntimeAdapter",
        lambda: fake,
    )


def _make_fake_dataset(tmp_path: Path) -> Path:
    """Create a minimal fake dataset directory."""
    ds = tmp_path / "dataset"
    ds.mkdir()
    (ds / "meta_data").mkdir()
    (ds / "meta_data" / "episode_0.json").write_text("{}")
    return ds


# ── 1. reference_test_vector is an allowed seed section ─────────────────


def test_reference_test_vector_in_allowed_sections():
    assert "reference_test_vector" in ALLOWED_SEED_SECTIONS


# ── 2. Extraction writes reference_test_vector to seed ──────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_extraction_with_reference_writes_seed_section(tmp_path, monkeypatch):
    _patch_adapter(monkeypatch)
    ds = _make_fake_dataset(tmp_path)

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(
        tmp_path,
        config_name="test",
        device="cpu",
        reference_dataset_path=ds,
        reference_episode_index=0,
        reference_start_frame=0,
        reference_num_frames=10,
    )

    validate_seed(seed)
    assert "reference_test_vector" in seed
    rtv = seed["reference_test_vector"]
    assert rtv["n_frames"] == 10
    assert rtv["input_state_path"] is not None
    assert rtv["input_state_hash"] is not None
    assert rtv["input_images_path"] is not None
    assert len(rtv["input_images_hash"]) > 0


# ── 3. State array saved as .npy with correct hash ──────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_state_array_saved_as_npy(tmp_path, monkeypatch):
    _patch_adapter(monkeypatch)
    ds = _make_fake_dataset(tmp_path)

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(
        tmp_path,
        config_name="test",
        device="cpu",
        reference_dataset_path=ds,
        reference_episode_index=0,
        reference_start_frame=0,
        reference_num_frames=5,
    )

    rtv = seed["reference_test_vector"]
    state_path = tmp_path / rtv["input_state_path"]
    assert state_path.exists()
    assert state_path.suffix == ".npy"

    actual_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
    assert rtv["input_state_hash"] == actual_hash

    loaded = np.load(state_path)
    assert loaded.shape[0] == 5


# ── 4. Image frames saved and hashed per camera/frame ───────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_image_frames_saved_and_hashed(tmp_path, monkeypatch):
    _patch_adapter(monkeypatch, FakeRuntimeAdapterWithReference(
        camera_keys=["front", "wrist"],
        n_frames=3,
    ))
    ds = _make_fake_dataset(tmp_path)

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(
        tmp_path,
        config_name="test",
        device="cpu",
        reference_dataset_path=ds,
        reference_episode_index=0,
        reference_start_frame=0,
        reference_num_frames=3,
    )

    rtv = seed["reference_test_vector"]
    images_dir = tmp_path / rtv["input_images_path"]
    assert images_dir.is_dir()

    assert "front" in rtv["input_images_hash"]
    assert "wrist" in rtv["input_images_hash"]
    assert len(rtv["input_images_hash"]["front"]) == 3
    assert len(rtv["input_images_hash"]["wrist"]) == 3

    for cam_key, hashes in rtv["input_images_hash"].items():
        for i, h in enumerate(hashes):
            img_path = images_dir / f"{cam_key}_{i:03d}.png"
            assert img_path.exists(), f"missing {img_path}"
            actual = hashlib.sha256(img_path.read_bytes()).hexdigest()
            assert h == actual


# ── 5. Prompt recorded in seed ──────────────────────────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_prompt_recorded_in_reference(tmp_path, monkeypatch):
    _patch_adapter(monkeypatch, FakeRuntimeAdapterWithReference(
        prompt="stack the red block",
    ))
    ds = _make_fake_dataset(tmp_path)

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(
        tmp_path,
        config_name="test",
        device="cpu",
        reference_dataset_path=ds,
        reference_episode_index=0,
        reference_start_frame=0,
        reference_num_frames=10,
    )

    assert seed["reference_test_vector"]["input_prompt"] == "stack the red block"


# ── 6. Notes include dataset path and frame range ───────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_notes_include_provenance(tmp_path, monkeypatch):
    _patch_adapter(monkeypatch)
    ds = _make_fake_dataset(tmp_path)

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(
        tmp_path,
        config_name="test",
        device="cpu",
        reference_dataset_path=ds,
        reference_episode_index=2,
        reference_start_frame=5,
        reference_num_frames=10,
    )

    notes = seed["reference_test_vector"]["notes"]
    assert "episode" in notes.lower() or "2" in notes
    assert "frame" in notes.lower() or "5" in notes


# ── 7. Without --device, no reference vector ────────────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_no_reference_without_device(tmp_path):
    ds = _make_fake_dataset(tmp_path)
    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(
        tmp_path,
        config_name="test",
        reference_dataset_path=ds,
    )

    validate_seed(seed)
    assert "reference_test_vector" not in seed


# ── 8. Missing dataset path fails clearly ───────────────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_missing_dataset_path_with_reference_args_fails(tmp_path, monkeypatch):
    _patch_adapter(monkeypatch)

    extractor = OpenPIExtractor()
    with pytest.raises(ValueError, match="reference-dataset-path"):
        extractor.extract_seed(
            tmp_path,
            config_name="test",
            device="cpu",
            reference_dataset_path=None,
            reference_episode_index=3,
            reference_start_frame=5,
            reference_num_frames=10,
        )


# ── 9. Dataset commit downgrade: soft signal not hard fail ──────────────


def test_missing_dataset_commit_is_soft_signal():
    """training_datasets_resolvable should soft signal (not hard fail)
    when repo IDs are present but commits are missing."""
    from checkpoint_passport import load_passport, Status
    from checkpoint_passport.kernel.input_expectation import (
        check_training_datasets_resolvable,
    )
    from checkpoint_passport.schema import ModelPassport, InputContract, TrainingDatasetSpec

    passport = ModelPassport(
        input_contract=InputContract(
            training_datasets=[
                TrainingDatasetSpec(repo="user/dataset-1", commit=None),
                TrainingDatasetSpec(repo="user/dataset-2", commit=None),
            ],
        ),
    )

    load = type("FakeLoad", (), {
        "has_passport": True,
        "passport": passport,
    })()

    obs = check_training_datasets_resolvable(load)
    assert obs.status is not Status.FAIL, (
        f"Expected soft signal or pass, got {obs.status}: {obs.message}"
    )


def test_present_dataset_commit_still_validated():
    """When commit IS present, it should still be validated as before."""
    from checkpoint_passport import Status
    from checkpoint_passport.kernel.input_expectation import (
        check_training_datasets_resolvable,
    )
    from checkpoint_passport.schema import ModelPassport, InputContract, TrainingDatasetSpec

    passport = ModelPassport(
        input_contract=InputContract(
            training_datasets=[
                TrainingDatasetSpec(repo="user/dataset-1", commit="abc1234"),
            ],
        ),
    )

    load = type("FakeLoad", (), {
        "has_passport": True,
        "passport": passport,
    })()

    obs = check_training_datasets_resolvable(load)
    assert obs.status is Status.PASS


def test_malformed_repo_still_fails():
    """Malformed repo IDs should still hard-fail."""
    from checkpoint_passport import Status
    from checkpoint_passport.kernel.input_expectation import (
        check_training_datasets_resolvable,
    )
    from checkpoint_passport.schema import ModelPassport, InputContract, TrainingDatasetSpec

    passport = ModelPassport(
        input_contract=InputContract(
            training_datasets=[
                TrainingDatasetSpec(repo="", commit=None),
            ],
        ),
    )

    load = type("FakeLoad", (), {
        "has_passport": True,
        "passport": passport,
    })()

    obs = check_training_datasets_resolvable(load)
    assert obs.status is Status.FAIL


# ── 10. Seed with reference_test_vector passes validation ───────────────


def test_seed_with_reference_passes_validation():
    seed = {
        "extractor": {
            "extractor_name": "openpi",
            "extractor_version": "0.2.0",
        },
        "stack": "openpi",
        "reference_test_vector": {
            "n_frames": 10,
            "input_state_path": "assets/reference_test_vector/input_states.npy",
            "input_state_hash": "abc123",
            "input_images_path": "assets/reference_test_vector/images",
            "input_images_hash": {"front": ["h1", "h2"]},
            "input_prompt": "test",
        },
    }
    validate_seed(seed)


# ── 11. Dummy reference vector generation ────────────────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_dummy_reference_vector_writes_files_and_seed(tmp_path, monkeypatch):
    """--dummy-reference-vector produces valid files + seed without model/dataset."""
    _patch_adapter(monkeypatch)

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(
        tmp_path,
        config_name="test",
        dummy_reference_vector=True,
    )

    validate_seed(seed)
    assert "reference_test_vector" in seed
    rtv = seed["reference_test_vector"]

    assert rtv["n_frames"] == 10
    assert "dummy" in rtv["notes"]

    state_path = tmp_path / rtv["input_state_path"]
    assert state_path.exists()
    loaded = np.load(state_path)
    assert loaded.shape == (10, 7)

    actual_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
    assert rtv["input_state_hash"] == actual_hash

    images_dir = tmp_path / rtv["input_images_path"]
    assert images_dir.is_dir()
    for cam, hashes in rtv["input_images_hash"].items():
        assert len(hashes) == 10
        for i, h in enumerate(hashes):
            img_path = images_dir / f"{cam}_{i:03d}.png"
            assert img_path.exists()
            assert hashlib.sha256(img_path.read_bytes()).hexdigest() == h


@pytest.mark.usefixtures("_fake_openpi")
def test_dummy_and_dataset_path_are_mutually_exclusive(tmp_path, monkeypatch):
    _patch_adapter(monkeypatch)
    ds = _make_fake_dataset(tmp_path)

    extractor = OpenPIExtractor()
    with pytest.raises(ValueError, match="mutually exclusive"):
        extractor.extract_seed(
            tmp_path,
            config_name="test",
            dummy_reference_vector=True,
            reference_dataset_path=ds,
        )


@pytest.mark.usefixtures("_fake_openpi")
def test_dummy_reference_vector_uses_action_dim_from_config(tmp_path, monkeypatch):
    """State dim should come from input_contract.actions.dim when available."""
    _patch_adapter(monkeypatch)

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(
        tmp_path,
        config_name="test",
        dummy_reference_vector=True,
    )

    state_path = tmp_path / seed["reference_test_vector"]["input_state_path"]
    loaded = np.load(state_path)
    assert loaded.shape[1] == seed.get("input_contract", {}).get("actions", {}).get("dim", 7)
