"""
Tests for OpenPI runtime enrichment (Task 4).

Uses monkeypatched fakes — does not require real OpenPI, missiontracker,
or any ML framework in the test environment.  The RuntimeAdapter protocol
is faked directly; no sys.modules hacking for missiontracker needed.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

from checkpoint_passport.passport_seed import validate_seed
from checkpoint_passport.runtime_extractors.base import MissingRuntimeError
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


# ── Fake RuntimeAdapter ─────────────────────────────────────────────────


class FakeRuntimeAdapter(RuntimeAdapter):
    """In-memory fake that implements the protocol for unit tests."""

    def __init__(
        self,
        *,
        total_params: int = 1_500_000,
        trainable_params: Optional[int] = 1_000_000,
        smoke_status: str = "pass",
        smoke_raises: bool = False,
        params_available: bool = True,
        smoke_available: bool = True,
    ):
        self._total_params = total_params
        self._trainable_params = trainable_params
        self._smoke_status = smoke_status
        self._smoke_raises = smoke_raises
        self._params_available = params_available
        self._smoke_available = smoke_available

    def load(self, checkpoint_dir: Path, *, device: str, **kwargs: Any) -> None:
        self._device = device

    def count_parameters(self) -> Optional[Dict[str, int]]:
        if not self._params_available:
            return None
        result: Dict[str, int] = {"total_params": self._total_params}
        if self._trainable_params is not None:
            result["trainable_params"] = self._trainable_params
        return result

    def smoke_inference(self) -> Optional[Dict[str, Any]]:
        if not self._smoke_available:
            return None
        if self._smoke_raises:
            return {"status": "fail", "error": "RuntimeError"}
        return {"status": self._smoke_status, "elapsed_ms": 42.0}

    def library_versions(self) -> Dict[str, str]:
        return {"python": "3.11.14", "openpi": "0.3.1", "jax": "0.5.3"}

    def extract_reference_sample(
        self,
        dataset_path,
        *,
        episode_index=0,
        start_frame=0,
        num_frames=10,
    ) -> Dict[str, Any]:
        import numpy as np
        return {
            "states": np.zeros((num_frames, 7), dtype=np.float32),
            "images": {},
            "prompt": "",
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


def _patch_adapter(monkeypatch, adapter: FakeRuntimeAdapter = None):
    """Monkeypatch OpenPIRuntimeAdapter with a fake."""
    fake = adapter or FakeRuntimeAdapter()
    monkeypatch.setattr(
        "checkpoint_passport.runtime_adapters.openpi.OpenPIRuntimeAdapter",
        lambda: fake,
    )


@pytest.fixture()
def _fake_adapter(monkeypatch):
    """Default fake adapter for enrichment tests."""
    _patch_adapter(monkeypatch)


# ── 1. Without device: no enrichment ────────────────────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_no_enrichment_without_device(tmp_path: Path):
    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test")

    validate_seed(seed)
    assert "device" not in seed["extractor"]
    assert "library_versions" not in seed.get("model_identity", {})
    assert "parameter_summary" not in seed.get("model_internals", {})
    assert "numerical_health" not in seed.get("model_internals", {})


# ── 2. With device: enrichment populated ────────────────────────────────


@pytest.mark.usefixtures("_fake_openpi", "_fake_adapter")
def test_enrichment_with_device(tmp_path: Path):
    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test", device="cpu")

    validate_seed(seed)
    assert seed["extractor"]["device"] == "cpu"


@pytest.mark.usefixtures("_fake_openpi", "_fake_adapter")
def test_enrichment_adds_library_versions(tmp_path: Path):
    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test", device="cpu")

    lib_versions = seed["model_identity"]["library_versions"]
    assert "python" in lib_versions
    assert "openpi" in lib_versions
    assert lib_versions["openpi"] == "0.3.1"


@pytest.mark.usefixtures("_fake_openpi", "_fake_adapter")
def test_enrichment_adds_parameter_summary(tmp_path: Path):
    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test", device="cpu")

    param_summary = seed["model_internals"]["parameter_summary"]
    assert param_summary["total_params"] == 1_500_000
    assert param_summary["trainable_params"] == 1_000_000


@pytest.mark.usefixtures("_fake_openpi", "_fake_adapter")
def test_enrichment_adds_smoke_result(tmp_path: Path):
    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test", device="cpu")

    smoke = seed["model_internals"]["numerical_health"]["smoke"]
    assert smoke["status"] == "pass"
    assert "elapsed_ms" in smoke


# ── 3. Smoke failure is recorded, not raised ────────────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_smoke_failure_recorded(tmp_path: Path, monkeypatch):
    _patch_adapter(monkeypatch, FakeRuntimeAdapter(smoke_raises=True))

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test", device="cpu")

    validate_seed(seed)
    smoke = seed["model_internals"]["numerical_health"]["smoke"]
    assert smoke["status"] == "fail"
    assert smoke["error"] == "RuntimeError"


# ── 4. Missing openpi raises clearly ────────────────────────────────────


def test_missing_openpi_raises(tmp_path: Path, monkeypatch):
    for mod_name in FAKE_OPENPI_MODULES:
        monkeypatch.delitem(sys.modules, mod_name, raising=False)

    with patch.dict(sys.modules, {
        "openpi": None,
        "openpi.training": None,
        "openpi.training.config": None,
    }):
        extractor = OpenPIExtractor()
        with pytest.raises(MissingRuntimeError, match="openpi"):
            extractor.extract_seed(tmp_path, config_name="test", device="cpu")


# ── 5. No params available → no param summary ──────────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_no_params_skips_summary(tmp_path: Path, monkeypatch):
    _patch_adapter(monkeypatch, FakeRuntimeAdapter(params_available=False))

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test", device="cpu")

    validate_seed(seed)
    assert "parameter_summary" not in seed.get("model_internals", {})


# ── 6. No smoke available → no smoke result ─────────────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_no_smoke_skips_health(tmp_path: Path, monkeypatch):
    _patch_adapter(monkeypatch, FakeRuntimeAdapter(smoke_available=False))

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test", device="cpu")

    validate_seed(seed)
    assert "numerical_health" not in seed.get("model_internals", {})


# ── 7. Enriched seed is JSON-serializable ───────────────────────────────


@pytest.mark.usefixtures("_fake_openpi", "_fake_adapter")
def test_enriched_seed_is_json_serializable(tmp_path: Path):
    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test", device="cpu")

    serialized = json.dumps(seed)
    roundtripped = json.loads(serialized)
    assert roundtripped == seed


# ── 8. Enriched seed passes validation ──────────────────────────────────


@pytest.mark.usefixtures("_fake_openpi", "_fake_adapter")
def test_enriched_seed_passes_validation(tmp_path: Path):
    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test", device="cpu")

    validate_seed(seed)
    assert seed["stack"] == "openpi"
    assert seed["extractor"]["extractor_name"] == "openpi"


# ── 9. CLI --device flag wiring ─────────────────────────────────────────


@pytest.mark.usefixtures("_fake_openpi", "_fake_adapter")
def test_cli_device_flag(tmp_path: Path, monkeypatch):
    from checkpoint_passport.cli.extract_passport_seed import main

    out_path = tmp_path / "PASSPORT_SEED.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "extract-passport-seed",
            "openpi",
            "--checkpoint-dir", str(tmp_path),
            "--out", str(out_path),
            "--openpi-config-name", "pi05_test",
            "--device", "cpu",
        ],
    )

    main()

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    validate_seed(data)
    assert data["extractor"]["device"] == "cpu"
    assert "library_versions" in data["model_identity"]
    assert "parameter_summary" in data["model_internals"]


# ── 10. CLI without --device still works ────────────────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_cli_no_device_flag(tmp_path: Path, monkeypatch):
    from checkpoint_passport.cli.extract_passport_seed import main

    out_path = tmp_path / "PASSPORT_SEED.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "extract-passport-seed",
            "openpi",
            "--checkpoint-dir", str(tmp_path),
            "--out", str(out_path),
            "--openpi-config-name", "pi05_test",
        ],
    )

    main()

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    validate_seed(data)
    assert "device" not in data["extractor"]
