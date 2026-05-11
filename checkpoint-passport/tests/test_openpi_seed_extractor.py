"""
Tests for the OpenPI passport seed extractor.

Uses monkeypatched fake modules — does not require real OpenPI in
the test environment.  Verifies the extractor produces valid passport
seeds with the correct sections populated from runtime pipeline data.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from checkpoint_passport.passport_seed import validate_seed, InvalidSeedError
from checkpoint_passport.runtime_extractors.base import (
    MissingRuntimeError,
    UnsupportedBackendError,
    SUPPORTED_BACKENDS,
)
from checkpoint_passport.runtime_extractors.openpi import OpenPIExtractor


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


def _make_fake_data_config(*, action_dim: int = 7, image_keys: list | None = None):
    """Fake instantiated DataConfig (result of data_factory.create())."""
    output_transform = _make_fake_output_transform(action_dim)
    input_transform = types.SimpleNamespace()
    input_transform.image_keys = image_keys or []

    data_transforms = types.SimpleNamespace()
    data_transforms.inputs = [input_transform] if image_keys else []
    data_transforms.outputs = [output_transform]

    dc = types.SimpleNamespace()
    dc.data_transforms = data_transforms
    return dc


def _make_fake_train_config(
    *,
    model_type: str = "pi05",
    action_dim: int = 32,
    action_horizon: int = 50,
    use_delta_actions: bool = True,
    default_prompt: str = "build a block tower",
    repo_id: str = "villekuosmanen/build_block_tower",
    joints_only: bool = True,
    delta_action_mask=None,
    image_keys: list | None = None,
    robot_action_dim: int = 7,
):
    """Build a fake TrainConfig mimicking the real nested structure."""
    model_type_enum = types.SimpleNamespace()
    model_type_enum.value = model_type

    model = types.SimpleNamespace()
    model.model_type = model_type_enum
    model.action_dim = action_dim
    model.action_horizon = action_horizon

    data_factory = types.SimpleNamespace()
    data_factory.use_delta_actions = use_delta_actions
    data_factory.default_prompt = default_prompt
    data_factory.repo_id = repo_id
    data_factory.joints_only = joints_only
    data_factory.delta_action_mask = delta_action_mask
    data_factory.create = lambda assets_dirs, model_config: _make_fake_data_config(
        action_dim=robot_action_dim,
        image_keys=image_keys,
    )

    cfg = types.SimpleNamespace()
    cfg.model = model
    cfg.data = data_factory
    cfg.assets_dirs = Path("/fake/assets")
    return cfg


@pytest.fixture()
def _fake_openpi(monkeypatch):
    """Install fake 'openpi' package into sys.modules."""
    for mod_name in FAKE_OPENPI_MODULES:
        monkeypatch.setitem(sys.modules, mod_name, types.ModuleType(mod_name))

    fake_config_mod = sys.modules["openpi.training.config"]
    fake_config_mod.get_config = lambda name: _make_fake_train_config()

    yield

    for mod_name in FAKE_OPENPI_MODULES:
        sys.modules.pop(mod_name, None)


# ── 1. Happy path: produces valid passport seed ─────────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_extractor_produces_valid_seed(tmp_path: Path, monkeypatch):
    """The extractor output passes passport seed validation."""
    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: _make_fake_train_config())

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(
        tmp_path,
        config_name="pi05_build_block_tower_baseline_6mix_joints_only",
    )

    validate_seed(seed)


@pytest.mark.usefixtures("_fake_openpi")
def test_extractor_emits_stack(tmp_path: Path, monkeypatch):
    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: _make_fake_train_config())

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test_config")

    assert seed["stack"] == "openpi"


@pytest.mark.usefixtures("_fake_openpi")
def test_extractor_emits_extractor_metadata(tmp_path: Path, monkeypatch):
    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: _make_fake_train_config())

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="my_config")

    assert seed["extractor"]["extractor_name"] == "openpi"
    assert seed["extractor"]["extractor_version"] == "0.1.0"
    assert seed["extractor"]["openpi_config_name"] == "my_config"


# ── 2. Input contract from runtime pipeline ─────────────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_extractor_gets_robot_action_dim_from_pipeline(tmp_path: Path, monkeypatch):
    """The extractor should use the output transform's action_dim (robot-facing),
    not model.action_dim (tokenized)."""
    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: _make_fake_train_config(
        action_dim=32,
        robot_action_dim=7,
    ))

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test")

    actions = seed["input_contract"]["actions"]
    assert actions["total_dim"] == 7
    assert actions["horizon"] == 50


@pytest.mark.usefixtures("_fake_openpi")
def test_extractor_gets_delta_actions(tmp_path: Path, monkeypatch):
    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: _make_fake_train_config(
        use_delta_actions=True,
        joints_only=True,
    ))

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test")

    actions = seed["input_contract"]["actions"]
    assert actions["use_delta_actions"] is True
    assert actions["joints_only"] is True


@pytest.mark.usefixtures("_fake_openpi")
def test_extractor_gets_language_prompt(tmp_path: Path, monkeypatch):
    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: _make_fake_train_config(
        default_prompt="stack the blocks",
    ))

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test")

    assert seed["input_contract"]["language"]["default_prompt"] == "stack the blocks"


@pytest.mark.usefixtures("_fake_openpi")
def test_extractor_prompt_override(tmp_path: Path, monkeypatch):
    """CLI-supplied prompt takes precedence over config prompt."""
    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: _make_fake_train_config(
        default_prompt="config prompt",
    ))

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(
        tmp_path, config_name="test", default_prompt="override prompt",
    )

    assert seed["input_contract"]["language"]["default_prompt"] == "override prompt"


@pytest.mark.usefixtures("_fake_openpi")
def test_extractor_gets_training_datasets(tmp_path: Path, monkeypatch):
    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: _make_fake_train_config(
        repo_id="[org/data1, org/data2]",
    ))

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test")

    datasets = seed["input_contract"]["training_datasets"]
    assert len(datasets) == 2
    assert datasets[0]["repo"] == "org/data1"
    assert datasets[1]["repo"] == "org/data2"


@pytest.mark.usefixtures("_fake_openpi")
def test_extractor_gets_image_keys_from_pipeline(tmp_path: Path, monkeypatch):
    """Image keys come from the instantiated data pipeline, not config attributes."""
    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: _make_fake_train_config(
        image_keys=["image_front", "image_wrist"],
    ))

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test", resize_size=224)

    images = seed["input_contract"]["images"]
    assert len(images) == 2
    assert images[0]["key"] == "image_front"
    assert images[0]["encoder_resize"] == [224, 224]


# ── 3. Model identity and internals ─────────────────────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_extractor_emits_model_identity(tmp_path: Path, monkeypatch):
    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: _make_fake_train_config(
        model_type="pi05",
    ))

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test")

    assert seed["model_identity"]["class_name"] == "pi05"
    assert seed["model_identity"]["resolved_via"] == "openpi.training.config"


@pytest.mark.usefixtures("_fake_openpi")
def test_extractor_emits_model_internals_shapes(tmp_path: Path, monkeypatch):
    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: _make_fake_train_config(
        action_dim=32,
        action_horizon=50,
    ))

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test")

    shapes = seed["model_internals"]["forward_graph"]["sample_output_shapes"]
    assert shapes["action_tokens"] == [1, 50, 32]


# ── 4. Checkpoint format detection ──────────────────────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_extractor_detects_orbax(tmp_path: Path, monkeypatch):
    (tmp_path / "params").mkdir()
    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: _make_fake_train_config())

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test")

    assert seed["extractor"]["checkpoint_format"] == "orbax"


@pytest.mark.usefixtures("_fake_openpi")
def test_extractor_detects_safetensors(tmp_path: Path, monkeypatch):
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 8)
    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: _make_fake_train_config())

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test")

    assert seed["extractor"]["checkpoint_format"] == "model.safetensors"


# ── 5. Missing runtime fails clearly ────────────────────────────────────


def test_missing_openpi_raises(tmp_path: Path, monkeypatch):
    for mod_name in FAKE_OPENPI_MODULES:
        monkeypatch.delitem(sys.modules, mod_name, raising=False)

    from unittest.mock import patch
    with patch.dict(sys.modules, {"openpi": None, "openpi.training": None, "openpi.training.config": None}):
        extractor = OpenPIExtractor()
        with pytest.raises(MissingRuntimeError, match="openpi"):
            extractor.extract_seed(tmp_path, config_name="anything")


# ── 6. Output is JSON-serializable ──────────────────────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_seed_is_json_serializable(tmp_path: Path, monkeypatch):
    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: _make_fake_train_config(
        image_keys=["front"],
    ))

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(
        tmp_path, config_name="test", default_prompt="go", resize_size=224,
    )

    serialized = json.dumps(seed)
    roundtripped = json.loads(serialized)
    assert roundtripped == seed


# ── 7. CLI contract ─────────────────────────────────────────────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_cli_writes_seed_file(tmp_path: Path, monkeypatch):
    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: _make_fake_train_config())

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
            "--default-prompt", "do the thing",
            "--resize-size", "224",
        ],
    )

    main()

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["stack"] == "openpi"
    assert data["extractor"]["extractor_name"] == "openpi"
    validate_seed(data)


def test_cli_unknown_backend_exits_nonzero(tmp_path: Path, monkeypatch):
    from checkpoint_passport.cli.extract_passport_seed import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "extract-passport-seed",
            "lerobot",
            "--checkpoint-dir", str(tmp_path),
            "--out", str(tmp_path / "out.json"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0


@pytest.mark.usefixtures("_fake_openpi")
def test_cli_missing_config_name_exits_nonzero(tmp_path: Path, monkeypatch):
    from checkpoint_passport.cli.extract_passport_seed import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "extract-passport-seed",
            "openpi",
            "--checkpoint-dir", str(tmp_path),
            "--out", str(tmp_path / "out.json"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0


# ── 8. Graceful fallback when pipeline instantiation fails ───────────────


@pytest.mark.usefixtures("_fake_openpi")
def test_extractor_works_without_pipeline(tmp_path: Path, monkeypatch):
    """If data.create() fails, the extractor still produces a valid seed
    with what it can get from config attributes alone."""
    def make_broken_config():
        cfg = _make_fake_train_config()
        cfg.data.create = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("missing assets"))
        cfg.assets_dirs = None
        return cfg

    config_mod = sys.modules["openpi.training.config"]
    monkeypatch.setattr(config_mod, "get_config", lambda name: make_broken_config())

    extractor = OpenPIExtractor()
    seed = extractor.extract_seed(tmp_path, config_name="test")

    validate_seed(seed)
    assert seed["stack"] == "openpi"
    # Should still get action_horizon from model config
    assert seed["input_contract"]["actions"]["horizon"] == 50
