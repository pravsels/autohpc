"""
OpenPI passport seed extractor.

Runs inside an OpenPI-capable environment to extract the inference contract
from OpenPI's runtime pipeline.  The contract does NOT exist in a static
config.json — it is constructed by cfg.data.create(), transforms, norm stats,
and adapter behavior.

This extractor:
    1. Loads the OpenPI TrainConfig by name.
    2. Instantiates the data pipeline (cfg.data.create) to inspect transforms.
    3. Reads norm stats from the checkpoint assets/ directory.
    4. Emits passport seed sections: stack, input_contract, output_spec,
       model_identity, model_internals (partial).

When --device is provided (runtime enrichment):
    5. Loads the model via the RuntimeAdapter protocol
       (checkpoint_passport.runtime_adapters.openpi).
    6. Collects library versions, parameter count, and smoke results.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from checkpoint_passport.runtime_extractors.base import (
    BaseExtractor,
    MissingRuntimeError,
)


_EXTRACTOR_VERSION = "0.3.0"


def _import_openpi_config():
    """Import openpi.training.config, raising MissingRuntimeError on failure."""
    try:
        import openpi.training.config as config_mod
        return config_mod
    except (ImportError, ModuleNotFoundError) as exc:
        missing = getattr(exc, "name", None) or "openpi"
        raise MissingRuntimeError(missing, "openpi") from exc


def _stringify(val: Any) -> Any:
    """Convert enum-like objects to their string value for JSON."""
    if val is None:
        return None
    if hasattr(val, "value"):
        return str(val.value)
    if hasattr(val, "name") and not isinstance(val, type):
        return str(val.name)
    return val


def _safe_getattr(obj: Any, *attrs: str) -> Any:
    """Walk a dotted attribute chain, returning None if any step is missing."""
    val = obj
    for attr in attrs:
        val = getattr(val, attr, None)
        if val is None:
            return None
    return val


def _detect_checkpoint_format(checkpoint_dir: Path) -> str:
    if (checkpoint_dir / "model.safetensors").exists():
        return "model.safetensors"
    if (checkpoint_dir / "params").is_dir():
        return "orbax"
    return "unknown"


def _read_norm_stats(checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
    """Read norm_stats.json from the checkpoint assets/ directory if present."""
    for candidate in [
        checkpoint_dir / "assets" / "norm_stats.json",
        checkpoint_dir / "norm_stats.json",
    ]:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text())
            except (json.JSONDecodeError, OSError):
                pass
    return None


def _extract_input_contract(
    cfg: Any,
    data_config: Any,
    norm_stats: Optional[Dict[str, Any]],
    *,
    default_prompt: Optional[str],
    resize_size: Optional[int],
) -> Dict[str, Any]:
    """Build input_contract from instantiated data config + transforms."""
    contract: Dict[str, Any] = {}

    model = getattr(cfg, "model", None)
    data_factory = getattr(cfg, "data", None)

    # Action spec
    action_horizon = _safe_getattr(model, "action_horizon")
    use_delta = _safe_getattr(data_factory, "use_delta_actions")
    joints_only = _safe_getattr(data_factory, "joints_only")

    # Determine actual robot-facing action dim from data config
    # For joints_only=True, the robot action dim comes from the policy output
    # transforms, not model.action_dim (which is the tokenized dim).
    action_dim = None
    if data_config is not None:
        # The data config's output transforms define the actual action dim
        output_transforms = _safe_getattr(data_config, "data_transforms", "outputs")
        if output_transforms:
            for t in output_transforms:
                ad = getattr(t, "action_dim", None)
                if ad is not None:
                    action_dim = ad
                    break

    actions: Dict[str, Any] = {}
    if action_horizon is not None:
        actions["horizon"] = action_horizon
    if action_dim is not None:
        actions["total_dim"] = action_dim
    if use_delta is not None:
        actions["use_delta_actions"] = use_delta
    if joints_only is not None:
        actions["joints_only"] = joints_only

    # Delta mask from data config
    delta_mask = _safe_getattr(data_factory, "delta_action_mask")
    if delta_mask is not None:
        actions["delta_mask"] = list(delta_mask)

    if actions:
        contract["actions"] = actions

    # Language spec
    prompt = default_prompt or _safe_getattr(data_factory, "default_prompt")
    if prompt is not None:
        contract["language"] = {"default_prompt": prompt}

    # Training datasets
    repo_id = _safe_getattr(data_factory, "repo_id")
    if repo_id is not None:
        if isinstance(repo_id, str) and repo_id.startswith("["):
            repos = [r.strip() for r in repo_id.strip("[]").split(",") if r.strip()]
        elif isinstance(repo_id, (list, tuple)):
            repos = list(repo_id)
        else:
            repos = [repo_id]
        contract["training_datasets"] = [{"repo": r} for r in repos]

    # Image spec (from data transforms if available)
    if data_config is not None:
        input_transforms = _safe_getattr(data_config, "data_transforms", "inputs")
        if input_transforms:
            image_keys = []
            for t in input_transforms:
                ik = getattr(t, "image_keys", None)
                if ik:
                    image_keys.extend(ik)
            if image_keys:
                images = [{"key": k} for k in image_keys]
                if resize_size is not None:
                    for img in images:
                        img["encoder_resize"] = [resize_size, resize_size]
                contract["images"] = images

    return contract


def _extract_output_spec(cfg: Any) -> Dict[str, Any]:
    """Build output_spec from model config."""
    model = getattr(cfg, "model", None)
    spec: Dict[str, Any] = {}

    action_horizon = _safe_getattr(model, "action_horizon")
    if action_horizon is not None:
        spec["actions"] = {"horizon": action_horizon}

    return spec


def _extract_model_identity(cfg: Any) -> Dict[str, Any]:
    """Build model_identity from config (without loading weights)."""
    model = getattr(cfg, "model", None)
    identity: Dict[str, Any] = {}

    model_type = _safe_getattr(model, "model_type")
    if model_type is not None:
        identity["class_name"] = _stringify(model_type)

    identity["resolved_via"] = "openpi.training.config"

    return identity


def _extract_model_internals(cfg: Any) -> Dict[str, Any]:
    """Build partial model_internals from config (no weight loading)."""
    model = getattr(cfg, "model", None)
    internals: Dict[str, Any] = {}

    # Forward graph shape hints from config
    action_dim = _safe_getattr(model, "action_dim")
    action_horizon = _safe_getattr(model, "action_horizon")
    if action_dim is not None and action_horizon is not None:
        internals["forward_graph"] = {
            "sample_output_shapes": {
                "action_tokens": [1, action_horizon, action_dim],
            },
        }

    return internals


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _extract_reference_test_vector(
    seed: Dict[str, Any],
    checkpoint_dir: Path,
    adapter: Any,
    *,
    reference_dataset_path: Path,
    reference_episode_index: int,
    reference_start_frame: int,
    reference_num_frames: int,
) -> None:
    """Extract reference data from a real dataset and write assets."""
    import numpy as np

    sample = adapter.extract_reference_sample(
        reference_dataset_path,
        episode_index=reference_episode_index,
        start_frame=reference_start_frame,
        num_frames=reference_num_frames,
    )

    rtv_dir = checkpoint_dir / "assets" / "reference_test_vector"
    rtv_dir.mkdir(parents=True, exist_ok=True)
    images_dir = rtv_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Save state array
    state_path = rtv_dir / "input_states.npy"
    np.save(state_path, sample["states"])
    state_hash = _sha256_file(state_path)

    # Save image frames
    images_hash: Dict[str, List[str]] = {}
    for cam_key, frames in sample["images"].items():
        cam_hashes = []
        for i, frame in enumerate(frames):
            img_path = images_dir / f"{cam_key}_{i:03d}.png"
            try:
                from PIL import Image
                img = Image.fromarray(frame)
                img.save(img_path)
            except ImportError:
                import struct
                import zlib
                _write_png_minimal(frame, img_path)
            cam_hashes.append(_sha256_file(img_path))
        images_hash[cam_key] = cam_hashes

    rel_state = str(state_path.relative_to(checkpoint_dir))
    rel_images = str(images_dir.relative_to(checkpoint_dir))

    seed["reference_test_vector"] = {
        "n_frames": reference_num_frames,
        "input_state_path": rel_state,
        "input_state_hash": state_hash,
        "input_images_path": rel_images,
        "input_images_hash": images_hash,
        "input_prompt": sample.get("prompt", ""),
        "notes": (
            f"dataset={sample.get('dataset_path', str(reference_dataset_path))}, "
            f"episode={reference_episode_index}, "
            f"frames={reference_start_frame}..{reference_start_frame + reference_num_frames - 1}"
        ),
    }


def _generate_dummy_reference_vector(
    seed: Dict[str, Any],
    checkpoint_dir: Path,
    *,
    num_frames: int = 10,
    state_dim: int = 7,
    image_size: int = 224,
    cameras: Optional[List[str]] = None,
    prompt: str = "dummy reference vector for testing",
) -> None:
    """Generate a synthetic reference test vector without model or dataset.

    Writes the same file layout and seed schema as the real extraction path
    so that downstream validation treats it identically.  Useful for MVP
    testing and CI where the full runtime stack isn't available.
    """
    import numpy as np

    if cameras is None:
        cameras = ["front", "wrist"]

    rtv_dir = checkpoint_dir / "assets" / "reference_test_vector"
    rtv_dir.mkdir(parents=True, exist_ok=True)
    images_dir = rtv_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    states = rng.randn(num_frames, state_dim).astype(np.float32)
    state_path = rtv_dir / "input_states.npy"
    np.save(state_path, states)
    state_hash = _sha256_file(state_path)

    images_hash: Dict[str, List[str]] = {}
    for cam in cameras:
        cam_hashes = []
        for i in range(num_frames):
            img = np.random.RandomState(42 + i).randint(
                0, 255, (image_size, image_size, 3), dtype=np.uint8,
            )
            img_path = images_dir / f"{cam}_{i:03d}.png"
            _write_png_minimal(img, img_path)
            cam_hashes.append(_sha256_file(img_path))
        images_hash[cam] = cam_hashes

    rel_state = str(state_path.relative_to(checkpoint_dir))
    rel_images = str(images_dir.relative_to(checkpoint_dir))

    seed["reference_test_vector"] = {
        "n_frames": num_frames,
        "input_state_path": rel_state,
        "input_state_hash": state_hash,
        "input_images_path": rel_images,
        "input_images_hash": images_hash,
        "input_prompt": prompt,
        "notes": "dummy reference vector (synthetic data, seed=42)",
    }


def _write_png_minimal(array: Any, path: Path) -> None:
    """Write a numpy uint8 HWC array as PNG without PIL."""
    import struct
    import zlib

    h, w = array.shape[:2]
    c = array.shape[2] if array.ndim == 3 else 1

    if c == 3:
        color_type = 2
    elif c == 1:
        color_type = 0
        array = array.reshape(h, w)
    else:
        color_type = 2

    raw_rows = b""
    for row in range(h):
        raw_rows += b"\x00" + array[row].tobytes()

    compressed = zlib.compress(raw_rows)

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr_data))
        f.write(_chunk(b"IDAT", compressed))
        f.write(_chunk(b"IEND", b""))


def _enrich_with_runtime(
    seed: Dict[str, Any],
    checkpoint_dir: Path,
    *,
    config_name: str,
    default_prompt: Optional[str],
    resize_size: Optional[int],
    device: str,
    reference_dataset_path: Optional[Path] = None,
    reference_episode_index: int = 0,
    reference_start_frame: int = 0,
    reference_num_frames: int = 10,
) -> None:
    """Load the model via the RuntimeAdapter protocol and enrich the seed."""
    from checkpoint_passport.runtime_adapters.openpi import OpenPIRuntimeAdapter

    adapter = OpenPIRuntimeAdapter()
    adapter.load(
        checkpoint_dir,
        device=device,
        config_name=config_name,
        default_prompt=default_prompt,
        resize_size=resize_size,
    )

    seed["extractor"]["device"] = device

    lib_versions = adapter.library_versions()
    if lib_versions:
        seed.setdefault("model_identity", {})
        seed["model_identity"]["library_versions"] = lib_versions

    param_summary = adapter.count_parameters()
    if param_summary is not None:
        seed.setdefault("model_internals", {})
        seed["model_internals"]["parameter_summary"] = param_summary

    smoke = adapter.smoke_inference()
    if smoke is not None:
        seed.setdefault("model_internals", {})
        seed["model_internals"]["numerical_health"] = {"smoke": smoke}

    if reference_dataset_path is not None:
        _extract_reference_test_vector(
            seed,
            checkpoint_dir,
            adapter,
            reference_dataset_path=reference_dataset_path,
            reference_episode_index=reference_episode_index,
            reference_start_frame=reference_start_frame,
            reference_num_frames=reference_num_frames,
        )


class OpenPIExtractor(BaseExtractor):
    """Extracts a passport seed from an OpenPI checkpoint."""

    def extract_seed(
        self,
        checkpoint_dir: Path,
        *,
        config_name: str,
        default_prompt: Optional[str] = None,
        resize_size: Optional[int] = None,
        device: Optional[str] = None,
        reference_dataset_path: Optional[Path] = None,
        reference_episode_index: int = 0,
        reference_start_frame: int = 0,
        reference_num_frames: int = 10,
        dummy_reference_vector: bool = False,
    ) -> Dict[str, Any]:
        # Validate reference args: if any frame selection is specified,
        # dataset path is required.
        has_ref_args = (
            reference_episode_index != 0
            or reference_start_frame != 0
            or reference_num_frames != 10
        )
        if reference_dataset_path is None and has_ref_args:
            raise ValueError(
                "--reference-dataset-path is required when specifying "
                "reference episode/frame selection"
            )
        if dummy_reference_vector and reference_dataset_path is not None:
            raise ValueError(
                "--dummy-reference-vector and --reference-dataset-path "
                "are mutually exclusive"
            )

        config_mod = _import_openpi_config()
        cfg = config_mod.get_config(config_name)

        # Try to instantiate the data pipeline for deeper introspection
        data_config = None
        try:
            data_factory = getattr(cfg, "data", None)
            model_config = getattr(cfg, "model", None)
            assets_dirs = getattr(cfg, "assets_dirs", None)
            if data_factory is not None and hasattr(data_factory, "create"):
                if assets_dirs is not None and model_config is not None:
                    data_config = data_factory.create(assets_dirs, model_config)
        except Exception:
            pass

        norm_stats = _read_norm_stats(checkpoint_dir)

        input_contract = _extract_input_contract(
            cfg, data_config, norm_stats,
            default_prompt=default_prompt,
            resize_size=resize_size,
        )
        output_spec = _extract_output_spec(cfg)
        model_identity = _extract_model_identity(cfg)
        model_internals = _extract_model_internals(cfg)

        seed: Dict[str, Any] = {
            "extractor": {
                "extractor_name": "openpi",
                "extractor_version": _EXTRACTOR_VERSION,
                "openpi_config_name": config_name,
                "checkpoint_format": _detect_checkpoint_format(checkpoint_dir),
            },
            "stack": "openpi",
        }

        if input_contract:
            seed["input_contract"] = input_contract
        if output_spec:
            seed["output_spec"] = output_spec
        if model_identity:
            seed["model_identity"] = model_identity
        if model_internals:
            seed["model_internals"] = model_internals

        if device is not None:
            _enrich_with_runtime(
                seed,
                checkpoint_dir,
                config_name=config_name,
                default_prompt=default_prompt,
                resize_size=resize_size,
                device=device,
                reference_dataset_path=reference_dataset_path,
                reference_episode_index=reference_episode_index,
                reference_start_frame=reference_start_frame,
                reference_num_frames=reference_num_frames,
            )

        if dummy_reference_vector:
            state_dim = 7
            ic = seed.get("input_contract", {})
            actions = ic.get("actions", {})
            if "dim" in actions:
                state_dim = actions["dim"]
            _generate_dummy_reference_vector(
                seed,
                checkpoint_dir,
                num_frames=reference_num_frames,
                state_dim=state_dim,
                prompt=default_prompt or "dummy reference vector for testing",
            )

        return seed
