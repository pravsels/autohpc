"""
OpenPI runtime adapter for model enrichment.

Uses OpenPI's own API directly — no missiontracker dependency.  The only
runtime requirements are ``openpi``, ``jax``, ``flax``, and ``numpy``,
which are already present in any OpenPI-capable environment.

This adapter:
    1. Loads the trained Policy via openpi.* (config → restore params → Policy).
    2. Counts parameters via flax.nnx / jax.tree.
    3. Runs a smoke inference by calling policy.infer() with an OpenPI-native
       dict (numpy arrays, not torch tensors or PolicyObservation).
    4. Reports library versions for Python, JAX, torch, OpenPI.
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

from checkpoint_passport.runtime_adapters.base import RuntimeAdapter


def _load_norm_stats(assets_dir: Path, asset_id: str) -> Any:
    """Load norm stats using OpenPI's normalize module."""
    from openpi.shared import normalize as _normalize
    try:
        from etils import epath
        norm_stats_dir = epath.Path(str(assets_dir)) / asset_id
    except ImportError:
        norm_stats_dir = assets_dir / asset_id
    return _normalize.load(norm_stats_dir)


def _create_policy(
    train_config: Any,
    checkpoint_dir: Path,
    *,
    default_prompt: Optional[str] = None,
) -> Any:
    """Build an OpenPI Policy from config + checkpoint weights.

    Inlined from the missiontracker loading path — uses only openpi.* APIs.
    """
    import jax.numpy as jnp
    import openpi.models.model as _model
    import openpi.policies.policy as _policy
    import openpi.transforms as transforms

    ckpt_str = str(checkpoint_dir)
    weight_path = os.path.join(ckpt_str, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        pytorch_device = "cuda"
        try:
            import torch
            if not torch.cuda.is_available():
                pytorch_device = "cpu"
        except ImportError:
            pytorch_device = "cpu"
    else:
        model = train_config.model.load(
            _model.restore_params(Path(ckpt_str) / "params", dtype=jnp.bfloat16)
        )
        pytorch_device = None

    data_config = train_config.data.create(
        train_config.assets_dirs, train_config.model,
    )

    norm_stats = None
    if data_config.asset_id is not None:
        norm_stats = _load_norm_stats(
            Path(ckpt_str) / "assets", data_config.asset_id,
        )

    repack = transforms.Group()

    return _policy.Policy(
        model,
        transforms=[
            *repack.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm,
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm,
            ),
            *data_config.data_transforms.outputs,
            *repack.outputs,
        ],
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )


class OpenPIRuntimeAdapter(RuntimeAdapter):
    """Loads an OpenPI checkpoint and provides enrichment data."""

    def load(
        self,
        checkpoint_dir: Path,
        *,
        device: str,
        config_name: str,
        default_prompt: Optional[str] = None,
        resize_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        from checkpoint_passport.runtime_extractors.base import MissingRuntimeError
        try:
            from openpi.training.config import get_config
        except (ImportError, ModuleNotFoundError) as exc:
            missing = getattr(exc, "name", None) or "openpi"
            raise MissingRuntimeError(missing, "openpi") from exc

        cfg = get_config(config_name)
        self._policy = _create_policy(cfg, checkpoint_dir, default_prompt=default_prompt)
        self._model = self._policy._model
        self._action_horizon = cfg.model.action_horizon
        self._default_prompt = default_prompt
        self._resize_size = resize_size or 224

        # The raw action dim for observations comes from the data config's
        # output transforms (e.g. 7 for joints-only), not cfg.model.action_dim
        # which is the tokenized dimension (e.g. 32).
        self._action_dim = self._infer_raw_action_dim(cfg)

    @staticmethod
    def _infer_raw_action_dim(cfg: Any) -> int:
        """Derive the raw observation action dim from config.

        The data config's output transforms expose the robot-facing action_dim
        (e.g. 7 for joints-only).  Falls back to the model's tokenized
        action_dim if the transforms don't expose one.
        """
        try:
            data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
            for t in getattr(
                getattr(data_config, "data_transforms", None), "outputs", []
            ):
                ad = getattr(t, "action_dim", None)
                if ad is not None:
                    return ad
        except Exception:
            pass
        return cfg.model.action_dim

    def count_parameters(self) -> Optional[Dict[str, int]]:
        try:
            import jax
            from flax import nnx
            state = nnx.state(self._model)
            leaves = jax.tree.leaves(state)
            if leaves and hasattr(leaves[0], "size"):
                total = sum(p.size for p in leaves)
                return {"total_params": total}
        except Exception:
            pass

        # PyTorch fallback
        parameters_fn = getattr(self._model, "parameters", None)
        if parameters_fn is not None:
            try:
                params = list(parameters_fn())
                total = sum(p.numel() for p in params)
                trainable = sum(
                    p.numel() for p in params
                    if getattr(p, "requires_grad", False)
                )
                return {"total_params": total, "trainable_params": trainable}
            except Exception:
                pass

        return None

    def smoke_inference(self) -> Optional[Dict[str, Any]]:
        try:
            import numpy as np
        except ImportError:
            return None

        obs = {
            "observation.images.front": np.zeros(
                (self._resize_size, self._resize_size, 3), dtype=np.uint8,
            ),
            "observation.images.wrist": np.zeros(
                (self._resize_size, self._resize_size, 3), dtype=np.uint8,
            ),
            "observation.state.pos": np.zeros(7, dtype=np.float32),
            "observation.state.eef_pose": np.zeros(7, dtype=np.float32),
            "actions": np.zeros(
                (self._action_horizon, self._action_dim), dtype=np.float32,
            ),
            "prompt": self._default_prompt or "smoke test",
        }

        try:
            start = time.monotonic()
            self._policy.infer(obs)
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            return {"status": "pass", "elapsed_ms": elapsed_ms}
        except Exception as exc:
            return {"status": "fail", "error": type(exc).__name__}

    def library_versions(self) -> Dict[str, str]:
        versions: Dict[str, str] = {"python": platform.python_version()}
        for lib_name, import_name in [
            ("torch", "torch"),
            ("jax", "jax"),
            ("openpi", "openpi"),
        ]:
            try:
                mod = __import__(import_name)
                v = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
                if v is not None:
                    versions[lib_name] = str(v)
            except (ImportError, ModuleNotFoundError):
                pass
        return versions

    def extract_reference_sample(
        self,
        dataset_path: Path,
        *,
        episode_index: int = 0,
        start_frame: int = 0,
        num_frames: int = 10,
    ) -> Dict[str, Any]:
        import numpy as np

        try:
            from robocandywrapper import make_dataset_without_config
        except ImportError as exc:
            raise ValueError(
                "RoboCandyWrapper is required for reference sample extraction. "
                "Install checkpoint-passport with its declared dependencies."
            ) from exc

        try:
            ds = make_dataset_without_config([str(dataset_path)])
        except Exception as exc:
            raise ValueError(
                "RoboCandyWrapper could not load reference dataset "
                f"{dataset_path} for episode {episode_index}, "
                f"frames {start_frame}:{start_frame + num_frames}: {exc}"
            ) from exc

        frame_indices = list(range(start_frame, start_frame + num_frames))

        states_list = []
        images: Dict[str, list] = {}

        for fi in frame_indices:
            try:
                sample = ds[fi]
            except (IndexError, KeyError) as exc:
                raise ValueError(
                    f"Cannot read frame {fi} from dataset at {dataset_path}: {exc}"
                ) from exc

            state_keys = sorted(
                k for k in sample
                if k.startswith("observation.state")
                and not isinstance(sample[k], str)
            )
            if state_keys:
                state_parts = []
                for sk in state_keys:
                    v = sample[sk]
                    if hasattr(v, "numpy"):
                        v = v.numpy()
                    state_parts.append(np.asarray(v).flatten())
                states_list.append(np.concatenate(state_parts))

            image_keys = sorted(
                k for k in sample if k.startswith("observation.image")
            )
            for ik in image_keys:
                cam_name = ik.replace("observation.images.", "").replace("observation.image.", "")
                img = sample[ik]
                if hasattr(img, "numpy"):
                    img = img.numpy()
                img = np.asarray(img)
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = np.transpose(img, (1, 2, 0))
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                images.setdefault(cam_name, []).append(img)

        states = np.stack(states_list) if states_list else np.zeros((num_frames, 0), dtype=np.float32)

        prompt_val = ""
        try:
            sample_0 = ds[start_frame]
            for pk in ("prompt", "language_instruction", "task"):
                if pk in sample_0 and isinstance(sample_0[pk], str):
                    prompt_val = sample_0[pk]
                    break
        except Exception:
            pass

        return {
            "states": states,
            "images": images,
            "prompt": prompt_val or (self._default_prompt or ""),
            "episode_index": episode_index,
            "start_frame": start_frame,
            "num_frames": num_frames,
            "dataset_path": str(dataset_path),
        }
