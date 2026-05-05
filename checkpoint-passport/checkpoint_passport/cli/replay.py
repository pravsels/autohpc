"""
CLI entry point: replay the reference test vector through an adapter.

Loads the golden input (images, state, prompt) from the checkpoint's
assets/ directory, runs the adapter's predict() method, and compares
the output against the stored expected_output.npy.  All asset hashes
are verified against the passport before running.

This is a dynamic check — it requires torch, the model package, and
GPU access.  Run it in the model's own environment, not the
checkpoint-passport uv env.

Usage:
    replay-reference-vector <checkpoint_dir> \\
        --adapter-module missiontracker.adapters.multitask_dit_adapter \\
        --adapter-class MultiTaskDiTAdapter \\
        [--device cuda:0] [--tolerance 1e-4]

Exit code 0 = output matches expected within tolerance.
Exit code 1 = mismatch or error.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PASSPORT_FILENAME = "MODEL_PASSPORT.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_passport(ckpt: Path) -> Dict[str, Any]:
    pp = ckpt / PASSPORT_FILENAME
    if not pp.is_file():
        print(f"error: {PASSPORT_FILENAME} not found in {ckpt}", file=sys.stderr)
        sys.exit(1)
    return json.loads(pp.read_text())


def _verify_hash(path: Path, expected: str, label: str) -> bool:
    """Returns True if hash matches, False otherwise."""
    actual = _sha256(path)
    if actual != expected:
        print(
            f"FAIL: {label} hash mismatch\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}",
            file=sys.stderr,
        )
        return False
    return True


def _load_adapter(
    module_name: str,
    class_name: str,
    checkpoint_dir: Path,
    device: str,
) -> Any:
    """Import the adapter class and instantiate via from_pretrained."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        print(
            f"error: cannot import adapter module '{module_name}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    cls = getattr(mod, class_name, None)
    if cls is None:
        print(
            f"error: class '{class_name}' not found in module '{module_name}'",
            file=sys.stderr,
        )
        sys.exit(1)

    return cls.from_pretrained(str(checkpoint_dir), device=device)


def replay(
    checkpoint_dir: Path,
    adapter_module: str,
    adapter_class: str,
    device: str = "cuda",
    tolerance: Optional[float] = None,
) -> bool:
    """Run the reference test vector through the adapter and compare.

    Returns True if output matches expected within tolerance.
    """
    import numpy as np
    import torch

    passport = _load_passport(checkpoint_dir)
    rtv = passport.get("reference_test_vector")
    if not rtv:
        print("FAIL: no reference_test_vector in passport", file=sys.stderr)
        return False

    input_state_path = rtv.get("input_state_path")
    input_state_hash = rtv.get("input_state_hash")
    expected_output_path = rtv.get("expected_output_path")
    expected_output_hash = rtv.get("expected_output_hash")
    input_images_path = rtv.get("input_images_path")
    input_images_hash: Dict[str, str] = rtv.get("input_images_hash", {})
    input_prompt = rtv.get("input_prompt", "")
    torch_seed = rtv.get("torch_seed", 0)
    tol = tolerance if tolerance is not None else rtv.get("tolerance", 1e-4)

    for field in ("input_state_path", "input_state_hash",
                  "expected_output_path", "expected_output_hash",
                  "input_images_path"):
        if not rtv.get(field):
            print(f"FAIL: reference_test_vector.{field} is missing", file=sys.stderr)
            return False
    if not input_images_hash:
        print("FAIL: reference_test_vector.input_images_hash is empty", file=sys.stderr)
        return False

    # -- verify and load input_state.npy --
    state_file = checkpoint_dir / input_state_path
    if not state_file.is_file():
        print(f"FAIL: {input_state_path} not found", file=sys.stderr)
        return False
    if not _verify_hash(state_file, input_state_hash, "input_state"):
        return False
    input_state = np.load(state_file, allow_pickle=False)
    print(f"  input_state: {input_state.shape} {input_state.dtype}")

    # -- verify and load expected_output.npy --
    output_file = checkpoint_dir / expected_output_path
    if not output_file.is_file():
        print(f"FAIL: {expected_output_path} not found", file=sys.stderr)
        return False
    if not _verify_hash(output_file, expected_output_hash, "expected_output"):
        return False
    expected_output = np.load(output_file, allow_pickle=False)
    print(f"  expected_output: {expected_output.shape} {expected_output.dtype}")

    # -- verify and load reference frame images --
    images_dir = checkpoint_dir / input_images_path
    if not images_dir.is_dir():
        print(f"FAIL: {input_images_path} not found or not a directory", file=sys.stderr)
        return False

    image_tensors: Dict[str, torch.Tensor] = {}
    for cam_key, expected_hash in input_images_hash.items():
        # cam_key is like "images.front" — the PNG filename uses the last part
        short_key = cam_key.split(".")[-1] if "." in cam_key else cam_key
        png_path = images_dir / f"{short_key}.png"
        if not png_path.is_file():
            print(f"FAIL: reference frame {png_path} not found", file=sys.stderr)
            return False
        if not _verify_hash(png_path, expected_hash, f"image:{cam_key}"):
            return False

        from PIL import Image
        img = Image.open(png_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0  # HWC [0,1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # CHW
        # Strip the "images." or "observation.images." prefix for the adapter
        adapter_key = cam_key
        if adapter_key.startswith("observation."):
            adapter_key = adapter_key[len("observation."):]
        if adapter_key.startswith("images."):
            adapter_key = "image_" + adapter_key[len("images."):]
        image_tensors[adapter_key] = img_tensor
        print(f"  {cam_key}: {img_tensor.shape}")

    # -- load adapter --
    print(f"\nLoading adapter: {adapter_module}.{adapter_class}")
    adapter = _load_adapter(adapter_module, adapter_class, checkpoint_dir, device)

    # -- build observation and run forward pass --
    # Dynamically import PolicyObservation from the same package as the adapter.
    # Falls back to a simple namespace if the adapter's package doesn't provide one.
    adapter_pkg = adapter_module.rsplit(".", 1)[0] if "." in adapter_module else adapter_module
    state_tensor = torch.from_numpy(input_state.astype(np.float32))
    try:
        policy_mod = importlib.import_module(f"{adapter_pkg}.policy")
        ObsCls = getattr(policy_mod, "PolicyObservation")
        obs = ObsCls(
            images=image_tensors,
            state=state_tensor,
            language_instruction=input_prompt if input_prompt else None,
        )
    except (ImportError, AttributeError):
        # Fallback: construct a simple namespace the adapter can consume
        class _Obs:
            pass
        obs = _Obs()
        obs.images = image_tensors
        obs.state = state_tensor
        obs.language_instruction = input_prompt if input_prompt else None

    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    print(f"Running forward pass (seed={torch_seed})...")
    with torch.no_grad():
        result = adapter.predict(obs)

    actual_output = result.actions.cpu().numpy()
    print(f"  actual output: {actual_output.shape} {actual_output.dtype}")

    # -- compare --
    if actual_output.shape != expected_output.shape:
        print(
            f"\nFAIL: shape mismatch\n"
            f"  expected: {expected_output.shape}\n"
            f"  actual:   {actual_output.shape}",
            file=sys.stderr,
        )
        return False

    abs_diff = np.abs(actual_output - expected_output)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    mismatches = int((abs_diff > tol).sum())
    total_elements = int(abs_diff.size)

    print(f"\n  tolerance:  {tol}")
    print(f"  max_diff:   {max_diff:.6e}")
    print(f"  mean_diff:  {mean_diff:.6e}")
    print(f"  mismatches: {mismatches}/{total_elements}")

    if max_diff <= tol:
        print(f"\nPASS: output matches expected within tolerance {tol}")
        return True

    # Print worst offenders
    flat_diff = abs_diff.flatten()
    worst_indices = np.argsort(flat_diff)[-5:][::-1]
    print(f"\nFAIL: output differs beyond tolerance {tol}")
    print("  worst mismatches (flat index, expected, actual, diff):")
    flat_expected = expected_output.flatten()
    flat_actual = actual_output.flatten()
    for idx in worst_indices:
        print(
            f"    [{idx}] expected={flat_expected[idx]:.6f} "
            f"actual={flat_actual[idx]:.6f} "
            f"diff={flat_diff[idx]:.6e}"
        )
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="replay-reference-vector",
        description=(
            "Replay the reference test vector through the model adapter "
            "and compare output against the stored golden expected output."
        ),
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument(
        "--adapter-module", required=True,
        help="Python module containing the adapter class",
    )
    parser.add_argument(
        "--adapter-class", required=True,
        help="adapter class name (must have from_pretrained classmethod)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="torch device (default: cuda)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=None,
        help="override passport tolerance for comparison",
    )
    args = parser.parse_args()

    ckpt = args.checkpoint_dir.resolve()
    if not ckpt.is_dir():
        print(f"error: {ckpt} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"replay-reference-vector")
    print(f"  checkpoint: {ckpt}")
    print(f"  adapter:    {args.adapter_module}.{args.adapter_class}")
    print(f"  device:     {args.device}")
    print()

    ok = replay(
        checkpoint_dir=ckpt,
        adapter_module=args.adapter_module,
        adapter_class=args.adapter_class,
        device=args.device,
        tolerance=args.tolerance,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
