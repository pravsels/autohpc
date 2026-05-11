"""
CLI: extract-passport-seed

Runs a framework-specific runtime extractor to produce a PASSPORT_SEED.json
for checkpoints that lack a passport-compatible static config.json.

Usage:
    extract-passport-seed openpi \
      --checkpoint-dir <ckpt> \
      --out <ckpt>/PASSPORT_SEED.json \
      --openpi-config-name <name> \
      --default-prompt <prompt> \
      --resize-size 224

Exit code 0 = seed written.  Exit code 1 = error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from checkpoint_passport.passport_seed import validate_seed, InvalidSeedError
from checkpoint_passport.runtime_extractors.base import (
    SUPPORTED_BACKENDS,
    UnsupportedBackendError,
    MissingRuntimeError,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="extract-passport-seed",
        description=(
            "Extract a passport seed from a checkpoint using a "
            "framework-specific runtime extractor."
        ),
    )
    parser.add_argument(
        "backend",
        help=f"framework backend ({', '.join(sorted(SUPPORTED_BACKENDS))})",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, required=True,
        help="path to the checkpoint directory",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="output path for the passport seed JSON",
    )
    parser.add_argument(
        "--openpi-config-name", type=str, default=None,
        help="OpenPI config registry name (required for openpi backend)",
    )
    parser.add_argument(
        "--default-prompt", type=str, default=None,
        help="default language prompt for the model",
    )
    parser.add_argument(
        "--resize-size", type=int, default=None,
        help="image resize dimension (e.g. 224)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="device for runtime enrichment (e.g. cuda, cpu); "
             "when set, loads model weights to extract library versions, "
             "parameter summary, and smoke results",
    )
    args = parser.parse_args()

    backend = args.backend.lower()

    if backend not in SUPPORTED_BACKENDS:
        print(
            f"error: {UnsupportedBackendError(backend)}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        if backend == "openpi":
            if not args.openpi_config_name:
                print(
                    "error: --openpi-config-name is required for the openpi backend",
                    file=sys.stderr,
                )
                sys.exit(1)

            from checkpoint_passport.runtime_extractors.openpi import OpenPIExtractor
            extractor = OpenPIExtractor()
            seed = extractor.extract_seed(
                args.checkpoint_dir,
                config_name=args.openpi_config_name,
                default_prompt=args.default_prompt,
                resize_size=args.resize_size,
                device=args.device,
            )

        validate_seed(seed)

    except (UnsupportedBackendError, MissingRuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except InvalidSeedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(seed, indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
