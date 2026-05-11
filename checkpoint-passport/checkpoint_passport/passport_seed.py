"""
Passport seed: the partially assembled passport content emitted by a runtime
extractor for checkpoints that lack a passport-compatible static config.json.

The seed contains sections whose content can only be determined by
instantiating the framework's runtime pipeline (e.g. OpenPI's
cfg.data.create(), transforms, norm stats).

The canonical assembler (assemble-passport) takes the seed, computes
weight_integrity, provenance, and metadata, then writes MODEL_PASSPORT.json.

Allowed seed sections (extractor may populate any subset):
    stack, input_contract, output_spec, model_identity,
    model_internals, transform_pipeline

The assembler owns and the seed must NOT contain:
    schema_version, generated_at, generated_by,
    weight_integrity, provenance
"""

from __future__ import annotations

from typing import Any, Dict, List, Set


ALLOWED_SEED_SECTIONS: Set[str] = {
    "stack",
    "input_contract",
    "output_spec",
    "model_identity",
    "model_internals",
    "transform_pipeline",
}

ASSEMBLER_OWNED_SECTIONS: Set[str] = {
    "schema_version",
    "generated_at",
    "generated_by",
    "weight_integrity",
    "provenance",
}

REQUIRED_EXTRACTOR_METADATA_KEYS: Set[str] = {
    "extractor_name",
    "extractor_version",
}


class InvalidSeedError(Exception):
    """Raised when a passport seed fails validation."""

    def __init__(self, reasons: List[str]):
        self.reasons = reasons
        super().__init__(
            "Invalid passport seed:\n" + "\n".join(f"  - {r}" for r in reasons)
        )


def validate_seed(seed: Dict[str, Any]) -> None:
    """Validate a passport seed dict.

    Checks:
        - No assembler-owned keys are present.
        - All top-level keys are in the allowed set or are 'extractor' metadata.
        - Extractor metadata block has required keys.

    Raises:
        InvalidSeedError with all violations.
    """
    errors: List[str] = []

    forbidden = set(seed.keys()) & ASSEMBLER_OWNED_SECTIONS
    if forbidden:
        errors.append(
            f"seed must not contain assembler-owned sections: "
            f"{', '.join(sorted(forbidden))}"
        )

    known_keys = ALLOWED_SEED_SECTIONS | {"extractor"}
    unknown = set(seed.keys()) - known_keys
    if unknown:
        errors.append(
            f"unknown top-level keys in seed: {', '.join(sorted(unknown))}"
        )

    extractor = seed.get("extractor")
    if extractor is None:
        errors.append("seed must contain an 'extractor' metadata block")
    elif isinstance(extractor, dict):
        missing = REQUIRED_EXTRACTOR_METADATA_KEYS - set(extractor.keys())
        if missing:
            errors.append(
                f"extractor metadata missing required keys: "
                f"{', '.join(sorted(missing))}"
            )

    if errors:
        raise InvalidSeedError(errors)
