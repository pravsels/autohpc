"""
Checkpoint validation: deterministic checks over (passport, checkpoint files,
optional dataset).

Public surface:
- Observation / Status: structured check result
- CheckpointExtraction: framework-agnostic read of checkpoint files
- ModelPassport / Signoff: schemas for the two on-disk passport files
- run_all_checks: top-level entry point used by the CLI and the autohpc
  self-validator
"""

from .observation import Observation, Status
from .extraction import (
    CheckpointExtraction,
    WeightManifest,
)
from .schema import (
    ModelPassport,
    Signoff,
    SCHEMA_VERSION,
    SIGNOFF_SCHEMA_VERSION,
)
from .passport import (
    PassportLoadResult,
    LoadStatus,
    load_passport,
    PASSPORT_FILENAME,
    SIGNOFF_FILENAME,
)
from .runner import (
    run_all_checks,
    run_static_checks,
    run_passport_checks,
    run_integrity_checks,
    format_report,
)
from .self_validate_passport import (
    self_validate_passport,
    SelfValidationResult,
)

__all__ = [
    "Observation",
    "Status",
    "CheckpointExtraction",
    "WeightManifest",
    "ModelPassport",
    "Signoff",
    "SCHEMA_VERSION",
    "SIGNOFF_SCHEMA_VERSION",
    "PassportLoadResult",
    "LoadStatus",
    "load_passport",
    "PASSPORT_FILENAME",
    "SIGNOFF_FILENAME",
    "run_all_checks",
    "run_static_checks",
    "run_passport_checks",
    "run_integrity_checks",
    "format_report",
    "self_validate_passport",
    "SelfValidationResult",
]
