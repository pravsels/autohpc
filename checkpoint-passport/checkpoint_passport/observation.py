"""
Structured observation schema for validation checks.

Every check returns an Observation. Status is one of pass / fail /
soft_signal / not_checked.

The schema is the contract between Layer 1 (deterministic checks) and
any future Layer 2 (LLM/VLM reasoning over observations). Stays
JSON-serialisable and torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SOFT_SIGNAL = "soft_signal"
    NOT_CHECKED = "not_checked"


@dataclass(frozen=True)
class Observation:
    check: str                  # Which check produced this
    status: Status              # pass / fail / soft_signal / not_checked
    message: str                # One-line human summary
    details: Dict[str, Any]     # Exact numbers for forensics and Layer 2
    category: str = "uncategorised"  # input_expectation | model_internals | output_spec | signoff | provenance

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d
