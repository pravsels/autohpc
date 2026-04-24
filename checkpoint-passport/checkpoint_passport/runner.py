"""
Collects checks across all kernel categories and renders verdicts.

Two layers run here:
  - STATIC_CHECKS  -- functions of `CheckpointExtraction` only.  Always
                      runnable, no passport required.
  - PASSPORT_CHECKS -- functions that consume a `PassportLoadResult`
                      (and sometimes `CheckpointExtraction` and/or a
                      dataset path).  When no passport is present we still
                      call them so they can emit NOT_CHECKED observations,
                      keeping the report shape stable for consumers.

Per-module check functions vary in their argument lists; we introspect
each function's signature once and pass through only the kwargs it
declares.  This keeps the kernel modules free of a "uniform check
signature" constraint while letting the runner stay generic.

Output behaviour preserves the existing `format_report` layout so legacy
CLI consumers don't see surprises.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .extraction import CheckpointExtraction
from .observation import Observation, Status
from .passport import LoadStatus, PassportLoadResult, load_passport
from .schema import KNOWN_PASSPORT_SECTIONS
from .kernel import (
    input_expectation,
    model_internals,
    output_spec,
    provenance,
    signoff,
)


# Categories any check can emit. Used by `--skip-section` for validation.
KNOWN_CATEGORIES = (
    "signoff",
    "input_expectation",
    "model_internals",
    "output_spec",
    "provenance",
)


# Order chosen so the report reads top-down by category. Each module's
# STATIC_CHECKS are deterministic functions of the extraction.
STATIC_CHECK_MODULES = [
    model_internals,
    input_expectation,
]


# Modules whose PASSPORT_CHECKS lists we run when a passport is present.
# `signoff` doesn't fit this list cleanly because its checks need
# `require_signoff` and we want them grouped first in the report.
PASSPORT_CHECK_MODULES = [
    model_internals,
    input_expectation,
    output_spec,
    provenance,
]


def run_static_checks(ext: CheckpointExtraction) -> List[Observation]:
    """Run every static check across the registered kernel modules."""
    results: List[Observation] = []
    for module in STATIC_CHECK_MODULES:
        for check in module.STATIC_CHECKS:
            results.append(check(ext))
    return results


def run_passport_checks(
    load: PassportLoadResult,
    ext: CheckpointExtraction,
    dataset_path: Optional[Path] = None,
    require_signoff: bool = False,
) -> List[Observation]:
    """Run every passport-driven check, dispatched by signature.

    Each module exposes `PASSPORT_CHECKS: List[Callable]`. The callables
    have heterogeneous signatures -- some take just `load`, some take
    `(load, ext)`, one takes an optional `dataset_path`, and signoff's
    take `require_signoff`. We introspect each signature and pass only
    the kwargs the function declares.

    `signoff` lives at the top of the report (security-critical), so it
    runs first.
    """
    available_kwargs: Dict[str, Any] = {
        "load": load,
        "ext": ext,
        "dataset_path": dataset_path,
        "require_signoff": require_signoff,
    }

    results: List[Observation] = []
    # Signoff first: anything else is meaningless if checksums don't match.
    for check in signoff.PASSPORT_CHECKS if hasattr(signoff, "PASSPORT_CHECKS") else _signoff_default_checks():
        results.append(_invoke_with_known_kwargs(check, available_kwargs))

    for module in PASSPORT_CHECK_MODULES:
        for check in getattr(module, "PASSPORT_CHECKS", []):
            results.append(_invoke_with_known_kwargs(check, available_kwargs))
    return results


def run_all_checks(
    checkpoint_dir: str | Path,
    dataset_path: Optional[Path] = None,
    require_signoff: bool = False,
) -> List[Observation]:
    """Top-level entry point: extract, load passport, run everything.

    Returns observations across signoff + every passport-driven check
    plus the legacy static checks. When no passport is present, the
    passport-driven checks emit NOT_CHECKED (or SOFT_SIGNAL for
    signoff-presence) so the report shape stays the same.
    """
    ext = CheckpointExtraction.from_checkpoint_dir(checkpoint_dir)
    load = load_passport(checkpoint_dir)

    observations: List[Observation] = []
    observations.extend(
        run_passport_checks(
            load, ext,
            dataset_path=dataset_path,
            require_signoff=require_signoff,
        )
    )
    observations.extend(run_static_checks(ext))
    return observations


# Back-compat alias: pre-passport callers still expect this name.
def run_integrity_checks(ext: CheckpointExtraction) -> List[Observation]:
    return run_static_checks(ext)


# ── Dispatch helper ─────────────────────────────────────────────────────


def _invoke_with_known_kwargs(
    check: Callable[..., Observation],
    available: Dict[str, Any],
) -> Observation:
    """Call `check` with whichever subset of `available` it declares.

    Positional-only and var-args are not used by the kernel checks, so a
    plain signature inspection is enough. We also fall through default
    values for parameters we have no value for.
    """
    sig = inspect.signature(check)
    kwargs: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in available and available[name] is not None:
            kwargs[name] = available[name]
        elif name in available and param.default is inspect.Parameter.empty:
            # Required parameter we don't have -- pass the None through;
            # the check is responsible for handling it.
            kwargs[name] = available[name]
    return check(**kwargs)


def _signoff_default_checks() -> List[Callable[..., Observation]]:
    """Fallback if signoff.py hasn't published a PASSPORT_CHECKS list."""
    return [
        signoff.check_signoff_present,
        signoff.check_signoff_schema_version_supported,
        signoff.check_passport_checksum_valid,
        signoff.check_weight_files_match_signoff,
    ]


# ── Report formatting ──────────────────────────────────────────────────


_STATUS_ICONS = {
    Status.PASS: "✅",
    Status.FAIL: "❌",
    Status.SOFT_SIGNAL: "⚠️ ",
    Status.NOT_CHECKED: "⬜",
}


_CATEGORY_LABELS = {
    "signoff": "Signoff",
    "model_internals": "Model Internals",
    "input_expectation": "Input Expectation",
    "output_spec": "Output Spec",
    "provenance": "Provenance",
    "uncategorised": "Other",
}


_LOAD_STATUS_LABELS = {
    LoadStatus.NO_PASSPORT: "no passport on disk",
    LoadStatus.PASSPORT_NO_SIGNOFF: "passport ok, no signoff",
    LoadStatus.PASSPORT_AND_SIGNOFF: "passport + signoff present",
    LoadStatus.PASSPORT_DEGRADED: "passport present but unparseable (degraded)",
}


def _format_header_lines(
    load: PassportLoadResult,
    skipped_sections: Optional[Iterable[str]] = None,
) -> List[str]:
    """One-shot header summarising the passport/signoff state.

    Operator-facing: the answer to "did this checkpoint actually arrive
    with a usable passport, and what was inside it?" should be visible
    without scrolling through the per-check observations.
    """
    lines: List[str] = []
    lines.append(f"Load status:    {_LOAD_STATUS_LABELS.get(load.status, load.status.value)}")

    if load.passport is not None:
        lines.append(f"Passport:       schema_version={load.passport.schema_version}")
    elif load.passport_present_on_disk:
        lines.append("Passport:       on disk but unparseable")
    else:
        lines.append("Passport:       not present")

    if load.signoff is not None:
        lines.append(f"Signoff:        schema_version={load.signoff.schema_version}")
    elif load.signoff_present_on_disk:
        lines.append("Signoff:        on disk but unparseable")
    else:
        lines.append("Signoff:        not present")

    if load.passport_raw is not None:
        present = sorted(
            s for s in KNOWN_PASSPORT_SECTIONS if s in load.passport_raw
        )
        missing = sorted(KNOWN_PASSPORT_SECTIONS - set(present))
        lines.append(
            "Sections:       "
            + (", ".join(present) if present else "(none populated)")
        )
        if missing:
            lines.append("  missing:      " + ", ".join(missing))
    elif load.passport_present_on_disk:
        # File was on disk but JSON didn't parse / wasn't an object.
        # Distinguish from "no passport" so the operator knows the file
        # exists and needs investigating.
        lines.append("Sections:       n/a (passport unparseable)")
    else:
        lines.append("Sections:       n/a (no passport)")

    skipped = sorted(set(skipped_sections or []))
    if skipped:
        lines.append("Skipped:        " + ", ".join(skipped))

    lines.append("=" * 40)
    return lines


def format_report(
    observations: Iterable[Observation],
    show_not_checked: bool = False,
    passport_load: Optional[PassportLoadResult] = None,
    skipped_sections: Optional[Iterable[str]] = None,
) -> str:
    """Render observations as a human-readable, category-grouped report.

    NOT_CHECKED rows are hidden by default to keep the operator-facing
    view small; pass show_not_checked=True for the full forensic view.

    When a `passport_load` is supplied, a one-line header summarises the
    signoff/passport state plus which top-level passport sections were
    populated, so the operator can tell at a glance how much of the
    passport was actually exercised.
    """
    obs_list = list(observations)
    lines = ["Model Integrity Report", "=" * 40]
    if passport_load is not None:
        lines.extend(_format_header_lines(passport_load, skipped_sections))

    by_category: dict[str, List[Observation]] = {}
    for o in obs_list:
        by_category.setdefault(o.category, []).append(o)

    category_order = list(_CATEGORY_LABELS.keys())
    seen_categories: set = set()
    for cat in category_order:
        bucket = by_category.get(cat)
        if not bucket:
            continue
        seen_categories.add(cat)
        visible = [
            o for o in bucket
            if show_not_checked or o.status is not Status.NOT_CHECKED
        ]
        if not visible:
            continue
        lines.append("")
        lines.append(f"── {_CATEGORY_LABELS[cat]} ──")
        for obs in visible:
            lines.append(f"{_STATUS_ICONS[obs.status]} {obs.check}")
            lines.append(f"   {obs.message}")
            if obs.status in (Status.FAIL, Status.SOFT_SIGNAL):
                for k, v in obs.details.items():
                    lines.append(f"   {k}: {v}")

    # Defensive: any categories not in the known list still get rendered.
    for cat, bucket in by_category.items():
        if cat in seen_categories or cat in _CATEGORY_LABELS:
            continue
        visible = [
            o for o in bucket
            if show_not_checked or o.status is not Status.NOT_CHECKED
        ]
        if not visible:
            continue
        lines.append("")
        lines.append(f"── {cat} ──")
        for obs in visible:
            lines.append(f"{_STATUS_ICONS[obs.status]} {obs.check}")
            lines.append(f"   {obs.message}")

    lines.append("")
    lines.append("=" * 40)

    n_pass = sum(1 for o in obs_list if o.status == Status.PASS)
    n_fail = sum(1 for o in obs_list if o.status == Status.FAIL)
    n_soft = sum(1 for o in obs_list if o.status == Status.SOFT_SIGNAL)
    n_skip = sum(1 for o in obs_list if o.status == Status.NOT_CHECKED)
    if n_fail > 0:
        lines.append(f"❌ DON'T SHIP -- {n_fail} check(s) failed")
    elif n_soft > 0:
        lines.append(f"⚠️  HUMAN LOOK HERE -- {n_soft} soft signal(s), "
                     f"{n_pass} passed, {n_skip} not checked")
    else:
        lines.append(f"✅ SHIP IT -- {n_pass} check(s) passed, "
                     f"{n_skip} not checked")

    return "\n".join(lines)
