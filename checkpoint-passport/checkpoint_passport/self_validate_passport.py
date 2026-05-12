"""
Public entry point for the deterministic checkpoint validator.

This is the function the autohpc skill calls pre-upload (to verify its
own passport is internally consistent before pushing to HF) and the
function the downstream consumer calls post-download (to verify nothing
changed in transit and the deployment environment can resolve the
declared classes).

It's a pure function over (passport, signoff, checkpoint files,
optional dataset).  It NEVER loads the model -- model loading is the
upstream skill's job, the result of which is what the passport records.

Returns a structured `SelfValidationResult` so callers can branch on the
verdict programmatically; pretty-printing is in `runner.format_report`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set

from .extraction import CheckpointExtraction
from .observation import Observation, Status
from .passport import PassportLoadResult, load_passport
from .runner import (
    KNOWN_CATEGORIES,
    format_report,
    run_passport_checks,
    run_static_checks,
)


@dataclass(frozen=True)
class SelfValidationResult:
    """Result of `self_validate_passport`.

    `verdict` collapses observations into the plan's three states:
      "ship_it" / "human_look_here" / "dont_ship"
    """

    checkpoint_dir: Path
    observations: List[Observation]
    passport_load: PassportLoadResult
    extraction: CheckpointExtraction
    notes: List[str] = field(default_factory=list)
    skipped_sections: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if any(o.status is Status.FAIL for o in self.observations):
            return "dont_ship"
        if any(o.status is Status.SOFT_SIGNAL for o in self.observations):
            return "human_look_here"
        return "ship_it"

    @property
    def has_failures(self) -> bool:
        return self.verdict == "dont_ship"

    def render(self, show_not_checked: bool = False) -> str:
        return format_report(
            self.observations,
            show_not_checked=show_not_checked,
            passport_load=self.passport_load,
            skipped_sections=self.skipped_sections,
        )


def self_validate_passport(
    checkpoint_dir: str | Path,
    *,
    dataset_path: Optional[str | Path] = None,
    target_repo: Optional[str | Path] = None,
    require_signoff: bool = False,
    extra_norm_stats: Optional[List[Path]] = None,
    skip_sections: Optional[Iterable[str]] = None,
) -> SelfValidationResult:
    """Run every Layer 1 check against a checkpoint directory.

    Parameters
    ----------
    checkpoint_dir
        Path to a checkpoint root.  Should contain weight files plus
        optionally MODEL_PASSPORT.json and SIGNOFF.json.
    dataset_path
        Optional path to a local LeRobot dataset directory.  Enables the
        `input_contract_vs_dataset` check; without it that check
        emits NOT_CHECKED unless the passport's training_datasets[0]
        resolves to a local HF cache snapshot.
    target_repo
        Optional path to the deployment target repo (e.g. alpha-robotics).
        When provided, the validator reports whether the repo's HEAD matches
        ``provenance.deployment_repo_commit`` and whether the working tree is
        clean.  Deployment repo drift is debug context only and is reported as
        a soft signal, not a load gate.
    require_signoff
        When true, missing SIGNOFF.json becomes a hard FAIL instead of
        a SOFT_SIGNAL.  Use this on production deployment paths.
    extra_norm_stats
        Optional list of stats JSON files outside the checkpoint dir to
        merge into `extraction.norm_stats`.  Some training stacks store
        stats in a sibling assets/ directory; this lets the validator
        see them.
    skip_sections
        Optional iterable of category names (e.g. {"signoff",
        "provenance"}).  Observations with `category` in this set are
        dropped before the verdict is computed -- useful during partial
        development when a section's checks are noisy or known to be
        broken.  Unknown category names are ignored.

        SECURITY: skipping `signoff` removes every tamper-detection check
        (passport checksum, weight-file checksums) from the verdict. It
        is therefore an error to combine `skip_sections={"signoff"}` with
        `require_signoff=True` -- the two options contradict and the
        combination would silently let a tampered checkpoint pass.

    Returns
    -------
    SelfValidationResult
        Bundles observations, the passport load result, and the raw
        extraction so callers can introspect what was checked.

    Raises
    ------
    ValueError
        If `require_signoff` is True and `"signoff"` is in
        `skip_sections`.
    """
    ckpt_path = Path(checkpoint_dir)
    skip_set: Set[str] = set(skip_sections or [])

    if require_signoff and "signoff" in skip_set:
        raise ValueError(
            "require_signoff=True is incompatible with skip_sections "
            "containing 'signoff': the first demands a valid signoff, "
            "the second discards every signoff observation. Pick one."
        )

    extraction = CheckpointExtraction.from_checkpoint_dir(ckpt_path)
    if extra_norm_stats:
        merged = dict(extraction.norm_stats)
        for ns_path in extra_norm_stats:
            ns = Path(ns_path)
            try:
                import json as _json
                merged[f"extra:{ns.name}"] = _json.loads(ns.read_text())
            except (OSError, ValueError):
                # The file isn't readable -- the checks themselves will
                # surface "no norm stats" if this leaves us with none.
                continue
        extraction = CheckpointExtraction(
            path=extraction.path,
            config=extraction.config,
            weights=extraction.weights,
            norm_stats=merged,
            config_files_found=extraction.config_files_found,
        )

    load = load_passport(ckpt_path)

    observations: List[Observation] = []
    observations.extend(
        run_passport_checks(
            load, extraction,
            dataset_path=Path(dataset_path) if dataset_path else None,
            require_signoff=require_signoff,
            target_repo=Path(target_repo) if target_repo else None,
        )
    )
    observations.extend(run_static_checks(extraction))

    # `--skip-section` filters by Observation.category. Done after the
    # checks run (rather than gating dispatch) so the runner stays
    # category-agnostic and a future check that emits multiple categories
    # still gets filtered correctly.
    notes = list(load.notes)
    skipped_actually_present: List[str] = []
    if skip_set:
        unknown = sorted(skip_set - set(KNOWN_CATEGORIES))
        if unknown:
            notes.append(
                "skip_sections: ignored unknown categories: "
                + ", ".join(unknown)
            )
        kept: List[Observation] = []
        for obs in observations:
            if obs.category in skip_set:
                if obs.category not in skipped_actually_present:
                    skipped_actually_present.append(obs.category)
                continue
            kept.append(obs)
        observations = kept

    return SelfValidationResult(
        checkpoint_dir=ckpt_path,
        observations=observations,
        passport_load=load,
        extraction=extraction,
        notes=notes,
        skipped_sections=sorted(
            (skip_set & set(KNOWN_CATEGORIES)) | set(skipped_actually_present)
        ),
    )


__all__ = ["self_validate_passport", "SelfValidationResult"]
