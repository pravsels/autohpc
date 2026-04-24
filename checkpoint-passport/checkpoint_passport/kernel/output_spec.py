"""
Output Spec: "are the predictions coming out reasonable?"

Operates on the passport's `output_spec` section. Smoke results were
computed upstream (single forward pass on a calibration batch); downstream
verifies they pass and that declared output structure is consistent with
the input contract.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List

from ..observation import Observation, Status
from ..passport import PassportLoadResult


CATEGORY = "output_spec"


def check_output_actions_vs_input_actions(load: PassportLoadResult) -> Observation:
    """`output_spec.actions` is consistent with `input_contract.actions`.

    The cleanest model is: the model's output IS its input action contract,
    so horizon and sub_keys must agree. The plan's example uses a literal
    "see input_contract.actions.sub_keys" string for sub_keys; that's
    accepted as a reference rather than a duplicate value.
    """
    if not load.has_passport:
        return _not_checked("output_actions_vs_input_actions", "no passport loaded")

    out = load.passport.output_spec.actions
    inp = load.passport.input_contract.actions

    if out is None and inp is None:
        return _not_checked(
            "output_actions_vs_input_actions",
            "neither input_contract.actions nor output_spec.actions populated",
        )
    if out is None:
        return Observation(
            check="output_actions_vs_input_actions",
            status=Status.SOFT_SIGNAL,
            message="output_spec.actions missing while input_contract.actions present",
            details={},
            category=CATEGORY,
        )
    if inp is None:
        return Observation(
            check="output_actions_vs_input_actions",
            status=Status.SOFT_SIGNAL,
            message="input_contract.actions missing while output_spec.actions present",
            details={},
            category=CATEGORY,
        )

    mismatches: List[Dict[str, Any]] = []

    if out.horizon is not None and inp.horizon is not None:
        if out.horizon != inp.horizon:
            mismatches.append({
                "field": "horizon",
                "input_contract": inp.horizon,
                "output_spec": out.horizon,
            })

    # sub_keys: accept a "see input_contract.actions.sub_keys" string as a
    # valid reference; only compare lists when both sides supply lists.
    if isinstance(out.sub_keys, list) and inp.sub_keys:
        if not _sub_keys_equal(out.sub_keys, inp.sub_keys):
            mismatches.append({
                "field": "sub_keys",
                "input_contract": [k.get("key") for k in inp.sub_keys],
                "output_spec": [k.get("key") for k in out.sub_keys],
            })

    if mismatches:
        return Observation(
            check="output_actions_vs_input_actions",
            status=Status.FAIL,
            message=f"{len(mismatches)} output/input action mismatch(es)",
            details={"mismatches": mismatches},
            category=CATEGORY,
        )

    return Observation(
        check="output_actions_vs_input_actions",
        status=Status.PASS,
        message="output_spec.actions consistent with input_contract.actions",
        details={
            "horizon": inp.horizon,
            "sub_keys_layout_reference": (
                out.sub_keys if isinstance(out.sub_keys, str) else None
            ),
        },
        category=CATEGORY,
    )


# Smoke result subfields that have a top-level "status" the upstream sets
# to "pass" / "fail". Keep this list narrow -- distribution and range_check
# also have status fields but encode their pass/fail via boolean fields too.
_STATUS_BEARING_SUBFIELDS = [
    "determinism",
    "nan_inf",
    "liveness",
    "distribution",
    "range_check",
]


def check_smoke_results_pass(load: PassportLoadResult) -> Observation:
    """Every smoke-test bucket in output_spec.smoke_results passes.

    Pass criteria, per bucket:
      determinism: status == "pass"  (recorded after running forward twice)
      nan_inf:     status == "pass"
      liveness:    status == "pass"
      distribution: ratio_in_acceptable_range == True (or status == "pass")
      range_check: in_expected_range == True          (or status == "pass")

    A bucket the upstream didn't populate is reported as not-checked but
    doesn't break the overall pass -- the autohpc skill controls how
    thorough the smoke run is.
    """
    if not load.has_passport:
        return _not_checked("smoke_results_pass", "no passport loaded")

    sr = load.passport.output_spec.smoke_results
    if sr is None:
        return _not_checked(
            "smoke_results_pass",
            "passport.output_spec.smoke_results not populated",
        )

    # Re-render typed sub-dataclasses as dicts so the per-bucket helper can
    # iterate uniformly.  An unset bucket (None) becomes {} -- treated as
    # not_run by `_bucket_verdict`.
    sr_dict = {
        "determinism": _bucket_to_dict(sr.determinism),
        "nan_inf": _bucket_to_dict(sr.nan_inf),
        "liveness": _bucket_to_dict(sr.liveness),
        "distribution": _bucket_to_dict(sr.distribution),
        "range_check": _bucket_to_dict(sr.range_check),
    }

    failures: List[Dict[str, Any]] = []
    not_run: List[str] = []
    passed: List[str] = []

    for name in _STATUS_BEARING_SUBFIELDS:
        block = sr_dict[name]
        if not block:
            not_run.append(name)
            continue
        verdict = _bucket_verdict(name, block)
        if verdict == "pass":
            passed.append(name)
        elif verdict == "fail":
            failures.append({"bucket": name, "block": dict(block)})
        else:  # "unknown"
            not_run.append(name)

    if failures:
        return Observation(
            check="smoke_results_pass",
            status=Status.FAIL,
            message=(
                f"{len(failures)} smoke-test bucket(s) failed: "
                f"{[f['bucket'] for f in failures]}"
            ),
            details={
                "failures": failures,
                "passed": passed,
                "not_run": not_run,
                "calibration_batch_source": sr.calibration_batch_source,
                "calibration_batch_size": sr.calibration_batch_size,
            },
            category=CATEGORY,
        )

    if not passed:
        return Observation(
            check="smoke_results_pass",
            status=Status.SOFT_SIGNAL,
            message="smoke_results present but no bucket recorded a verdict",
            details={"not_run": not_run},
            category=CATEGORY,
        )

    return Observation(
        check="smoke_results_pass",
        status=Status.PASS,
        message=f"{len(passed)} smoke-test bucket(s) passed; "
                f"{len(not_run)} not run",
        details={
            "passed": passed,
            "not_run": not_run,
            "calibration_batch_source": sr.calibration_batch_source,
            "calibration_batch_size": sr.calibration_batch_size,
        },
        category=CATEGORY,
    )

# ── Helpers ─────────────────────────────────────────────────────────────


def _not_checked(name: str, reason: str) -> Observation:
    return Observation(
        check=name,
        status=Status.NOT_CHECKED,
        message=reason,
        details={},
        category=CATEGORY,
    )


def _bucket_to_dict(block: Any) -> Dict[str, Any]:
    """Coerce a smoke-test bucket to a flat dict for verdict inspection.

    Buckets are typed sub-dataclasses on modern passports but can also be
    bare dicts on older / hand-edited ones.  Either way the verdict logic
    below reads `.get("status")` and friends, so we normalise here.
    """
    if block is None:
        return {}
    if is_dataclass(block):
        return asdict(block)
    if isinstance(block, dict):
        return block
    return {}


def _bucket_verdict(name: str, block: Dict[str, Any]) -> str:
    """Decide whether a smoke-test sub-bucket passed.

    Returns "pass" / "fail" / "unknown". Unknown means the upstream skill
    didn't populate enough to judge -- treated as not-run rather than fail.
    """
    status = block.get("status")
    if status == "pass":
        return "pass"
    if status == "fail":
        return "fail"

    # Per-bucket fallbacks for blocks that encode pass/fail differently.
    if name == "distribution":
        v = block.get("ratio_in_acceptable_range")
        if v is True:
            return "pass"
        if v is False:
            return "fail"
    if name == "range_check":
        v = block.get("in_expected_range")
        if v is True:
            return "pass"
        if v is False:
            return "fail"
    if name == "nan_inf":
        # if "samples_checked" recorded with no failures column, treat as pass
        if block.get("samples_checked") and not block.get("failures"):
            return "pass"
    return "unknown"


def _sub_keys_equal(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> bool:
    """Sub-keys are equal if their (key, dim) tuples line up in order."""
    def tup(x):
        return (x.get("key"), x.get("dim"))
    return [tup(x) for x in a] == [tup(x) for x in b]


PASSPORT_CHECKS: List[Callable[..., Observation]] = [
    check_output_actions_vs_input_actions,
    check_smoke_results_pass,
]
