---
name: deployment-protocol
description: Use when deploying a signed checkpoint onto a robot or inference rig and the agent needs to perform first-run or fresh-run preflight checks by reading MODEL_PASSPORT.json, the target repo's runner code, and live device samples.
---

# Deployment Protocol

## Overview

Use after a checkpoint has `MODEL_PASSPORT.json` and `SIGNOFF.json`, and the
next step is a robot or inference-rig run. The goal is to prove the live runner
feeds the signed checkpoint the same way the passport says it should.

Preflight compares three sources of truth:

1. `MODEL_PASSPORT.json` - model-side contract.
2. Target repo runner/container - what deployment code actually does.
3. Live or replay rig samples - what the device path produces.

## When to Use

- A checkpoint is signed and ready for first deployment.
- Hardware, code, bindings, or checkpoint changed since the last run.
- The user wants a repeatable preflight record before running inference.

Do not use this for training or eval. Eval uses `eval-tracking/SKILL.md`.

## Agent Algorithm

Follow this order. Later sections are reference material.

1. **Gate on signed checkpoint**
   - Run `validate-checkpoint <ckpt_dir> --require-signoff`.
   - If the CLI is unavailable, install/use
     `checkpoint-passport` from this repo.
   - Stop on any hard failure.

2. **Collect sources of truth**
   - Load `MODEL_PASSPORT.json` as expected model contract.
   - Identify the target repo runner/container and real model-loading path.
   - Capture live or replay rig samples for the modalities the passport declares.

3. **Build verification script**
   - Write code that loads expected passport values and observed runner/device
     values.
   - Use assertions for deterministic comparisons.
   - Save the script with the audit artifact.

4. **Run chain audit**
   - Validate input keys, shapes, dtypes, image semantics, normalization,
     model load, forward pass, output shape, unnormalization/post-processing, and
     final emission command.
   - Missing data or failed assertions are failures, not judgment calls.

5. **Record evidence**
   - Save the audit under the target repo, for example
     `deployment_preflight/<date>_<checkpoint>.md`.
   - Save script output, failures, screenshots/samples if relevant, and the exact
     deployment command that would run.
   - Do not patch code or relax assertions during preflight.

6. **Decide handoff**
   - If all checks pass, record deploy-ready status.
   - If any check fails, report the mismatch and stop for user/owner decision.

## Evidence Standard

The agent's primary job is to write and run verification code. For each chain
element, the script loads expected values from the passport, loads observed
values from the real runner or sample, asserts equality/ranges, and prints a
structured pass/fail result.

Rules:

- This is validation, not debugging. Report mismatches; do not fix them.
- Never weaken an assertion after it fails.
- Passport values are `expected`; live/runner values are `observed`.
- Missing data is a failure, not a reason to substitute passport values.
- The verification script and its output are the audit evidence.
- Agent judgment is allowed only for genuinely semantic items, such as whether a
  camera view matches the declared mount.

## Validator vs Preflight

Run `validate-checkpoint` first. It owns deterministic artifact checks: signoff
hashes, required files, schema, runtime constraints, dimensions, smoke-test
buckets, reference vectors, and normalization results.

This deployment protocol owns live/contextual checks: source selection, bindings,
runner path, semantic camera matching, transform path, and would-be emission.

Hard validator failures are terminal for preflight.

## Chain Audit Checklist

Audit these ten elements in order. Each element must be backed by code output or
a short judgment note where automation cannot decide.

1. **Source identity** - live rig, replay log, dataset, simulator, or adapter.
2. **Raw source sample** - observed keys, shapes, dtypes, ranges, timestamps.
3. **Source-to-passport bindings** - mapping from raw keys to passport keys.
4. **Checkpoint identity/integrity** - `validate-checkpoint --require-signoff`.
5. **Runtime model load path** - class/function, checkpoint path, assets.
6. **Preprocessing/transforms** - resize, color order, dtype, layout, state rules.
7. **Final model input contract** - exact tensors/arrays fed to the model.
8. **Model output sanity** - shape, dtype, finite values, expected horizon/dim.
9. **Output unnormalization/post-processing** - action units and clipping.
10. **Would-be emission behavior** - command/topic/API that would receive output.

## Outcomes

- `pass` - all required checks executed and matched.
- `blocked_by_missing_data` - a required source, sample, runner, or binding is
  unavailable.
- `fail_contract_mismatch` - observed behavior conflicts with the passport.
- `fail_runner_error` - runner/import/checkpoint load failed.
- `manual_review_required` - only semantic judgment remains.

Do not report deploy-ready unless every required element is `pass`.

## Audit Artifact

Save a markdown audit under the target repo, for example:

```markdown
# Deployment Preflight - <checkpoint/task>

## Run Header / Provenance
- checkpoint: `<path>`
- passport: `<path>/MODEL_PASSPORT.json`
- signoff verdict: `pass` | `soft_signal`
- target repo: `<path>` at `<commit>`
- runner entry point: `<command or module>`
- source mode: `live` | `replay` | `dataset` | `sim`

## Source Bindings
| passport key | source key/path | observed shape/dtype | status |
|--------------|-----------------|----------------------|--------|

## Chain Ledger
| element | status | evidence |
|---------|--------|----------|
| source_identity | pass | ... |

## Final Verdict
- verdict: `pass` | `blocked_by_missing_data` | `fail_contract_mismatch` | `fail_runner_error` | `manual_review_required`
- reason: <one sentence>

## Remaining Gaps / Next Actions
```

Save the verification script next to the audit so another engineer can rerun it.

## Minimal Code Pattern

Use one script with small helper functions rather than hand-written tables:

```python
expected = load_passport(passport_path)
observed = load_runner_or_sample(...)

assert observed["state"].shape == tuple(expected["input_contract"]["state"]["shape"])
assert observed["state"].dtype == expected["input_contract"]["state"]["dtype"]
print({"element": "final_model_input_contract", "status": "pass"})
```

If an assertion fails, the element fails. Do not relax the check in the same
preflight.

## Stop Gates

Stop and report when:

- `validate-checkpoint --require-signoff` fails.
- The target runner cannot load the checkpoint.
- Required device/replay samples are unavailable.
- Raw source keys cannot be mapped to passport keys.
- Output actions cannot be traced to the would-be emission path.
- Any assertion fails.

## Re-Audit Triggers

Run this protocol again when the checkpoint, target repo commit, container,
bindings, camera/sensor config, normalization assets, robot firmware, or runner
entry point changes.

## Common Mistakes

- Treating `validate-checkpoint` alone as deployment preflight.
- Filling audit tables by hand instead of writing assertions.
- Accepting "close enough" shapes/ranges.
- Testing a helper script instead of the real deployment runner path.
- Ignoring output post-processing and emission semantics.
- Declaring pass when samples were missing or substituted.
