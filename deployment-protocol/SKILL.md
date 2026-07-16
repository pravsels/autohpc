---
name: deployment-protocol
description: Use before running a checkpoint on a robot or inference rig to audit the real sensor-to-action chain using saved model assets, runner code, and live or replay samples.
---

# Deployment Protocol

## Overview

Use before a checkpoint's first robot or inference-rig run, and repeat when the
checkpoint, runner, hardware, or configuration changes. The goal is to audit the
actual chain from sensor input through preprocessing, model inference,
postprocessing, and would-be robot command.

The integrity manifest proves only that checkpoint files did not change. This
protocol establishes whether the real runtime uses those files correctly.

## Sources Of Truth

Reconcile these sources rather than relying on a separately authored contract:

1. Checkpoint files: saved config, processor/preprocessor, normalization and
   unnormalization assets, weights.
2. Model implementation and the code path that loads those saved files.
3. Target runner/container and its sensor/action bindings.
4. Live or replay rig samples and would-be emitted commands.

If a semantic expectation is not encoded in checkpoint assets or executable
code, mark it missing or inferred and cite its source. Do not invent it.

## Agent Algorithm

1. **Verify artifact integrity**
   - Run `verify-checkpoint <ckpt_dir>`.
   - If unavailable, install `checkpoint-integrity` from this repo.
   - Stop if a declared file is missing or changed.

2. **Trace the real runtime**
   - Identify the exact deployment command, container, model-loading function,
     saved config/processors/assets, input adapters, output postprocessing, and
     final robot API/topic.
   - Capture live or replay samples for every required modality.

3. **Build one verification script**
   - Import or invoke the real runner path; do not reproduce its behavior in a
     toy implementation.
   - Instrument boundaries so the script records raw inputs, final model inputs,
     raw outputs, postprocessed actions, and would-be emissions.
   - Assert deterministic properties derived from saved assets and real model
     interfaces: keys, shapes, dtypes, finite values, normalization behavior,
     action dimensions, units, clipping, and mappings.

4. **Run in no-act mode**
   - Load the real checkpoint and run the full chain without sending commands to
     the robot.
   - Missing evidence and failed assertions are failures.

5. **Record evidence**
   - Save the audit and rerunnable script under the target repo, for example
     `deployment_preflight/<date>_<checkpoint>.md`.
   - Include exact revisions, commands, output, failures, and relevant samples.
   - Do not patch runtime code or weaken assertions during the audit.

6. **Decide handoff**
   - Report deploy-ready only when every required chain element passes.
   - Otherwise report the precise failed or missing boundary and stop for a
     user/owner decision.

## Chain Audit Checklist

Audit these elements in order:

1. Source identity: live rig, replay log, dataset, simulator, or adapter.
2. Raw source sample: keys, shapes, dtypes, ranges, timestamps.
3. Source-to-runner bindings: camera/state names and ordering.
4. Checkpoint identity: passing `verify-checkpoint`.
5. Runtime load path: class/function, weights, config, processors, assets.
6. Preprocessing: resize, color order, dtype/layout, state transforms and norm.
7. Final model inputs: exact tensors/arrays crossing the model boundary.
8. Raw model outputs: shape, dtype, finite values, horizon/action dimensions.
9. Postprocessing: unnormalization, units, clipping, temporal selection.
10. Would-be emission: exact command/topic/API and final values, with actuation
    disabled.

## Evidence Standard

- The verification script and its output are the primary evidence.
- Expected values come from saved artifacts, executable interfaces, or an
  explicitly cited hardware/config source.
- Missing data is a failure, not permission to fill in a plausible value.
- Semantic judgment is allowed only where code cannot decide, such as whether a
  camera image corresponds to the intended physical mount.
- Integrity success does not imply deployment safety.

## Outcomes

- `pass` — every required check executed and matched.
- `blocked_by_missing_data` — a source, sample, asset, or binding is unavailable.
- `fail_contract_mismatch` — saved assets, runner, or live behavior disagree.
- `fail_runner_error` — import, model load, or inference failed.
- `manual_review_required` — only a named semantic judgment remains.

## Audit Artifact

```markdown
# Deployment Preflight - <checkpoint/task>

## Provenance
- checkpoint: `<path or HF repo/revision>`
- manifest: `<path>/CHECKPOINT_MANIFEST.json` (sha256: `<hash>`)
- target repo: `<path>` at `<commit>`
- runner entry point: `<command or module>`
- container/image: `<immutable reference>`
- source mode: `live` | `replay` | `dataset` | `sim`

## Saved Runtime Assets
| role | path | loaded by | status |
|------|------|-----------|--------|

## Source Bindings
| model input | source key/path | observed shape/dtype | status |
|-------------|-----------------|----------------------|--------|

## Chain Ledger
| element | status | evidence |
|---------|--------|----------|
| source_identity | pass | ... |

## Final Verdict
- verdict: `pass` | `blocked_by_missing_data` | `fail_contract_mismatch` | `fail_runner_error` | `manual_review_required`
- reason: <one sentence>

## Remaining Gaps / Next Actions
```

Save the verification script next to the audit.

## Stop Gates

- `verify-checkpoint` fails.
- The real runner cannot load the checkpoint and saved assets.
- Required live/replay samples are unavailable.
- Raw source keys cannot be mapped to model inputs.
- Output actions cannot be traced through postprocessing to no-act emission.
- Any assertion fails.

## Re-Audit Triggers

Repeat when the checkpoint, manifest, target repo commit, container, bindings,
camera/sensor config, normalization assets, robot firmware, or runner changes.

## Common Mistakes

- Treating a passing integrity check as deployment preflight.
- Replacing missing runtime semantics with a handwritten metadata document.
- Testing a helper rather than the real deployment runner.
- Comparing only shapes while ignoring order, normalization, units, and frames.
- Sending actions during an audit instead of stopping at would-be emission.
- Fixing code or relaxing assertions in the same run that discovered a failure.
