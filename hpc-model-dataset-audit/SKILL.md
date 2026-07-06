---
name: hpc-model-dataset-audit
description: Use before committing to a full training run to reconcile a base model's baked-in expectations against the training config and the actual dataset, catching semantic mismatches (normalization, feature order, units, encoding) that build and run fine but silently ruin results.
---

# HPC Model–Dataset Audit

## Overview

Use this to catch the failures that `hpc-container-promotion` and
`hpc-dataset-adaptation` cannot: the image builds, the loader emits the right
keys/shapes/dtypes, the smoke test passes — and every trained policy is still
uniformly bad because a *semantic* contract is violated.

Structural checks see shapes and dtypes. They do not see that the base
checkpoint was pretrained expecting quantile-normalized state, that the state
vector's joint order is permuted, that images are BGR, or that gripper units are
raw counts instead of radians. Those pass every existing gate and then quietly
destroy training.

Core principle: **three sources of truth must agree — and structural checks
can't see them disagree.** Reconcile them, report discrepancies, and stop before
a full run if a hard mismatch exists.

## When to Use

- Before the first full training run of a model on a given dataset.
- Finetuning from a base checkpoint (its baked-in contract must match your data).
- Policies build, run, and smoke-test fine but results are uniformly bad.
- Switching a working setup to a new dataset, embodiment, or base checkpoint.

Skip only when this exact model+dataset pairing has already been audited and
neither the checkpoint, config, nor dataset has changed.

## Placement

Run this **after** Phase 1 (the Docker image builds and runs) and after or
alongside `hpc-dataset-adaptation`, but **before** committing to a full training
run. Run it **inside the container** so it can load the real checkpoint, the real
processor, and the real dataset with the repo's own libraries.

## The Three Sources

- **(A) Base-checkpoint expectations** — what the pretrained model was built to
  receive: saved processor config, normalization mode, baked-in stats, input
  token order, expected value ranges. This is authoritative and is reloaded
  verbatim at inference. If the base has a `MODEL_PASSPORT.json`, its
  `input_contract` is source (A) directly (see `checkpoint-passport/SKILL.md`).
  If there is no saved contract, infer it from the base's training repo/docs and
  mark those fields as inferred.
- **(B) Training config** — the model/processor config the run will use:
  normalization mapping, image/feature keys, control-mode hints, and any
  overrides.
- **(C) Dataset reality** — the actual per-feature names/order, dtypes, value
  ranges, available stats, units, and metadata.

## Check Categories

1. **Normalization contract** — does the mode + stats the base expects match
   what the config uses *and* what the dataset can support? A convenience
   override (e.g. `MEAN_STD` to skip computing quantiles) can contradict the
   model's core design and saturate or clip its inputs/targets.
2. **Layout & ordering (positional semantics)** — element order of vector
   features (state/action layout) and input/token order (image/camera order). A
   permutation passes every shape check but corrupts the meaning of every token.
3. **Dtype, range & encoding** — post-preprocess numeric range and clamping,
   image scale (0–1 vs 0–255), and channel order (RGB vs BGR).
4. **Units & conventions** — physical units and representation conventions:
   absolute vs delta, coordinate frame, calibration/convention version.
5. **Train↔eval reload consistency** — the *saved* artifact that inference
   reloads must encode the intended stats and masks, not just the in-memory
   training config.

## Agent Algorithm

1. **Locate the three sources.**
   - (A) Base checkpoint: find the saved processor/config artifacts inference
     will reload (or a `MODEL_PASSPORT.json` `input_contract`). If none exist,
     the contract is implicit — infer it from the base's training repo/docs and
     flag those fields as inferred, not verified.
   - (B) Training config: the config + overrides the run will actually use.
   - (C) Dataset: path(s) reachable inside the container.

2. **Extract each contract with native libraries, inside the container.**
   - Do not inspect data on the host. Use the format's own library.
   - Dump, for A / B / C: normalization mode and stat availability, feature
     names and order, dtypes, value ranges, image scale and channel order, units,
     and relevant metadata.

3. **Reconcile A↔B↔C per check category.**
   - For each feature/field, produce a row: expected (A) / configured (B) /
     actual (C) / verdict (`match` / `mismatch` / `unverifiable`).

4. **Cross-check against a known-good reference (optional, best-effort).**
   - If a *different* model trained on the *same* data and hardware works, use it
     to localize:
     - Reference works, audited model fails → bug is specific to the audited
       model's contract (e.g. normalization / tokenization), not the data.
     - Reference also fails → bug is upstream in the shared data/hardware.
   - If no such reference exists, skip — do not fabricate one.

5. **Verify train↔eval reload consistency.**
   - Load the saved processor/passport the eval path will reload and confirm it
     encodes the intended stats, masks, and order — not just what the training
     config says in memory.

6. **Report and gate.**
   - Write the audit record (see below).
   - Any hard mismatch is a **stop gate**: do not start a full training run until
     it is resolved or explicitly waived by the user.

## Audit Record

Write a dated markdown record to `audit_logs/` in the target repo (parallel to
`run_logs/` and `eval_logs/`), with a `timeline.md` index. Do not scaffold
AutoHPC files into the target repo — only the audit record.

Name it `<date>_<model>_<dataset>_audit.md` (date-prefixed for chronological
sort). Maintain `audit_logs/timeline.md` as a chronological index.

```markdown
# <model> on <dataset> — model–dataset audit

## Sources
- base_contract: `<path to saved processor/config or MODEL_PASSPORT.json>` (or `inferred from <ref>`)
- training_config: `<path>`
- dataset: `<path>` (`<name>`)
- known_good_reference: `<model or n/a>`

## Verdict
- overall: `pass` | `mismatch` | `unverifiable`
- gate: `clear to train` | `blocked — <reason>`

## Reconciliation
| feature | category | expected (A) | configured (B) | actual (C) | verdict |
|---|---|---|---|---|---|
| state[0..6] order | layout | ... | ... | ... | match |
| STATE norm | normalization | quantiles | mean_std | mean_std, no q01/q99 | mismatch |
| images | encoding | RGB, 0-1 | RGB | RGB (LeRobot) | match |
| gripper units | units | 0–4.5 rad | raw 0–100 | raw 0–100 | mismatch |

## Evidence
<dumped stats/keys/ranges that back each row>

## Physical checks to confirm with user (agent cannot run)
- <cameras/arms not swapped, record+playback, BC sanity — as applicable>

## Next
- <resolve each mismatch, or waive with justification, before full training>
```

## Robotics / VLA Appendix

Concrete instantiations of each category for robot-policy work. Each maps back to
a general category above.

- **Normalization contract** — the mode the base was pretrained under vs. the
  config's `normalization_mapping`, and whether the dataset actually has the
  stats that mode needs (e.g. `QUANTILES` design vs. `MEAN_STD` override with no
  q01/q99). Trace the downstream consequences: clamping to `[-1, 1]` and
  discretization into bins mean the wrong norm scheme saturates state tokens and
  clips action targets, embodiment-independent.
- **Layout & ordering** — the state-vector joint order (and gripper position in
  it) matches the base's expected order; image/camera key order matches the
  processor's positional image-token order (e.g. `top/wrist/front`); no silently
  permuted features. Dump `meta/info.json` `observation.state.names` and
  `observation.images.*` and eyeball them.
- **Dtype, range & encoding** — images are RGB not BGR (the OpenCV
  `cv2.imread`/capture trap outside the framework's camera path); image scale is
  0–1 vs 0–255 as expected; post-preprocess clamp range is what the base assumes.
- **Units & conventions** — gripper units per hardware (e.g. ARX 0–4.5 rad, i2rt
  1→0); action space is ee-pose vs joint-angle as the base expects; absolute vs
  delta targets; joint-convention/calibration version (e.g. LeRobot v2.1 vs
  v3.0 records) matches between training data and eval hardware.
- **Train↔eval reload** — the saved processor/passport `input_contract` the eval
  path reloads encodes the intended stats and masks (e.g. gripper normalization
  on/off), so inference follows the same contract training baked in.

**Physical checks the agent cannot run — flag them to the user:**

- Cameras not swapped and arms not swapped on the physical rig.
- Record-and-playback on the same hardware replays faithfully (decouples model
  from hardware/calibration).
- A small BC sanity run (e.g. a simple pick task, ~10 demos) trains a sane
  policy — or compare against a working baseline model on the same data instead.

## Quick Reference

| Goal | Approach |
|---|---|
| Read base contract | Saved processor/config, or `MODEL_PASSPORT.json` `input_contract` |
| Read training config | The config + overrides the run will use |
| Inspect dataset semantics | Native library inside the container; dump names/order/dtype/range/stats |
| Reconcile | Per-feature table: expected (A) / configured (B) / actual (C) / verdict |
| Localize a bug | Compare a known-good model on the same data (optional) |
| Confirm inference follows | Load the saved processor the eval path reloads |
| Record | `audit_logs/<date>_<model>_<dataset>_audit.md` + `timeline.md` |

## Stop Gates

- A hard mismatch in any category (normalization, layout/order, encoding, units)
  before a full run.
- Base contract is implicit and cannot be inferred with confidence.
- Dataset stats required by the configured normalization mode do not exist.
- The saved processor the eval path reloads disagrees with training intent.

## Common Mistakes

- Treating a passing smoke test as proof of correctness — it proves the code
  runs, not that the semantics match.
- Overriding the base model's normalization scheme for convenience (e.g. to skip
  computing quantile stats) against its core design.
- Checking shapes/dtypes but not the *order* of features or image tokens.
- Assuming RGB — a custom OpenCV path can silently feed BGR.
- Auditing the in-memory training config but not the saved artifact inference
  reloads.
- Passing raw hardware gripper units where the base expects normalized/physical
  units.
- Mixing joint-convention/calibration versions between training data and eval
  hardware.
- Skipping the known-good cross-check when a working baseline on the same data
  exists — it is the fastest way to localize the bug.
- Not writing an `audit_logs/` record — then "why were all these policies bad?"
  has no answer on disk.
