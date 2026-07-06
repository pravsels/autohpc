# hpc-model-dataset-audit — Design

Date: 2026-07-06

## Motivation

A MolmoAct2 postmortem exposed a class of failures that the current AutoHPC
pipeline does not catch: policies that **build fine, run fine, and pass every
structural check, yet are uniformly bad** because the mismatch is *semantic*,
not structural.

Concrete examples from that postmortem:

- Normalization scheme fights the model's design (`MEAN_STD` override where the
  base was designed for `QUANTILES`; missing q01/q99 stats) — silently saturates
  discretized state tokens and clips action targets.
- Permuted state-vector joint order corrupts every positional state token.
- Camera/image key order swapped relative to the processor's positional tokens.
- Images accidentally BGR (OpenCV path) instead of RGB.
- Wrong gripper units per hardware (ARX 0–4.5 rad, i2rt 1→0).
- Action space mismatch (ee-pose vs joint-angle; absolute vs delta).
- Joint-convention/calibration version mismatch (LeRobot v2.1 vs v3.0).

`hpc-container-promotion` verifies the image *builds and runs*.
`hpc-dataset-adaptation` verifies the loader emits the right *keys/shapes/dtypes*.
Neither reconciles the **base checkpoint's baked-in expectations** against the
**training config** against the **actual dataset semantics**. That gap is what
this skill fills.

## Scope decisions (validated with user)

- **Name:** `hpc-model-dataset-audit` (pipeline `hpc-` prefix, general — not
  finetune-specific, since it applies to any base-model + dataset pairing).
- **Home:** a new standalone skill, distinct from loader adaptation.
- **Placement:** after Phase 1 Docker build (image runs) and after/alongside
  `hpc-dataset-adaptation`, but *before* committing to a full training run. Runs
  inside the container so it can load the real checkpoint + processor + dataset
  with the repo's own libraries.
- **Generality:** a domain-agnostic core algorithm plus a robotics/VLA appendix
  that instantiates each general category with the concrete checklist as worked
  examples.
- **Record:** writes a dated markdown audit to `audit_logs/` in the target repo
  (parallel to `run_logs/` / `eval_logs/`), with a `timeline.md` index.
- **Cross-check:** using a known-good reference model on the same data is an
  optional/best-effort step, not a hard requirement.

## Core mental model — three sources of truth

The audit reconciles three sources that structural checks cannot see disagreeing:

- **(A) Base-checkpoint expectations** — what the pretrained model was built to
  receive: its saved processor config, normalization mode, baked-in stats, input
  token order, value ranges. Authoritative, and (per the eval trace) reloaded
  verbatim at inference. When the base has a `MODEL_PASSPORT.json`, its
  `input_contract` is source (A) directly.
- **(B) Training config** — the model/processor config the run will use
  (`normalization_mapping`, `image_keys`, control-mode hints, overrides).
- **(C) Dataset reality** — actual per-feature layout, dtype, value ranges,
  computed stats, units, and metadata.

## Check taxonomy (general categories)

1. **Normalization contract** — does the mode + stats the base expects match
   what the config uses *and* what the dataset can support?
2. **Layout & ordering (positional semantics)** — element order of vector
   features and input/token order, where a permutation passes shape checks but
   corrupts meaning.
3. **Dtype, range & encoding** — post-preprocess numeric range/clamping, image
   scale (0–1 vs 0–255), channel order (RGB vs BGR).
4. **Units & conventions** — physical units and representation conventions
   (absolute vs delta, coordinate/calibration version).
5. **Train↔eval reload consistency** — verify the *saved* artifact inference
   reloads equals training intent (not just the in-memory training config).

## Agent Algorithm

1. **Locate the three sources** (base checkpoint artifacts, training config,
   dataset paths reachable inside the container).
2. **Extract each contract with native libraries** inside the container (no host
   inspection). Dump expected/configured/actual for every relevant field.
3. **Reconcile A↔B↔C per check category.** Produce a per-feature row:
   expected (A) / configured (B) / actual (C) / verdict.
4. **Cross-check against a known-good reference when one exists** (optional).
   Reference works + audited model fails → bug is model-contract-specific.
   Reference also fails → bug is upstream in shared data/hardware.
5. **Verify train↔eval reload consistency** — the saved processor/passport
   `input_contract` encodes the intended stats + masks.
6. **Report and gate.** Write the `audit_logs/` record; any hard mismatch is a
   stop gate before full training.

## Record artifact

- Location: `audit_logs/` in the target repo, with `timeline.md` index.
- Name: `<date>_<model>_<dataset>_audit.md` (date-prefixed for sort).
- Body: reconciliation table, overall verdict (`pass` / `mismatch` /
  `unverifiable`), dumped evidence (stats/keys/ranges), and `Next`.

## Robotics/VLA appendix

Each general category instantiated with the postmortem checklist:

- Normalization contract → base's pretrain norm mode vs `normalization_mapping`;
  dataset stat availability; downstream clamp/discretization consequences.
- Layout & ordering → state-vector joint order; image/camera key order vs
  processor positional tokens.
- Dtype/range/encoding → RGB not BGR; image scale; clamp range.
- Units & conventions → gripper units per hardware; ee-pose vs joint; absolute
  vs delta; joint-convention/calibration version.
- Train↔eval reload → saved processor/passport `input_contract` encodes intended
  stats + masks.

Plus physical checks the agent cannot run but must flag to the user: cameras/arms
not swapped on the rig, record-and-playback on same hardware, small BC sanity run.

## Integration

- Add to README Phase Router and Skill Map.
- Cross-reference `checkpoint-passport` (`input_contract`) and
  `hpc-dataset-adaptation` (structural adaptation vs semantic audit).
