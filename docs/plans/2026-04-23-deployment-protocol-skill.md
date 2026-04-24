# Deployment Protocol Skill — implementation plan

**Status**: drafting
**Author**: agent (claude-opus-4.7)
**Date**: 2026-04-23
**Target**: `~/Desktop/code/autohpc/deployment-protocol/SKILL.md`
            (+ runnable Python package `deployment_protocol/` in the same folder)

## Goal

Codify, as a sibling skill to `checkpoint-passport/`, the workflow that
takes a signed checkpoint plus a physical robot/inference rig and
**audits, once before each session, that every link in the chain is in
place as expected** — physical sensor stream → preprocessing →
model load → selected layers → normalization → un-normalization →
post-processing → action emission. The audit walks the entire pipeline
under controlled conditions, captures one comprehensive per-stage
record, and gates session start on the result.

The artifact that comes out of the audit is also the documentation:
when something goes wrong later, the engineer looking at it has a
complete, signed record of *exactly* what was bound to what, what
shape/dtype/units flowed at each stage, what intermediate activations
looked like, and how every transformation was named. Auditing for
trust at startup; documentation for debuggability afterwards. Those
are the only two things this skill is responsible for.

`checkpoint-passport` answers *"is the model itself well-formed?"*.
`deployment-protocol` answers *"before this session starts, is every
link in the chain — bindings, devices, preprocessing, model load,
norm, post-processing, emission — present, healthy, and consistent
with the passport?"*. The latter consumes the former — every protocol
check is parameterised on `MODEL_PASSPORT.json`.

What happens during the session — runtime safety, monitoring,
telemetry, action verification, e-stop wiring — is **explicitly out
of scope** for this skill. Those are the robot driver's, the safety
layer's, and the monitoring system's concerns. Our contract ends when
preflight returns success and writes the audit artifact.

The skill is shaped as **a one-shot preflight audit** (a fixed set of
checks plus one full controlled forward pass with per-stage capture)
plus **deployment-local config** (which physical device maps to which
logical input key, and which model modules to inspect during the
audit's forward pass). It is **not** a third signed artifact mirroring
the model passport — a signed artifact describing a rig that changes
every reboot would either lie or rot. The protocol is the contract;
bindings are config; the preflight audit artifact is the
session-grade record.

Re-audit triggers (all explicit, all human-initiated): rig hardware
changes, bindings file edits, model passport changes, on operator
request. Whether to re-audit based on what happens during a session is
a decision for the runtime/monitoring system, not this skill.

## Inputs (what the skill consumes)

- A signed checkpoint (`MODEL_PASSPORT.json` + passing `SIGNOFF.json`)
  in the deployment environment.
- A live inference rig — robot + cameras + state sensors + compute,
  reachable from the inference process.
- A deployment-local config file (`deployment_bindings.yaml`, see
  schema sketch below) authored once per rig and updated whenever
  hardware changes.

## Outputs (what the skill produces)

- `deployment_bindings.yaml` at the deployment root (deployment-local
  config; not signed).
- A `preflight-check` CLI invocation at the start of every inference
  session, returning non-zero on any protocol violation.
- A `PREFLIGHT_AUDIT.json` per session, written under
  `preflight_audits/<audit_id>/`. Records every kernel check's
  verdict plus one comprehensive per-stage walk of the full pipeline
  (sensor read → preprocessing → batch assembly → model forward with
  optional named-module activation taps → output un-normalization →
  action post-processing → action emission) produced by the audit's
  controlled forward pass, plus operator-supplied session metadata
  (operator id, task description, scene notes). This is *the*
  artifact — both the gate and the record.

## Architectural decisions

1. **Protocol, not passport.** The mandatory checks live in code (the
   `deployment_protocol` package), versioned with the package. There is
   no signed JSON describing the deployment. Code is the right place
   for "must run these checks every time" because it can be diffed,
   reviewed, and audited as a unit.
2. **Bindings file is plain config, not signed.** Hardware identity
   (camera serial → logical key) is brittle (cables get re-plugged,
   USB ports re-enumerate), and re-signing on every change wouldn't
   happen in practice. The protocol re-verifies bindings on every
   preflight, so the bindings file doesn't need a tamper seal — if
   it's wrong, preflight fails.
3. **One artifact per session: `PREFLIGHT_AUDIT.json`.** Same kind
   of role as `run_logs/` and `eval_logs/` elsewhere in autohpc —
   the deployment-side per-session record — but written once at
   session start, finalised before the session begins, and never
   touched again. No run log, no telemetry, no session-end summary.
4. **Some additions land in the model passport schema, not here.**
   Specifically: per-sub-key `units` and `coord_frame` on state and
   actions, plus a `norm_round_trip` smoke bucket. These describe
   what the model was *trained* on — they belong in the model
   passport, and the deployment protocol cross-checks them against
   bindings-declared units / live samples. See "Model passport
   additions" section below.
5. **Audit is one-shot at preflight. Runtime is out of scope.** The
   full per-stage walk happens once, under controlled conditions,
   before the session starts. After preflight passes and the audit
   artifact is written, this skill is done. Whether the runner
   monitors actions for NaN, clips against a safety envelope,
   responds to e-stop, or restarts on faults is the runner / driver /
   safety layer's responsibility — not codified here.
6. **The preflight audit covers every stage; bindings declare which
   model internals get inspected during the audit's forward pass.**
   The 10-stage audit pipeline (camera.raw → camera.preprocessed →
   state.raw → state.transformed → batch_assembled →
   model.intermediate.<tap> → model.output_normalized →
   action.unnormalized → action.post_processed → action.emitted) is
   fixed by the protocol. Which named modules to hook for the
   `model.intermediate` stage is per-deployment and per-debug-need —
   bindings declare a list of module names from the passport's
   `module_hierarchy`; the audit installs forward hooks on them for
   the single preflight forward pass and removes them before the
   audit completes.
7. **The audit artifact is the documentation.** No separate run log,
   session log, or telemetry stream. When something needs debugging
   later, the engineer pulls the relevant `PREFLIGHT_AUDIT.json` and
   has a complete record of what was bound, what flowed at each
   stage, what the activations and norms looked like — the inputs
   to a "what changed since the last clean audit?" diff. Deeper
   runtime forensics (action streams, per-call latency, abort
   causes) come from whatever runtime/monitoring system is layered on
   top — out of scope here.

## Bindings schema sketch (deployment-local config, not signed)

YAML for human authoring; the package parses it into typed dataclasses
mirroring the YAML keys. This is a sketch — the real schema lives in
`deployment_protocol/schema.py` once we implement.

Bindings carry only what the audit needs to verify, plus a little
runner-facing documentation (e.g. how the runner uses the action
horizon — required so the audit's forward pass can be representative,
but not enforced at runtime by us).

```yaml
schema_version: "0.1"               # bindings file format version
deployment_id: "lab-rig-3"          # free-text identifier for the rig
authored_at: "2026-04-23T10:00:00Z"
authored_by: "username"
target_passport: "/path/to/MODEL_PASSPORT.json"  # which model this is bound to
target_passport_sha256: "abcd..."   # sha of the passport at binding time

images:
  - logical_key: "observation.images.front"
    device:                          # how to find the physical source
      kind: "v4l2"                   # v4l2 | ros_topic | rtsp | gstreamer | ...
      address: "/dev/video0"
      identifier: "Logitech_C920_serial_ABC123"  # what the OS reports
    units: "uint8_rgb_0_255"         # cross-checked against passport image spec
    color_order: "RGB"               # cross-checked against passport
    channel_layout: "HWC"            # cross-checked
    calibration_reference:           # rebuilt by `bind-deployment`
      captured_at: "2026-04-23T10:00:00Z"
      perceptual_hash: "phash:abc123..."   # a-hash / d-hash / phash
      mean_luminance: 0.42
      capture_resolution: [480, 640]

state:
  - logical_sub_key: "observation.state"   # dim 0..6 in the passport
    source:
      kind: "ros_topic"
      topic: "/joint_states"
      message_field: "position"
    units: "rad"                     # cross-checked against passport sub_keys[].units
    coord_frame: "joint_local"
  - logical_sub_key: "observation.eef_6d_pose"  # dim 7..12
    source:
      kind: "ros_topic"
      topic: "/tf"
      message_field: "transforms[0]"
    units: "m_xyz_axisangle_rad"
    coord_frame: "robot_base"

control:
  rate_hz: 30                        # cross-checked against passport temporal.control_rate_hz
  action_topic: "/policy/action"     # documentation of where actions are published; not gated
  action_consumption: "first_of_chunk"  # first_of_chunk | replan_each_step | overlap_blend
                                        # used by the audit forward pass to be representative
  warmup_history: "repeat_first"        # repeat_first | zero_pad | wait_until_full
                                        # used by the audit forward pass to be representative

temporal:
  max_inter_stream_skew_ms: 20       # max allowed clock skew between sources at preflight

audit:
  module_taps:                       # which named_modules to forward-hook during
    - "policy.denoiser.blocks.0"     # the preflight audit's controlled forward pass
    - "policy.denoiser.blocks.11"
    - "policy.denoiser.final_norm"
  capture_tensor_dumps: true         # write .npy sidecars for each stage of the
                                     # preflight forward pass; small, one-shot
  tensor_dump_dir: "preflight_tensors"
```

`bind-deployment` (a CLI in the package) is the recommended way to
author/update this file — it walks the model passport, prompts for
device addresses, captures the calibration reference frames, suggests
sensible default `module_taps` (typically: first block input, last
block output, final norm), and writes a fresh
`deployment_bindings.yaml`. Manual editing is fine too.

## Preflight audit data model (every link in the chain)

The audit fixes a 10-stage pipeline. At session start, the audit runs
ONE controlled forward pass — fetching one frame from each camera, one
state sample from each source, and pushing them through the runner's
real preprocessing → real model → real post-processing → emission
(emission is held back from the robot in audit mode; see "Preflight
CLI contract"). It writes ONE `PREFLIGHT_AUDIT.json` carrying one
record per stage, plus the kernel-check verdicts and full sidecar
tensor dumps when `capture_tensor_dumps: true`. This is a once-per-
session artifact, not a per-call stream.

```json
{
  "audit_id": "2026-04-23T10:00:01.000_lab-rig-3",
  "ran_at": "2026-04-23T10:00:01.000Z",
  "verdict": "ship_it",                  // ship_it | human_look_here | dont_ship
  "checkpoint_passport_sha256": "abcd...",
  "bindings_sha256": "ef12...",
  "session_intent": {                    // captured at preflight-check time, free-text
    "operator_id": "username",
    "task_description": "stack red block on blue block",
    "scene_notes": "lab lighting, default object set"
  },
  "checks": [ ... Observation list, identical shape to validate-checkpoint ... ],
  "controlled_forward_pass": {
    "rng_seed": 0,
    "library_versions_observed": {"torch": "...", "cuda": "...", "<model_pkg>": "..."},
    "loaded_dtype": "float32",           // cross-checked against passport-declared dtype
    "stages": [
      {
        "stage": "camera.raw",
        "logical_key": "observation.images.front",
        "device_identifier": "Logitech_C920_serial_ABC123",
        "captured_at": "2026-04-23T10:00:01.012Z",
        "shape": [480, 640, 3], "dtype": "uint8", "color_order": "RGB",
        "value_min": 12, "value_max": 248, "value_mean": 117.4,
        "sha256": "abc123...",
        "tensor_dump": "preflight_tensors/camera.raw__observation.images.front.npy"
      },
      {
        "stage": "camera.preprocessed",
        "logical_key": "observation.images.front",
        "transformations": [
          {"op": "resize", "to": [224, 224]},
          {"op": "to_float", "scale": "1/255"},
          {"op": "normalize", "mean": [...], "std": [...]}
        ],
        "shape": [3, 224, 224], "dtype": "float32",
        "value_min": -2.11, "value_max": 2.64, "value_mean": 0.012,
        "sha256": "def456..."
      },
      {"stage": "state.raw", "logical_sub_key": "observation.state",
       "shape": [7], "dtype": "float64", "values": [...], "units": "rad",
       "source_message_timestamp": "..."},
      {"stage": "state.transformed", "logical_sub_key": "observation.eef_6d_pose",
       "transformations": [{"op": "axisangle_to_rot6d"}],
       "shape": [9], "dtype": "float32", "values": [...]},
      {"stage": "batch_assembled",
       "keys_and_shapes": {"observation.images": [2, 3, 224, 224],
                           "observation.state": [16],
                           "language": "..."}},
      {"stage": "model.intermediate", "tap": "policy.denoiser.blocks.0",
       "shape": [..., ...], "dtype": "float32",
       "value_min": ..., "value_max": ..., "norm": ...,
       "tensor_dump": "..."},
      {"stage": "model.output_normalized",
       "shape": [32, 17], "dtype": "float32",
       "value_min": -1.02, "value_max": 0.97, "value_mean": ...,
       "sha256": "..."},
      {"stage": "action.unnormalized",
       "applied_norm_inverse": {"type": "ramen", "stats_sha256": "..."},
       "shape": [32, 17], "value_min": ..., "value_max": ...},
      {"stage": "action.post_processed",
       "selected_step": 0, "clip_applied": true,
       "shape": [17], "values": [...]},
      {"stage": "action.emitted",
       "topic": "/policy/action", "would_have_emitted_at": "...",
       "values": [...], "actually_sent": false}
    ]
  },
  "inter_stream_sync": {                  // populated by `time_sync_within_window` check
    "max_inter_stream_skew_ms": 8.4,
    "tolerance_ms": 20
  },
  "norm_round_trip": {                    // populated by the round-trip checks
    "state_max_abs_diff": 1.2e-7,
    "action_max_abs_diff": 8.1e-8
  }
}
```

Invariants the audit enforces on this trace:
- Every stage record carries `shape`, `dtype`, and a value-range
  summary (`min`, `max`, `mean`). Sensor stages additionally carry
  the source identifier and a capture timestamp.
- Each transformation step is named, with parameters captured (no
  "magic preprocessing"). If the runner can't name a transformation,
  it's a runner bug — the audit trace must explain everything that
  changed between adjacent stages.
- `model.intermediate` records may appear 0..N times depending on how
  many `module_taps` are configured.
- `action.emitted.actually_sent: false` for the audit forward pass
  — the audit must not move the robot. Bindings can declare a
  `dry_run: true` mode for the same reason during regular sessions.

The runner integrates by calling a small `AuditRecorder` helper from
the package — `recorder.stage("camera.raw", ...)` calls feed records
into the audit. The recorder handles sha256-ing, sidecar writing, and
audit-file flushing. Runners that already have their own
instrumentation can implement the recorder interface themselves; the
schema is the contract, not the helper. The recorder is **only active
during the preflight forward pass**. After preflight returns and the
audit artifact is written, this skill's runtime footprint is zero —
the runner removes any installed hooks and proceeds with whatever
inference loop it would normally run.

## Mandatory protocol checks (the kernel)

Same shape as `checkpoint_passport/kernel/*.py` — one file per check
section, each returning a list of `Observation(check, status, message,
details)` records. Status vocabulary identical to the checkpoint
passport: `PASS / SOFT_SIGNAL / FAIL / NOT_CHECKED`.

**Bindings vs passport (static, runs without the live rig):**
- `bindings_schema_version_supported` — bindings' `schema_version`
  is one this version of the package understands. Hard fail on
  major-version mismatch, soft signal on minor.
- `bindings_target_passport_match` — bindings' `target_passport_sha256`
  matches the on-disk passport sha. Hard fail if drift.
- `bindings_image_keys_complete` — every `input_contract.images[].key`
  in the passport has exactly one binding.
- `bindings_state_subkeys_complete` — every `state.sub_keys[].name`
  has exactly one binding.
- `bindings_units_match_passport` — bindings' declared units match the
  passport's per-sub-key `units` (requires the model passport schema
  addition below).
- `bindings_coord_frame_match_passport` — same for `coord_frame`.
- `bindings_module_taps_resolvable` — every name in
  `audit.module_taps` is present in the passport's
  `model_internals.module_hierarchy`. Hard fail (typos here mean
  silent loss of stage capture during the audit's forward pass).

**Live device probes (require the rig to be reachable):**
- `image_device_reachable` — each binding's image device returns a
  frame within a small timeout. Hard fail otherwise.
- `image_frame_shape_dtype_match` — live frame shape, dtype, channel
  layout, color order match `input_contract.images[].raw_shape`. Hard
  fail.
- `image_frame_value_range_match` — live frame's min/max fall inside
  passport `value_range`. Soft signal on margin violation, hard fail
  if completely outside.
- `image_calibration_reference_match` — perceptual hash + mean
  luminance of the current frame are within tolerance of the binding's
  recorded `calibration_reference`. Soft signal — scenes legitimately
  change, but a complete mismatch likely means the camera was swapped.
- `state_source_reachable` — each binding's state source publishes a
  message within timeout. Hard fail.
- `state_dim_match` — published state dim matches passport
  `state.total_dim` (per-sub-key, then aggregated). Hard fail.
- `state_value_range_match` — sampled state values fall inside
  passport `state.normalization.per_dim_q02 / q98` (the wide envelope,
  not strict q01/q99). Soft signal.
- `control_rate_match` — measured publish rate of the binding's
  control loop input matches passport `temporal.control_rate_hz`
  within ±10%. Soft signal at ±10–20%, hard fail beyond.
- `time_sync_within_window` — sample timestamps from each binding's
  source(s) over a short window and confirm inter-stream skew (latest
  vs oldest source timestamp at the moment of policy step) is below
  bindings' `temporal.max_inter_stream_skew_ms`. Hard fail if skew
  exceeds tolerance — feeding a model state from time T with images
  from time T-200ms is a silent feeding bug that no other check
  catches.
- `gpu_memory_headroom` — measured free GPU memory ≥ a small
  multiplier × the model's measured peak resident memory (capture
  during the controlled forward pass). Soft signal if tight; hard
  fail if predicted peak exceeds free.
- `loaded_dtype_matches_passport` — the dtype the runner actually
  loaded the model in matches the passport's recorded weight dtype.
  Hard fail. Catches silent fp16/int8 conversions that a
  well-meaning ops engineer might add to save memory; if the
  conversion is intentional, it requires a re-passport.

**Audit forward pass (the controlled run; the heart of the audit):**
- `model_loads_against_passport` — model class resolves, weights load
  with no missing/unexpected keys, passport's `weight_integrity`
  matches on-disk hashes. Same code path as
  `validate-checkpoint --require-signoff`. Hard fail.
- `state_norm_round_trip_live` — take a real published state sample,
  apply `normalize` → `unnormalize` using the passport's recorded
  norm stats, confirm `max_abs_diff < eps`. Hard fail. Catches: stats
  with zero stddev, off-by-one dim layouts, sign-flipped quantile
  bounds — all of which silently corrupt actions otherwise.
- `action_norm_round_trip_synthetic` — same on a synthetic action
  drawn from `[-1, 1]^action_dim`, using the action-side norm stats.
  Hard fail. Decouples norm-correctness from policy-output range.
- `audit_forward_pass_completes` — run the public inference method
  once with the recorder active, with emission held back from the
  robot. Hard fail on NaN/Inf or any of the 10 stages missing.
- `audit_forward_pass_within_smoke_envelope` — output falls inside
  the passport's smoke `range_check` bounds. Soft signal if outside
  (smoke was synthetic; real may land slightly off).
- `audit_pipeline_transformations_named` — every adjacent-stage
  transition in the recorded audit trace has a named transformation
  block — no unexplained shape/dtype/value changes. Hard fail.
  Closes the "magic preprocessing in the runner" loophole.

The full check list lives in `deployment_protocol/kernel/`. Adding a
check means adding a function in a kernel module — same workflow as
the checkpoint passport validator. Every check in the kernel runs at
preflight; nothing in the kernel runs during the inference session.

## Preflight CLI contract

```
preflight-check <bindings.yaml>
    [--passport <path>]
    [--skip-live]
    [--out <audit_dir>]
    [--operator <id>] [--task <description>] [--scene-notes <text>]
    [--json]
```

- Reads bindings, resolves the passport (either explicit `--passport`
  or `bindings.target_passport`).
- Runs every kernel check, including the audit forward pass (with
  emission held back from the robot driver — the audit never moves
  the rig). `--skip-live` runs only the static bindings-vs-passport
  checks (useful in CI for the bindings file itself; not a substitute
  for a real preflight on the rig).
- Captures session intent (`--operator`, `--task`, `--scene-notes`)
  into the audit artifact. These are free-text and optional, but
  recommended — they make the audit useful as a debugging document
  later.
- Writes `<audit_dir>/PREFLIGHT_AUDIT.json` (the durable artifact —
  observations + controlled forward pass's per-stage record +
  sidecar tensor dumps if `audit.capture_tensor_dumps`).
- Prints a report identical in shape to `validate-checkpoint`'s.
- Exits non-zero on any hard failure.
- With `--json`, emits the same structured report to stdout for
  embedding in tooling.

The inference runner's startup script is expected to call
`preflight-check` and abort if it returns non-zero. On success, the
runner proceeds with whatever runtime loop / safety stack / monitoring
it normally uses; this skill has nothing more to say at that point.

## What we don't produce

There is no `RUN_LOG.json`, no per-call telemetry, no session-end
verdict, no stream of inference traces. The `PREFLIGHT_AUDIT.json` is
the only artifact this skill is responsible for. If the runtime /
monitoring system layered on top wants to record what happened during
the session, it does so on its own terms; it can reference the audit
artifact's path for context, but it doesn't need anything from us
beyond a successful exit code from `preflight-check`.

## Model passport additions

Four small, additive changes to `checkpoint_passport/schema.py`.
These are the only edits to the existing checkpoint-passport package
needed to support deployment protocol.

**`StateSubKey`** (the dict entries inside `StateSpec.sub_keys`) gains
two optional fields:
- `units: Optional[str]` — free-text but conventionally one of a small
  vocab (`"rad"`, `"deg"`, `"m"`, `"mm"`, `"normalized_-1_1"`,
  `"unitless"`, etc.). The vocab grows as needed.
- `coord_frame: Optional[str]` — same shape (`"joint_local"`,
  `"robot_base"`, `"world"`, `"camera"`, etc.).

**`ActionSpec.sub_keys`** gains the same two fields. (`ActionSpec`
already has `delta_dims` to mark which action channels are deltas, but
units of the delta — e.g. m vs mm — were never captured.)

**`SmokeResults`** gains a new typed sub-bucket
`norm_round_trip: Optional[NormRoundTripResult]`, mirroring the
existing typed buckets (`DeterminismResult`, etc.). It records a
norm → unnorm round-trip on synthetic state and action samples drawn
from the per-dim quantile envelope, with `max_abs_diff` per stream
and `status: pass | fail`. Generated at passport time; consumed by
the deployment protocol's `state_norm_round_trip_live` /
`action_norm_round_trip_synthetic` checks (which compare against the
threshold the passport already validated against — keeps both sides
honest about what counts as "round-trippable"). A new
`norm_round_trip` check goes into the checkpoint passport's
`output_spec` kernel module to gate the bucket.

**`ModelInternals.parameters`** gains an aggregated `weight_dtype:
Optional[str]` field — the dtype the safetensors were saved in
(`"float32"` etc.). Cross-checked at deployment time by
`loaded_dtype_matches_passport`. Without this, a runner silently
loading the model in fp16 to save memory passes every existing check
while producing measurably different actions; with it, downcasting
forces a re-passport (or an explicit
`output_spec.deployment_dtype_overrides` field, which we can add
later if downcasting becomes a common practice).

Existing checkpoints stay valid (all fields are `Optional` and default
to `None`); they simply produce `NOT_CHECKED` on the units /
coord-frame / norm-round-trip / dtype protocol checks until someone
backfills.

## Skill structure (sections)

```
deployment-protocol/SKILL.md
  YAML frontmatter (name + "Use when..." description)
  # Deployment Protocol
  ## Overview
  ## When to Use
  ## Authoritative References  (link out to schema.py + kernel + passport schema)
  ## Phase 1 — Author Bindings
    - what `bind-deployment` does
    - the calibration reference capture
    - manual editing notes
    - common bindings mistakes
  ## Phase 2 — Preflight Audit (every session)
    - preflight-check CLI + session-intent flags
    - the static checks (bindings vs passport)
    - the live device probes (incl. inter-stream sync, gpu memory,
      loaded dtype)
    - the controlled forward pass + per-stage capture
    - the norm-round-trip check (and why a round-trip failure means
      the action stream would be silently corrupted)
    - the AuditRecorder (only active during preflight, disabled after)
    - module-tap selection (which named_modules are worth hooking)
    - PREFLIGHT_AUDIT.json — the durable record of "every link was
      in place when we said go"
    - typical soft signals + how to triage
  ## Phase 3 — Re-binding / re-auditing
    - what triggers re-binding (hardware change, bindings edit)
    - what triggers re-auditing without re-binding (passport change,
      operator request, anything in the runtime / monitoring system
      that suggests something has drifted)
    - what triggers re-passporting (model code or weight changes,
      not rig changes — out of this skill, see checkpoint-passport)
  ## Phase 4 — Reading an audit artifact for debugging
    - the artifact as a documentation source ("here is exactly what
      was bound, what flowed at each stage, what activations looked
      like the last time things were known-good")
    - diffing two PREFLIGHT_AUDIT.json's across runs (the canonical
      "what changed?" workflow)
    - common signals when something downstream of the audit goes
      wrong: dtype mismatch, norm-round-trip drift, inter-stream
      skew, frame-shape mismatch caught late at preflight
  ## Quick Reference  (commands table)
  ## Common Mistakes
```

## Non-goals

- The skill does NOT do anything during the inference session.
  Once `preflight-check` returns success and writes the audit
  artifact, the skill is done. No runtime hooks, no telemetry, no
  in-loop checks, no session log, no abort-on-NaN, no safety
  envelope clipping. Those are runtime / driver / monitoring
  concerns and live elsewhere.
- The skill does NOT script the inference loop itself. It assumes
  the user has a runner; the runner calls `preflight-check` at
  startup, optionally drives an `AuditRecorder` during the preflight
  forward pass (or implements the recorder interface), and otherwise
  proceeds however it normally would.
- The skill does NOT prescribe a specific transport for cameras
  (v4l2 vs ROS vs gstreamer vs RTSP) or state (ROS topic vs zeromq vs
  shared memory). Bindings carry a `kind` + `address` and the protocol
  treats them as opaque — the package provides a small set of `kind`
  handlers and an extension point for custom ones.
- The skill does NOT replicate the model passport's contents in the
  audit artifact — the artifact carries the passport sha and points
  at the passport file.

## Success criteria

1. A second engineer (or fresh agent session) can take a signed
   checkpoint, follow the skill, author bindings, and run preflight
   — without re-asking what each check means or how to express their
   hardware in the bindings file.
2. Swapping a camera (different physical device on the same logical
   key, e.g. plugging the wrist cam into the front cam's port)
   produces a hard failure on `image_calibration_reference_match`
   and `image_device_reachable` at preflight, before the audit
   forward pass even runs.
3. Loading a checkpoint whose passport's `state.sub_keys[].units` say
   `rad` against a bindings file declaring `deg` produces a hard
   failure on `bindings_units_match_passport`.
4. A passport whose recorded norm stats are not actually invertible
   (zero stddev on a dim, mis-shaped quantile arrays) hard-fails
   `state_norm_round_trip_live` at preflight.
5. Loading the model in fp16 against a passport recording fp32
   weights hard-fails `loaded_dtype_matches_passport` at preflight —
   a silent ops-side optimisation can't bypass the audit.
6. The `PREFLIGHT_AUDIT.json` is sufficient to reconstruct, for the
   audit's controlled forward pass, *every transformation* between
   the camera/sensor read and the would-have-been-emitted action —
   including which named module activations were captured, what the
   norm-inverse did, and whether any stage triggered a soft signal.
   No "magic preprocessing" is allowed inside the audited region
   without a stage record.
7. Two `PREFLIGHT_AUDIT.json` artifacts from different runs of the
   same rig+checkpoint can be diff'd and the diff is meaningful:
   stage-by-stage values, dtypes, activations, captured-frame
   fingerprints. This is the "what changed since last good run?"
   workflow and it's the main payoff of insisting every transformation
   is captured.
8. Swapping the model code without re-passporting the checkpoint is
   detected at preflight (passport sha drift) **and** the audit's
   controlled forward pass is sufficient to localize which stage
   diverged from what the passport described.
9. Skill scope stays clean: nothing in the SKILL or the package's
   public API does anything during the inference session. Reviewing
   the package, you should be able to confidently say "everything in
   here runs at or before `preflight-check` exit; nothing runs
   inside the inference loop."

## Validation plan

After writing the skill + package:

1. Take `dit_block_tower_norm_fix` (already passport-signed) and
   author a `deployment_bindings.yaml` against it for a synthetic /
   recorded "rig" (replay a saved episode as the live source). This
   tests the static + audit checks without needing a real robot.
2. Mutate the bindings to introduce known errors (wrong units, wrong
   image shape, swapped cameras, typo'd `module_taps`, wrong
   `schema_version`) and confirm the right preflight checks fire.
3. Add the `units` / `coord_frame` / `weight_dtype` fields to the
   existing `dit_block_tower_norm_fix` passport, populate the new
   `norm_round_trip` smoke bucket, re-sign, and confirm the new
   protocol checks resolve cleanly when bindings agree.
4. Run preflight against the replayed episode with
   `audit.capture_tensor_dumps: true`, then verify the
   `PREFLIGHT_AUDIT.json` carries a complete 10-stage record and
   that each adjacent-stage transition is named (no unexplained
   shape/dtype/value changes).
5. Deliberately corrupt the inference runner's preprocessing (skip
   the resize, swap RGB→BGR, drop a normalization step, downcast to
   fp16) and confirm preflight catches each one —
   `audit_pipeline_transformations_named` for unnamed mutations,
   `loaded_dtype_matches_passport` for the downcast, the norm-round-
   trip checks for the dropped normalization.
6. Run preflight twice in a row against an unchanged rig+checkpoint
   and confirm the two `PREFLIGHT_AUDIT.json` artifacts diff cleanly
   (only timestamps + a few sensor-noise floats differ; structure,
   transformations, activations are stable). This validates the
   "diff two audits to find drift" workflow.
7. Once a real robot is available, run `bind-deployment` against it
   and validate the live-device probes from a clean session.

## Framework-agnosticism

Same principle as `checkpoint-passport/SKILL.md`: the skill describes
the *protocol* and *what each check guarantees*, not how a specific
robot stack publishes joint state or exposes a camera. The bindings
schema's `kind` + `address` fields are the abstraction line — adding
support for a new transport means writing a small handler in
`deployment_protocol/transports/<kind>.py`, not branching the skill.

## Out of scope — owned by other systems

Anything that happens after `preflight-check` returns success is not
this skill's concern. Listed here so future readers don't ask "should
this go in deployment-protocol?":

- **Runtime safety hooks** (NaN/Inf detect on emitted actions,
  joint-limit clipping, e-stop wiring, action-rate watchdogs).
  Belongs in the runner / robot driver / safety layer.
- **Runtime telemetry / monitoring** (per-call latency, frame-drop
  counters, action-norm trends, stuck-action detection,
  inter-stream skew during the session, GPU utilisation).
  Belongs in the runtime / monitoring system.
- **Session logs** (run id, start/end time, abort cause, episode
  markers, runtime stats, success/failure annotations). Belongs in
  whatever per-session log the runtime / experiment-tracking system
  already produces.
- **Closed-loop action verification** (did the robot actually
  execute the command we sent?). Belongs in the driver's feedback
  loop.
- **In-loop re-validation against the passport.** Explicitly
  rejected — the audit happens once, before the run.

## Out of scope — for this iteration

- **A `DEPLOYMENT_SIGNOFF.json`-style signed artifact.** Explicitly
  rejected by the architectural decision above. If we ever need a
  tamper seal on the bindings file (e.g. compliance), it can be added
  as a thin wrapper around the existing `sign-checkpoint` machinery —
  but absent a concrete need, the protocol-as-code shape stays.
- **Audit replay tooling.** Phase 4 of the SKILL describes reading
  `PREFLIGHT_AUDIT.json` by hand. A dedicated `replay-audit` CLI
  that re-runs the captured forward pass offline against a different
  runner build (for divergence analysis between trainer and
  inference code) is a clear next deliverable but out of this
  iteration's scope.
- **Multi-policy deployments.** The bindings schema assumes one
  policy per rig. Multi-policy / policy-switching deployments need a
  richer schema; deferred until we have a real use case.
- **Camera intrinsics / extrinsics in bindings.** The current
  `calibration_reference` is a perceptual-hash-and-luminance
  fingerprint — sufficient to detect "wrong camera" but not "camera
  moved 5 cm". Full intrinsics/extrinsics handling deserves its own
  iteration once we have a concrete model that depends on them.
- **Privacy / image redaction in audit captures.** The audit can
  capture full camera frames; if those frames contain people, some
  deployments will need face-blur or full redaction before
  persistence. Out of scope until a deployment requires it.
- **Eval-as-deployment (dataset-replay binding kind).** The bindings
  schema's transport abstraction (`device.kind`) makes a future
  `kind: dataset_replay` plausible — feeding the model from a saved
  episode for offline eval, reusing the same protocol. Worth doing
  but deferred.
