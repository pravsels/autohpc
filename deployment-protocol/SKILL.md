---
name: deployment-protocol
description: Use when deploying a signed checkpoint onto a robot or inference rig and the agent needs to perform first-run or fresh-run preflight checks by reading MODEL_PASSPORT.json, the target repo's runner code, and live device samples.
---

# Deployment Protocol

## Overview
Use this after a checkpoint has already been passported and signed. The
goal is to verify, before the first run on a rig or before a fresh run
after something changed, that the live rig
still matches the signed checkpoint and that the target repo's real
deployment path still feeds the model the way the passport says it should.

Core principle: the passport is the model-side contract; the preflight is
an agent-run procedure that compares that contract against the actual rig,
the actual runner, and one controlled dry-run forward pass.

The point is whole-chain confidence. By the end of preflight, the agent
should know what raw device data is arriving, how it is transformed, what
the model actually receives, what code/class path loads the checkpoint,
what comes out of the model, how outputs are unnormalized or post-processed,
and what would be emitted if the real inference script were started.

The key phrase is **full chain audit**. That means starting at the top of
the chain and walking downward stage by stage, recording what was actually
observed, what the passport says should happen, and whether those two agree.
It is not enough to say "the checkpoint validates" or "the script ran."

## When to Use

Use when:

- a checkpoint already has `MODEL_PASSPORT.json` and `SIGNOFF.json`
- the next step is robot or rig deployment, not training or eval
- you need to confirm the live sensors, preprocessing, model load,
  normalization, post-processing, and emission path still match the
  passport
- you need a repeatable preflight record before the first run, or before a
  fresh run after hardware, code, bindings, or checkpoint changes

## Core Pattern

Treat preflight as a comparison across three sources of truth:

1. `MODEL_PASSPORT.json`: what the model expects
2. target repo runner/container: what deployment code actually does
3. live rig samples: what the hardware is currently producing

The agent's job is to line these up and prove that they agree.

## Preflight Rubric

Use this rubric explicitly.

### Gold standard

**Live-rig preflight** is the gold standard.

That means:

- real live device streams
- the target repo's real inference path
- one safe dry-run through the full chain
- a durable audit record

### Acceptable for now

**Replay preflight** is acceptable for now when a live rig is unavailable.

That means:

- the audit note says why live-rig preflight could not be run now
- the replay source still starts high enough in the chain to expose raw
  observations, bindings, transformations, model input, model output,
  post-processing, and would-be emission
- the replay source is fed through the target repo's real inference path
- the audit note clearly says this was replay mode and that live-rig
  preflight still remains to be done later

Examples that usually qualify:

- recorded camera frames plus recorded state from the deployment stack
- eval samples that still expose the raw observation keys consumed by the
  runner
- logged observations before preprocessing and batch assembly

Examples that do not qualify:

- already-normalized tensors
- already-assembled model batches
- checkpoint-only validation
- a model-load smoke test with no source-side evidence

### Not preflight

Running only checkpoint validation, passport validation, or a model-load
smoke test is not deployment preflight. Those checks are useful inputs, but
they do not count unless the full inference flow is exercised.

### Definition of a full chain audit

A full chain audit means the agent checks and records these ten required
elements in this exact order:

1. source identity
2. raw source sample
3. source-to-passport bindings
4. checkpoint identity and integrity
5. runtime model load path and model internals
6. preprocessing and transformation steps
7. final model input contract
8. model output shape and value sanity
9. output unnormalization and post-processing
10. would-be emission behavior

Those ten elements are the chain. The audit artifact should name them
explicitly and should not silently collapse them into fewer stages.

If the run fails at any stage, the audit must say exactly where it failed
and explicitly mark all later stages as unreached.

### Valid outcomes

- `PASS (live)` — all ten chain elements were checked on the real rig
- `PASS (replay, provisional)` — all ten chain elements were checked through
  replay data; a live-rig preflight is still required later
- `FAIL` — one or more executed chain elements did not match; later elements
  may be unreached
- `NOT RUN` — the preflight never reached a decisive chain check, so the
  audit could not actually begin

## Step-by-Step Procedure

### Step 1: Choose the mode up front

Decide and state which mode you are running:

- `live-rig`
- `replay`

If no live rig is available, state why, switch to replay mode, and choose a
replay source that still exposes the top of the chain. Use representative
raw observations from eval data, recorded rollouts, or logs, but still drive
them through the target repo's real inference path. If the available replay
source starts after preprocessing or after model input assembly, that is not
a full chain audit.

### Step 2: Read the passport first

Extract at least:

- image keys, raw shape, dtype, color order, channel layout, value range
- state sub-keys, dims, units, coord frames
- action dims, output shape, and any normalization/post-processing
  assumptions
- temporal assumptions such as control rate or history shape
- model dtype, class/module identity, and any notable model internals that
  matter for debugging
- any norm stats or smoke evidence that describe how data should round-trip
  through normalize/unnormalize paths

### Step 3: Inspect the target repo's real deployment path

Find the code that:

- reads cameras and state
- preprocesses observations
- assembles the model batch
- loads the checkpoint
- runs inference
- unnormalizes/post-processes actions
- emits or publishes actions

The target repo is the source of truth for how deployment actually works.
The agent should identify the exact runtime path end to end rather than
inferring it from config names alone. Record the concrete entrypoint,
container/image, relevant config, and code paths that implement each chain
element.

### Step 4: Build a simple deployment bindings record

This can be a YAML file, markdown table, or audit note in the target repo.
It only needs to record the deployment-local facts the agent is checking,
for example:

- which physical camera maps to which passport image key
- which state source maps to which passport state sub-key
- units and coord frames claimed by each source
- expected control rate
- which runner entry point performs inference
- whether a safe dry-run / no-emit mode exists

Keep it local to the deployment target. Do not create a general-purpose
package for it here.

### Step 5: Run the preflight checks

Run the audit in order. Do not jump ahead. Use the same ten chain elements
defined in the rubric. For each element, record:

- observed: what was actually seen or executed
- expected: what the passport, signoff, bindings, or runner says should
  happen
- result: `pass`, `fail`, or `unreached`
- evidence: where the proof lives, such as sample IDs, file paths, command
  lines, logs, saved tensors, or code paths
- notes: the mismatch or reason, if any

#### Element 1: Source identity

- in `live-rig` mode, identify exactly which camera, sensor, topic, device,
  or stream is bound to each logical source
- in `replay` mode, identify exactly which dataset, log, recording, sample
  ID, and timestamp/index is being used
- record enough source provenance that another engineer can fetch the same
  source again later

#### Element 2: Raw source sample

- inspect at least one raw sample per source before trusting the rest of the
  run
- record raw key names, shapes, dtypes, value ranges, and simple summaries
- for images, record layout, channel order, and whether values appear raw or
  already transformed
- for state, record sub-keys, dims, units, coord frames, and obvious sanity

#### Element 3: Source-to-passport bindings

- every passport image key and state sub-key has exactly one source
- units, coord frames, timing, ordering, and history expectations match the
  passport
- if replay keys differ from passport keys, the binding/remap layer is made
  explicit and checked here
- if a source starts after the raw-observation boundary, mark that as a chain
  gap instead of pretending the top of the chain was checked

#### Element 4: Checkpoint identity and integrity

- checkpoint has a valid passport and signoff
- deployment bindings point at the intended checkpoint, passport, and signoff
- recompute and verify signed artifact hashes against `SIGNOFF.json`
- confirm any declared norm stats files or config references resolve to the
  intended artifacts

Static checkpoint validation supports this element, but by itself it is still
not preflight.

#### Element 5: Runtime model load path and model internals

- runner is loading the intended checkpoint path
- actual class/module path used at runtime matches the passport
- actual dtype used at runtime matches the passport
- model structure on load is consistent with what the passport says should be
  there
- if the runner falls back to a different loader, class path, config, or
  dtype, that is a fail

#### Element 6: Preprocessing and transformation steps

- trace the transformations from raw source sample toward model input
- confirm resize, crop, dtype conversion, color order, layout changes,
  history assembly, normalization, and batch assembly match the passport and
  runner code
- when norm stats are part of the contract, inspect them directly and check
  that normalize -> unnormalize behaves as expected on representative data
- if the chain cannot explain how a raw key becomes a model input key, this
  element fails

#### Element 7: Final model input contract

- capture the actual tensors or structured inputs presented to the model
- record model input keys, shapes, dtypes, sequence/history axes, and value
  summaries
- confirm the observed final model inputs match the passport contract, not
  just the runner author's intent
- this element is about what the model actually received, not just what the
  code appears to prepare

#### Element 8: Model output shape and value sanity

- run one safe dry-run inference through the target repo's real preprocessing
  and model code
- record output keys, shapes, dtypes, and simple value sanity
- confirm the observed model outputs match the checkpoint/passport
  expectations closely enough to continue

#### Element 9: Output unnormalization and post-processing

- confirm output-side unnormalization and post-processing match what the
  deployment path is expected to do
- record the reconstructed action or command payload after output-side
  transforms
- if there are output masks, clipping, frame conversions, or delta-to-absolute
  reconstruction steps, check them here

#### Element 10: Would-be emission behavior

- confirm the final would-be emitted action has the expected shape and
  semantics
- confirm which code path would publish, actuate, or send the action
- confirm the action is not actually sent to the robot during preflight
- if no safe dry-run / no-emit path exists, stop before actuation, record the
  gap, and classify later elements accordingly

If the target repo has no safe dry-run path, stop and ask the user before
attempting anything that could move hardware.

### Step 6: Decide the result using the rubric

Use the rubric literally:

- `PASS (live)` only if the whole chain ran on the real rig
- `PASS (replay, provisional)` only if the whole chain ran in replay mode
- `FAIL` if any executed chain element did not match, even if later elements
  are unreached
- `NOT RUN` only if the audit never reached a decisive chain check or never
  exercised the real inference flow

Do not call checkpoint-only validation a preflight.

### Step 7: Write a durable audit artifact

Write one preflight record per first-run or fresh-run check. Think of it as a
fill-in audit note with evidence, not a narrative essay.

The easiest way to write it is in this order:

1. `Run Header / Provenance`
2. `Source Bindings`
3. `Chain Ledger`
4. `Final Verdict`
5. `Remaining Gaps / Next Actions`

### What goes in each section

#### `Run Header / Provenance`

Capture the facts that tell another engineer exactly what was checked:

- audit timestamp
- operator or agent name
- mode: `live-rig` or `replay`
- if `replay`, why live-rig preflight was not run now
- target repo path
- target repo commit and whether the tree was dirty
- runtime environment: host, container image/tag/digest, or equivalent
- exact runner entrypoint, command, and config used
- checkpoint path
- passport path
- signoff path
- norm stats / aux artifact paths used by the run
- replay dataset/log path plus stable sample identifiers if replay was used

#### `Source Bindings`

List each real or replay source and how it maps into the passport contract:

- source name
- mapped passport key
- units and coord frame
- rate / temporal notes
- explicit remap layer or adapter, if any

#### `Chain Ledger`

Walk through all ten chain elements in order. Do not skip later elements just
because an earlier one failed. Mark them `unreached` instead.

For each element, write:

- `Observed`
- `Expected`
- `Result` using only `pass`, `fail`, or `unreached`
- `Evidence`
- `Notes`

#### `Final Verdict`

State exactly one of:

- `PASS (live)`
- `PASS (replay, provisional)`
- `FAIL`
- `NOT RUN`

#### `Remaining Gaps / Next Actions`

Say plainly:

- which elements were unreached and why
- what must be done before the next preflight
- if replay was used, what still remains to be checked on the live rig

Prefer storing it near the deployment work, for example under
`preflight_audits/` in the target repo or deployment notes area.

### Suggested audit layout

Prefer an expanded ledger that reads top-to-bottom.

Copy this structure and fill it in:

```markdown
## Chain Ledger

### 1. Source identity
- Observed:
- Expected:
- Result: `pass` | `fail` | `unreached`
- Evidence:
- Notes:

### 2. Raw source sample
- Observed:
- Expected:
- Result: `pass` | `fail` | `unreached`
- Evidence:
- Notes:

### 3. Source-to-passport bindings
- Observed:
- Expected:
- Result: `pass` | `fail` | `unreached`
- Evidence:
- Notes:

### 4. Checkpoint identity and integrity
- Observed:
- Expected:
- Result: `pass` | `fail` | `unreached`
- Evidence:
- Notes:

### 5. Runtime model load path and model internals
- Observed:
- Expected:
- Result: `pass` | `fail` | `unreached`
- Evidence:
- Notes:

### 6. Preprocessing and transformation steps
- Observed:
- Expected:
- Result: `pass` | `fail` | `unreached`
- Evidence:
- Notes:

### 7. Final model input contract
- Observed:
- Expected:
- Result: `pass` | `fail` | `unreached`
- Evidence:
- Notes:

### 8. Model output shape and value sanity
- Observed:
- Expected:
- Result: `pass` | `fail` | `unreached`
- Evidence:
- Notes:

### 9. Output unnormalization and post-processing
- Observed:
- Expected:
- Result: `pass` | `fail` | `unreached`
- Evidence:
- Notes:

### 10. Would-be emission behavior
- Observed:
- Expected:
- Result: `pass` | `fail` | `unreached`
- Evidence:
- Notes:
```

Use these short reminders when filling it in:

1. `Source identity`
   Name the exact device, topic, dataset, log, sample ID, timestamp, or frame
   index used.

2. `Raw source sample`
   Record the raw keys, shapes, dtypes, value ranges, and basic sanity notes.

3. `Source-to-passport bindings`
   Show how each real source maps onto each passport key, including any remap
   or adapter layer.

4. `Checkpoint identity and integrity`
   Name the exact checkpoint bundle and the recomputed hash results.

5. `Runtime model load path and model internals`
   Name the actual runtime class, module, dtype, loader path, and major
   structural facts checked on load.

6. `Preprocessing and transformation steps`
   Describe the real path from raw sample to model-facing tensors, including
   resize, dtype conversion, normalization, history assembly, and batch
   assembly.

7. `Final model input contract`
   Capture what the model actually received: keys, shapes, dtypes, axes, and
   value summaries.

8. `Model output shape and value sanity`
   Record the actual output keys, shapes, dtypes, and simple value sanity.

9. `Output unnormalization and post-processing`
   Show the reconstructed output payload after output-side transforms.

10. `Would-be emission behavior`
    Name the exact no-emit path and the exact code path that would publish,
    actuate, or send actions.

If you want a compact summary table, add it after the expanded ledger, not
instead of it.

If an element fails, mark every later element `unreached` unless they were
actually exercised and evidenced.

## Quick Reference

| Check area | Compare |
|------------|---------|
| Passport integrity | `SIGNOFF.json` and checkpoint contents |
| Images | live frame vs passport image contract |
| State | live sample vs passport state contract |
| Units / frames | deployment notes vs passport sub-key metadata |
| Artifact integrity | recomputed hashes vs `SIGNOFF.json` |
| Runner load path | actual checkpoint + dtype vs passport |
| Model identity | loaded class / code path / structure vs passport |
| Transformations | observed resize / normalize / layout changes vs expectations |
| Norm stats | configured stats and round-trip behavior |
| Final model input | actual tensors presented to model vs passport contract |
| Forward pass output | real dry-run outputs vs documented deployment stack |
| Output reconstruction | actual unnorm / post-process behavior |
| Emission safety | no-emit path plus would-be publish path |
| Verdict | `PASS (live)` / `PASS (replay, provisional)` / `FAIL` / `NOT RUN` |

## Minimal Code Rule

Default: no new `autohpc` code.

If something is missing, only add the smallest possible repo-local hook in
the target repo/container, such as:

- a dry-run flag that suppresses action emission
- a helper that prints one live frame/state sample
- a debug hook that records intermediate tensors for one forward pass

Do not respond to missing hooks by creating a reusable package in
`autohpc` unless the user explicitly asks for one.

## Re-Audit Triggers

Run preflight again when:

- hardware changes
- the checkpoint or passport changes
- preprocessing or runner code changes
- the operator asks for a fresh preflight
- a prior clean deployment starts behaving differently and you need a fresh
  "last known good" record

## Common Mistakes

- stopping at checkpoint validation instead of exercising the full chain
- forgetting to state whether the run was `live-rig` or `replay`
- treating replay as equal to live-rig instead of provisional
- using replay sources that start after preprocessing and calling that
  "full chain"
- skipping source identity and raw sample checks at the top of the chain
- failing to verify the actual class/module/dtype used at load time
- failing to recompute and compare signed artifact hashes
- checking raw samples but not the transformations between sample and model
- checking transforms without capturing the final model input actually
  presented to the model
- checking shapes without checking normalization, unnormalization, or
  post-processing
- narrating the failure without saying which later stages were unreached
- writing an audit note with no provenance, no sample IDs, or no evidence
  pointers
- running a forward pass that can emit real actions during preflight
