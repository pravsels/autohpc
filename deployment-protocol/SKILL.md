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

## Evidence Standard: Code-First Verification

The agent's primary job during preflight is to **write and execute verification
code**, not to fill in audit tables by hand. For each chain element, the agent
writes a Python block that:

1. Loads the passport (expected values)
2. Loads the real artifact (dataset sample, model, adapter, output tensor)
3. Compares them with `assert` statements
4. Prints a structured pass/fail result

The script output IS the audit evidence. If an assertion fails, that element
fails. If code cannot run (e.g. missing data), the element fails. The agent
never eyeballs two values and decides if they match -- Python does that.

### What the agent writes vs. what Python checks

| Agent responsibility | Python responsibility |
|---|---|
| Choose the right entry point and loading code | Execute it |
| Decide which passport fields to compare | Assert equality/ranges |
| Interpret semantic questions (is this the right camera?) | N/A -- agent judgment |
| Write the verification script | Catch mismatches deterministically |

### Rules

- **This is validation, not debugging.** The agent's job is to report what
  it finds, not to fix problems. If a check fails, report the failure. Do not
  modify the assertion to make it pass. Do not patch the code to work around
  the issue. Do not "help" by accepting a close-enough match. A mismatch is
  a mismatch -- report it and stop.
- **Never modify an assertion after it fails.** Write the check, run it. If
  it fires, that element fails. Do not go back and relax the check to accept
  the observed value. The whole point of preflight is to catch exactly these
  mismatches before they reach production.
- **Passport values are "expected", never "observed."** The script loads the
  passport into an `expected` dict and the real artifact into an `observed`
  dict. Assertions compare them.
- **No manual pass/fail decisions.** If the code runs without assertion errors,
  the element passes. If an assertion fires, it fails. The agent does not
  override.
- **Missing data is a code error, not a judgment call.** If loading a sample
  throws KeyError or a modality is absent, the script catches the exception
  and reports fail. The agent does not substitute passport values.
- **The script is the artifact.** Save it alongside the audit markdown so
  another engineer can re-run it later.
- **Agent judgment is reserved** for elements that genuinely cannot be
  automated: source selection, semantic camera matching, deployment context
  safety. For those, the agent writes a brief rationale instead of code.

## Hybrid Preflight (Validator + Agent)

Preflight is hybrid. `validate-checkpoint` owns deterministic artifact and
schema checks: signoff hashes, required files, runtime version constraints,
state/action dimensions, dtype summaries, smoke-test buckets, reference test
vectors, and normalization round-trip results. The deployment protocol owns
the contextual chain checks that code cannot fully decide: which live or replay
source is bound to each passport key, whether a camera view semantically
matches the declared mount, whether a target runner's adapter actually applies
the declared transforms, and whether the final would-be command has the
expected robot semantics.

Do not blur those responsibilities. Run the validator first and record its
coverage, then run the ten-element chain audit to cover the semantic and
runtime gaps. If a mismatch could have been caught deterministically, record it
as validator backlog. If it requires live context or judgment, record it as
procedure backlog.

Hard validator failures are terminal for this preflight. If
`validate-checkpoint <ckpt> --require-signoff` exits non-zero or reports any
hard failure, Element 4 fails: set the final verdict to `FAIL`, stop before
model load or dry-run, and mark all later chain elements `unreached`. A later
successful model load, forward pass, or replay dry-run cannot override a failed
checkpoint identity/integrity gate.

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

Start from the actual deployment entry point -- the script or command that
would be run in production. Follow it through any config loading, factory
routing, architecture detection, or adapter selection to the final model class.
Name every file and function in that chain. If the entry point uses a config
file, open it and record the values that control model class, architecture,
stats path, dtype, and any other deployment-critical settings. Compare those
values against the passport before proceeding to the chain audit.

Do not skip the routing layer by going directly to the adapter or model class.
A real deployment goes through the entry point, and faults in the routing layer
(wrong architecture in config, factory fallback to wrong class) are invisible
if you bypass it.

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

Every binding must come from actually reading the target repo's code or config,
not from guessing based on the passport. If the runner has no image input, the
image binding should be empty or absent -- do not fill it from the passport and
assume it works. If you cannot determine a binding from the target repo, record
it as `unknown` and note the gap.

### Step 5: Write and run the preflight verification script

Write a single Python script (or a small set of scripts) that exercises the
full chain. The script loads the passport, loads the real artifacts, and uses
`assert` statements to compare them. Run the script and paste its output as
the audit evidence. Save the script alongside the audit markdown.

Structure the script with one section per element. Each section prints a
header, runs the checks, and prints PASS or FAIL. If an earlier element fails
hard (especially element 4), skip later sections and print UNREACHED.

Below is the pattern for each element. Adapt the code to the specific
checkpoint, adapter, and dataset -- these are templates, not copy-paste.

#### Element 1: Source identity (agent judgment + code)

```python
# Element 1: Source identity
# Agent judgment: is this the right source for this deployment?
# Use whatever dataset loader the target repo uses (e.g. HF datasets,
# a custom loader, or the framework's dataset class).
ds = load_dataset(REPLAY_REPO_ID)  # adapt to target repo's loader
sample = ds[0]
print(f"source: {REPLAY_REPO_ID}")
print(f"num_samples: {len(ds)}")
print(f"keys: {sorted(sample.keys())}")
# AGENT: confirm this source is appropriate for the deployment context
```

#### Element 2: Raw source sample (code-checkable)

```python
# Element 2: Raw source sample — compare against passport
import json, torch
passport = json.load(open(PASSPORT_PATH))
sample = ds[0]

# --- Check all passport image keys exist ---
for img_spec in passport["input_contract"].get("images", []):
    key = img_spec["key"]
    assert key in sample, f"FAIL E2: passport image key '{key}' missing from dataset"
    val = sample[key]
    assert isinstance(val, torch.Tensor), f"FAIL E2: {key} is not a tensor"
    print(f"  {key}: shape={val.shape}, dtype={val.dtype}, "
          f"min={val.min():.4f}, max={val.max():.4f}")

# --- Check state keys exist ---
for state_spec in passport["input_contract"].get("state", []):
    key = state_spec["key"]
    assert key in sample, f"FAIL E2: passport state key '{key}' missing from dataset"
    val = sample[key]
    print(f"  {key}: shape={val.shape}, dtype={val.dtype}, "
          f"min={val.min():.4f}, max={val.max():.4f}")

# --- Check action keys exist ---
action_key = "action"
assert action_key in sample, f"FAIL E2: '{action_key}' missing from dataset"
print(f"  {action_key}: shape={sample[action_key].shape}")

print("Element 2: PASS")
```

#### Element 3: Source-to-passport bindings (code-checkable)

```python
# Element 3: Source-to-passport bindings
input_contract = passport["input_contract"]
for key, spec in input_contract.items():
    val = sample[key]
    if "shape" in spec and isinstance(val, torch.Tensor):
        expected_shape = tuple(spec["shape"])
        # Compare relevant dims (e.g. last dim for state, C/H/W for images)
        assert val.shape[-len(expected_shape):] == expected_shape, \
            f"FAIL: {key} shape {val.shape} vs expected {expected_shape}"
print("Element 3: PASS")
```

#### Element 4: Checkpoint identity and integrity (validator)

```python
# Element 4: Run validator
import subprocess, sys
result = subprocess.run(
    ["validate-checkpoint", CKPT_PATH, "--require-signoff", "--show-not-checked"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0 or "❌" in result.stdout:
    print("Element 4: FAIL — hard validator failure, stopping here")
    for i in range(5, 11):
        print(f"Element {i}: UNREACHED")
    sys.exit(1)
print("Element 4: PASS")
```

#### Element 5: Runtime model load path (code-checkable)

This is the most important code-checkable element. Every field below MUST be
asserted -- do not skip any, do not treat mismatches as "not blocking."

```python
# Element 5: Load model through the DEPLOYMENT entry point and check class
# Adapt imports to the target repo's actual entry point / factory
from <target_repo>.adapters.factory import load_policy_adapter
info = load_policy_adapter(policy_path=CKPT_PATH, device="cpu")
adapter = info.adapter
policy = adapter.policy

# --- Class identity (MUST match exactly) ---
expected_class = passport["model_identity"]["class_name"]
actual_class = type(policy).__name__
assert actual_class == expected_class, \
    f"FAIL E5: class_name mismatch: runtime={actual_class}, passport={expected_class}"

expected_module = passport["model_identity"]["class_module"]
actual_module = type(policy).__module__
assert actual_module == expected_module, \
    f"FAIL E5: class_module mismatch: runtime={actual_module}, passport={expected_module}"

# --- Temporal structure (MUST match passport exactly) ---
expected_horizon = passport["output_spec"]["actions"]["horizon"]
actual_horizon = policy.config.horizon
assert actual_horizon == expected_horizon, \
    f"FAIL E5: horizon mismatch: runtime={actual_horizon}, passport={expected_horizon}"

expected_action_dim = passport["input_contract"]["actions"]["total_dim"]
actual_action_dim = policy.config.action_feature.shape[0]
assert actual_action_dim == expected_action_dim, \
    f"FAIL E5: action_dim mismatch: runtime={actual_action_dim}, passport={expected_action_dim}"

actual_n_obs_steps = policy.config.n_obs_steps
actual_n_action_steps = policy.config.n_action_steps
print(f"  n_obs_steps={actual_n_obs_steps}, n_action_steps={actual_n_action_steps}")
# n_action_steps <= horizon is required; if passport records it, assert equality
assert actual_n_action_steps <= actual_horizon, \
    f"FAIL E5: n_action_steps ({actual_n_action_steps}) > horizon ({actual_horizon})"

# --- Print summary ---
print(f"Element 5: PASS — {actual_class} from {actual_module}, "
      f"horizon={actual_horizon}, action_dim={actual_action_dim}, "
      f"n_obs_steps={actual_n_obs_steps}, n_action_steps={actual_n_action_steps}")
```

If any assertion fails, this element fails. Do not continue to element 6.

#### Element 6: Preprocessing and transformation steps (code-checkable)

```python
# Element 6: Run sample through adapter preprocessing, check shapes
raw_sample = ds[0]
# Use the adapter's own preprocessing (not manual transforms)
preprocessed = adapter.preprocess(raw_sample)  # adapt to actual API
for key, tensor in preprocessed.items():
    print(f"  {key}: {tensor.shape} {tensor.dtype}")
    # Assert shapes match what the model expects
print("Element 6: PASS")
```

#### Element 7: Final model input contract (code-checkable)

```python
# Element 7: Capture actual model input shapes
# Hook or inspect the batch dict right before model.forward()
batch = adapter.build_batch(raw_sample)  # adapt to actual API
for key, val in batch.items():
    if isinstance(val, torch.Tensor):
        expected = passport["input_contract"].get(key, {})
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        if "shape" in expected:
            # check relevant dimensions
            pass
print("Element 7: PASS")
```

#### Element 8: Model output shape and value sanity (code-checkable)

```python
# Element 8: Run forward pass, check output against passport
with torch.no_grad():
    output = adapter.predict(sample)  # adapt to actual API

print(f"  output shape: {output.shape}")
print(f"  output dtype: {output.dtype}")
print(f"  output range: [{output.min():.4f}, {output.max():.4f}]")
print(f"  output mean: {output.mean():.4f}")

# --- Sanity (MUST pass) ---
assert not torch.isnan(output).any(), "FAIL E8: NaN in output"
assert not torch.isinf(output).any(), "FAIL E8: Inf in output"

# --- Action dim (MUST match passport) ---
expected_action_dim = passport["input_contract"]["actions"]["total_dim"]
assert output.shape[-1] == expected_action_dim, \
    f"FAIL E8: action_dim mismatch: output={output.shape[-1]}, passport={expected_action_dim}"

# --- Horizon / action chunk length (MUST match passport) ---
expected_horizon = passport["output_spec"]["actions"]["horizon"]
actual_chunk_len = output.shape[-2] if output.dim() >= 2 else 1
# Output chunk may be n_action_steps (a slice of horizon). Both are valid
# but the chunk length must be <= horizon and > 0.
assert 0 < actual_chunk_len <= expected_horizon, \
    f"FAIL E8: chunk length {actual_chunk_len} outside [1, {expected_horizon}]"

# --- Cross-check with Element 5 config values ---
assert actual_chunk_len == policy.config.n_action_steps, \
    f"FAIL E8: output chunk {actual_chunk_len} != config.n_action_steps {policy.config.n_action_steps}"

print(f"Element 8: PASS — shape={output.shape}, range=[{output.min():.4f}, {output.max():.4f}]")
```

If a forward pass cannot run, this element fails. Do not fill in shapes from
config or passport attributes.

#### Element 9: Output unnormalization and post-processing (code-checkable)

```python
# Element 9: Unnormalize output and check physical plausibility
unnormed = adapter.unnormalize(output)  # adapt to actual API
print(f"unnormalized range: [{unnormed.min():.4f}, {unnormed.max():.4f}]")
# Agent judgment: are these values physically plausible for this robot?
# e.g. joint angles in [-pi, pi], velocities within motor limits
print("Element 9: PASS (pending agent review of value ranges)")
```

#### Element 10: Would-be emission behavior (agent judgment)

```python
# Element 10: Emission check
# Agent judgment: confirm dry-run mode, no robot commands sent
# Verify the deployment path has a no-emit / dry-run flag
print("Element 10: agent confirms no-emit mode is active")
```

### Adapting the templates

The code above is a starting pattern. The agent must adapt it to the target
repo's actual API -- different adapters, different factory functions, different
predict methods. The key constraint is: **every comparison between expected and
observed must be an `assert` statement in code, not a manual judgment.**

If an element genuinely requires agent judgment (source selection, semantic
camera matching, physical plausibility), the agent writes a brief rationale
in a comment or print statement. But shapes, class names, config values,
dtypes, and ranges are always code-checked.
that would have been used, and confirm the payload semantics match the robot
controller.

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
- validator command and exit code
- validator report path, if generated
- validator hard failures, soft signals, and not-checked rows
- explicit statement of whether each trial fault was caught by validator,
  deployment protocol, agent judgment, or missed
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

Separate gaps into:

- validator gaps: deterministic checks that should become code
- schema/passport gaps: missing contract fields needed for a future check
- procedure gaps: agent/deployment instructions that need to become clearer

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

## Validator Coverage Map

When `validate-checkpoint` can produce a coverage report, include it in the
preflight artifact. Until then, create a manual coverage note with three lists:

- `covered_static`: validator checks that ran and passed or failed.
- `not_checked_static`: schema paths that should be deterministic but were not
  checked by the current validator.
- `requires_agent`: schema paths or chain claims that require live context,
  semantic judgment, or target-runner inspection.

The preflight verdict must not imply that `not_checked_static` paths were
validated. If an adversarial trial exposes a missed deterministic fault, add a
validator-kernel backlog item naming the exact schema path and failure mode.

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
