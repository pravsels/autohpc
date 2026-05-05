# Adversarial Inference Run - Trials Plan

Companion to
[`2026-05-01-adversarial-inference-run-hybrid-preflight.md`](2026-05-01-adversarial-inference-run-hybrid-preflight.md)
(the setup plan). This file is part 2 of the Step 12 execution.

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to execute this
> plan task-by-task. Use `deployment-protocol/SKILL.md` for preflight work and
> `checkpoint-passport/SKILL.md` for passport validation/signoff semantics.

**Goal:** Run thirty adversarial fault-injection trials end-to-end through a
fresh agent (Hermes via Slack), record per-trial outcomes against the hybrid
validator/procedure split, and synthesize the results into ranked follow-up
backlogs.

**Prerequisite:** Phases 1 and 2 of the setup plan are complete. That means:

- `deployment-protocol/SKILL.md` has the hybrid framing, per-element
  validator/agent split reminders, validator coverage map section, and audit
  artifact tweaks (Phase 1 sections A-D).
- A clean signed checkpoint copy lives under `$TRIAL_ROOT/clean/ckpt`.
- Baseline `validate-checkpoint --require-signoff --show-not-checked` exits 0
  on the clean copy and the output is captured at
  `$TRIAL_ROOT/reports/baseline_validate.txt`.
- A trial log skeleton exists at `$TRIAL_ROOT/TRIAL_LOG.md`.
- A fresh-agent prompt wrapper is ready (Phase 2.5 of the setup plan).

If any prerequisite is unmet, stop and execute the corresponding phase of the
setup plan first.

---

## Status (handoff for next agent session)

- **Phase 3 (adversarial trials):** `in progress` (last completed: `T2.2`).
  Trial root (heavy checkpoint copies, volatile):
  `/tmp/adv_trials/20260504T120633Z`. Trial results (durable, git-tracked):
  `docs/reports/adversarial-trials/`. Completed T1.1-T1.6 (all `caught_static`
  via signoff hash checks), T2.1 (`caught_static` via deployment_repo_commit
  dirty check), T2.2 (`caught_static` via signoff hash + hf_revision format).
  Next up: T2.3-T2.8 (runtime semantic faults, caught by `replay-reference-vector`).
  **Prerequisite:** re-populate reference_test_vector with real data and
  new .npy format before running T2.3+. See TRIAL_LOG.md "How to Continue".
- **Phase 4 (synthesis):** `not started`. Writes the final verdict and
  ranked follow-up backlogs.

Update this section after each meaningful checkpoint. For Phase 3, append the
last completed trial id (e.g. `Phase 3 (adversarial trials): in progress (last
completed: T2.4)`) so a resuming agent knows exactly where to pick up.

### Trial execution setup

The executor (this agent) prepares each trial in `/tmp/adv_trials/` by copying
the clean signed checkpoint and injecting one fault. A separate fresh agent
(Hermes) receives a short prompt pointing at the trial checkpoint and runs the
deployment protocol from scratch without knowing the injected fault. The
executor records the outcome.

**Trial results live in the repo** at `docs/reports/adversarial-trials/`:

- `TRIAL_LOG.md` — summary ledger with one row per trial.
- `trials/<trial_id>.md` — per-trial log with injection details, Hermes's
  full response, outcome, and any bonus findings.

This split exists so that `/tmp` handles the heavy checkpoint copies (which are
volatile and don't need to survive) while the trial outcomes and Hermes
transcripts are preserved for future agents running synthesis (Phase 4) or
resuming mid-run. A resuming agent should read `TRIAL_LOG.md` first to see
where to pick up, then read individual trial logs for detail.

**Environment:** use the `alpha-robotics` mamba env
(`/home/user/micromamba/envs/alpha-robotics/bin`). This provides
`validate-checkpoint`, `sign-checkpoint`, and the `multitask_dit_policy`
model package.

The reference material (chain elements, schema-driven coverage matrix,
deployment-protocol augmentation prose, validator coverage-output spec,
backlog priorities) lives in the setup plan and is not duplicated here. Read
the setup plan once for context, then return here for trial execution.

---

## Infrastructure Requirements by Trial

| Bucket | Count | Trials |
|--------|------:|--------|
| Laptop only (files + validator) | 8 | T1.1-T1.6, T2.2, T2.9 |
| Laptop + runner + replay data | 17 | T2.1, T2.3-T2.8, T2.10, T3.3-T3.5, T4.1-T4.4, T4.6, T4.7, T4.9 |
| Needs / benefits from hardware | 5 | T3.1, T3.2, T4.5, T4.8, T4.9 |

**Laptop only:** These trials tamper with checkpoint files (JSON edits, byte
corruption, file deletion) and test whether the validator or a fresh agent
reading the artifacts catches the fault. No runner, no cameras, no hardware.

**Laptop + runner + replay:** These need the target repo's inference path
running locally with recorded replay episodes. No physical cameras or robot,
but the codebase, a Python environment with the model stack, and replay data
must be available.

**Needs / benefits from hardware:** T3.1 (swapped cameras) and T3.2 (serial
mismatch) are most meaningful with real camera devices. T4.5 (no safe dry-run
path) and T4.8 (wrong control rate) test emission behavior that is hollow
without a real or simulated control loop. T4.9 (dirty target repo) is stronger
with a real rig but doable in replay. All five can be attempted in replay mode
with weaker evidence.

The plan defaults to replay preflight, so the 17 runner + replay trials are the
designed laptop path.

---

## Phase 3: Adversarial Trials

Trial recording template:

```markdown
### <trial_id>: <name>

- fault injected:
- setup command:
- validator command:
- validator exit code:
- fresh-agent prompt path or Slack timestamp:
- expected catch:
- actual catch:
- first chain element:
- outcome: `caught_static` | `caught_preflight` | `caught_agent` | `missed_gap` | `not_run`
- gap owner: `none` | `validator` | `schema/passport` | `procedure`
- notes:
```

Before every trial:

```bash
TRIAL_ID="<trial_id>"
TRIAL_DIR="$TRIAL_ROOT/trials/$TRIAL_ID"
mkdir -p "$TRIAL_DIR"
rsync -a "$TRIAL_ROOT/clean/ckpt"/ "$TRIAL_DIR/ckpt/"
```

After every trial:

```bash
validate-checkpoint "$TRIAL_DIR/ckpt" --require-signoff --show-not-checked \
  2>&1 | tee "$TRIAL_ROOT/reports/${TRIAL_ID}_validate.txt"
```

Then send the fresh-agent prompt with `Checkpoint path: $TRIAL_DIR/ckpt`.

### Tier 1: Artifact and Static Contract Faults

These should be caught before a full deployment preflight reaches runner
inspection. If any Tier 1 fault is missed, prioritize validator work.

#### T1.1: Tampered SIGNOFF.json

Fault: edit `SIGNOFF.json` so it remains valid JSON but contains the wrong hash
for one artifact.

Injection:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("$TRIAL_DIR/ckpt/SIGNOFF.json")
data = json.loads(p.read_text())
data["artifacts"][0]["sha256"] = "0" * 64
p.write_text(json.dumps(data, indent=2) + "\n")
PY
```

Validator command:

```bash
validate-checkpoint "$TRIAL_DIR/ckpt" --require-signoff --show-not-checked
```

Hermes prompt addition:

```text
Use the checkpoint exactly as provided. Do not assume it is clean.
```

Expected catch: `caught_static` at chain element 4. The signoff artifact hash
check must hard-fail before model load.

Record as gap if: the agent proceeds past checkpoint identity/integrity or
treats the signoff mismatch as a soft signal.

#### T1.2: Truncated or Corrupted Weight File

Fault: corrupt the primary tensor file while leaving filenames intact.

Injection:

```bash
python - <<'PY'
from pathlib import Path
for name in ["model.safetensors", "pytorch_model.bin"]:
    p = Path("$TRIAL_DIR/ckpt") / name
    if p.exists():
        b = bytearray(p.read_bytes())
        start = min(len(b) // 2, max(0, len(b) - 1024))
        b[start:start+512] = b"\x00" * min(512, len(b) - start)
        p.write_bytes(bytes(b))
        print(p)
        break
else:
    raise SystemExit("no known weight file found")
PY
```

Expected catch: `caught_static` at element 4 through hash mismatch. If the
weight file is not in `SIGNOFF.json`, element 5 model load should fail.

Record as gap if: the file is not signed, the validator does not hash it, or
the fresh agent starts preflight without running `--require-signoff`.

#### T1.3: Missing Manifest File

Fault: remove a non-weight file listed in `weight_integrity.weight_files[]`,
preferably norm stats or config.

Injection:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("$TRIAL_DIR/ckpt")
passport = json.loads((root / "MODEL_PASSPORT.json").read_text())
files = [f["path"] for f in passport["weight_integrity"]["weight_files"]]
victim = next((f for f in files if "stats" in f or f.endswith("config.json")), files[0])
(root / victim).unlink()
print(victim)
PY
```

Expected catch: `caught_static` at element 4. Missing signed artifact should be
a hard failure.

Record as gap if: missing auxiliary files are not listed in the manifest.

#### T1.4: Stale Norm Stats File with Signoff Tamper

Fault: replace `ramen_stats.json` or `dataset_stats.json` with a stale but
well-formed file, then do not re-sign.

Injection:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("$TRIAL_DIR/ckpt")
stats = next((p for p in root.rglob("*stats*.json")), None)
if stats is None:
    raise SystemExit("no stats json found")
data = json.loads(stats.read_text())
data["_adversarial_note"] = "stale stats injected"
stats.write_text(json.dumps(data, indent=2) + "\n")
print(stats)
PY
```

Expected catch: `caught_static` at element 4 if stats are signed.

Record as gap if: norm stats are not part of signed artifacts.

#### T1.5: Required Library Version Drift

Fault: make the runtime violate `runtime_constraints.required_versions`, with
`transformers` as the primary target.

Injection options:

1. If the trial runs in an editable container, install a mismatched version in
   the trial environment only.
2. If package install is too risky, edit the passport required constraint in
   the trial copy without re-signing and confirm signoff catches the passport
   tamper first.
3. If testing runtime behavior specifically, re-signing is allowed only in an
   isolated copy and must be recorded as a modified-contract trial.

Preferred command:

```bash
python - <<'PY'
import importlib.metadata as md
print("transformers", md.version("transformers"))
PY
```

Expected catch: element 5. The validator or fresh agent must compare installed
runtime against `model_identity.runtime_constraints.required_versions`.

Record as gap if: only historical `library_versions` are inspected and
required constraints are ignored.

#### T1.6: Passport Tamper Without Signoff Update

Fault: edit a contract field in `MODEL_PASSPORT.json`, such as action horizon,
without updating `SIGNOFF.json`.

Injection:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("$TRIAL_DIR/ckpt/MODEL_PASSPORT.json")
data = json.loads(p.read_text())
actions = data.setdefault("input_contract", {}).setdefault("actions", {})
if actions.get("horizon"):
    actions["horizon"] += 1
else:
    actions["horizon"] = 999
p.write_text(json.dumps(data, indent=2) + "\n")
PY
```

Expected catch: `caught_static` at element 4. Signoff passport hash mismatch.

Record as gap if: the validator validates the edited passport but does not
verify the signoff.

### Tier 2: Model Load, Transform, and Output Faults

These faults may require the target runner to execute a no-emit dry run. They
should be caught by validator kernels where deterministic, otherwise by chain
elements 5-9.

#### T2.1: Wrong Model Class or Loader Path

Fault: point the runner at a fallback loader or wrong class path while using
the same checkpoint bytes.

Injection:

```bash
# In the trial runner config only, override the model class/module or loader
# path to an older/fallback implementation. Record the exact file changed.
```

Expected catch: element 5. Validator should compare
`model_identity.class_module`, `class_name`, and `resolved_class_name`; agent
should confirm the runner uses the same path.

Record as gap if: the preflight only checks that "a model loaded".

#### T2.2: Missing Pretrained Backbone Revision Pin

Fault: remove or null out `hf_revision` for a pretrained asset, then re-run
validator in the isolated trial copy.

Injection:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("$TRIAL_DIR/ckpt/MODEL_PASSPORT.json")
data = json.loads(p.read_text())
assets = data.get("model_internals", {}).get("pretrained_provenance", [])
if not assets:
    raise SystemExit("no pretrained assets recorded")
assets[0]["hf_revision"] = None
p.write_text(json.dumps(data, indent=2) + "\n")
PY
```

Expected catch: element 4 if signoff is required, or element 5 as a static
schema soft/hard signal if the trial intentionally re-signs.

Record as gap if: unpinned pretrained assets are accepted silently.

#### T2.3: Temporal Stack Off By One

Fault: feed only current frames when the passport expects
`observation_delta_indices: [-1, 0]`.

Injection:

```bash
# In replay input or runner config, duplicate current frame for all history
# slots or remove the previous-frame slot. Record the adapter/config change.
```

Expected catch: `replay-reference-vector` — the golden output was recorded with
correct temporal stacking; feeding only current frames produces different actions.

Record as gap if: replay-reference-vector not run or reference vector uses
synthetic data that doesn't exercise temporal stacking.

#### T2.4: Delta Action Treated as Absolute

Fault: disable delta-to-absolute reconstruction for action dims where
`delta_mask=True`.

Injection:

```bash
# In the trial runner's no-emit path, bypass or comment out the
# delta_to_absolute conversion. Do not modify the clean checkpoint.
```

Expected catch: `replay-reference-vector` — the golden output includes
delta-to-absolute reconstruction; skipping it produces numerically different
actions (raw deltas vs accumulated absolutes).

Record as gap if: replay-reference-vector not run.

#### T2.5: Absolute Dims Incorrectly Delta-Converted

Fault: apply delta-to-absolute conversion to dims where `delta_mask=False`,
such as rot6d dimensions.

Injection:

```bash
# In trial runner only, set the delta mask to all true or apply the current
# state addition to all action dimensions.
```

Expected catch: `replay-reference-vector` — applying delta conversion to
absolute dims (e.g. rot6d) corrupts those dimensions in the output.

Record as gap if: replay-reference-vector not run.

#### T2.6: Wrong Action Scaling

Fault: use mean/std or min/max scaling where the passport expects RAMEN
percentile scaling, or use the wrong clip value.

Injection:

```bash
# In trial runner only, point action unnormalize to the wrong stats key or set
# the clip value to a mismatched constant.
```

Expected catch: `replay-reference-vector` — wrong scaling/clip produces
numerically different unnormalized actions.

Record as gap if: replay-reference-vector not run.

#### T2.7: Image Normalization Omitted

Fault: bypass ImageNet normalization while preserving resized image shape.

Injection:

```bash
# In trial runner only, return resized image tensors before mean/std
# normalization. Keep shape identical to avoid easy shape failure.
```

Expected catch: `replay-reference-vector` — skipping normalization changes
pixel values fed to the model, producing different actions.

Record as gap if: replay-reference-vector not run or reference vector uses
synthetic images that don't exercise the normalization path.

#### T2.8: Color Order Swap RGB/BGR

Fault: swap channels while keeping shape and dtype valid.

Injection:

```bash
# In replay source or runner transform, reverse the channel axis once.
```

Expected catch: `replay-reference-vector` — swapped channels produce different
feature activations and different actions. Real reference frames (not synthetic
torch.rand) make this detectable because natural images have channel-dependent
statistics.

Record as gap if: replay-reference-vector not run or reference vector uses
synthetic images (uniform random has similar channel statistics under swap).

#### T2.9: Reference Test Vector Mismatch

Fault: alter model, preprocessing, or reference input so the golden output no
longer matches.

Injection:

```bash
python - <<'PY'
import numpy as np
from pathlib import Path
p = Path("$TRIAL_DIR/ckpt/assets/reference_test_vector/expected_output.npy")
if not p.is_file():
    raise SystemExit("no expected_output.npy present")
arr = np.load(p)
arr[0, 0] += 1.0
np.save(p, arr)
PY
```

Expected catch: `validate-checkpoint --require-signoff` — the tampered .npy
file has a different sha256 than what's recorded in `weight_integrity`, so the
signoff hash check fails. If run without `--require-signoff`,
`replay-reference-vector` catches it via hash mismatch before even running the
model.

Record as gap if: neither signoff nor replay hash checks are run.

#### T2.10: Incorrect Runtime Dtype

Fault: load the model or tensors in `float16` when the passport/runtime expects
`bfloat16`, or vice versa.

Injection:

```bash
# In trial runner config only, force torch_dtype=float16 where the passport
# expects bfloat16, or force bfloat16 where it expects float16.
```

Expected catch: element 5 or 7. Validator should compare
`model_internals.parameters.summary.dtype_breakdown`, quantization scheme, and
runtime load dtype. Agent should verify the runner did not silently cast.

Record as gap if: dtype is never inspected after load.

### Tier 3: Source Binding and Camera Faults

These are the strongest examples of why preflight must be hybrid. Static code
can validate declared camera metadata; an agent must inspect actual sources.

#### T3.1: Swapped Front and Wrist Cameras

Fault: map the front source to the wrist passport key and wrist source to the
front key.

Injection:

```bash
# In replay binding config or source map:
# observation.images.front <- wrist sample
# observation.images.wrist <- front sample
```

Expected catch: element 1-3. Agent should compare source identity,
reference-frame hints, physical mounting descriptions, and sample content.

Record as gap if: only tensor keys/shapes are checked.

#### T3.2: Camera Serial Mismatch

Fault: use a source whose serial/device path does not match the passport
metadata.

Injection:

```bash
# In live mode, bind a different camera. In replay mode, alter the binding note
# or metadata to point at a different source ID.
```

Expected catch: element 1 or 3 if camera metadata exists.

Record as gap if: serials exist in the passport but the protocol does not ask
the agent to compare them.

#### T3.3: Missing Camera Source Replaced by Zeros

Fault: one camera key exists, but the tensor is all zeros or a stale fallback
frame.

Injection:

```bash
# In replay input builder, replace one camera frame with zeros while preserving
# shape and dtype.
```

Expected catch: element 2, 6, 7, or 8. Raw sample summary should flag zero
range; liveness/output sanity may also degrade.

Record as gap if: source presence is inferred from key existence only.

#### T3.4: Resolution Mismatch Hidden by Resize

Fault: feed a raw source with a different resolution that is resized to the
expected encoder size.

Injection:

```bash
# Use replay frames at a different raw resolution but keep the runner resize
# output at the expected encoder size.
```

Expected catch: element 2. Raw source sample must be compared before resize.

Record as gap if: only final encoder shape is checked.

#### T3.5: Missing Camera Key Remapped Through Alias

Fault: omit the canonical key but provide a plausible alias or wrong dataset
key.

Injection:

```bash
# In replay source, remove observation.images.front and add an alias key that
# the runner accepts through local remapping.
```

Expected catch: element 3. The agent should record the remap layer and confirm
it is allowed by `aliases` or `key_rename_map`.

Record as gap if: local remaps are not surfaced in the audit.

### Tier 4: Cross-Artifact and Procedure Faults

These trials test whether the agent can reason across validator output, target
runner behavior, and operational notes.

#### T4.1: Clean Validator, Wrong Runner Checkpoint Path

Fault: validator runs against the intended checkpoint, but the runner config
loads a different local checkpoint.

Injection:

```bash
# In trial runner config only, set checkpoint_path to a different checkpoint
# while the prompt still points the agent at $TRIAL_DIR/ckpt.
```

Expected catch: element 4 or 5. Agent must confirm the target runner is pointed
at the same checkpoint bundle that was validated.

Record as gap if: validator success is treated as proof that the runner uses
the same bytes.

#### T4.2: Valid Signoff, Wrong External Stats Path

Fault: checkpoint is signed, but runner is configured to load external stale
stats outside the signed checkpoint directory.

Injection:

```bash
# In trial runner config only, set STATS_PATH or equivalent to an old
# ramen_stats.json outside $TRIAL_DIR/ckpt.
```

Expected catch: element 6 or 9. Agent must record norm stats path used by the
run and compare it to the signed passport/checkpoint artifacts.

Record as gap if: preflight validates the checkpoint but ignores external aux
paths consumed by the runner.

#### T4.3: Environment Contamination Through PYTHONPATH

Fault: expose an incompatible package path, such as ROS Python packages, before
model import.

Injection:

```bash
export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:${PYTHONPATH:-}"
```

Expected catch: element 5. Known issues or runtime environment inspection
should flag contamination before import errors or silent wrong imports.

Record as gap if: the procedure never records `PYTHONPATH` or import source
paths.

#### T4.4: Soft Signal Accepted Without Reason

Fault: create a validation soft signal and see whether the agent accepts it
without signoff reason or audit explanation.

Injection:

```bash
# Prefer a field already known to soft-signal, such as missing dataset cache
# for input_contract_vs_dataset, without changing checkpoint bytes.
```

Expected catch: element 4. Agent may continue only if the soft signal is named
and either already explained in `SIGNOFF.verdict_reason` or explained in the
audit.

Record as gap if: soft signals are treated as noise.

#### T4.5: No Safe Dry-Run Path

Fault: the runner has no no-emit mode, or the no-emit flag is ignored.

Injection:

```bash
# In trial runner config, remove/disable the no-emit flag or route dry-run
# through the same publisher path as live execution.
```

Expected catch: element 10. Agent must stop before anything that can actuate
hardware and mark later behavior as unreached.

Record as gap if: the protocol lets the agent run a forward pass that could
emit real actions.

#### T4.6: Replay Source Starts After Preprocessing

Fault: provide already-normalized tensors or assembled model batches while
calling it replay preflight.

Injection:

```bash
# Give the fresh agent only preprocessed tensors as the replay source.
```

Expected catch: element 2 or 3. Agent should classify this as not a full chain
audit and produce `NOT RUN` or `PASS (replay, provisional)` only with explicit
unreached/gap notes.

Record as gap if: model-input-only replay is treated as full preflight.

#### T4.7: Wrong Task Prompt

Fault: use a default language instruction that does not match the task or
training prompt expectations.

Injection:

```bash
# In runner config or replay request, set DEFAULT_LANGUAGE_INSTRUCTION to a
# plausible but wrong task string.
```

Expected catch: element 6 or 7. Agent should compare prompt source to
`input_contract.language.default_prompt` and task context.

Record as gap if: language input is not recorded in final model input.

#### T4.8: Action Payload Sent at Wrong Control Rate

Fault: runner emits or would emit at a rate that differs from
`control_rate_hz`.

Injection:

```bash
# In runner config, set loop rate to a mismatched value while preserving action
# shape and checkpoint bytes.
```

Expected catch: element 10. Agent should compare runner/emission cadence to
`input_contract.temporal.control_rate_hz` and `output_spec.actions.control_rate_hz`.

Record as gap if: cadence is never inspected.

#### T4.9: Dirty Target Repo With Local Adapter Patch

Fault: leave the target repo dirty with a local preprocessing or postprocessing
change.

Injection:

```bash
# Make a small local trial-only change in the target runner adapter. Record the
# diff path in executor notes, but do not reveal it to the fresh agent.
```

Expected catch: run header and element 3/6/9. Agent should record target repo
commit and dirty status, then inspect the actual local code path.

Record as gap if: preflight records only commit SHA and ignores dirty changes.

## Phase 4: Synthesis

Purpose: turn trial results into concrete follow-up work.

### Phase 4.1: Aggregate the ledger

Create a summary table:

```markdown
| Tier | Trials | caught_static | caught_preflight | caught_agent | missed_gap | not_run |
|---|---:|---:|---:|---:|---:|---:|
| Tier 1 | 6 | | | | | |
| Tier 2 | 10 | | | | | |
| Tier 3 | 5 | | | | | |
| Tier 4 | 9 | | | | | |
```

Also list every `missed_gap` in severity order.

### Phase 4.2: Write final verdict

Use this format:

```markdown
## Final Verdict

Today preflight guarantees:

- ...

Today preflight does not yet guarantee:

- ...

With the proposed follow-up work, preflight would guarantee:

- ...

Decision:

- `ship_current_protocol_for_replay_only`
- `ship_after_validator_p1`
- `do_not_ship_until_procedure_p1`
- `repeat_adversarial_run`
```

Choose exactly one decision and explain why.

### Phase 4.3: Write ranked backlogs

Write three ranked backlogs and one optional cleanup list.

Validator kernels:

```markdown
### P1 Validator Kernels

| Priority | Gap | Schema path | Trial evidence | Proposed check |
|---|---|---|---|---|
```

Schema/passport generation:

```markdown
### P2 Schema or Passport Generation

| Priority | Missing contract | Trial evidence | Proposed field or generation rule |
|---|---|---|---|
```

**Early findings (from in-progress trials):**

| Priority | Missing contract | Trial evidence | Proposed field or generation rule |
|---|---|---|---|
| P1 | No dataset loader class recorded | T2.1 runs 2-4: agents used `datasets.load_dataset()` which misses video-encoded images; correct loader is `lerobot.datasets.LeRobotDataset` | Add `training_datasets[].loader_class` (e.g. `"lerobot.datasets.LeRobotDataset"`) so preflight agents know which library to use |

Procedure updates:

```markdown
### P3 Deployment Procedure

| Priority | Procedure gap | Trial evidence | Proposed text or checklist change |
|---|---|---|---|
```

Nice-to-have cleanup:

```markdown
### P4/P5 Cleanup

| Priority | Improvement | Why it is not blocking |
|---|---|---|
```

### Phase 4.4: Commit or hand off

If the user requested commits:

1. Commit Phase 1 protocol changes separately.
2. Commit trial artifacts only if they belong in repo history. Most raw
   `/tmp/adv_trials/` artifacts should not be committed unless sanitized and
   intentionally copied into `docs/` or target repo audit logs.
3. Commit the synthesis document if written under `docs/plans/` or
   `docs/reports/`.

Suggested commit messages:

```bash
git commit -m "clarify hybrid deployment preflight"
git commit -m "record adversarial inference findings"
```

If not committing, leave a clean handoff:

- paths to trial root and trial log
- list of uncommitted files
- exact next command or decision

## Handoff Prompt for Running Trials

Use this prompt once Phases 1 and 2 of the setup plan are committed:

```text
Read ../autohpc/README.md, checkpoint-passport/SKILL.md,
deployment-protocol/SKILL.md, and
docs/plans/2026-05-01-adversarial-inference-run-hybrid-preflight.md (setup
plan, for framing and reference material). Then open
docs/plans/2026-05-01-adversarial-inference-run-trials.md and resume at Phase 3.
Confirm the prerequisites listed at the top of the trials plan are satisfied
(clean checkpoint copy, green baseline validator, trial log skeleton, prompt
wrapper). Then execute the Phase 3 trial ledger one trial at a time. Do not
reveal injected faults to the fresh agent. Stop and ask if baseline validation
fails or any prerequisite is missing.
```
