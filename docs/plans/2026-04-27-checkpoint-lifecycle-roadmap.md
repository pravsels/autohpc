# Checkpoint Lifecycle Roadmap

## Purpose

AutoHPC started as a way to speed up the path from local container work to
training on HPC. That worked, but it created the next bottleneck: checkpoints
are now easy to produce, upload, and forget. The next workflow should help an
agent and human operator decide which signed checkpoints deserve more
attention, which should be archived, and which have enough evidence to move
toward simulation, deployment preflight, or a controlled robot run.

This is not an automation plan yet. Follow the five-step order:

1. Make the requirements less wrong.
2. Delete unnecessary process.
3. Simplify and standardize the manual path.
4. Accelerate with small templates or thin helpers.
5. Automate last.

The first version should be manual, evidence-rich, and file-backed. It should
teach us what the real repeated work is before we build triggers, dashboards,
queues, or scoring systems.

## Execution Model

Most of this lifecycle is expected to be carried out by an AI coding agent,
not by a human manually stepping through every command. "Manual-first" means
agent-executed and human-supervised, not fully human-operated.

The intended split is:

- The human sets the goal, names the checkpoint family or target task, and
  approves ambiguous promotion decisions.
- The agent gathers repo context, reads the relevant AutoHPC skills, inspects
  checkpoint artifacts, runs validation/eval commands, writes logs, and
  reports evidence.
- The agent stops instead of guessing when a gate is missing, a command fails,
  eval infrastructure is not reusable, or a promotion decision depends on
  task judgment the human has not supplied.
- The durable output of each step is a file-backed record: run log, eval log,
  promotion note, preflight audit, or roadmap update.

This matters because the first system we need is not background automation. It
is a reliable agent procedure that produces evidence humans can inspect. Later
automation should replace repeated agent actions only after those actions have
proved stable across real checkpoint batches.

## Current Anchors

The roadmap builds on existing AutoHPC boundaries:

- `checkpoint-passport/SKILL.md` defines the post-train gate:
  `MODEL_PASSPORT.json`, `SIGNOFF.json`, `validate-checkpoint`, and
  `sign-checkpoint`.
- `hpc-run-tracking/SKILL.md` defines training logs and the HF upload handoff.
- `eval-tracking/SKILL.md` defines per-eval logs with provenance, metrics,
  qualitative notes, and a verdict.
- `deployment-protocol/SKILL.md` defines the deployment-side full-chain audit
  from source sample to would-be action emission.
- `../alpha-robotics/missiontracker/examples/run_backtest.py` and
  `../alpha-robotics/missiontracker/backtest/` are the likely first offline
  backtest backend for post-HF checkpoint triage.

Keep the existing hard rule: passport and signoff happen before a checkpoint
moves anywhere, including HF upload. The eval harness is itself a passport
consumer, so eval-before-passport can create misleading eval results.

## Current State

This plan intentionally separates what already exists from what still needs to
be built or proven.

### Already Exists

- Training/HPC workflow guidance exists in `hpc-container-promotion/`,
  `hpc-training-operations/`, `hpc-dataset-adaptation/`, and
  `hpc-run-tracking/`.
- Run logs already have conventions for recording training status, results,
  W&B links, and HF upload details.
- Checkpoint passport tooling already exists as an installable Python package
  under `checkpoint-passport/`.
- `validate-checkpoint` already validates signed checkpoint artifacts.
- `sign-checkpoint` already writes `SIGNOFF.json` after validation.
- `eval-tracking/SKILL.md` already defines basic per-eval markdown logs.
- `deployment-protocol/SKILL.md` already defines the full-chain robot
  preflight audit concept.
- `../alpha-robotics` appears to contain legacy MissionTracker backtest code.
  It may become the first offline eval backend, but its reusability still needs
  to be assessed.

### Partially Exists

- HF publishing guidance exists, but it stops at uploading and recording the
  upload. It does not define post-upload triage or promotion.
- Eval logs exist as a convention, but they do not yet distinguish normal evals
  from post-HF checkpoint triage.
- The passport contains calibration/smoke evidence, but there is no documented
  workflow that uses that evidence as the first cheap promotion gate.
- Deployment preflight is described as an agent-run procedure, but it is not
  connected to a checkpoint promotion decision record.

### Not Yet Done

- Define the manual post-HF checkpoint triage procedure.
- Define the promotion note format and where it should live.
- Assess whether the legacy MissionTracker backtest code is reusable for
  signed checkpoint evals.
- Run the first signed HF checkpoint through the full manual ladder once an
  eval backend is known: artifact gate, cheap behavior checks, offline
  backtest or explicit eval-gap note, eval log, promotion note.
- Decide how MissionTracker backtest configs should be pinned for repeatable
  checkpoint comparison.
- Decide how simulation evidence should be logged and linked into promotion
  notes.
- Review several real checkpoint batches before adding templates, helper
  scripts, or automation.

## Lifecycle Map

```mermaid
flowchart TD
    trainRun["Training run finishes"] --> syncWandb["Sync W&B and preserve run evidence"]
    syncWandb --> passport["Generate MODEL_PASSPORT.json"]
    passport --> signoff["Validate and write SIGNOFF.json"]
    signoff --> publish["Upload signed checkpoint bundle to HF"]
    publish --> artifactGate["Artifact gate: validate signoff"]
    artifactGate --> cheapEval["Cheap behavior checks"]
    cheapEval --> backtest["Offline backtest"]
    backtest --> triage["Eval log and promotion note"]
    triage --> decision{"Next justified action?"}
    decision -->|"reject"| archive["Archive with reason"]
    decision -->|"needs_more_eval"| moreEval["Run more eval"]
    decision -->|"promote_to_sim"| simEval["Run simulation"]
    decision -->|"promote_to_preflight"| preflight["Deployment preflight audit"]
    moreEval --> triage
    simEval --> triage
    preflight --> robotDecision{"Preflight passed?"}
    robotDecision -->|"no"| diagnose["Localize failure area"]
    robotDecision -->|"yes"| robotRun["Controlled robot run"]
```

The key shift is that "uploaded" is no longer a meaningful end state. Upload
means the signed artifact is available for consumers. Promotion records explain
what should happen next.

## Artifact Boundaries

Do not make one artifact carry every kind of evidence. Keep each artifact's job
small and stable.

### Passport And Signoff

`MODEL_PASSPORT.json` and `SIGNOFF.json` describe the checkpoint contract and
artifact integrity. They answer:

- What does this model expect on input?
- What does it emit?
- What architecture, code identity, and runtime assumptions produced it?
- Which files are part of the inference bundle?
- Do the signed bytes still match?

They should not become a dumping ground for every later eval. Later evals
should reference the passport/signoff hash instead.

### Training And HF Records

`run_logs/` and the HF model card answer:

- Which training run produced the checkpoint?
- Which config, dataset, branch, commit, and W&B run explain it?
- Which checkpoint files were uploaded?
- Which signed artifacts shipped with the upload?

The HF upload record is transport and provenance evidence. It is not a quality
verdict.

### Eval Logs

`eval_logs/` answer:

- Which checkpoint revision was evaluated?
- Which passport/signoff was used as the feeding contract?
- Which dataset, backtest config, sim scenario, or evaluation harness ran?
- What metrics and qualitative evidence came out?
- What did that specific eval conclude?

Eval logs are per-evaluation. They can disagree with each other. That is
acceptable; the promotion note exists to interpret the combined evidence.

### Promotion Notes

A promotion note answers:

- Given all evidence available now, what is the next justified action?
- Which signals support that decision?
- Which signals are weak, missing, or contradictory?
- What should be run next, if anything?

Promotion notes should live in the target repo, probably near `eval_logs/`, and
reference the signed checkpoint, HF revision, source run log, and relevant eval
logs. They are human decision records, not immutable model artifacts.

### Preflight Audits

`preflight_audits/` answer:

- Did the real deployment path feed this model according to the passport?
- Which live or replay sources were bound to each passport input?
- What transformations occurred from raw sensor sample to final model input?
- What output and post-processing were observed?
- Would the runner emit the expected command shape without actually actuating?

Preflight audits are deployment readiness evidence. They are not replacements
for eval logs, and checkpoint validation alone does not count as preflight.

## Evaluation Ladder

Use increasing cost and confidence. Do not start with simulation or robot runs
when cheaper checks can reject obvious failures.

### 1. Artifact Gate

Every eval job starts with:

```bash
validate-checkpoint <ckpt_dir> --require-signoff
```

Hard failure means stop. Missing or stale signoff is not an eval failure; it is
an artifact-readiness failure.

### 2. Cheap Behavior Checks

Use the passport's calibration/smoke evidence and, where available, a fresh
single-batch forward pass through the public inference path. Check:

- output keys and shapes
- dtype and numeric range
- NaN/Inf absence
- determinism where expected
- liveness, such as non-constant action output
- obvious action scale problems after unnormalization

These checks catch catastrophic mis-feeding and broken checkpoints. They do not
prove task quality.

### 3. Offline Backtest

Run a fixed backtest dataset through the target eval harness. For
`alpha-robotics`, the first likely backend is MissionTracker:

- `../alpha-robotics/missiontracker/examples/run_backtest.py`
- `../alpha-robotics/missiontracker/backtest/`
- `../alpha-robotics/missiontracker/adapters/`

The backtest should record:

- HF repo and revision or exact local snapshot
- passport/signoff hash used
- backtest config path
- validation dataset repo, revision, and selected episodes
- metrics such as loss or task-specific error
- representative qualitative failures

Treat high backtest loss as a strong diagnostic signal. Treat low backtest loss
as encouraging but not sufficient for robot readiness.

### 4. Qualitative Review

For checkpoints that are not immediately rejected, inspect representative
outputs, failures, or trajectories. The review should explain what the metrics
missed: temporal drift, camera-specific failures, action jitter, collapse to
mean behavior, language-conditioning mistakes, or implausible state/action
relations.

### 5. Simulation

Run simulation only for checkpoints that pass cheap gates and are worth the
extra cost. Simulation should be treated as a promotion stage:

- It can catch catastrophic behavior, unsafe contacts, table-banging, or gross
  task misunderstanding.
- It may reveal unexpectedly promising policies.
- It should not be treated as a final real-world guarantee because sim-to-real
  gaps can dominate.

### 6. Deployment Preflight

Before a robot run, follow `deployment-protocol/SKILL.md`. The required output
is a full-chain audit, not merely a successful model load. It must localize the
chain from source identity through would-be emission behavior.

### 7. Controlled Robot Run

Only after preflight passes should a checkpoint become a robot candidate. The
first run should be controlled, observable, and recorded as a new eval or
deployment record depending on the target repo's convention.

## Promotion Vocabulary

Do not create a global numeric score in the first version. Use conservative
stage-specific actions:

- `reject`: Archive with a short reason. No further eval planned unless new
  evidence appears.
- `needs_more_eval`: Evidence is missing, weak, or contradictory. The next
  action is another specific eval, not deployment.
- `promote_to_sim`: Cheap checks and offline evidence justify simulation.
- `promote_to_preflight`: Eval and/or sim evidence justify a deployment
  preflight audit.
- `candidate_for_robot`: Deployment preflight passed and a controlled robot run
  is the next justified action.

Each promotion note should include:

```markdown
# checkpoint promotion - <checkpoint name or HF revision>

## Decision
- action: `reject` | `needs_more_eval` | `promote_to_sim` | `promote_to_preflight` | `candidate_for_robot`
- decided_at: `<ISO timestamp>`
- decided_by: `<human/agent>`

## Artifact
- hf_repo:
- hf_revision:
- checkpoint_path_or_snapshot:
- passport_sha256:
- signoff_sha256:
- source_run_log:

## Evidence Reviewed
- eval_log:
- backtest_config:
- dataset:
- simulation_log:
- preflight_audit:

## Signals
- hard_gates:
- positive_signals:
- negative_signals:
- missing_evidence:
- contradictory_evidence:

## Rationale
<Short explanation of why this action is justified now.>

## Next
<One concrete next step, or "none".>
```

The promotion note should make uncertainty visible. It should not hide weak
evidence behind a single score.

## Manual First Operating Procedure

For each new checkpoint batch:

1. Confirm training finished and W&B or equivalent run evidence is intact.
2. Generate and sign the checkpoint passport before upload.
3. Upload the signed bundle to HF, including `README.md`, `TRAINING_LOG.md`,
   `MODEL_PASSPORT.json`, `SIGNOFF.json`, weights, configs, and assets.
4. Record the HF repo/revision in the source run log.
5. Run the artifact gate from the exact snapshot being evaluated.
6. Run cheap behavior checks and/or inspect passport smoke evidence.
7. Run the first offline backtest on pinned data, if Step 1 has proven the
   eval backend is usable.
8. Write an eval log with provenance, metrics, qualitative notes, and verdict.
9. Write or update a promotion note for the checkpoint or batch.
10. If promoted, run the next stage: more eval, sim, deployment preflight, or a
    controlled robot run.

Keep this manual until the team has several real checkpoint batches worth of
promotion notes. The notes are the evidence for what should be automated later.

## Thin Helpers Later

After repeated manual use, add only helpers that remove proven friction:

- an eval log template generator
- a promotion note template generator
- a local summarizer that scans eval logs and promotion notes
- a local report that groups checkpoints by current promotion action

These helpers should consume existing artifacts. They should not create a new
source of truth, and they should not submit jobs or update HF automatically.

## Automation Last

Defer HF triggers, background workers, dashboards, automatic backtest
submission, and automatic promotion until these conditions hold:

- Eval log structure is stable across several checkpoint batches.
- Promotion actions have remained useful and understandable.
- Backtest configs and datasets are pinned and repeatable.
- Failure modes are known well enough to route automatically.
- GPU scheduling, credentials, storage, retry behavior, and ownership are clear.
- Humans trust the manual notes enough to ask for less manual execution.

When automation arrives, it should execute the proven lifecycle. It should not
define the lifecycle.

## Work Plan

This section is the actual execution checklist. Each step should produce a
concrete artifact or decision before moving on.

The agent should execute these steps by default. Human involvement is called
out only where judgment, approval, or ambiguous external access is required.

### Step 0: Write The Roadmap

Status: done.

What already happened:

- This roadmap document was created.

Output:

- `docs/plans/2026-04-27-checkpoint-lifecycle-roadmap.md`

Do not update operational skills yet. First test the workflow against real
checkpoints.

### Step 1: Inventory What Eval Capability Actually Exists

Status: done.

What already happened (2026-04-28):

- Full assessment in `docs/plans/2026-04-28-step1-missiontracker-eval-readiness.md`.
- MissionTracker's `MultiTaskDiTAdapter` was extended to support DiT checkpoints
  with RAMEN normalization (Steps 1.1-1.6 of the sub-plan).
- `from_pretrained` loads the DiT policy, detects RAMEN vs LeRobot stats format,
  and routes to the correct normalization path.
- `predict()` runs a full forward pass on `dit_block_tower_norm_fix` with correct
  output shapes and no NaN. Delta-to-absolute conversion is implemented.
- Verdict: `usable_with_small_adapter` (the adapter we built).
- Remaining debts (transformers pin/shim, PYTHONPATH ROS leak guard, n_obs_steps
  queue logic) do not block Step 6.

Previously:

Actions:

1. Read the likely MissionTracker entry points:
   `../alpha-robotics/missiontracker/examples/run_backtest.py`,
   `../alpha-robotics/missiontracker/backtest/`, and
   `../alpha-robotics/missiontracker/adapters/`.
2. Identify which policy types it can load today, such as ACT, RewACT, OpenPi,
   or other adapters.
3. Identify what inputs it expects: policy path, HF repo, local checkpoint,
   dataset repo, validation episodes, config JSON, cache paths, and runtime
   dependencies.
4. Identify what outputs it produces: metrics, artifacts, logs, videos,
   per-shard caches, or only Python objects.
5. Identify whether it can run against a signed HF checkpoint snapshot without
   code changes.
6. Identify whether it validates `MODEL_PASSPORT.json` and `SIGNOFF.json`
   before loading. Assume it does not until proven otherwise.
7. Identify whether it feeds the model according to the passport
   `input_contract`, or whether it relies on its own adapter assumptions.
8. Write a short eval-readiness note with one of these verdicts:
   `usable_as_is`, `usable_with_small_adapter`, `needs_repair`,
   `not_reusable`.
9. Ask the human to review the verdict if the backend is marked `needs_repair`
   or `not_reusable`, because that may redirect the whole lifecycle effort.

Output:

- An eval-readiness note in the target repo or this plan's follow-up notes.
- A list of exact commands needed to run the smallest possible backtest, if the
  backend is usable.
- A list of gaps if the backend is not usable.

Exit criteria:

- The team knows whether MissionTracker can be used for Step 4.
- The team knows the smallest eval that can be run on one checkpoint.

### Step 2: Choose One Checkpoint For A Manual Trial

Status: done.

What already happened (2026-04-28):

- Selected `dit_block_tower_norm_fix` (HF: `pravsels/dit_block_tower_norm_fix`).
- Local path: `../alpha-robotics/checkpoints/dit_block_tower_norm_fix/`.
- W&B: `https://wandb.ai/pravsels/dit_block_tower_norm_fix/runs/ksuxe451`.
- Training repo commit: `af0a43a512841aa1f4d6bb2f93755e5358dca8cb`.

Previously:

Actions:

1. Identify candidate training runs and source run logs.
2. Confirm W&B or equivalent training evidence is intact.
3. Identify checkpoint directories and whether any have already been uploaded
   to HF.
4. Identify whether each candidate already has `MODEL_PASSPORT.json`.
5. Identify whether each candidate already has `SIGNOFF.json`.
6. Ask the human to choose exactly one checkpoint if there are multiple
   reasonable candidates.
7. Record the current state before changing anything.

Output:

- A one-checkpoint trial note with:
  - training run
  - checkpoint path
  - HF repo/revision if any
  - passport status
  - signoff status
  - intended eval backend from Step 1

Exit criteria:

- There is exactly one checkpoint selected for the first manual trial.
- Its current artifact state is clear.

### Step 3: Make The Checkpoint A Signed Artifact

Status: done.

What already happened (2026-04-28):

- Passport schema revised from v0.1 to v0.2 (plan:
  `docs/plans/2026-04-28-passport-v02-revision.md`).
- v0.1 passport (170KB) deleted and regenerated as v0.2 (22KB, 87% smaller).
- Key v0.2 additions: 12-step `transform_pipeline`, `runtime_constraints`,
  structured `DeltaSpec`, `known_issues` (3 entries), `observation_delta_indices`,
  camera identity fields on `ImageSpec`, checkpoint lineage on `Provenance`.
- Key v0.2 removals: `parameters.by_name` (130KB), module_hierarchy slimmed to
  top 2 levels.
- New validator checks: `runtime_constraints` (PASS), `reference_test_vector`
  (SOFT_SIGNAL — not yet populated, needs GPU inference),
  `camera_identity` (SOFT_SIGNAL — hardware metadata not yet captured).
- Signed with 4 soft signals documented:
  `state_dim_consistency` (rot6d 13→16 intentional),
  `reference_test_vector` (not yet populated),
  `camera_identity` (not yet captured),
  `training_datasets_resolvable` (commit SHA not captured at train time).
- Full validation: 23 passed, 4 soft signals, 0 failures.

Previously:

Actions:

1. Generate `MODEL_PASSPORT.json` if it does not exist, following
   `checkpoint-passport/SKILL.md`.
2. Run:

   ```bash
   validate-checkpoint <ckpt_dir> --show-not-checked
   ```

3. Fix passport-authoring hard failures, or stop if the failure indicates the
   checkpoint itself is not usable.
4. Ask the human to approve any accepted soft signals that require task
   judgment.
5. Run:

   ```bash
   sign-checkpoint <ckpt_dir> --reason '<reason if soft signals exist>'
   ```

6. Re-run:

   ```bash
   validate-checkpoint <ckpt_dir> --require-signoff
   ```

Output:

- `MODEL_PASSPORT.json`
- `SIGNOFF.json`
- Validation output or a note explaining why signing stopped.

Exit criteria:

- The checkpoint passes `validate-checkpoint --require-signoff`, or it is
  rejected before eval.

### Step 4: Establish The Exact Snapshot To Evaluate

Status: done.

What already happened (2026-04-28):

- `MODEL_PASSPORT.json` (v0.2) and `SIGNOFF.json` (v0.2) uploaded to HF.
- HF repo: `pravsels/dit_block_tower_norm_fix`.
- HF revision: `d1a5f7fb1fe85040b452d293a0ae94fa29bf0083`.
- Snapshot downloaded to flat directory (no symlinks) and validated with
  `validate-checkpoint --require-signoff`: 23 passed, 4 soft signals, 0
  failures.
- Note: HF cache symlink layout causes path-escape false positives in the
  validator. Use `snapshot_download(local_dir=...)` to get a flat copy for
  validation. Validator symlink handling is a future improvement.
- Passport sha256: `a6b95c662a91ea27ee05c4d5ab67ff6db8e9e05095d354a53d7be319e20c9103`.
- Signoff sha256: `cd12a18be4da42507a670a68dd9d91a464f513ed2ba646f48c064853f1f93602`.
- Eval snapshot path: `../alpha-robotics/checkpoints/dit_block_tower_norm_fix/`
  (replaced local pre-upload copy with exact HF snapshot downloaded via
  `snapshot_download(local_dir=...)`).

Previously:

The eval should run against the same artifact a downstream consumer would load,
not an ambiguous local directory.

Actions:

1. If the checkpoint is not on HF yet, prepare the signed bundle and ask the
   human before upload if credentials, visibility, or destination are
   ambiguous.
2. Record the HF repo and exact revision.
3. Download, cache, or otherwise resolve the exact snapshot that the eval
   backend will load.
4. Run:

   ```bash
   validate-checkpoint <hf_snapshot_or_ckpt_dir> --require-signoff
   ```

5. Record passport and signoff hashes for later eval/provenance notes.

Output:

- HF repo/revision or exact local snapshot path.
- Verified signed artifact gate result.

Exit criteria:

- There is no ambiguity about which bytes the eval will consume.

### Step 5: Run The Cheapest Behavior Evidence

Status: done.

What already happened (2026-04-28):

- **Passport smoke results** (from Phase 2 calibration batch): all 5 buckets
  pass — determinism (max_abs_diff=0.0), NaN/Inf (0/0 in 1088 samples),
  liveness (std=0.579), distribution (all per-dim stats recorded),
  range_check (actual [−0.91, 1.46] within expected [−1.5, 1.5]).
- **Numerical health**: determinism passed (max_abs_diff=0.0), no NaN/Inf,
  no dropout modules active, no batchnorm running stats issues.
- **Fresh forward pass through MissionTracker adapter** on synthetic
  observations matching passport `input_contract`:
  - Inputs: state (16,) float32 randn, images front+wrist (3,480,640)
    float32 rand [0,1], language "build a block tower".
  - Output shape: action (17,), action_chunk (32, 17) — matches passport
    action_dim=17, horizon=32.
  - Output dtype: float32.
  - NaN/Inf: 0 in both action and chunk.
  - Range: [−2.10, +1.95] — reasonable post-unnormalization action scale.
  - Liveness: chunk std across time = 0.033, across dims = 1.12 — non-constant,
    temporally smooth, dimensionally varied.
  - Determinism: max abs diff = 0.0 with same seed — fully deterministic.
  - No obvious action-scale blowup or collapse.
- **Verdict**: no brokenness detected. Model plumbing is intact through the
  full normalize → forward → unnormalize → delta-to-absolute path.

Previously:

Before backtest, collect the cheapest signal that the model is not obviously
broken.

Actions:

1. Inspect the passport's calibration/smoke results.
2. Record determinism, NaN/Inf, liveness, distribution, and range-check status
   if present.
3. If the target repo has an easy public inference smoke command, run one safe
   batch through that path.
4. Record output keys, shapes, dtypes, value ranges, and obvious action-scale
   issues.
5. Do not treat this as quality evidence. Treat it as brokenness evidence.

Output:

- Cheap behavior section in the eval log or trial note.

Exit criteria:

- Catastrophic model/output issues have either been ruled out or documented.

### Step 6: Run One Offline Eval If Step 1 Found A Usable Backend

Status: done.

What already happened (2026-04-28):

- MissionTracker was usable after three targeted fixes:
  1. `BacktestConfig` gained `stats_path` field to pass external RAMEN
     normalization stats to the `MultiTaskDiTAdapter`.
  2. `backtest.py` passes `stats_path` through to `load_policy_adapter` in the
     `multitask_dit` branch.
  3. `run_backtest.py` wires `POLICY_ARCHITECTURE`, `STATS_PATH`, and
     `DEFAULT_LANGUAGE_INSTRUCTION` from the JSON config into `BacktestConfig`.
- A critical state-assembly bug was fixed in `backtest.py`: the DiT policy's
  `dataset_schema` declares separate `observation.state` (7D joint) and
  `observation.eef_6d_pose` (6D → 9D via RPY-to-rot6d) sub-features that must
  be concatenated to the 16D state the model expects. The backtest's
  `_run_policy` previously picked a single state key, so a new
  `_assemble_dit_state` method was added to assemble the full 16D vector
  matching the training pipeline's `dataset_adapter.assemble_vector` logic.
- `RoboCandyWrapper` submodule updated to `c9481ed` (lerobot 0.5.2 support).
- `scikit-learn` installed for MissionTracker's `ClusterAnalyser`.
- Backtest config: `missiontracker/examples/configs/dit_block_tower_smallest.json`.
- Validation dataset: `villekuosmanen/build_block_tower_val`, episode 0 (557 frames).
- Policy path: `checkpoints/dit_block_tower_norm_fix/checkpoints/29000/params`.
- Stats path: `checkpoints/dit_block_tower_norm_fix/assets/ramen_stats.json`.
- Language instruction: "build a block tower".
- Results:
  - 557 frames, 117 anomalies (21.0% anomaly rate).
  - `VALIDATION_LOSS_TOO_HIGH`: 112 frames.
  - `INCOHERENT_ACTION_PREDICTIONS`: 5 frames.
  - `policy_loss`: mean=0.0066, p50=0.0033, p95=0.023, max=0.080.
  - `max_joint_delta`: mean=0.034, p50=0.032, p95=0.072, max=0.087.
  - `time_coherence`: mean=0.0015, very temporally coherent.
  - `perturbation_coherence`: N/A for multitask_dit.
  - Cluster analysis: no policy activations collected (DiT adapter doesn't
    expose forward hooks yet).
- Artefact: `eval_outputs/artefacts/build_block_tower_val_params.tar.gz`.
- Key observation: the 21% anomaly rate is dominated by validation-loss
  anomalies. The validation loss threshold is calibrated for ACT/RewACT
  policies; DiT may need its own threshold. This does NOT indicate the model
  is broken — Step 5's cheap checks confirmed plumbing is intact.

Previously:

If MissionTracker is usable, run the smallest possible backtest. If it is not
usable, do not pretend eval exists; write the gap and stop here.

Actions if usable:

1. Pin the validation dataset repo/revision and selected episodes.
2. Pin the backtest config.
3. Point the policy path at the exact signed snapshot from Step 4.
4. Run the smallest backtest that exercises the full policy adapter.
5. Save metrics, logs, artifacts, and representative failures.
6. Note any mismatch between MissionTracker feeding assumptions and the passport
   `input_contract`.

Actions if not usable:

1. Write why it is not usable.
2. Identify the smallest repair or adapter needed.
3. Ask the human whether repairing eval is now the next work item.

Output:

- One backtest result, or one explicit eval-gap note.

Exit criteria:

- The first checkpoint has either real offline eval evidence or a clear reason
  why eval infrastructure must be repaired first.

### Step 7: Write The First Eval Log

Status: done.

What already happened (2026-04-28):

- Eval log written to `alpha-robotics/eval_logs/dit_block_tower_norm_fix/2026-04-28-backtest-eval.md`.
- Covers: checkpoint identity, signed artifacts, source run, eval backend,
  dataset, cheap behavior evidence (Step 5), backtest results (Step 6),
  qualitative review (not performed — flagged for human), and eval verdict.
- Verdict: `needs_more_eval`. Model passes all cheap gates and produces
  plausible actions, but anomaly thresholds need DiT-specific calibration,
  more episodes should be tested, and human qualitative review is pending.

Actions:

1. Create one eval log in the target repo's `eval_logs/`.
2. Record checkpoint path or HF revision.
3. Record passport/signoff paths and hashes.
4. Record source run log and config snapshot.
5. Record eval backend, command, config, dataset, and selected episodes.
6. Record cheap behavior evidence.
7. Record backtest metrics if Step 6 produced them.
8. Record qualitative notes or say that qualitative review was not available.
9. Ask the human to review any qualitative claims that require task judgment.
10. Record a narrow eval verdict.

Output:

- One eval log.

Exit criteria:

- Another engineer can tell what was evaluated, with which contract, against
  which data, and what happened.

### Step 8: Write The First Promotion Note

Status: done (pending human approval of promotion action).

What already happened (2026-04-28):

- Promotion note written to
  `alpha-robotics/eval_logs/dit_block_tower_norm_fix/2026-04-28-promotion-note.md`.
- Drafted action: `needs_more_eval`.
- Key rationale: checkpoint passes all hard gates and produces healthy metrics,
  but evidence base is too narrow (1 episode, no DiT-calibrated thresholds, no
  human qualitative review) for `promote_to_sim`.
- Next action: multi-episode backtest with calibrated thresholds, then human
  qualitative review.

Actions:

1. Use the promotion-note template in this roadmap.
2. Reference the signed artifact from Step 4.
3. Reference the eval log from Step 7.
4. Draft exactly one action:
   `reject`, `needs_more_eval`, `promote_to_sim`, `promote_to_preflight`, or
   `candidate_for_robot`.
5. List positive signals, negative signals, missing evidence, and
   contradictions.
6. Ask the human to approve or change the promotion action.
7. Write one concrete next action.

Output:

- One promotion note.

Exit criteria:

- The checkpoint has a clear next justified action.

### Step 9: Simulation Evaluation

Status: deferred (pending sim environment research).

Simulation provides two signals that val loss cannot:

1. **Rejection signal** — does the policy do obviously bad things (crash into
   the table, freeze, miss the workspace, drop objects)? A policy that fails
   in sim is very likely bad. Strong negative evidence.
2. **Ranking signal** — given N candidate policies, does the sim ranking
   correlate with real-world ranking? Even with a sim-to-real gap, relative
   ordering may transfer. This makes sim useful for comparing policies within
   the same architecture/task family without burning robot time.

Simulation is not a real-world guarantee. Sim-to-real gap means a sim-passing
policy can still fail on hardware. Treat sim as a filter and a ranker, not a
certifier.

Inputs:

- Signed checkpoint with passing artifact gate.
- Eval log from Step 7 with verdict `needs_more_eval` or better.
- A sim environment that can run the task (to be determined).
- Task specification: success criteria, episode length, reset conditions.
- The policy's input contract (from passport) to ensure correct feeding in sim.

Outputs:

- Success rate over N episodes (N >= 20-50 to be meaningful).
- Per-episode: task success (binary), time to completion, failure mode
  classification (if failed).
- Representative video rollouts: at least 1 success and 1 failure.
- Optional: contact forces, workspace violations, sim-specific safety metrics.
- A sim eval log (same structure as backtest eval log, different backend).

Use cases:

- *Single policy triage*: does it control the robot sensibly at all? If success
  rate is near zero, that is a strong reject signal.
- *Policy ranking*: run the same sim eval on multiple policies, rank by success
  rate. Use to pick which policies are worth real robot time. The ranking
  hypothesis (sim ranking holds in real) should be validated once real-world
  comparison data exists.

Promotion interaction:

- A passing sim eval supports `promote_to_preflight`.
- A failing sim eval is strong evidence for `reject` or `needs_more_eval`.

Open questions:

- Which sim environment and physics engine for the target robot platform.
- How to define task success programmatically (object pose checks, contact
  sensors, etc.).
- What success rate threshold justifies `promote_to_preflight` vs
  `needs_more_eval`.
- Whether to run sim on the same machine or offload (GPU sim vs CPU sim).
- How to validate that sim ranking transfers to real (needs real-world
  comparison data).

### Step 10: Update Skills Only After The Manual Trial

Status: deferred.

Still in the manual phase — one checkpoint has been triaged end-to-end but
skills should not be updated until the workflow has been repeated enough times
to know what's stable. The learnings from this first pass are recorded in the
roadmap itself (Steps 4-8) and in the eval log and promotion note under
`alpha-robotics/eval_logs/dit_block_tower_norm_fix/`.

When ready, the key things to codify in skills are:

- MissionTracker backtest config format and DiT-specific gotchas (state
  assembly, RAMEN stats path, loss dims, threshold calibration).
- Promotion vocabulary and note template.
- The eval log checklist (what to record after a backtest).

Previously:

Only update operational docs after Steps 1-9 expose what actually works.

### Step 11: Add Thin Helpers Only If Repetition Is Proven

Status: deferred.

Possible helpers:

- Generate an eval log skeleton.
- Generate a promotion note skeleton.
- Summarize checkpoint promotion notes in a local markdown report.
- Check that promotion notes reference real eval logs and signed artifacts.

Exit criteria:

- Each helper removes repeated manual work observed in Steps 2-9.
- Helpers consume existing files and do not create a new source of truth.

### Step 12: Adversarial Inference Run

Status: `plan written` — split across two executable plans:

- `docs/plans/2026-05-01-adversarial-inference-run-hybrid-preflight.md`
  (setup): hybrid preflight framing, schema-driven coverage matrix,
  deployment-protocol augmentations (Phase 1), trial harness setup
  (Phase 2), and ranked backlogs.
- `docs/plans/2026-05-01-adversarial-inference-run-trials.md` (trials):
  30 fault-injection trials across 4 tiers (Phase 3) and the synthesis
  writeup (Phase 4).

Execute the setup plan first, then move to the trials plan. Each plan's
Status section tracks its own phases.

Red-team the passport-based inference protocol. Deliberately introduce one
fault at a time into a signed checkpoint's environment, then hand it to a
fresh agent session that has never seen the checkpoint and only has the
passport and operational skills. The agent should follow the standard
passport-based inference protocol from scratch. The question is: does the
protocol surface the fault, or does the agent silently proceed?

Faults to inject (one per trial):

- Wrong library version (e.g. transformers minor version mismatch).
- Swapped or missing camera input (wrong image key or resolution).
- Stale or mismatched normalization stats.
- Truncated or corrupted weight file (single tensor zeroed).
- Wrong action space scaling or delta-vs-absolute mismatch.
- Tampered `SIGNOFF.json` (valid JSON, wrong hash).
- Missing a file listed in the passport manifest.
- Incorrect dtype (float16 vs bfloat16).

Each trial should record:

- Which fault was injected.
- Whether the agent's protocol caught it, and at which step.
- Whether the failure was surfaced as a hard gate, a soft signal, or missed.
- How far the agent got before stopping or producing wrong output.

Exit criteria:

- Every injected fault is either caught by the protocol or logged as a known
  gap that needs a new validator or passport field.
- The passport and inference protocol are updated to close any gaps found.

### Step 13: End-To-End Agent Automation Trial

Status: deferred.

Once the manual lifecycle is stable and the adversarial run has closed
protocol gaps, test whether an external agent (OpenClaw, Hermes, or
equivalent) can execute the full checkpoint triage loop from a single
natural-language request — e.g. a user asking through Slack "evaluate the
latest checkpoint from this training run."

The agent should, without human hand-holding:

1. Locate the checkpoint and its signed artifacts.
2. Run the artifact gate.
3. Run cheap behavior checks.
4. Run the offline backtest on pinned data.
5. Write the eval log.
6. Draft a promotion note.
7. Report the result back to the requesting channel.

Trial criteria:

- The agent has access only to the operational skills, passport tooling, and
  eval backend — no roadmap, no prior chat history, no hints.
- The human reviews the output for correctness, not to steer execution.
- Record where the agent got stuck, hallucinated, or asked unnecessary
  clarifying questions.

Exit criteria:

- The agent can complete the loop end-to-end on a known-good checkpoint, or
  the failure points are documented as skill/tooling gaps to fix first.
- Do not ship this as a production workflow until at least 3 checkpoint
  batches have succeeded without human correction.
