# AutoHPC

Take any AI / ML repo from local Docker build to HPC training and eval — with an AI agent doing the work.

## Quick Start

Clone this repo alongside your target ML repo, then paste this prompt into your AI coding agent:

```
Read ../autohpc/README.md. Assess the current phase (see phase table), confirm with user, then follow the matching skill or README section.
```

Adjust the path if your clone location differs.

## Repo Layout

- `hpc-container-promotion/SKILL.md`
  - Local Docker build/test and promotion to the cluster-native container artifact/runtime.
- `hpc-training-operations/SKILL.md`
  - Slurm submission, monitoring, observability, and debugging (Slurm targets only).
- `hpc-dataset-adaptation/SKILL.md`
  - Adapting code to read the user's dataset format without converting data.
- `hpc-run-tracking/SKILL.md`
  - Maintaining per-run logs for submitted jobs: config, status, results, next steps.
- `eval-tracking/SKILL.md`
  - Maintaining per-eval logs for checkpoint evaluations: provenance, metrics, qualitative assessment, verdict.
- `checkpoint-passport/`
  - Skill **and** runnable Python package for producing and verifying `MODEL_PASSPORT.json` / `SIGNOFF.json`. Generated **right after training, before the checkpoint moves anywhere** (HF upload, copy to eval box, copy to robot, hand to colleague) — every downstream consumer, including the eval harness itself, reads the passport. Installable: `uv pip install -e checkpoint-passport`.
- `deployment-protocol/`
  - Docs-only skill for first-run / fresh-run deployment preflight on a real robot or inference rig. Uses `MODEL_PASSPORT.json`, the target repo's real runner code, and live device samples to verify the whole chain before the first run.
- `autoresearch/`
  - Git submodule for autonomous post-baseline experiment loops once replication runs and evals are stable.
- `cluster-profiles/`
  - One file per cluster with docs links and cluster-specific notes.

Most skill folders are docs-only (one `SKILL.md` describing commands the agent already has — `docker`, `sbatch`, etc.). A few skills also ship bespoke tooling alongside the SKILL.md (e.g. `checkpoint-passport/` ships a pip-installable Python package). When a skill ships code, it lives inside that skill's folder so the workflow and the tool stay in lockstep.

## Adding A Cluster

Create `cluster-profiles/<cluster_name>.md` with:
- authoritative docs links
- scheduler/storage/container notes
- any cluster-specific caveats

Never store secrets in profile files.

## If You Are An Agent

This repo is a reference — read and follow the docs here, then apply them to whatever target repo you are working in. Do **not** copy or scaffold these files into the target repo.

The Docker image is the build artifact — for local work **and** for remote deployment. The runtime may be Docker or Apptainer depending on the target. Do not install dependencies on the host or use conda/mamba/venv as an alternative. Build the image first, run everything inside it.

Keep commit messages short — a few words, not a paragraph. Check `git log` in the target repo and match its style.

### Assessing current phase

When resuming work on a repo, check these signals to determine where you are before doing anything else:

| Signal | Phase |
|--------|-------|
| No Dockerfile or broken image build | Phase 1 — local Docker |
| Dockerfile works but training fails on user's data format | Phase 2 — dataset adaptation |
| Image works locally, no remote deployment done yet | Phase 3 — ask user for target environment |
| `slurm/` scripts exist, no `run_logs/` or empty `run_logs/` | Phase 3 — first submission (Slurm) |
| `run_logs/` has run logs with results | Ongoing — run tracking |
| A run log just landed (training finished, `wandb sync` succeeded) and the checkpoint dir has no `MODEL_PASSPORT.json` / `SIGNOFF.json` | Post-train — checkpoint passport |
| User is about to upload a checkpoint to HF / copy it to an eval box / copy it to a robot / hand it off, and there's no `MODEL_PASSPORT.json` / `SIGNOFF.json` | Post-train — checkpoint passport (gate it first) |
| Checkpoint dir has a passing `SIGNOFF.json` and the next step is the first run on a robot or inference rig, or a fresh run after hardware / runner / checkpoint changes | Post-passport — deployment protocol |
| Checkpoint dir has a passing `SIGNOFF.json` and `eval_logs/` has eval logs with metrics | Ongoing — eval tracking |
| Signed checkpoint needs a structured quality verdict and promotion decision (not just one-off eval logs) | Checkpoint triage (below — no separate SKILL.md) |

Report what you find and your best guess to the user (e.g. "Dockerfile exists and builds, slurm scripts are present, run_logs/ has 12 logged runs — looks like you're in the ongoing phase. Are you still focused on replication baselines, or are you now in experimentation mode?"). Wait for confirmation before continuing — the signals above are heuristics, and the user may know better (e.g. the image builds but is stale, run_logs exist from a previous attempt that was abandoned, or experiment logs exist even though the current goal is still baseline replication).

### Phase 1 — Local Docker

Read `hpc-container-promotion/SKILL.md` (in this repo) and follow Phase 1 for the target repo. Nothing else.

Do **not** read Phase 2 or 3 yet. Do **not** plan for dataset adaptation, remote deployment, ask which target to use, scan for Slurm scripts, or create cluster config files. Those are later concerns and you are not there yet.

Do not move to Phase 2 until the image builds and basic sanity checks pass inside the container using the repo's own data.

### Phase 2 — Dataset adaptation (skip if using repo data as-is)

Only begin this phase after Phase 1 is complete. If the user's data is already in the format the repo expects, skip to Phase 3.

Follow `hpc-dataset-adaptation/SKILL.md` (in this repo) to adapt the target repo's code to read the user's dataset. Do not move to Phase 3 until training runs end-to-end with the user's data.

### Phase 3 — Remote deployment

Only begin this phase after Phase 1 (and Phase 2 if applicable) is complete and the Docker image works locally.

The workflow is simple: push code, upload image and data, run training. Do not create helper scripts, wrapper scripts, or multi-stage pipelines. Run commands directly.

1. Identify the target environment and read `cluster-profiles/<cluster_name>.md` (in this repo). The profile determines the deployment path — not all targets are Slurm clusters.
2. **If the target is a cloud VM** (e.g. GCloud): the cluster profile is the workflow — follow it directly for instance creation, environment setup, building the image on the VM, and running training. Skip `hpc-container-promotion` Phase 3 and `hpc-training-operations` — they don't apply.
3. **If the target uses Slurm** (e.g. Isambard): follow Phase 3 of `hpc-container-promotion/SKILL.md` to promote the image, then `hpc-training-operations/SKILL.md` to write sbatch scripts and submit. The only scripts you should create in the target repo's `slurm/` are training and eval sbatch scripts.

Once training is running (container launched on a cloud VM, or job submitted on Slurm), move to the Ongoing phase below.

### Ongoing — Run tracking

Once you start running training, follow `hpc-run-tracking/SKILL.md` (in this repo) for every run. This is not a one-time setup step — it is an ongoing practice.

- Create a run log in `run_logs/` when you start a training run.
- Update it when checking status or collecting results.
- For replication runs, a single log per task is enough.
- For experiment runs, maintain a comparison summary across variations.

When a run finishes and `wandb sync` (or equivalent) confirms the training trace is intact, move to the next phase before doing anything else with the checkpoint.

### Post-train — Checkpoint passport

A trained checkpoint sitting on the training filesystem is not yet a deliverable. The instant anyone — the trainer, an eval job, a colleague, a robot, future-you — copies those files anywhere else, they need to know what the model expects on its inputs, what comes out, what's inside, and that the bytes haven't been tampered with. That contract lives in two artifacts at the checkpoint root: `MODEL_PASSPORT.json` and `SIGNOFF.json`.

**Trigger:** training run finished, `wandb sync` succeeded, the user is considering uploading to HF / copying to an eval box / copying to a robot / handing off. The passport is generated **before** any of those movements happen, not after.

**Why this ordering:** the eval harness itself is a passport consumer. It reads `input_contract` to know how to feed the model (image dtype, value range, color order, channel layout, state sub-key layout, action post-processing) instead of re-deriving it from training code or hardcoding guesses. If you eval first and passport later, an eval feeding bug will silently corrupt your eval numbers and you'll blame the model. If you upload to HF first and passport later, the HF snapshot is permanently unsigned and any cached copies people made in the interim never get a passport. Passport-then-move closes both holes.

Follow `checkpoint-passport/SKILL.md` (in this repo) to produce the passport, then sign it.

The skill ships a runnable Python package — install once per environment:

```bash
uv pip install -e ../autohpc/checkpoint-passport
```

Then for each checkpoint:

1. Generate `MODEL_PASSPORT.json` per the SKILL (mix of static file inspection and a single forward-pass smoke run inside the model's own container).
2. Run `validate-checkpoint <ckpt_dir>` and iterate until there are no hard failures (soft signals are OK if documented).
3. Run `sign-checkpoint <ckpt_dir> --reason '<one-liner if any soft signals>'` to write `SIGNOFF.json`.
4. From that point on, any consumer (eval harness, robot loader, colleague's repo) can run `validate-checkpoint <ckpt_dir> --require-signoff` as a load-time gate; non-zero exit means do not load this checkpoint.

Do **not** treat passport generation as optional. A checkpoint without a passing signoff has no integrity story — there is nothing to detect a corrupted weight file, a mismatched config, a silent dependency drift between training and inference, or an eval harness silently mis-feeding the model.

### Post-passport — Deployment protocol

Once a checkpoint has a passing `SIGNOFF.json`, and the next step is the
first run on a robot or inference rig (or a fresh run after hardware,
runner, bindings, or checkpoint changes), follow
`deployment-protocol/SKILL.md`.

This is the deployment-side gate. The skill tells the agent to read the
passport, inspect the target repo's real inference path, compare live
device samples against the passport's contract, and do one controlled
dry-run preflight so the whole chain is understood before the first run.

Use the target repo's own code/container for this work. Do not build a
generic `autohpc` package for deployment preflight unless the user
explicitly asks for one.

### Ongoing — Eval tracking

Once a checkpoint has a passing `SIGNOFF.json`, evaluating it is a separate, ongoing practice. Follow `eval-tracking/SKILL.md` (in this repo) for every eval.

- Create an eval log in `eval_logs/` for each evaluation.
- Record provenance (which checkpoint, which data), metrics, qualitative assessment, and verdict.
- The eval harness loads the checkpoint via the passport's `input_contract` — confirm `validate-checkpoint <ckpt_dir> --require-signoff` is part of the eval-job startup so a missing or stale signoff fails fast.

### Checkpoint triage

Unlike the other phases above (which each map to a single skill), triage is a
**multi-step workflow that chains several skills in order**. This section is
the checklist — follow it top to bottom, jumping into each referenced skill
as needed and returning here for the next step.

When a signed checkpoint needs a structured quality verdict — not just another
eval log, but a decision about what should happen next (more eval, simulation,
deployment preflight, or rejection) — follow this procedure.

**When this applies vs. "Ongoing — eval tracking":** eval tracking is for
recording individual evaluations. Checkpoint triage is the end-to-end flow
that produces a **promotion note** — a checkpoint-level decision informed by
one or more eval logs. If you already have eval logs and need a promotion
decision, start at step 5 below.

#### Environment setup

Before running any triage commands:

1. Install passport tools (once per environment):

   ```bash
   uv pip install -e ../autohpc/checkpoint-passport
   ```

2. Activate the target ML repo's Python environment. Use whatever the repo
   provides — micromamba, uv, venv, Docker. The model's own dependencies
   (torch, transformers, diffusers, etc.) must be available.

3. Guard against PYTHONPATH contamination. If the host shell exports
   incompatible site-packages (e.g. `/opt/ros/*/python3.x/site-packages`
   leaking into a different Python version), `unset PYTHONPATH` before
   running model code.

4. Check the passport's `runtime_constraints.required_versions` (if present)
   against the current environment before loading the model. Library version
   drift — especially `transformers` — is a known failure mode that wastes
   debugging time when the passport already has the answer.

#### Triage procedure

1. **Locate the checkpoint** and confirm training evidence is intact (W&B run,
   run log, config snapshot). If evidence is missing, stop and ask.

2. **Generate passport** if `MODEL_PASSPORT.json` does not exist — follow
   `checkpoint-passport/SKILL.md`. This requires the model's own
   container with GPU access for Phase 2 (dynamic extraction / smoke tests).

3. **Validate and sign** — follow `checkpoint-passport/SKILL.md` Phases 3-4.

4. **Establish the exact eval snapshot.** Either upload to HF and record the
   revision, or pin a local snapshot. Record the HF repo/revision or local
   path so there is no ambiguity about which bytes the eval consumes.

5. **Artifact gate:**

   ```bash
   validate-checkpoint <ckpt_dir_or_snapshot> --require-signoff
   ```

   Non-zero exit means stop. Record passport and signoff sha256 hashes.

6. **Cheap behavior checks.** Inspect the passport's `output_spec.smoke_results`
   (determinism, NaN/Inf, liveness, distribution, range). Optionally run a
   fresh single-batch forward pass through the target repo's public inference
   path and check output shapes, dtypes, value ranges, and liveness. This
   catches catastrophic brokenness, not task quality.

7. **Offline backtest** via MissionTracker (see below). If no usable eval
   backend exists, write an explicit eval-gap note and stop — do not skip
   this step silently.

8. **Write an eval log** — follow `eval-tracking/SKILL.md`.

9. **Write a promotion note** — follow the promotion notes section in
   `eval-tracking/SKILL.md`. This is the output of triage: a structured
   decision with evidence, not just metrics.

#### MissionTracker backtests

MissionTracker is the primary offline eval backend. It lives in the target ML
repo under `missiontracker/` and is not pip-installable — import it via
`sys.path` from the repo checkout.

**Config format.** Create a JSON config under
`missiontracker/examples/configs/`. Required fields:

- `VAL_DATASET_REPO_IDS` — HF dataset repo IDs (LeRobot format).
- `POLICY_PATHS` — HF repo IDs or local checkpoint paths.
- `VAL_EPISODES` — pin specific episodes for reproducibility, or `null` for all.

Architecture-specific fields (set as needed):

- `POLICY_ARCHITECTURE` — e.g. `"multitask_dit"`, `"act"`, `"rewact"`, `"openpi"`.
- `STATS_PATH` — path to normalization stats if the architecture needs external
  stats (e.g. RAMEN-format `ramen_stats.json`).
- `DEFAULT_LANGUAGE_INSTRUCTION` — text prompt for language-conditioned policies.

Set optional auxiliary model fields (`POLICY_SMALL_PATH`, `SAE_MODEL_REPO_ID`,
`REWARD_AE_PATH`, etc.) to `null` to disable extra checks you don't need.

**Running:**

```bash
cd <target_repo>/missiontracker/examples
python run_backtest.py configs/<your_config>.json
```

**Output.** A compressed artefact tarball under `eval_outputs/artefacts/`
containing `metrics.json`, `summary.json`, per-episode parquet files, and
videos. Key metrics to record in the eval log:

- `policy_loss` (mean, p50, p95) — primary reconstruction quality signal.
- `max_joint_delta` — action smoothness.
- `time_coherence` — temporal consistency of predictions.

**Caveats:**

- **Anomaly thresholds are architecture-specific.** Default thresholds are
  calibrated for the architecture family they were tuned on. For a new
  architecture or checkpoint family, use raw `metrics.json` numbers and
  ignore `summary.total_anomalies` until thresholds have been calibrated.
- **Stats format.** Checkpoints may ship LeRobot-format `dataset_stats.json`
  or RAMEN-format `ramen_stats.json`. The adapter detects the format
  automatically, but record which format was used in the eval log.
- **HF revision not recorded.** MissionTracker records `policy_path` but not
  the HF revision. Pin the revision externally before running and record it
  in the eval log.
- **Passport-blind feeding.** MissionTracker feeds the model from training
  dataset stats, not from the passport's `input_contract`. Record both the
  passport's `input_contract.training_datasets[]` and the dataset
  MissionTracker actually loaded in the eval log, so any divergence is
  visible.
- **Open-loop only.** This is offline replay (predicted vs ground-truth
  actions), not closed-loop simulation. Do not conflate backtest results
  with simulation evidence.

### After replication baselines are stable

Once replication training runs and checkpoint evals are working end-to-end, you can use the `autoresearch/` submodule to branch into experiment-driven work.

Treat this as a second phase after baseline reproduction: first make sure the original training setup is reproducible and the eval loop is trustworthy, then use `autoresearch` to run controlled variations and compare different ideas.

That split keeps replication and experimentation separate: use this repo's run and eval tracking to establish the baseline, then use `autoresearch` to drive higher-variance research experiments on top of that foundation.

Even in ongoing work, do not infer experimentation mode from files alone. Confirm with the user whether the current objective is still replication or whether they want to switch into experiment exploration.

## Acknowledgement

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
