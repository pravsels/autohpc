---
name: hpc-run-tracking
description: Use when submitting, monitoring, or reviewing HPC training runs to maintain a persistent log of each run's config, status, results, and next steps.
---

# HPC Run Tracking

## Overview

Use this after Phase 3 setup is complete and you are in the ongoing submit/monitor/iterate loop. Every job submission gets a run log file that tracks what was submitted, what happened, and what to do next.

There are three modes of operation:

- **Replication** — verifying the repo's training works on your cluster with your data. Usually a small number of runs. The goal is "does this work?" Once it does, you're done or you move to experiments.
- **Experiments** — iterating on the setup: trying different hyperparameters, architectures, data configurations. The goal is "which variation is best?" Many runs, compared against each other.
- **Pipeline** — supporting jobs that feed into training but aren't training themselves: embedding generation, data preprocessing, format conversion, dataset construction. The goal is "produce an artifact that a training run needs." Use a parenthetical to specify the kind, e.g. `pipeline (data generation)`, `pipeline (embedding extraction)`, `pipeline (format conversion)`.

All three use the same run log format. Experiments add a comparison summary.

## When to Use

- Submitting a training, eval, or pipeline job
- Checking on a running or completed job
- Resuming a run after walltime interruption
- Comparing experiment results to decide what to try next

## Cluster Access

When checking remote run status, use the cluster profile and try direct SSH yourself before asking the user for command output. For Isambard, the user may already have run `clifton auth`; try `ssh isambard "<command>"` first and only ask the user to refresh Clifton auth after a current SSH attempt fails.

## Agent Algorithm

Use this lifecycle algorithm. The sections below define the log format and
reference details.

1. **Preflight**
   - Read the target repo's existing `run_logs/`, `slurm/`, configs, and recent
     run log timelines.
   - Identify run type: replication, experiment, or pipeline.
   - Identify cluster profile and remote paths if Slurm/cloud state is involved.

2. **On submit**
   - Create or update the run log before/at submission.
   - Record objective, script/command, config, dataset, key settings, code state,
     and submission time.
   - Submit using the repo's normal training/eval command.
   - Record execution ID immediately.

3. **On poll**
   - Query scheduler/VM state directly; do not ask the user for output unless
     access fails.
   - Check real progress, not just "running": output log step counts, loss/metric
     movement, error log, disk usage, and GPU telemetry where available.
   - Append timestamped status with concrete evidence.

4. **On complete**
   - Capture runtime, final status/exit code, final step/metrics, checkpoint path,
     config snapshot, W&B local dir and synced URL if available.
   - Fill `Results`, W&B notes, and `Next`.
   - Do not recommend eval/upload before checkpoint passport/signoff.

5. **On resume**
   - If resuming the same objective/config after walltime or interruption, append a
     `Job (resumed)` block to the same run log.
   - Only create a new run log for a meaningfully different config/data/task.

6. **On publish or handoff**
   - Follow `checkpoint-passport/SKILL.md` first.
   - Sanitize any public `TRAINING_LOG.md` by removing cluster-specific paths,
     node names, job IDs, and secret-adjacent details while keeping training
     dynamics and W&B links.

7. **Status updates**
   - During long remote jobs, report what is done, what is running, what remains,
     and the last observed scheduler/log evidence.

## Run Logs

Run logs live in `run_logs/` in the target repo. One markdown file per run.

### Directory structure

Group run logs into subdirectories when there are distinct groupings — by component (encoder, decoder), by task variant (arx5_multitask, libero_subtask), or by pipeline stage. Discuss the grouping with the user — the right split depends on the project. Create subdirs proactively when you know variants are coming, don't wait for the flat list to get messy.

```
run_logs/
  arx5_multitask/
    timeline.md
    2026-03-22_train.md
    ...
  libero_subtask/
    timeline.md
    ...
```

For simpler repos with a single training target, a flat `run_logs/` is fine.

Each subdirectory gets a `timeline.md` — a chronological index of all runs in that group with human-readable dates. The timeline header should match the subdirectory:

```markdown
# ARX5 Multitask Timeline

1. `2026-02-20_train.md` — Friday, Feb 20th
2. `2026-02-21_train.md` — Saturday, Feb 21st
3. `2026-02-25_eval.md` — Wednesday, Feb 25th
```

### Creating a run log

Create the file when you submit the job. Name it `<date>_<task>.md` — date-prefix ensures chronological sorting in the file explorer. Job IDs go inside the file, not in the filename, since they change on every resubmit.

Examples: `2026-03-11_train_front_cam.md`, `2026-03-15_eval.md`.

Include at minimum:

```markdown
# <task> — <short description>

## Mode
- run_type: <replication, experiment, or pipeline (subtype)>
- objective: <one line — what this run is trying to verify or test>

## Config
- script: `slurm/<script>.sh` or `docker run` command
- config: `configurations/<config>.yaml`
- dataset: `<actual filename>` at `<path>` (if hosted online, link: `<URL>`)
- key settings: <whatever matters for this run — learning rate, batch size, resume, etc.>

## Job
- execution_id: <Slurm job_id, or instance_name/zone for cloud VMs>
- submitted/start: `<ISO timestamp>`
- start_human: `<Wednesday, Feb 25th, 2026>`
- end: `<ISO timestamp>`
- end_human: `<Thursday, Feb 26th, 2026>`
- runtime: `<HH:MM:SS>`
- node: <from squeue/logs> (Slurm only — for cloud VMs the instance is already in execution_id)

## Status

## Results

## W&B
- local: `<offline run dir, e.g. wandb/offline-run-...>`
- synced: `<URL after wandb sync, e.g. https://wandb.ai/team/project/runs/abc123>`
- notes: <brief qualitative read of the curves — discuss with user after reviewing the dashboard>

## Next
```

### Updating a run log

When checking on a job, append evidence-backed entries to **Status**. A running
process with GPU activity is not proof of progress; check logs for advancing
steps/metrics and errors before recording health.

```markdown
## Status
- 2026-03-11 15:00 — running, step 2400, train_loss 0.085
- 2026-03-12 14:30 — completed, exit code 0
```

When the job finishes, fill in **Results** and **W&B**:

```markdown
## Results
- runtime: `<HH:MM:SS>` (start `<ISO timestamp>`, end `<ISO timestamp>`)
- final step: 100000
- start_train_loss: `<first logged value>`
- end_train_loss: `<last logged value>`
- start_val_loss: `<first logged value or n/a>`
- end_val_loss: `<last logged value or n/a>`
- loss_one_liner: <one-sentence qualitative summary of the loss progression>
- checkpoint: `<path to checkpoint on remote storage>`
- config_snapshot: `<path to resolved config from run output>`

## W&B
- local: `wandb/offline-run-20260311_150000-abc123`
- synced: `https://wandb.ai/team/project/runs/abc123`
- notes: <brief qualitative read of the curves>
```

Use the exact config snapshot from the run output; repo config files can drift.
Make `loss_one_liner` a human-readable takeaway, not a restatement of numbers.
If W&B is not yet synced, write `pending — run wandb sync <local>`.

### W&B sync

`wandb` is typically not installed on the host. Run `wandb sync` inside the container:

```bash
# Slurm / Apptainer
apptainer exec --bind /scratch/... <container.sif> bash -lc \
  'export WANDB_API_KEY="$(cat ~/.wandb_key)"; wandb sync <offline-run-dir>'

# Cloud VM / Docker
docker run --rm --network=host \
  -e WANDB_API_KEY="$(cat ~/.wandb_key)" \
  -v /path/to/repo:/workspace/repo \
  <image>:<tag> wandb sync /workspace/repo/wandb/<offline-run-dir>
```

If sync fails with "No API key configured", ask the user to place their key in a dotfile (e.g. `~/.wandb_key`). Don't hardcode the key in scripts or run logs.

### Per-job-block W&B URLs

When a run log has multiple job blocks (original + resumptions), record the synced URL in each job block it belongs to, not only in the W&B section at the bottom. This way you can find the right dashboard from whichever block you're reading without scrolling.

Suggest next steps in **Next**:

```markdown
## Next
- resume for more steps: edit `LOAD_CKPT_PATH` in the sbatch script (or pass checkpoint path to `docker run`), then resubmit
- or generate `MODEL_PASSPORT.json` + `SIGNOFF.json` for the checkpoint (see `checkpoint-passport/SKILL.md`), then eval / upload / hand off
```

**Do not** suggest "eval this checkpoint" or "upload this checkpoint" as the next step without passport generation in between — the passport is a hard prerequisite for both. The eval harness reads the passport's `input_contract` to drive how the model is fed; uploading without a passport puts an unsigned snapshot on HF that downstream consumers cannot verify.

### Resumptions

When resuming a walltime-interrupted run, don't create a new file. Append to the same run log:

```markdown
## Job (resumed)
- execution_id: 12346
- submitted: 2026-03-12 15:00 UTC
- resumed from: `<path to checkpoint>`
```

Only create a new file when the run represents a meaningfully different experiment (different config, different data, different task).

## Replication Runs

For replication, a single run log per training task is usually enough. The goal is confirming the setup works end-to-end. Once results look reasonable, you're done — note the outcome and move on.

## Experiment Runs

For experiments, each variation gets its own run log. Additionally, maintain a `run_logs/experiments.md` summary that compares results across runs:

```markdown
# Experiments

| run | description | val_loss | status | notes |
|-----|-------------|----------|--------|-------|
| train_baseline_2026-03-11 | front cam, default LR | 0.028 | keep | baseline |
| train_lr_sweep_2026-03-12 | LR 1e-3 -> 5e-4 | 0.025 | keep | small improvement |
| train_both_cams_2026-03-13 | front + wrist | 0.031 | discard | worse than single cam |
```

Before submitting a new experiment, read `run_logs/experiments.md` and recent run logs to understand what's been tried and what worked. Use this to decide what to try next — don't repeat failed variations.

Each experiment run should be on its own git branch or tagged commit so you can recover the exact code that produced a given result.

## Publishing Checkpoints

When a checkpoint is worth sharing or backing up, follow
`checkpoint-passport/SKILL.md` first. Do not upload, copy, eval, or hand off a
checkpoint before `MODEL_PASSPORT.json` and `SIGNOFF.json` exist and validate.

The public package normally contains:

```
README.md                    # Model card: description, config, loss table, usage
TRAINING_LOG.md              # Sanitized run log (see below)
MODEL_PASSPORT.json          # Feeding contract + integrity manifest (generated by checkpoint-passport)
SIGNOFF.json                 # Verdict + sha256 of passport and weight files (generated by sign-checkpoint)
assets/                      # Norm stats, dataset list, valid indices, etc.
checkpoints/<step>/params/   # Model weights (inference + fine-tuning)
```

Create `TRAINING_LOG.md` from the run log, stripped of cluster-specific details: Slurm job IDs, node names, scratch paths, `wandb_id.txt` references. Keep the training dynamics, config, loss progression, and W&B link.

### W&B visibility

If the main W&B project is private but you want to share training curves, sync the run to a separate public project:

```bash
wandb sync --project <public-project-name> <offline-run-dir>
```

Then set the project to public in the W&B UI. Link this public URL in the README, not the private project URL.

### Recording in the run log

After uploading, add a `## HuggingFace` section to the run log:

```markdown
## HuggingFace
- repo: https://huggingface.co/<user>/<repo>
- uploaded checkpoints: <which steps, params only or full>
- includes: README, TRAINING_LOG, MODEL_PASSPORT.json, SIGNOFF.json, assets, ...
- signoff verdict: `pass` | `soft_signal` (if soft_signal, copy the verdict_reason here verbatim)
```

## Common Mistakes

- Not creating a run log — then you forget what config produced which checkpoint
- Creating a new file for every resumption of the same run
- Logging status without the step count or loss — timestamps alone aren't useful
- Forgetting to record the checkpoint path in results — makes resumption a guessing game
- Not recording run_type (replication vs experiment vs pipeline) — makes intent unclear when reviewing later
- Running experiments without updating the comparison summary — then you lose track of what's been tried
- Not branching/tagging experiment code — then you can't recover what produced a good result
- Not recording the W&B synced URL — then you have to hunt through `wandb/` dirs or re-sync to find training curves
- Putting the synced URL only in the W&B section when there are multiple job blocks — then you can't find it from the block you're reading
- Naming files `<task>_<date>.md` instead of `<date>_<task>.md` — breaks chronological sorting in file explorers
- Dumping all run logs flat in `run_logs/` instead of grouping by project — becomes unreadable past ~10 files
- Only recording ISO timestamps without human-readable dates — forces mental parsing every time you open a file
- Not maintaining `timeline.md` per subdirectory — then you have to open individual files to reconstruct order
- Uploading a checkpoint to HF (or copying it to an eval box) without first generating `MODEL_PASSPORT.json` + `SIGNOFF.json` — the snapshot is permanently unsigned, downstream consumers can't verify the feeding contract, and any cached copies people pulled in the interim will never get a passport
- Rolling a custom sha256-of-sha256s manifest in the README instead of using `sign-checkpoint` — loses the verdict, loses the per-file breakdown, and the consumer-side `validate-checkpoint --require-signoff` gate doesn't work against it
