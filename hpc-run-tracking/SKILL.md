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
- script: `slurm/<script>.sh`
- config: `configurations/<config>.yaml`
- dataset: `<actual filename>` at `<path on scratch>` (if hosted online, link: `<URL>`)
- key settings: <whatever matters for this run — learning rate, batch size, resume, etc.>

## Job
- job_id: <filled after sbatch>
- submitted/start: `<ISO timestamp>`
- start_human: `<Wednesday, Feb 25th, 2026>`
- end: `<ISO timestamp>`
- end_human: `<Thursday, Feb 26th, 2026>`
- runtime: `<HH:MM:SS>`
- node: <filled from squeue/logs>

## Status

## Results

## W&B
- local: `<offline run dir, e.g. wandb/offline-run-...>`
- synced: `<URL after wandb sync, e.g. https://wandb.ai/team/project/runs/abc123>`
- notes: <brief qualitative read of the curves — discuss with user after reviewing the dashboard>

## Next
```

### Updating a run log

When checking on a job (via `squeue`, log tailing, or `sacct`), append to the **Status** section:

```markdown
## Status
- 2026-03-11 15:00 — running, step 2400, train_loss 0.085
- 2026-03-11 18:00 — running, step 28000, train_loss 0.041
- 2026-03-12 14:30 — completed, exit code 0
```

When the job finishes, fill in **Results**:

```markdown
## Results
- runtime: `<HH:MM:SS>` (start `<ISO timestamp>`, end `<ISO timestamp>`)
- final step: 100000
- start_train_loss: `<first logged value>`
- end_train_loss: `<last logged value>`
- start_val_loss: `<first logged value or n/a>`
- end_val_loss: `<last logged value or n/a>`
- loss_one_liner: <one-sentence qualitative summary of the loss progression>
- checkpoint: `/scratch/.../checkpoints/step_100000.pt`
- config_snapshot: `<path to resolved config from run output>`
```

When archiving a checkpoint, include the exact config snapshot from the run output, not a reference to the repo config file — repo configs are mutable and may not match what actually produced the checkpoint.

The `loss_one_liner` should be a brief human-readable takeaway, not a restatement of numbers. Examples:

```markdown
- loss_one_liner: Train loss dropped steadily from 0.82 to 0.10; val loss flat at 0.15, likely plateaued.
- loss_one_liner: Both losses decreased healthily with no sign of overfitting.
- loss_one_liner: Train loss decreased but val loss crept up after step 20k — overfitting.
- loss_one_liner: Loss metrics were not logged for this run.
```

And fill in the **W&B** section. Record the local offline dir immediately, then add the synced URL after running `wandb sync`:

```markdown
## W&B
- local: `wandb/offline-run-20260311_150000-abc123`
- synced: `https://wandb.ai/team/project/runs/abc123`
```

The synced link is a clickable URL to the W&B dashboard where training dynamics (loss curves, metrics, system stats) can be inspected. If not yet synced, leave it as `pending — run wandb sync <local>`.

### W&B sync on HPC

On HPC, `wandb` typically isn't installed on the host. Run `wandb sync` inside the container:

```bash
apptainer exec --bind /scratch/... <container.sif> bash -lc \
  'export WANDB_API_KEY="$(cat ~/.wandb_key)"; wandb sync <offline-run-dir>'
```

If an interactive `apptainer exec` isn't available, use a short `srun` allocation.

If sync fails with "No API key configured", ask the user to place their key in a dotfile on the cluster (e.g. `~/.wandb_key`). Don't hardcode the key in scripts or run logs.

### Per-job-block W&B URLs

When a run log has multiple job blocks (original + resumptions), record the synced URL in each job block it belongs to, not only in the W&B section at the bottom. This way you can find the right dashboard from whichever block you're reading without scrolling.

After syncing, review the dashboard with the user and add a `notes` field — a brief qualitative read of how training went. This is subjective and best written together. Examples:

```markdown
- notes: loss drops steadily, no plateau, looks healthy
- notes: loss plateaued around step 40k, minimal improvement after that
- notes: loss unstable in first 5k steps then settled, final value reasonable
- notes: val_loss diverged from train_loss around step 20k, possible overfitting
```

The point is that months later, you (or someone else) can open the log and get the takeaway without re-opening the dashboard.

And suggest next steps in **Next**:

```markdown
## Next
- resume for more steps: `sbatch --export=ALL,LOAD_CKPT_PATH=/scratch/.../step_100000.pt slurm/...`
- or start eval run with this checkpoint
```

### Resumptions

When resuming a walltime-interrupted run, don't create a new file. Append to the same run log:

```markdown
## Job (resumed)
- job_id: 12346
- submitted: 2026-03-12 15:00 UTC
- resumed from: `/scratch/.../checkpoints/step_50000.pt`
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
