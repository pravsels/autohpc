---
name: hpc-run-tracking
description: Use when submitting, monitoring, or reviewing HPC training runs to maintain a persistent log of each run's config, status, results, and next steps.
---

# HPC Run Tracking

## Overview

Use this after Phase 3 setup is complete and you are in the ongoing submit/monitor/iterate loop. Every job submission gets a run log file that tracks what was submitted, what happened, and what to do next.

There are two modes of operation:

- **Replication** — verifying the repo's training works on your cluster with your data. Usually a small number of runs. The goal is "does this work?" Once it does, you're done or you move to experiments.
- **Experiments** — iterating on the setup: trying different hyperparameters, architectures, data configurations. The goal is "which variation is best?" Many runs, compared against each other.

Both use the same run log format. Experiments add a comparison summary.

## When to Use

- Submitting a training or eval job
- Checking on a running or completed job
- Resuming a run after walltime interruption
- Comparing experiment results to decide what to try next

## Run Logs

Run logs live in `runs/` in the target repo. One markdown file per run.

### Creating a run log

Create the file when you submit the job. Name it `<task>_<short_description>_<date>.md`, e.g. `train_front_cam_2026-03-11.md`, `eval_checkpoint_50k_2026-03-15.md`.

Include at minimum:

```markdown
# <task> — <short description>

## Mode
- run_type: <replication or experiment>
- objective: <one line — what this run is trying to verify or test>

## Config
- script: `slurm/<script>.sh`
- config: `configurations/<config>.yaml`
- dataset: <path on scratch>
- key settings: <whatever matters for this run — learning rate, batch size, resume, etc.>

## Job
- job_id: <filled after sbatch>
- submitted: <timestamp>
- node: <filled from squeue/logs>

## Status

## Results

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
- final step: 100000
- val_loss: 0.028
- checkpoint: `/scratch/.../checkpoints/step_100000.pt`
- wandb: `runs/offline-run-...' (sync with `wandb sync`)
```

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

For experiments, each variation gets its own run log. Additionally, maintain a `runs/experiments.md` summary that compares results across runs:

```markdown
# Experiments

| run | description | val_loss | status | notes |
|-----|-------------|----------|--------|-------|
| train_baseline_2026-03-11 | front cam, default LR | 0.028 | keep | baseline |
| train_lr_sweep_2026-03-12 | LR 1e-3 -> 5e-4 | 0.025 | keep | small improvement |
| train_both_cams_2026-03-13 | front + wrist | 0.031 | discard | worse than single cam |
```

Before submitting a new experiment, read `runs/experiments.md` and recent run logs to understand what's been tried and what worked. Use this to decide what to try next — don't repeat failed variations.

Each experiment run should be on its own git branch or tagged commit so you can recover the exact code that produced a given result.

## Common Mistakes

- Not creating a run log — then you forget what config produced which checkpoint
- Creating a new file for every resumption of the same run
- Logging status without the step count or loss — timestamps alone aren't useful
- Forgetting to record the checkpoint path in results — makes resumption a guessing game
- Not recording run_type (replication vs experiment) — makes intent unclear when reviewing later
- Running experiments without updating the comparison summary — then you lose track of what's been tried
- Not branching/tagging experiment code — then you can't recover what produced a good result
