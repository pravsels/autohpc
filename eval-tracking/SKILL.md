---
name: eval-tracking
description: Use when evaluating checkpoints to maintain a persistent log of what was evaluated, the metrics, qualitative assessment, and verdict.
---

# Eval Tracking

## Overview

Use this when evaluating a trained checkpoint — running it against a dataset to assess quality. Each evaluation gets its own log file that records what was evaluated, the quantitative results, a qualitative assessment, and a verdict.

Eval logs are separate from run logs. Run logs track training; eval logs track evaluation of the artifacts that training produces.

## When to Use

- Evaluating a checkpoint against a test or validation set
- Running inference to assess reconstruction quality, generation quality, etc.
- Comparing eval results across checkpoints to decide which to keep

## Eval Logs

Eval logs live in `eval_logs/` in the target repo, parallel to `run_logs/`. One markdown file per evaluation.

### Directory structure

Same convention as run logs — use subdirectories when there are distinct components, flat when simple. Each subdirectory gets a `timeline.md` with human-readable dates.

```
eval_logs/
  encoder/
    timeline.md
    2026-03-16_recon_quality_job-2875100.md
    ...
  decoder/
    timeline.md
    ...
```

### Creating an eval log

Create the file when you submit the eval job. Same naming convention as run logs: `<date>_<task>_job-<jobid>.md`.

Include at minimum:

```markdown
# eval — <short description>

## Provenance
- checkpoint: `<path to checkpoint being evaluated>`
- source run log: `<path to run log that produced this checkpoint>`
- config_snapshot: `<path to resolved config from the training run>`
- dataset: `<actual filename>` at `<path>` (if hosted online, link: `<URL>`)

## Job
- job_id: <filled after sbatch>
- submitted/start: `<ISO timestamp>`
- start_human: `<Wednesday, Feb 25th, 2026>`
- end: `<ISO timestamp>`
- end_human: `<Thursday, Feb 26th, 2026>`
- runtime: `<HH:MM:SS>`
- node: <filled from squeue/logs>

## Metrics
<quantitative results — whatever metrics the eval produces>

## Qualitative
<subjective assessment of outputs — discuss with user>

## Verdict
<one-liner: is this checkpoint good enough? what's the recommendation?>

## Next
```

### Provenance

The Provenance section links the eval back to its source. Always include:
- The exact checkpoint path being evaluated
- The run log that produced that checkpoint (so you can trace back to training config, loss curves, etc.)
- The config snapshot from the training run
- The dataset used for evaluation (actual filename, not just a mount path)

### Metrics

Record whatever quantitative results the eval produces. This is task-dependent — val_loss, accuracy, FID, MSE, PSNR, etc. Include the metric names so they're unambiguous.

### Qualitative

A subjective assessment of the outputs, written together with the user after reviewing them. Examples:

```markdown
## Qualitative
- Reconstructions look sharp, colors accurate, fine details preserved.
- Front camera reconstructions are good but wrist camera is blurry.
- Generated frames are temporally consistent but objects have soft edges.
- Predictions diverge from ground truth after ~5 steps.
```

### Verdict

A one-liner recommendation. This is the first thing someone reads when they open the eval log — it should answer "is this checkpoint usable?" Examples:

```markdown
- verdict: Good enough for Stage 2 training. Encoder/decoder quality is solid.
- verdict: Not ready. Reconstruction quality degrades badly on wrist camera.
- verdict: Marginal. Loss is reasonable but qualitative check shows artifacts.
```

## Common Mistakes

- Eval log without provenance — then you can't trace metrics back to the checkpoint or training run that produced them
- Skipping the qualitative section — metrics alone don't capture perceptual quality issues
- Not recording a verdict — then you have to re-analyze the eval to decide if the checkpoint is usable
- Not linking to the source run log — then you lose the connection between eval results and training dynamics
