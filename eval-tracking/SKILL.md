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

**Prerequisite:** the checkpoint must already have a `MODEL_PASSPORT.json` and a passing `SIGNOFF.json` at its root. The eval harness consumes the passport's `input_contract` to drive how the model is fed (image dtype, value range, color order, channel layout, state sub-key layout, action post-processing) — without it, you risk silently mis-feeding the model and blaming the model for what's a feeding bug. If the checkpoint has no signoff, stop and follow `checkpoint-passport/SKILL.md` (in this repo) first. As a defensive gate, the eval job's startup should run `validate-checkpoint <ckpt_dir> --require-signoff` and refuse to proceed on non-zero exit.

## Eval Logs

Eval logs live in `eval_logs/` in the target repo, parallel to `run_logs/`. One markdown file per evaluation.

### Directory structure

Same convention as run logs — group into subdirectories by component, task variant, or pipeline stage. Create subdirs proactively when you know variants are coming. Each subdirectory gets a `timeline.md` with a header matching the subdirectory and human-readable dates.

```
eval_logs/
  arx5_multitask/
    timeline.md
    2026-03-16_recon_quality.md
    ...
  libero_subtask/
    timeline.md
    ...
```

### Creating an eval log

Create the file when you submit the eval job. Same naming convention as run logs: `<date>_<task>.md`. Job IDs go inside the file, not in the filename.

Include at minimum:

```markdown
# eval — <short description>

## Provenance
- checkpoint: `<path to checkpoint being evaluated>`
- passport: `<path to checkpoint>/MODEL_PASSPORT.json` (signoff verdict: `pass` | `soft_signal`)
- source run log: `<path to run log that produced this checkpoint>`
- config_snapshot: `<path to resolved config from the training run>`
- dataset: `<actual filename>` at `<path>` (if hosted online, link: `<URL>`)

## Job
- execution_id: <Slurm job_id, or instance_name/zone for cloud VMs>
- submitted/start: `<ISO timestamp>`
- start_human: `<Wednesday, Feb 25th, 2026>`
- end: `<ISO timestamp>`
- end_human: `<Thursday, Feb 26th, 2026>`
- runtime: `<HH:MM:SS>`
- node: <from squeue/logs> (Slurm only — for cloud VMs the instance is already in execution_id)

## Metrics
<quantitative results — whatever metrics the eval produces>

## Qualitative
<subjective assessment of outputs — discuss with user>

## Verdict
<one-liner: is this checkpoint good enough? what's the recommendation?>

## Next
```

### Monitoring an eval job

The same health check applies as for training jobs: a running process with GPU activity is not proof that eval is progressing. Check the output log for advancing progress (samples processed, metrics being computed) and the error log for failures. On Slurm, check `slurm-<job_id>.out` and `.err`. On cloud VMs, check the persisted log file or `docker logs`.

### Provenance

The Provenance section links the eval back to its source. Always include:
- The exact checkpoint path being evaluated
- The passport that was loaded (and the signoff verdict it shipped with) — pins which feeding contract this eval was actually run against, so a future passport revision can't silently re-interpret these results
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

If the verdict marks the checkpoint as deploy-ready, the passport is already in place (it was generated post-train, before this eval ran) and the same `validate-checkpoint <ckpt_dir> --require-signoff` gate the eval harness used will gate the next consumer too. The verdict here is purely about *performance*; the *intact files + known feeding contract* story was already settled by the signoff.

## Promotion Notes

A promotion note is a **checkpoint-level decision** informed by one or more eval
logs. Where an eval log's **Verdict** answers "is this specific eval's result
good enough?", a promotion note's **Decision** answers "given all available
evidence, what should happen next with this checkpoint?"

Promotion notes live alongside eval logs in `eval_logs/`, in the same
subdirectory as the eval logs they reference. Name them
`<date>-promotion-note.md`.

### Promotion vocabulary

Use exactly one of these actions per promotion note:

- `reject` — archive with a short reason. No further eval planned unless new
  evidence appears.
- `needs_more_eval` — evidence is missing, weak, or contradictory. The next
  action is another specific eval, not deployment.
- `promote_to_sim` — cheap checks and offline evidence justify spending
  simulation compute.
- `promote_to_preflight` — eval and/or simulation evidence justify a deployment
  preflight audit.
- `candidate_for_robot` — deployment preflight passed and a controlled robot
  run is the next justified action.

Do not create a numeric score. The vocabulary is deliberately conservative —
each action names the next concrete step, not an abstract quality level.

### Promotion note template

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

### Guidance

- **Make uncertainty visible.** List missing evidence explicitly — do not hide
  gaps behind a confident-sounding action. A `needs_more_eval` with clear
  "missing_evidence" is more useful than a premature `promote_to_sim`.
- **Each note has exactly one next step.** If the next action is "run more eval",
  say which eval. If the next action is "none", say why.
- **Contradictory evidence is normal.** If some signals are positive and others
  negative, list both and explain the interpretation in Rationale. Do not
  suppress contradictions.
- **Reference eval logs, don't repeat them.** The promotion note should link to
  eval logs for detailed metrics, not duplicate them. Keep the note focused on
  the decision and its justification.

## Common Mistakes

- Eval log without provenance — then you can't trace metrics back to the checkpoint or training run that produced them
- Skipping the qualitative section — metrics alone don't capture perceptual quality issues
- Not recording a verdict — then you have to re-analyze the eval to decide if the checkpoint is usable
- Not linking to the source run log — then you lose the connection between eval results and training dynamics
- Writing a promotion note without referencing specific eval logs — the note becomes an unsupported opinion
- Hiding missing evidence behind a confident promotion action — be honest about what you don't know
