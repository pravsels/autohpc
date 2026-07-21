---
name: eval-tracking
description: Use when evaluating checkpoints to maintain a persistent log of what was evaluated, the metrics, qualitative assessment, and verdict.
---

# Eval Tracking

## Overview

Use when evaluating a checkpoint. Eval logs record artifact provenance,
metrics, qualitative observations, and a verdict. Run logs track training; eval
logs track the artifacts training produced.

## When to Use

- Evaluating a checkpoint against a test/validation set.
- Running inference to assess quality.
- Comparing eval results before a promotion decision.

## Agent Algorithm

1. **Verify artifact integrity**
   - Run or require `verify-checkpoint <ckpt_dir>`.
   - If the CLI is unavailable, install/use
     `checkpoint-integrity` from this repo.
   - If a declared file is missing or changed, stop and follow
     `checkpoint-integrity/SKILL.md`.

2. **Create eval log**
   - Create `eval_logs/<group>/<date>_<task>.md` before submission.
   - Record checkpoint, manifest path/hash, source run log, config snapshot,
     dataset, command/script, and objective.

3. **Submit/run eval**
   - Use the repo's real eval entry point.
   - Ensure the eval uses the checkpoint's real saved config, processor, and
     normalization assets rather than reconstructing them from memory.
   - Record execution ID and start time.

4. **Monitor progress**
   - Check output logs for advancing sample counts or metrics.
   - Check error logs for failures.
   - Append status updates with concrete evidence.

5. **Complete the eval log**
   - Record runtime, metrics, artifact links, qualitative notes, and verdict.
   - Tie the verdict to observed metrics/artifacts, not just impression.

6. **Handoff**
   - If more evals are needed, create separate eval logs.
   - If promoting or deploying, write/update the promotion note and follow the
     relevant next skill.

## Eval Logs

Eval logs live in `eval_logs/`, parallel to `run_logs/`. Group into
subdirectories by component/task when useful, and keep a `timeline.md` per group.

Name logs `<date>_<task>.md`. Include:

```markdown
# eval - <short description>

## Provenance
- checkpoint: `<path>`
- manifest: `<path>/CHECKPOINT_MANIFEST.json` (sha256: `<hash>`)
- source_run_log: `<path>`
- config_snapshot: `<path>`
- dataset: `<path or URL>`

## Job
- execution_id:
- submitted/start:
- end:
- runtime:

## Metrics

## Qualitative

## Verdict
- verdict: <one-line recommendation grounded in evidence>

## Next
```

## Promotion Notes

A promotion note is a checkpoint-level decision informed by one or more eval
logs. It answers: "given all available evidence, what should happen next?"

Name it `eval_logs/<group>/<date>-promotion-note.md`.

Use exactly one action:

- `reject`
- `needs_more_eval`
- `promote_to_sim`
- `promote_to_preflight`
- `candidate_for_robot`

Template:

```markdown
# checkpoint promotion - <checkpoint name or HF revision>

## Decision
- action: `reject` | `needs_more_eval` | `promote_to_sim` | `promote_to_preflight` | `candidate_for_robot`
- decided_at:
- decided_by:

## Artifact
- hf_repo:
- hf_revision:
- checkpoint_path_or_snapshot:
- manifest_sha256:
- source_run_log:

## Evidence Reviewed
- eval_log:
- simulation_log:
- preflight_audit:

## Signals
- hard_gates:
- positive_signals:
- negative_signals:
- missing_evidence:
- contradictory_evidence:

## Rationale

## Next
```

## Stop Gates

- Missing or failing `CHECKPOINT_MANIFEST.json`.
- Eval does not load the checkpoint's saved config/processors/assets.
- Missing checkpoint, dataset, or source run log.
- Metrics/logs are missing or contradictory.
- Promotion note lacks concrete eval evidence.

## Common Mistakes

- Eval without checkpoint/manifest provenance.
- Metrics without qualitative artifact review.
- Verdicts not tied to evidence.
- Promotion notes that hide missing or contradictory evidence.
