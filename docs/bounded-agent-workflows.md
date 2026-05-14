# Bounded Agent Workflows

## Purpose

AutoHPC workflows should be executable by agents without requiring open-ended
runtime discovery. If a workflow step depends on model loading, deployment
routing, dataset interpretation, or environment setup, the repo should provide a
documented script, command, artifact schema, or stop-and-ask gate.

The goal is not to remove agents from the loop. The goal is to keep agents on
rails: run the known command, collect the known artifact, validate it, record
the result, and stop when the required contract is missing.

## Core Rule

Agents execute documented workflows; they do not discover workflows.

Every agent-facing step should be one of:

- Run a documented command or script.
- Fill a documented artifact with a fixed schema.
- Run a validator and record its output.
- Update a run, eval, passport, or deployment log from concrete evidence.
- Stop and ask the user when a required script, artifact, environment, or input
  is missing.

Avoid instructions that ask agents to "figure out", "inspect until understood",
"use judgment", "write a small script", or "adapt this template" unless the
allowed scope and exit conditions are explicit.

## Why This Matters

Open-ended agent discovery has already wasted significant time, especially in
older checkpoint passport flows. Loading a trained robotics policy and
exercising the correct inference path is architecture-specific. When agents are
asked to infer that path from source code, they can spend hours reverse
engineering code that should have been captured as a reusable script.

What has worked better:

- Provide scripts that know how to load each policy family.
- Make agents run those scripts in the right container or environment.
- Emit structured JSON that `checkpoint-passport` can merge and validate.
- Treat missing scripts as a workflow gap, not an invitation to improvise.

## Checkpoint Passport Direction

`checkpoint-passport/SKILL.md` is the source of truth. It defines two valid
passport creation paths:

1. **Config-bearing checkpoints** — run `generate-passport` against an existing
   real `config.json`, then validate and sign.
2. **OpenPI checkpoints** — run `extract-passport-seed openpi` inside the OpenPI
   runtime environment, then run `assemble-passport`, validate, and sign.

Do not add an intermediate "generate config" phase for OpenPI. OpenPI's
inference contract is built by `cfg.data.create()`, transforms, norm stats, and
adapter behavior; the supported artifact is `PASSPORT_SEED.json`, not a
synthetic `config.json`.

Agents should not discover model loading. If the supported extractor cannot run
in the current environment, or a required argument such as `--openpi-config-name`
or `--reference-dataset-path` is missing, the agent should stop and ask.

Supported extractor anchors:

- OpenPI checkpoints: `extract-passport-seed openpi`, implemented in
  `checkpoint-passport/checkpoint_passport/runtime_extractors/openpi.py`.
- Config-bearing safetensors/checkpoint families: `generate-passport --config`
  when the checkpoint already ships a real config that fully describes the
  input/output contract.

## Repo-Wide Audit Backlog

### Root README

Add a short "Bounded Agent Contract" section near "If You Are An Agent".

Recommended text:

- Agents must run documented commands and skills.
- Agents must report observed signals, proposed phase, and uncertainty before
  continuing.
- Agents must stop when a required artifact or script is missing.
- Agents must not create ad hoc helper scripts unless the workflow explicitly
  permits it and defines where the script lives.

Tighten wording such as "best guess" into "observed signals plus proposed
phase".

### Checkpoint Passport

Keep the two-path workflow explicit:

- Config-bearing checkpoint: `generate-passport --config`.
- OpenPI checkpoint: `extract-passport-seed openpi` → `assemble-passport`.
- Missing extractor, missing runtime environment, missing config name, or
  missing reference dataset details means stop and ask.
- Soft-signal acceptance should require a decision record with signal IDs and
  evidence.

### Deployment Protocol

Replace broad inspection with a runner-profile contract:

- Require a deployment runner profile with repo commit, image digest, entry
  command, config path, and dry-run mode.
- Require a single bindings artifact, for example `bindings.yaml`, with schema
  validation.
- Require a checked-in preflight script per repo family.
- Limit agent source traversal. If the routing map is missing or incomplete,
  fail the preflight and ask.
- Move physical plausibility and safety judgments to user-attested sections
  unless they are backed by numeric limits from robot config, URDF, or
  controller bounds.

### Dataset Adaptation

Replace exploratory inspection with a dataset contract:

- Save a schema probe artifact such as `samples/schema_probe.json`.
- Write a mapping spec before adapter code.
- Save a golden sample fixture.
- Add a contract test that asserts keys, shapes, dtypes, frame counts, and
  language/action/state mappings.
- Run the repo's fixed training smoke command only after the contract test
  passes.

### Container Promotion

The current "do not create wrapper scripts" rule conflicts with reproducible
workflows. Replace it with:

- Prefer existing Makefile, justfile, CI, or documented commands.
- Fixed script targets are allowed when they are committed and documented.
- Ad hoc shell pipelines are not allowed unless copied into the run/build log
  as the exact command that was executed.
- Any dependency/version deviation from repo pins requires stop-and-ask and a
  recorded build parameter artifact.

### Training Operations

Tighten Slurm workflows with concrete contracts:

- sbatch filenames should follow a fixed pattern.
- resource thresholds should come from cluster profiles.
- job health checks should use numeric thresholds for queue state, log
  freshness, GPU utilization, disk space, and W&B sync state.
- worktrees should be required for objective triggers such as multiple
  concurrent experiment branches, not left to broad judgment.

### Run Tracking

Keep logs evidence-based:

- Define required config keys for every run: dataset revision, container digest,
  code commit, seed, batch size, LR, checkpoint path, and command.
- Qualitative notes must cite raw metrics, dashboard links, or log snippets.
- Next-step recommendations should reference an experiment row, hypothesis, or
  explicit user decision.

### Eval Tracking

Make eval verdicts rule-based where possible:

- Define metric names, units, commands, and thresholds per eval type.
- Qualitative assessment should link to artifacts, videos, or sampled outputs.
- Promotion notes should cite eval logs and quoted metric lines.
- Missing evidence should block promotion unless the user records an explicit
  override.

### Cluster Profiles

Cluster profiles are mostly procedural and should stay that way.

Tighten only where loops are unbounded:

- Add max retry or max wall-clock limits for cloud zone creation loops.
- Keep secret handling patterns fixed.
- Avoid vague cluster-specific advice that encourages agents to debug provider
  behavior by trial and error.

### Autoresearch

`autoresearch/program.md` is intentionally autonomous. Keep it isolated from the
normal AutoHPC workflow.

Add a warning that autoresearch is a special opt-in mode:

- It can loop indefinitely.
- It can modify `train.py`.
- It should run only in a dedicated branch/worktree/session.
- Its "never stop" behavior should not be imported into container promotion,
  deployment, passporting, training operations, or eval workflows.

## Suggested Implementation Order

1. Add the root README bounded-agent contract.
2. Keep checkpoint passport docs aligned with the two-path workflow.
3. Add or specify additional runtime extractors only when a new checkpoint family needs one.
4. Keep extractor outputs schema-validated and assembled by first-party commands.
5. Tighten deployment protocol around runner profiles and bindings artifacts.
6. Tighten dataset adaptation around schema probes and contract tests.
7. Update run/eval tracking to require evidence-cited qualitative notes.
8. Mark autoresearch as a separate autonomous mode.

## Acceptance Criteria

The workflow docs are improved when an agent can answer these questions before
doing work:

- What exact command or script do I run?
- What exact artifact should it produce?
- What validator checks that artifact?
- What evidence do I record?
- What condition makes me stop and ask the user?

If any workflow step cannot answer those questions, it is still too loose.
