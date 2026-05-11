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
checkpoint passport Phase 2. Loading a trained robotics policy and exercising
the correct inference path is architecture-specific. When agents are asked to
infer that path from source code, they can spend hours reverse engineering code
that should have been captured as a reusable script.

What has worked better:

- Provide scripts that know how to load each policy family.
- Make agents run those scripts in the right container or environment.
- Emit structured JSON that `checkpoint-passport` can merge and validate.
- Treat missing scripts as a workflow gap, not an invitation to improvise.

## Checkpoint Passport Direction

`checkpoint-passport/SKILL.md` currently makes Phase 2 too open-ended. It asks
the agent to load the model, identify the public inference path, build inputs,
run a forward pass, inspect internals, and merge dynamic JSON.

The intended replacement is:

1. Phase 1 produces a static passport draft with `generate-passport`.
2. Phase 2 runs an architecture-specific dynamic extractor script.
3. The extractor emits a structured dynamic JSON fragment.
4. A first-party merge command combines static and dynamic data.
5. `validate-checkpoint` reports hard failures, soft signals, and not-checked
   items.
6. `sign-checkpoint` writes `SIGNOFF.json` only after hard checks pass.

Agents should not discover model loading during Phase 2. If no extractor exists
for the checkpoint architecture, the agent should stop and ask.

Known extractor anchors:

- OpenPI checkpoints: repurpose the loading path from
  `../alpha-robotics/hw_control/new_lerobot_integrations/deploy_policy.py`,
  especially `missiontracker.adapters.load_policy_adapter(...)` with
  `force_architecture="openpi"`, `openpi_config_name`, prompt, resize size, and
  the adapter inference path.
- MultiTask DiT safetensors checkpoints: repurpose
  `MultiTaskDiTPolicy.load(<checkpoint_dir>)`, which loads `config.json` and
  `model.safetensors`, then run the public `select_action` or
  `predict_action_chunk` path.

Open questions for implementation:

- Should extractor scripts live in `checkpoint-passport`, in the target model
  repo, or both?
- Should `checkpoint-passport` provide only a generic extractor interface plus
  schemas, or also ship first-party OpenPI and MultiTask DiT extractors?
- What should the exact dynamic JSON schema be, and should it be a separate
  dataclass from `MODEL_PASSPORT.json`?
- Should the merge command refuse unknown keys and require schema version
  compatibility?

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

Highest priority.

Replace Phase 2 with a script-first dynamic extraction contract:

- Require a runtime extractor script for each supported architecture.
- Require fixed CLI arguments: checkpoint path, output JSON path, device,
  optional config name, optional prompt, optional dataset reference.
- Require the extractor to run inside the model's own container/environment.
- Require the extractor to emit schema-validated JSON.
- Add a first-party merge command instead of "a small Python script merging the
  two dicts is fine".
- For missing runtime extractors, say: stop and ask; do not infer model loading.

Also tighten:

- "Left null (Phase 2 / judgment)" should become "filled by script",
  "not applicable with reason", or "stop and ask".
- `library_versions` should come from a fixed environment export, not an
  open-ended dependency list.
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
2. Rewrite checkpoint passport Phase 2 around extractor scripts.
3. Add or specify OpenPI and MultiTask DiT dynamic extractors.
4. Add a schema-validated dynamic JSON fragment and merge command.
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
