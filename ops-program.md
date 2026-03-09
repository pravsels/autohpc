# ops-program

This program defines an autonomous improvement loop for HPC operations workflows.

## Setup

Before starting a run:

1. Ask which cluster is in scope for this session.
2. Load `cluster-profiles/<cluster_name>.md`.
3. Confirm target repository/workflow and agree on a run tag (for logs).
4. Create or reuse an untracked `ops-results.tsv` log for the run.
5. Confirm safety boundaries before executing anything.

## Scope

Allowed to edit:
- `hpc-container-promotion/SKILL.md`
- `hpc-training-operations/SKILL.md`
- `hpc-workflows.md`
- `cluster-profiles/<cluster_name>.md`
- `README.md`
- this file

Out of scope:
- storing secrets in files
- modifying unrelated project code without explicit approval

## Goal

Improve operational quality over time:
- fewer failed jobs
- faster setup and submission flow
- better reproducibility
- clearer recovery steps

## Safety Rules

- Never inline secrets (tokens, API keys, credentials).
- Ask for confirmation before destructive actions (`scancel`, overwrite sync, cleanup).
- Prefer parameterized commands over hardcoded usernames/paths.
- If cluster behavior is unclear, check `cluster-profiles/<cluster_name>.md` first.

## Change Control

Default behavior is operational stability, not continuous process experimentation.

Only change workflow docs/skills when:
- the user explicitly requests a change, or
- a clear mismatch/bug is confirmed against cluster behavior.

For any proposed change:
1. make the minimal edit required,
2. validate with representative commands or dry-run checks,
3. summarize what changed and why.
