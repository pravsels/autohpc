# AutoHPC

Take an AI / ML repo from local Docker build to remote training, checkpoint
publishing, eval, and deployment preflight, with an AI agent doing the work.

## Quick Start

Clone this repo alongside your target ML repo, then paste this into your AI agent:

```text
Read ../autohpc/README.md. Assess the current phase, confirm with user, then follow the matching skill.
```

Adjust the path if your clone location differs.

## Phase Router

When resuming work, inspect the target repo and report your best guess to the
user. Wait for confirmation before continuing.

| Signal | Phase |
|--------|-------|
| No Dockerfile or broken image build | Phase 1 — local Docker |
| Image works but training fails on user's data format | Phase 2 — dataset adaptation |
| Before the first full run, or policies are uniformly bad | Pre-train — model–dataset audit |
| Image works locally, no remote deployment yet | Phase 3 — ask user for target environment |
| `slurm/` scripts exist but `run_logs/` is empty | Phase 3 — first submission |
| `run_logs/` has results | Ongoing — run tracking |
| Checkpoint is ready to move or publish | Post-train — checkpoint integrity |
| Downloaded checkpoint needs evaluation | Ongoing — eval tracking |
| First robot/inference run is next | Pre-deploy — deployment protocol |
| Evaluated checkpoint needs a promotion decision | Checkpoint triage — see `eval-tracking/SKILL.md` |

## Skill Map

- `hpc-container-promotion/SKILL.md` — local Docker build/test and container promotion.
- `hpc-dataset-adaptation/SKILL.md` — adapt code to the user's dataset format.
- `hpc-model-dataset-audit/SKILL.md` — reconcile base model, config, and dataset semantics before a full run (catches normalization/order/units/encoding mismatches that build and run fine).
- `hpc-training-operations/SKILL.md` — Slurm submission, monitoring, and debugging.
- `hpc-run-tracking/SKILL.md` — per-run training logs.
- `checkpoint-integrity/SKILL.md` — lightweight SHA-256 manifest generation and verification.
- `wandb-sync/` — runnable W&B offline sync helper used by run tracking.
- `eval-tracking/SKILL.md` — per-eval logs and promotion notes.
- `deployment-protocol/` — first-run deployment preflight on robot or inference rig.
- `cluster-profiles/` — cluster-specific docs and caveats.
- `autoresearch/` — post-baseline autonomous experiment loops.

Most folders are docs-only skills. `checkpoint-integrity/` and `wandb-sync/`
also ship installable Python packages:

```bash
uv pip install -e ../autohpc/checkpoint-integrity
uv pip install -e ../autohpc/wandb-sync
```

## Agent Contract

- This repo is a reference. Apply the docs to the target ML repo; do not copy
  or scaffold AutoHPC files into the target repo.
- In-repo AutoHPC skills are canonical over global or personal skills with
  similar names. Read the relevant `SKILL.md` and follow its **Agent Algorithm**
  before using reference sections or external skill memory.
- Confirm the phase with the user before continuing. The phase table is a
  heuristic, not ground truth.
- The Docker image is the build artifact for local and remote work. Do not
  install dependencies on the host or use conda/mamba/venv as an alternative.
- Keep the workflow direct: build image, push/upload as needed, run training,
  log runs, manifest checkpoint bundles before moving them.
- Keep commit messages short and match the target repo's style.

## Adding A Cluster

Create `cluster-profiles/<cluster_name>.md` with docs links, scheduler/storage
notes, container/runtime notes, and caveats. Never store secrets in profile
files.

## Acknowledgement

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
