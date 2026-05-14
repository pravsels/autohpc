# AutoHPC

Take an AI / ML repo from local Docker build to remote training, checkpoint signoff, eval and deployment preflight, with an AI agent doing the work.

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
| Dockerfile works but training fails on user's data format | Phase 2 — dataset adaptation |
| Image works locally, no remote deployment done yet | Phase 3 — ask user for target environment |
| `slurm/` scripts exist, no `run_logs/` or empty `run_logs/` | Phase 3 — first submission |
| `run_logs/` has run logs with results | Ongoing — run tracking |
| Training finished and checkpoint has no `MODEL_PASSPORT.json` / `SIGNOFF.json` | Post-train — checkpoint passport |
| User is about to upload/copy/hand off a checkpoint without passport/signoff | Post-train — checkpoint passport |
| Checkpoint has passing `SIGNOFF.json`; next step is first robot/inference run | Post-passport — deployment protocol |
| Checkpoint has passing `SIGNOFF.json` and eval logs exist | Ongoing — eval tracking |
| Signed checkpoint needs a promotion decision | Checkpoint triage — follow `eval-tracking/SKILL.md` promotion notes |

## Skill Map

- `hpc-container-promotion/SKILL.md` — local Docker build/test and container promotion.
- `hpc-dataset-adaptation/SKILL.md` — adapt code to the user's dataset format.
- `hpc-training-operations/SKILL.md` — Slurm submission, monitoring, and debugging.
- `hpc-run-tracking/SKILL.md` — per-run training logs.
- `checkpoint-passport/SKILL.md` — canonical `MODEL_PASSPORT.json` / `SIGNOFF.json` tooling and workflow.
- `eval-tracking/SKILL.md` — per-eval logs and promotion notes.
- `deployment-protocol/` — first-run deployment preflight on robot or inference rig.
- `cluster-profiles/` — cluster-specific docs and caveats.
- `autoresearch/` — post-baseline autonomous experiment loops.

Most folders are docs-only skills. `checkpoint-passport/` also ships an installable Python package:

```bash
uv pip install -e ../autohpc/checkpoint-passport
```

## Agent Contract

- This repo is a reference. Apply the docs to the target ML repo; do not copy
  or scaffold AutoHPC files into the target repo.
- Confirm the phase with the user before continuing. The phase table is a
  heuristic, not ground truth.
- The Docker image is the build artifact for local and remote work. Do not
  install dependencies on the host or use conda/mamba/venv as an alternative.
- Keep the workflow direct: build image, push/upload as needed, run training,
  log runs, passport checkpoints before moving them.
- Keep commit messages short and match the target repo's style.

## Adding A Cluster

Create `cluster-profiles/<cluster_name>.md` with docs links, scheduler/storage
notes, container/runtime notes, and caveats. Never store secrets in profile
files.

## Acknowledgement

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
