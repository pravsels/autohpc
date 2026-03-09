# autoHPC

Reusable HPC workflow docs and skills for local container setup, cluster deployment, and Slurm job operations.

## Repo Layout

- `hpc-workflows.md`
  - Common flow across local setup, deployment, and training operations.
- `hpc-container-promotion/SKILL.md`
  - Local Docker build/test and promotion to the cluster-native container artifact/runtime.
- `hpc-training-operations/SKILL.md`
  - Slurm submission, monitoring, observability, debugging, and cleanup.
- `cluster-profiles/`
  - One file per cluster with docs links and cluster-specific notes.
- `ops-program.md`
  - General operating constraints for how this repo should be maintained.

## Adding A Cluster

Create `cluster-profiles/<cluster_name>.md` with:
- authoritative docs links
- scheduler/storage/container notes
- any cluster-specific caveats

Never store secrets in profile files.

## Quick Start

Paste this to your agent:

> Read ../autohpc/README.md and follow its agent instructions starting from Phase 1. Apply them to this repo.

## If You Are An Agent

This repo is a reference — read and follow the docs here, then apply them to whatever target repo you are working in. Do **not** copy or scaffold these files into the target repo.

The Docker container is the execution environment — for local work **and** for the cluster. Do not install dependencies on the host or use conda/mamba/venv as an alternative. Build the image first, run everything inside it.

### Phase 1 — Local Docker

Read `hpc-container-promotion/SKILL.md` (in this repo) and follow Part 1 for the target repo. Nothing else.

Do **not** read Phase 2 yet. Do **not** plan for cluster deployment, ask which cluster to target, scan for Slurm scripts, or create cluster config files. Those are Phase 2 concerns and you are not there yet.

Do not move to Phase 2 until the image builds and basic sanity checks pass inside the container.

### Phase 2 — Cluster deployment

Only begin this phase after Phase 1 is complete and the Docker image works locally.

1. Follow Part 2 of `hpc-container-promotion/SKILL.md` (in this repo) to promote the image for the target cluster.
2. Identify the target cluster and read `cluster-profiles/<cluster_name>.md` (in this repo) to decide the deployment path.
3. Follow `hpc-training-operations/SKILL.md` (in this repo) for cluster submission, monitoring, debugging, and observability.

## Inspiration

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
