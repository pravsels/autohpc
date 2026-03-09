---
name: hpc-container-promotion
description: Use when preparing cluster-ready containers by validating a repo locally with Docker and promoting the result into the container artifact or runtime required by the target cluster.
---

# HPC Container Promotion

## Overview

Use this for repo -> Docker test -> cluster artifact promotion workflows.
Core principle: verify locally first, then publish in the format required by the target cluster.

## When to Use

- Building/testing training code from a repo with Docker before HPC runs
- Converting Docker images into cluster-native artifacts when needed (for example `.sif`)
- Using a helper image/toolchain (for example, a dockerized Singularity environment)
- Publishing promoted container artifacts to cluster storage or registries

## Authoritative Docs

If cluster behavior, modules, container runtime, or policy is unclear, check:
`cluster-profiles/<cluster_name>.md`.

If docs imply this skill is stale, propose a patch and ask for approval before changing the skill.

## Part 1 — Local Docker build and test (do this first)

Do **not** skip ahead to Part 2. Do **not** plan for cluster deployment, ask about target clusters, or parameterize SSH/remote paths yet.

Docker scripts and the Dockerfile go in `docker/` in the target repo.

1. Set local params only: `REPO_DIR`, `IMAGE_NAME`, `IMAGE_TAG`.
2. If repo has `docker/` scripts, use those first; otherwise use raw `docker build`.
3. Before building, check the Dockerfile's `FROM` base image includes CUDA/GPU support (e.g. `nvidia/cuda:*`, `pytorch/pytorch:*-cuda*`). A CPU-only base will waste the entire build. To choose the right base image:
   - Check the repo's dependency files (`requirements.txt`, `conda_env.yaml`, `pyproject.toml`) for pinned `torch` and CUDA versions.
   - Check local CUDA support (`nvidia-smi`) — the base image's CUDA version must not exceed the driver's supported version.
   - Pick a base image that satisfies both constraints.
   - Verify the image tag actually exists (`docker pull`) before writing the Dockerfile.
4. The Dockerfile must install **all** runtime dependencies (use `requirements.txt`, `conda_env.yaml`, or equivalent from the repo). The image must be able to run the application, not just import the package.
5. Build the Docker image.
5. Smoke test inside the container. A smoke test means running the actual application workflows that will run on the HPC — inference with provided weights, or a short training run on a small batch. The point is to verify the exact flow that will execute on the cluster. It does **not** mean `python -V` or a bare import check.
6. Do not proceed to Part 2 until the image builds and smoke tests pass.

### Part 1 Quick Reference

| Goal | Command Template |
|---|---|
| Script-first build | `cd <repo_dir>/docker && ./build_docker.sh <platform>` |
| Build local image | `docker build -t <image>:<tag> <repo_dir>` |
| Smoke test (inference) | `docker run --rm --gpus all <image>:<tag> python <inference_script> <args>` |
| Smoke test (training) | `docker run --rm --gpus all <image>:<tag> python <train_script> --batch_size=1 --max_steps=10` |

### Part 1 Example

```bash
export REPO_DIR="$PWD"
export IMAGE_NAME="<project_image_name>"
export IMAGE_TAG="$(date +%Y%m%d-%H%M)"

# If the repo has docker scripts, prefer them:
cd "$REPO_DIR/docker"
./build_docker.sh amd64

# Smoke test with GPU: run an actual application workflow inside the container
# e.g. inference with provided weights, or a short training run
docker run --rm --gpus all "${IMAGE_NAME}:${IMAGE_TAG}" python <inference_or_train_script> <args>
```

## Part 2 — Cluster promotion (only after Part 1 passes)

Only begin this after the Docker image builds and smoke tests pass locally.

6. Parameterize cluster paths: `SSH_ALIAS`, `REMOTE_ARTIFACT_TARGET`.
7. Export Docker image tar (`docker save`).
8. Check the cluster profile to decide whether deployment uses Docker directly or requires conversion.
9. If required, check cluster runtime/version (`apptainer` vs `singularity`) before choosing conversion path.
10. Convert to the required artifact format (for example `.sif`) only when the cluster requires it.
11. Upload/publish the final artifact and verify path/tag/size.
12. Keep promotion metadata (tag, date, source commit) next to artifact.
13. Hand off to Slurm scripts in `slurm/` with an explicit image path/tag.

### Part 2 Quick Reference

| Goal | Command Template |
|---|---|
| Check cluster deployment mode | `read cluster-profiles/<cluster_name>.md and determine docker vs converted artifact` |
| Export tar | `docker save -o <image>_<tag>.tar <image>:<tag>` |
| Check cluster runtime | `ssh <ssh_alias> "command -v apptainer || command -v singularity; apptainer --version || singularity --version"` |
| Convert tar to sif | `apptainer build <image>_<tag>.sif docker-archive://<image>_<tag>.tar` |
| Upload artifact | `rsync -avP <artifact> <ssh_alias>:<remote_target>/` |
| Verify remote artifact | `ssh <ssh_alias> "ls -lh <remote_target>/<artifact>"` |

### Part 2 Example

```bash
export SSH_ALIAS="<your_cluster_alias>"
export REMOTE_ARTIFACT_TARGET="/scratch/<project_code>/<unix_user>/<project_name>/container"
export OUT_DIR="$REPO_DIR/dist"
mkdir -p "$OUT_DIR"

# Choose deployment path from cluster profile:
# - If cluster accepts Docker directly, publish/tag/push the Docker image there.
# - If cluster requires a converted artifact, use a local conversion approach that fits
#   your environment, then upload and verify the resulting artifact.

# Slurm handoff (repo has slurm/*.sh scripts)
# sbatch "$REPO_DIR/slurm/<job_script>.sh"
```

## Common Mistakes

- Reusing mutable tags (`latest`) so runs are not reproducible
- Treating `python -V` or a bare import as a smoke test — run real application workflows
- Skipping local smoke tests before conversion
- Building an image without installing all runtime dependencies from the repo
- Ignoring repo-provided `docker/` wrappers and rebuilding ad-hoc
- Uploading to `~/...` when job scripts expect `/scratch/...`
- Hardcoding old aliases/usernames in remote paths
- Inlining secrets for private pulls instead of secure auth flow
