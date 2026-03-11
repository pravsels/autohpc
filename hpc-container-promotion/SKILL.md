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

## Phase 1 — Local Docker build and test (do this first)

Do **not** skip ahead to Phase 3. Do **not** plan for cluster deployment, ask about target clusters, or parameterize SSH/remote paths yet.

Docker scripts and the Dockerfile go in `docker/` in the target repo.

1. Set local params only: `REPO_DIR`, `IMAGE_NAME`, `IMAGE_TAG`.
2. If repo has `docker/` scripts, use those first; otherwise use raw `docker build`.
3. Before building, check the Dockerfile's `FROM` base image includes CUDA/GPU support (e.g. `nvidia/cuda:*`, `pytorch/pytorch:*-cuda*`). A CPU-only base will waste the entire build. To choose the right base image:
   - Check the repo's dependency files (`requirements.txt`, `conda_env.yaml`, `pyproject.toml`) for pinned `torch` and CUDA versions.
   - Check local CUDA support (`nvidia-smi`) — the base image's CUDA version must not exceed the driver's supported version.
   - Check the target cluster's architecture (e.g. `uname -m` via SSH). Prefer a multi-arch base image (`docker manifest inspect`) that supports both local and cluster architectures so you don't have to rebuild later.
   - Pick a base image that satisfies all three constraints.
   - Verify the image tag actually exists (`docker pull`) before writing the Dockerfile.
4. The Dockerfile must install **all** runtime dependencies (use `requirements.txt`, `conda_env.yaml`, or equivalent from the repo). The image must be able to run the application, not just import the package.
5. Build the Docker image.
5. Smoke test inside the container. A smoke test means running the actual application workflows that will run on the HPC — inference with provided weights, or a short training run on a small batch. The point is to verify the exact flow that will execute on the cluster. It does **not** mean `python -V` or a bare import check.
6. Do not proceed to Phase 3 until the image builds and smoke tests pass.

### Phase 1 Quick Reference

| Goal | Command Template |
|---|---|
| Script-first build | `cd <repo_dir>/docker && ./build_docker.sh <platform>` |
| Build local image | `docker build -t <image>:<tag> <repo_dir>` |
| Smoke test (inference) | `docker run --rm --gpus all <image>:<tag> python <inference_script> <args>` |
| Smoke test (training) | `docker run --rm --gpus all <image>:<tag> python <train_script> --batch_size=1 --max_steps=10` |

### Phase 1 Example

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

## Phase 3 — Cluster promotion (only after Phase 1 passes)

Only begin this after the Docker image builds and smoke tests pass locally.

If the cluster architecture differs from local (e.g. arm64 vs amd64), you will likely need to rebuild for that architecture. Before starting a cross-arch build, verify that pinned packages in `requirements.txt` have wheels available for the target architecture and Python version. Builds are expensive in time — check availability first, resolve all blockers, then build once. When a specific package fails during a build, verify its wheel exists for the target platform before changing the Dockerfile and rebuilding.

Do not create promotion scripts, preflight scripts, or submission wrappers. Run commands directly.

Run SSH commands yourself — do not ask the user to run them for you. Request `all` permissions so you can access the user's SSH config and certificates. Only fall back to asking the user if SSH fails after that (e.g. expired certificate).

Open an SSH ControlMaster connection at the start and reuse it for all subsequent commands:

```bash
export SSH_CTRL="/tmp/ssh-ctrl-%r@%h:%p"
ssh -fNM -o ControlPath="$SSH_CTRL" "$SSH_ALIAS"            # open once
ssh -o ControlPath="$SSH_CTRL" "$SSH_ALIAS" "<command>"      # reuse for each command
ssh -o ControlPath="$SSH_CTRL" -O exit "$SSH_ALIAS"          # close when done
```

The deployment workflow:

1. Push code to GitHub and pull on the HPC.
2. Read `cluster-profiles/<cluster_name>.md` to check the container runtime. Not all clusters run Docker — many require `.sif` images via `apptainer`/`singularity`. This determines what you upload.
3. Export the Docker image (`docker save`). If the cluster requires `.sif`, convert locally (`apptainer build`) before uploading.
4. Upload the container artifact and any datasets to HPC scratch.
5. Hand off to `hpc-training-operations/SKILL.md` for job submission.

### Phase 3 Quick Reference

| Goal | Command |
|---|---|
| Export tar | `docker save -o <image>_<tag>.tar <image>:<tag>` |
| Convert tar to sif | `apptainer build <image>_<tag>.sif docker-archive://<image>_<tag>.tar` |
| Upload artifact | `rsync -avP <artifact> <ssh_alias>:<remote_path>/` |
| Verify remote artifact | `ssh <ssh_alias> "ls -lh <remote_path>/<artifact>"` |

## Common Mistakes

- Creating promotion scripts, submission wrappers, or preflight scripts — just run the commands directly
- Uploading a Docker tar without checking if the cluster even runs Docker — read the cluster profile for the container runtime first
- Reusing mutable tags (`latest`) so runs are not reproducible
- Treating `python -V` or a bare import as a smoke test — run real application workflows
- Skipping local smoke tests before conversion
- Building an image without installing all runtime dependencies from the repo
- Ignoring repo-provided `docker/` wrappers and rebuilding ad-hoc
- Uploading to `~/...` when job scripts expect `/scratch/...`
- Hardcoding old aliases/usernames in remote paths
- Inlining secrets for private pulls instead of secure auth flow
