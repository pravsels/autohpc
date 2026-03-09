---
name: hpc-container-promotion
description: Use when preparing HPC-ready containers by validating a repo locally with Docker, converting the image to a Singularity or Apptainer SIF (including dockerized conversion toolchains), and publishing the artifact to cluster paths.
---

# HPC Container Promotion

## Overview

Use this for repo -> Docker test -> SIF conversion -> cluster upload workflows.
Core principle: verify locally first, then convert and publish with pinned names and explicit paths.

## When to Use

- Building/testing training code from a repo with Docker before HPC runs
- Converting Docker images into `.sif` for cluster jobs
- Using a helper image/toolchain (for example, a dockerized Singularity environment)
- Uploading promoted `.sif` artifacts to project directories in `$HOME` or `/scratch`

Do not use for direct cloud registry deployments without SIF artifacts.

## Authoritative Docs

If cluster behavior, modules, container runtime, or policy is unclear, check:
`cluster-profiles/<cluster_name>.md`.

If cluster is unknown, ask the user at session start which cluster they are using, then load the matching profile file.

If docs imply this skill is stale, propose a patch and ask for approval before changing the skill.

## Core Pattern

1. Parameterize repo/image/target paths (`REPO_DIR`, `IMAGE_NAME`, `IMAGE_TAG`, `SSH_ALIAS`, `REMOTE_CONTAINER_DIR`).
2. If repo has `docker/` scripts, use those first; otherwise use raw `docker build`.
3. Build + smoke test Docker image locally.
4. Export Docker image tar (`docker save`).
5. Check cluster runtime/version first (`apptainer` vs `singularity`) before choosing conversion path.
6. Convert tar -> SIF with selected toolchain (local `apptainer` or dockerized converter).
7. Upload SIF to cluster storage and verify path/size.
8. Keep promotion metadata (tag, date, source commit) next to artifact.
9. Hand off to Slurm scripts in `slurm/` with an explicit image path/tag.

## Quick Reference

| Goal | Command Template |
|---|---|
| Script-first build | `cd <repo_dir>/docker && ./docker_build.sh <platform>` |
| Build local image | `docker build -t <image>:<tag> <repo_dir>` |
| Smoke test image | `docker run --rm <image>:<tag> <cmd>` |
| Export tar | `docker save -o <image>_<tag>.tar <image>:<tag>` |
| Check cluster runtime | `ssh <ssh_alias> "command -v apptainer || command -v singularity; apptainer --version || singularity --version"` |
| Convert tar to sif | `apptainer build <image>_<tag>.sif docker-archive://<image>_<tag>.tar` |
| Upload SIF | `rsync -avP <image>_<tag>.sif <ssh_alias>:<remote_dir>/` |
| Verify remote artifact | `ssh <ssh_alias> "ls -lh <remote_dir>/<image>_<tag>.sif"` |

## Implementation Example

```bash
export REPO_DIR="$PWD"
export IMAGE_NAME="<project_image_name>_arm64"
export IMAGE_TAG="$(date +%Y%m%d-%H%M)"
export SSH_ALIAS="<your_cluster_alias>"
export REMOTE_CONTAINER_DIR="/scratch/<project_code>/<unix_user>/<project_name>/container"
export OUT_DIR="$REPO_DIR/dist"
mkdir -p "$OUT_DIR"

# If the repo has docker scripts, prefer them:
cd "$REPO_DIR/docker"
./docker_build.sh amd64
./docker_build.sh arm64

# Generic smoke test (replace with repo-specific check if needed)
docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" python -V
docker save -o "${OUT_DIR}/${IMAGE_NAME}_${IMAGE_TAG}.tar" "${IMAGE_NAME}:${IMAGE_TAG}"

# Option A: local apptainer/singularity installed
apptainer build "${OUT_DIR}/${IMAGE_NAME}_${IMAGE_TAG}.sif" \
  "docker-archive://${OUT_DIR}/${IMAGE_NAME}_${IMAGE_TAG}.tar"

# Option B: dockerized converter image/toolchain
# docker run --rm --privileged -v "${OUT_DIR}:/work" <converter_image> \
#   apptainer build "/work/${IMAGE_NAME}_${IMAGE_TAG}.sif" \
#   "docker-archive:///work/${IMAGE_NAME}_${IMAGE_TAG}.tar"

rsync -avP "${OUT_DIR}/${IMAGE_NAME}_${IMAGE_TAG}.sif" "${SSH_ALIAS}:${REMOTE_CONTAINER_DIR}/"
ssh "$SSH_ALIAS" "ls -lh \"${REMOTE_CONTAINER_DIR}/${IMAGE_NAME}_${IMAGE_TAG}.sif\""

# Slurm handoff (repo has slurm/*.sh scripts)
# sbatch "$REPO_DIR/slurm/<job_script>.sh"
```

## Common Mistakes

- Reusing mutable tags (`latest`) so runs are not reproducible
- Skipping local smoke tests before conversion
- Ignoring repo-provided `docker/` wrappers and rebuilding ad-hoc
- Uploading to `~/...` when job scripts expect `/scratch/...`
- Hardcoding old aliases/usernames in remote paths
- Inlining secrets for private pulls instead of secure auth flow
