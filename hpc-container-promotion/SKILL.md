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

## Agent Algorithm

Follow this order. The phase sections below provide detailed commands.

1. **Classify the target**
   - If no local Docker image builds and runs yet, stay in Phase 1.
   - If the image works locally and the target is a Slurm/container cluster,
     proceed to Phase 3 only after Phase 1 passes.
   - If the target is a cloud VM with Docker, follow the cluster profile rather
     than exporting/converting images.

2. **Phase 1: build and test locally**
   - Verify base image GPU/CUDA support and architecture compatibility.
   - Build using existing `docker/` scripts where present.
   - Smoke test the real application workflow inside the container.
   - Stop on build or smoke-test failure; do not promote a broken image.

3. **Phase 3: promote for target runtime**
   - Read the cluster profile.
   - Before uploading or building remote artifacts, check free space on the
     target filesystem that will receive containers, datasets, logs,
     checkpoints, and W&B outputs.
   - For Slurm/Apptainer targets, build the `.sif` locally and upload only the
     `.sif` — do not upload the Docker tar (it is typically 2x+ the `.sif` size).
     Build from `docker-daemon://` when apptainer is local, or via a dockerized
     apptainer toolchain otherwise. Only `docker save` + upload a tar + convert
     on the cluster as a last resort when apptainer cannot run locally at all.
     Build the correct architecture image and place heavy artifacts on scratch.
   - For Docker-native VMs, clone/build on the VM if the cluster profile says so.

4. **Verify promoted artifact**
   - On the target, run a minimal container command that imports/runs the actual
     application entry point.
   - Record image tag/path, architecture, runtime, and smoke-test command.

5. **Handoff**
   - Once the artifact is verified, follow `hpc-training-operations/SKILL.md`
     for submission and monitoring.

## Phase 1 — Local Docker build and test (do this first)

Docker scripts and the Dockerfile go in `docker/` in the target repo.

1. Set local params only: `REPO_DIR`, `IMAGE_NAME`, `IMAGE_TAG`.
2. If repo has `docker/` scripts, use those first; otherwise use raw `docker build`.
3. Before building, verify the base image has GPU/CUDA support, exists, and
   supports the local and target architectures. Check pinned compiled packages
   for target-architecture wheels before starting expensive cross-builds.
4. The Dockerfile must install **all** runtime dependencies (use `requirements.txt`, `conda_env.yaml`, or equivalent from the repo). The image must be able to run the application, not just import the package.
5. Build the Docker image. Use `--network host` to avoid DNS resolution failures inside the build container (see "Docker build networking" below).
6. Smoke test inside the container. Open a shell in the container (`./docker/run_script.sh` or `docker run --rm --gpus all -it <image> bash`) and run the Python command directly. Do **not** create smoke test wrapper scripts — they add nothing over typing the command yourself. A smoke test means running the actual application workflows that will run on the HPC — inference with provided weights, or a short training run on a small batch. It does **not** mean `python -V` or a bare import check.
7. Do not proceed to Phase 3 until the image builds and smoke tests pass.

### Docker Build Script Conventions

When the repo has `docker/build_docker.sh`, it must support explicit platform selection:

- **Accept a platform argument.** Default to the current host architecture (detect it, don't hardcode), but allow overriding for cross-builds.
- **Tag images with the platform** so different architectures don't overwrite each other. Default the image tag to the platform name.
- **Use `docker buildx` for cross-architecture builds.** Plain `docker build` only targets the host. For non-host platforms, use `docker buildx build --platform linux/<arch> --load`.

```bash
./docker/build_docker.sh          # detects host arch
./docker/build_docker.sh arm64    # cross-build for cluster
```

### Docker build networking

Docker builds run in an isolated network namespace by default. DNS resolution frequently fails inside build containers, causing `apt-get update` to produce `Temporary failure resolving` errors. Every subsequent package install then fails with "unable to locate package" — but the root cause is DNS, not missing packages.

Fix: always pass `--network host` to `docker build` / `docker buildx build`. Build scripts should default to this (overridable via `DOCKER_BUILD_NETWORK` env var).

When diagnosing a failed build, look for `Temporary failure resolving` in the `apt-get update` output before investigating package-level errors — the package errors are just fallout from missing indexes.

### Phase 1 Quick Reference

| Goal | Command Template |
|---|---|
| Script-first build | `cd <repo_dir>/docker && ./build_docker.sh <platform>` |
| Build local image | `docker build -t <image>:<tag> <repo_dir>` |
| Smoke test (inference) | `docker run --rm --gpus all <image>:<tag> python <inference_script> <args>` |
| Smoke test (training) | `docker run --rm --gpus all <image>:<tag> python <train_script> --batch_size=1 --max_steps=10` |

## Phase 3 — Remote deployment (only after Phase 1 passes)

Only begin this after the Docker image builds and smoke tests pass locally.

**Read the cluster profile first.** The profile determines the deployment path — not all targets work the same way.

### Cloud VMs with Docker (e.g. GCloud)

If the target runs Docker natively, the export/convert/upload workflow does not apply. Instead:

1. Follow the cluster profile to create and configure the VM.
2. Push code to GitHub, clone on the VM, and build the image natively there.
3. Check VM disk space before Docker build/pull and before training.
4. Run training directly with `docker run --gpus all`.

This avoids cross-architecture builds and multi-GB image uploads. The cluster profile has the full workflow — instance creation, environment setup, training, and cleanup.

### Slurm clusters with Apptainer/Singularity (e.g. Isambard)

If the cluster architecture differs from local (e.g. arm64 vs amd64), rebuild for that architecture. Before cross-building, verify pinned compiled packages have target-architecture wheels.

Do not create promotion scripts, preflight scripts, or submission wrappers. Run commands directly.

Run SSH commands yourself and use `hpc-training-operations/SKILL.md` for the SSH ControlMaster/auth pattern.

The deployment workflow:

1. Push code to GitHub and pull on the HPC.
2. Read `cluster-profiles/<cluster_name>.md` to check the container runtime. Not all clusters run Docker — many require `.sif` images via `apptainer`/`singularity`. This determines what you upload.
3. Check remote free space for the destination directory before uploading:
   `df -h <remote_container_dir> <remote_scratch_dir>` and
   `du -h --max-depth=1 <remote_project_dir> | sort -hr`.
4. **Build the `.sif` locally and upload only the `.sif`. Do not upload the Docker tar.** The `.sif` is the only artifact the cluster needs; everything else is throwaway transport. An uncompressed `docker save` tar is typically 2x+ the size of the resulting `.sif`, so uploading the tar and converting on the cluster moves far more bytes than necessary and wastes scratch — the conversion produces the `.sif` you could have built and uploaded directly.
   - **Apptainer available locally (run `which apptainer`):** build straight from the Docker daemon — `apptainer build <image>_<tag>.sif docker-daemon://<image>:<tag>` — so there is no `docker save`, no `.tar`, and no gzip-header failure. The cluster never sees a tar.
   - **No native apptainer locally:** use a dockerized apptainer/singularity toolchain (see Overview) to build the `.sif` locally, or install apptainer. Prefer this over uploading a tar — you still upload only the small `.sif`.
   - **Last resort only — no way to run apptainer locally at all:** `docker save` an *uncompressed* `.tar`, upload it, convert on the cluster (`apptainer build ... docker-archive://`), then immediately delete the tar(s) to reclaim scratch. A gzipped `.tar.gz` fails with `gzip: invalid header`. Most HPC clusters have `apptainer` as a module — check the cluster profile.
5. Upload the container artifact (the `.sif` when built locally; otherwise the tar) and any datasets to HPC scratch.
6. Re-check free space after upload/conversion. Clean up tar files and stale intermediates before handing off.
7. Hand off to `hpc-training-operations/SKILL.md` for job submission.

### Phase 3 Quick Reference

| Goal | Command |
|---|---|
| Check local apptainer (do this first) | `which apptainer` |
| Build sif locally, no tar (preferred) | `apptainer build <image>_<tag>.sif docker-daemon://<image>:<tag>` |
| Export tar (fallback, only if no local apptainer) | `docker save -o <image>_<tag>.tar <image>:<tag>` |
| Convert tar to sif locally (if you already have a tar) | `apptainer build <image>_<tag>.sif docker-archive://<image>_<tag>.tar` |
| Convert tar to sif on cluster (fallback) | Upload uncompressed `.tar`, then on the cluster: `apptainer build <image>_<tag>.sif docker-archive://<image>_<tag>.tar` |
| Upload artifact (prefer the `.sif`) | `rsync -avP <artifact> <ssh_alias>:<remote_path>/` |
| Verify remote artifact | `ssh <ssh_alias> "ls -lh <remote_path>/<artifact>"` |
| Reclaim scratch after cluster-side convert | `ssh <ssh_alias> "rm -f <remote_path>/<image>_<tag>.tar <remote_path>/<image>_<tag>.tar.gz"` |

## Common Mistakes

- Uploading a Docker tar without checking if the cluster even runs Docker — read the cluster profile for the container runtime first
- Uploading a large intermediate transport artifact (e.g. a `docker save` tar, often 2x+ the final image size) to convert remotely, instead of producing the cluster-native artifact locally and uploading only that
- Leaving intermediate transport artifacts on scratch after the final image is built
- Promoting containers without checking target disk space first — a full
  container/checkpoint filesystem can break upload, Slurm logging, or later
  checkpoint writes
- Reusing mutable tags (`latest`) so runs are not reproducible
- Treating `python -V` or a bare import as a smoke test — run real application workflows
- Skipping local smoke tests before conversion
- Building an image without installing all runtime dependencies from the repo
- Ignoring repo-provided `docker/` wrappers and rebuilding ad-hoc
- Building without explicit platform selection — silently produces wrong architecture for the target cluster
- Using the same image tag for different architectures — overwrites and causes silent failures
- Cross-building and uploading a multi-GB image to a cloud VM that runs Docker natively — just clone and build on the VM
- Uploading to `~/...` when job scripts expect `/scratch/...`
- Hardcoding old aliases/usernames in remote paths
- Inlining secrets for private pulls instead of secure auth flow
