# HPC Workflow Index

Use this file as the high-level entry point for HPC workflow skills.

## Skills

- Ops workflow: `hpc-training-operations/SKILL.md`
  - Setup, submit, monitor, debug, and cleanup on Slurm-based clusters.
- Container promotion workflow: `hpc-container-promotion/SKILL.md`
  - Repo review, local Docker build/test, Docker tar -> SIF conversion, and upload to cluster storage.

## Suggested Usage

1. Use `hpc-container-promotion` when preparing a new image.
2. Use `hpc-training-operations` when running or debugging jobs with that image.
3. Pass concrete values each run (`SSH_ALIAS`, paths, script names, image tag) to avoid stale notes.
4. If uncertain about cluster behavior, check `cluster-profiles/<cluster_name>.md` before executing.

## Cluster Profiles

- Add one file per cluster: `cluster-profiles/<cluster_name>.md`
- Keep cluster-specific docs links and operational notes there.
- Example: `cluster-profiles/isambard.md`

## Repo Pattern Example (Generic)

1. Build/test from `docker/` (prefer `docker/docker_build.sh` and `docker/run_docker_container.sh`).
2. Convert promoted Docker image to `.sif` and upload to `/scratch/.../<project_name>/container`.
3. Submit from `slurm/` scripts (for example `sbatch slurm/<job_script>.sh <profile>`).
