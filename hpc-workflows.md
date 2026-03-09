# HPC Workflow Index

Use this file as the high-level entry point for HPC workflow skills.

## Skills

- Ops workflow: `hpc-training-operations/SKILL.md`
  - Setup, submit, monitor, debug, and cleanup on Slurm-based clusters.
- Container promotion workflow: `hpc-container-promotion/SKILL.md`
  - Repo review, local Docker build/test, and promotion to the cluster-native container artifact/runtime.

## Suggested Usage

1. Use `hpc-container-promotion` when preparing a new image.
2. Use `hpc-training-operations` when running or debugging jobs with that image.
3. Pass concrete values each run (`SSH_ALIAS`, paths, script names, image tag) to avoid stale notes.
4. If uncertain about cluster behavior, check `cluster-profiles/<cluster_name>.md` before executing.
5. During/after runs, collect observability data (logs, `sacct`, `seff`, and W&B sync).

## Suggested Flow

1. Start with local repo setup, Docker build, and smoke testing.
2. After local validation, check `cluster-profiles/<cluster_name>.md` to choose the deployment path.
3. Use the cluster-native runtime/artifact for deployment:
   Docker image where allowed, or converted `.sif`/similar artifact where required.
4. Then hand off to cluster submission and monitoring.

## Cluster Profiles

- Add one file per cluster: `cluster-profiles/<cluster_name>.md`
- Keep cluster-specific docs links and operational notes there.
- Example: `cluster-profiles/isambard.md`

## Repo Pattern Example (Generic)

1. Build/test from `docker/` (prefer `docker/build_docker.sh` and `docker/run_docker.sh`).
2. Choose the cluster-native deployment artifact from the cluster profile (`docker`, `.sif`, or similar).
3. Submit from `slurm/` scripts (for example `sbatch slurm/<job_script>.sh <profile>`).
