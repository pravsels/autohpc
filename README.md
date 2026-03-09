# autoHPC

Reusable HPC workflow skills for container promotion and Slurm job operations.

## What is in this repo

- `hpc-container-promotion/SKILL.md`
  - Build/test Docker locally, convert to `.sif`, upload to cluster storage.
- `hpc-training-operations/SKILL.md`
  - Submit, monitor, debug, and clean up Slurm jobs.
- `cluster-profiles/`
  - Cluster-specific docs and operational notes (one file per cluster).
- `hpc-workflows.md`
  - High-level flow and how the skills fit together.

## Quick start

1. Pick your cluster profile (or create one):
   - `cluster-profiles/<cluster_name>.md`
2. Run container workflow:
   - Use `hpc-container-promotion` to prepare and publish a `.sif`.
3. Run training workflow:
   - Use `hpc-training-operations` to submit and operate jobs using that image.

## Adding a new cluster

Create `cluster-profiles/<cluster_name>.md` with:
- authoritative docs links
- scheduler/storage/container notes
- any cluster-specific caveats

Never store secrets in profile files.

## Inspiration

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
