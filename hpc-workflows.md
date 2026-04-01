# HPC Workflow Index

See `README.md` for the phased workflow (Phase 1 → 2 → 3 → Ongoing) and skill routing. When resuming ongoing work, confirm with the user whether the current goal is baseline replication or experimentation.

## Skills

| Skill | Purpose |
|-------|---------|
| `hpc-container-promotion/SKILL.md` | Local Docker build/test (Phase 1) and cluster image promotion (Phase 3) |
| `hpc-dataset-adaptation/SKILL.md` | Adapting code for user's dataset format (Phase 2) |
| `hpc-training-operations/SKILL.md` | Sbatch scripts, Slurm submission, monitoring, debugging (Phase 3) |
| `hpc-run-tracking/SKILL.md` | Per-run logs for replication and experiment tracking (Ongoing) |
| `eval-tracking/SKILL.md` | Per-eval logs for checkpoint evaluations: metrics, assessment, verdict (Ongoing) |
| `autoresearch/` | After replication runs and evals are stable, use this submodule to run controlled experiment variations |
| `cluster-profiles/<cluster>.md` | Cluster-specific details: storage, modules, runtime, access |
