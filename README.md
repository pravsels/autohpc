# AutoHPC

Take any AI / ML repo from local Docker build to HPC training and eval — with an AI agent doing the work.

## Quick Start

Clone this repo alongside your target ML repo, then paste this prompt into your AI coding agent:

```
Read ../autohpc/README.md. Assess the current phase (see phase table), confirm with user, then follow the matching skill.
```

Adjust the path if your clone location differs.

## Repo Layout

- `hpc-container-promotion/SKILL.md`
  - Local Docker build/test and promotion to the cluster-native container artifact/runtime.
- `hpc-training-operations/SKILL.md`
  - Slurm submission, monitoring, observability, and debugging (Slurm targets only).
- `hpc-dataset-adaptation/SKILL.md`
  - Adapting code to read the user's dataset format without converting data.
- `hpc-run-tracking/SKILL.md`
  - Maintaining per-run logs for submitted jobs: config, status, results, next steps.
- `eval-tracking/SKILL.md`
  - Maintaining per-eval logs for checkpoint evaluations: provenance, metrics, qualitative assessment, verdict.
- `autoresearch/`
  - Git submodule for autonomous post-baseline experiment loops once replication runs and evals are stable.
- `cluster-profiles/`
  - One file per cluster with docs links and cluster-specific notes.

## Adding A Cluster

Create `cluster-profiles/<cluster_name>.md` with:
- authoritative docs links
- scheduler/storage/container notes
- any cluster-specific caveats

Never store secrets in profile files.

## If You Are An Agent

This repo is a reference — read and follow the docs here, then apply them to whatever target repo you are working in. Do **not** copy or scaffold these files into the target repo.

The Docker image is the build artifact — for local work **and** for remote deployment. The runtime may be Docker or Apptainer depending on the target. Do not install dependencies on the host or use conda/mamba/venv as an alternative. Build the image first, run everything inside it.

Keep commit messages short — a few words, not a paragraph. Check `git log` in the target repo and match its style.

### Assessing current phase

When resuming work on a repo, check these signals to determine where you are before doing anything else:

| Signal | Phase |
|--------|-------|
| No Dockerfile or broken image build | Phase 1 — local Docker |
| Dockerfile works but training fails on user's data format | Phase 2 — dataset adaptation |
| Image works locally, no remote deployment done yet | Phase 3 — ask user for target environment |
| `slurm/` scripts exist, no `run_logs/` or empty `run_logs/` | Phase 3 — first submission (Slurm) |
| `run_logs/` has run logs with results | Ongoing — run tracking |
| `eval_logs/` has eval logs with metrics | Ongoing — eval tracking |

Report what you find and your best guess to the user (e.g. "Dockerfile exists and builds, slurm scripts are present, run_logs/ has 12 logged runs — looks like you're in the ongoing phase. Are you still focused on replication baselines, or are you now in experimentation mode?"). Wait for confirmation before continuing — the signals above are heuristics, and the user may know better (e.g. the image builds but is stale, run_logs exist from a previous attempt that was abandoned, or experiment logs exist even though the current goal is still baseline replication).

### Phase 1 — Local Docker

Read `hpc-container-promotion/SKILL.md` (in this repo) and follow Phase 1 for the target repo. Nothing else.

Do **not** read Phase 2 or 3 yet. Do **not** plan for dataset adaptation, remote deployment, ask which target to use, scan for Slurm scripts, or create cluster config files. Those are later concerns and you are not there yet.

Do not move to Phase 2 until the image builds and basic sanity checks pass inside the container using the repo's own data.

### Phase 2 — Dataset adaptation (skip if using repo data as-is)

Only begin this phase after Phase 1 is complete. If the user's data is already in the format the repo expects, skip to Phase 3.

Follow `hpc-dataset-adaptation/SKILL.md` (in this repo) to adapt the target repo's code to read the user's dataset. Do not move to Phase 3 until training runs end-to-end with the user's data.

### Phase 3 — Remote deployment

Only begin this phase after Phase 1 (and Phase 2 if applicable) is complete and the Docker image works locally.

The workflow is simple: push code, upload image and data, run training. Do not create helper scripts, wrapper scripts, or multi-stage pipelines. Run commands directly.

1. Identify the target environment and read `cluster-profiles/<cluster_name>.md` (in this repo). The profile determines the deployment path — not all targets are Slurm clusters.
2. **If the target is a cloud VM** (e.g. GCloud): the cluster profile is the workflow — follow it directly for instance creation, environment setup, building the image on the VM, and running training. Skip `hpc-container-promotion` Phase 3 and `hpc-training-operations` — they don't apply.
3. **If the target uses Slurm** (e.g. Isambard): follow Phase 3 of `hpc-container-promotion/SKILL.md` to promote the image, then `hpc-training-operations/SKILL.md` to write sbatch scripts and submit. The only scripts you should create in the target repo's `slurm/` are training and eval sbatch scripts.

Once training is running (container launched on a cloud VM, or job submitted on Slurm), move to the Ongoing phase below.

### Ongoing — Run and eval tracking

Once you start running training, follow `hpc-run-tracking/SKILL.md` (in this repo) for every run. This is not a one-time setup step — it is an ongoing practice.

- Create a run log in `run_logs/` when you start a training run.
- Update it when checking status or collecting results.
- For replication runs, a single log per task is enough.
- For experiment runs, maintain a comparison summary across variations.

When evaluating checkpoints, follow `eval-tracking/SKILL.md` (in this repo) for every eval.

- Create an eval log in `eval_logs/` for each evaluation.
- Record provenance (which checkpoint, which data), metrics, qualitative assessment, and verdict.

### After replication baselines are stable

Once replication training runs and checkpoint evals are working end-to-end, you can use the `autoresearch/` submodule to branch into experiment-driven work.

Treat this as a second phase after baseline reproduction: first make sure the original training setup is reproducible and the eval loop is trustworthy, then use `autoresearch` to run controlled variations and compare different ideas.

That split keeps replication and experimentation separate: use this repo's run and eval tracking to establish the baseline, then use `autoresearch` to drive higher-variance research experiments on top of that foundation.

Even in ongoing work, do not infer experimentation mode from files alone. Confirm with the user whether the current objective is still replication or whether they want to switch into experiment exploration.

## Inspiration

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
