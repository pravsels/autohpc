---
name: hpc-training-operations
description: Use when running ML training workflows on Slurm-based HPC clusters, including environment setup, data and image transfer, job submission, monitoring, debugging, and storage cleanup, especially when notes are ad-hoc, usernames differ, or commands may expose secrets.
---

# HPC Training Operations

## Overview

Use this when submitting and managing training jobs on Slurm-based HPC clusters.
Core principle: push code, upload artifacts, submit job. Keep it simple.

## When to Use

- Slurm workflows (`sbatch`, `squeue`, `scancel`)
- Containerized jobs (`apptainer`/`singularity`)
- Dataset/checkpoint transfer (`scp`, `rsync`)
- Runtime debugging (`srun`, `nvidia-smi`, interactive shell)
- Scratch usage checks (`du`, `sort`)

Do not use for non-Slurm local training or cloud-only workflows.

## Authoritative Docs

When command behavior, scheduler settings, modules, or storage guidance is unclear, consult:
`cluster-profiles/<cluster_name>.md`.

If docs contradict this skill, propose updates and confirm before editing the skill text.

## Agent Algorithm

Follow this order for Slurm-based training operations. Later sections provide
templates and caveats.

1. **Preflight**
   - Identify target cluster and read `cluster-profiles/<cluster_name>.md`.
   - Set `SSH_ALIAS`, `UNIX_USER`, `PROJECT_NAME`, `PROJECT_CODE`,
     `PROJECT_DIR`, and `SCRATCH_DIR`.
   - Try SSH yourself using the configured alias. Only ask the user for auth or
     command output after a current SSH attempt fails.
   - Open/reuse an SSH ControlMaster connection for repeated commands.

2. **Verify remote state**
   - Confirm repo path, branch/commit, container path, dataset path, scratch space,
     and relevant modules/runtime.
   - For user data, run `ls`/size checks on the cluster; do not assume local paths
     match remote paths.
   - Never print or inline secrets; use token files, `GIT_ASKPASS`, or secure env
     handling.

3. **Prepare code and artifacts**
   - Push local code, then pull on the HPC.
   - Keep code in home and heavy artifacts on scratch.
   - Upload or verify container/dataset artifacts with resumable commands.
   - Record exact artifact paths for the run log.

4. **Submit**
   - Use a checked-in `slurm/<script>.sh`; submission should be just
     `sbatch slurm/<script>.sh`.
   - Do not pass experiment settings via ephemeral `sbatch --export` or shell env
     overrides.
   - Record job ID in the run log immediately.

5. **Monitor**
   - Check scheduler state, output log progress, error log, disk usage, and GPU
     telemetry. GPU allocation alone is not proof of progress.
   - Report status with concrete evidence and update the run log.

6. **Debug or intervene**
   - Use `srun` for interactive debugging.
   - Pause for confirmation before high-impact actions: `scancel`, destructive
     cleanup, overwrite syncs, or large uploads.

7. **Complete**
   - Capture accounting (`sacct`/`seff` when available), runtime, exit code,
     checkpoint paths, W&B sync state, and next step.
   - Handoff to `hpc-run-tracking/SKILL.md` for log completion and
     `checkpoint-passport/SKILL.md` before eval/upload.

## Core Pattern

The `slurm/` directory in the target repo is only for training and eval sbatch scripts. Do not create helper scripts, wrapper scripts, preflight scripts, or promotion scripts there.

Run SSH commands yourself and reuse a ControlMaster connection:

```bash
export SSH_CTRL="/tmp/ssh-ctrl-%r@%h:%p"
ssh -fNM -o ControlPath="$SSH_CTRL" "$SSH_ALIAS"            # open once
ssh -o ControlPath="$SSH_CTRL" "$SSH_ALIAS" "<command>"      # reuse for each command
ssh -o ControlPath="$SSH_CTRL" -O exit "$SSH_ALIAS"          # close when done
```

### Worktrees for multi-experiment setups

When the user is running multiple experiments from the same repo — different configs, different datasets, different branches — suggest using git worktrees on the HPC. Each worktree gives the experiment its own directory, which keeps things cleanly separated:

- **Uncommitted config edits stay isolated.** Slurm scripts often need per-worktree changes (e.g. `repo_dir` pointing to the worktree path) that you don't want to commit. These stay in their worktree without affecting others.
- **Slurm logs don't mix.** Each job's `slurm-<job_id>.out` and `.err` files land in the worktree that submitted them, not in a shared directory with dozens of other experiments' logs.
- **Checkpoints, assets, and caches are naturally separated.** Each worktree can point to its own scratch subdirectory.

Create worktrees from the main repo clone on the HPC:

```bash
cd /home/<project_code>/<username>/<project>
git fetch origin <branch>
git worktree add ../<project>_<experiment> origin/<branch>
cd ../<project>_<experiment>
git checkout -b <branch> origin/<branch>   # avoid detached HEAD
```

This isn't always needed — a single clone is fine for one-at-a-time runs. Suggest worktrees when you see the user setting up parallel experiments or when slurm logs and uncommitted edits would start colliding.

## Writing Sbatch Scripts

**Submission must be just `sbatch script.sh`.** No env vars on the command line. Settings that end up on the `sbatch` line are ephemeral — invisible to reviewers, absent from git history, lost when the terminal closes.

Configuration has two layers — keep them separate:

| Layer | Lives in | Controls | Examples |
|---|---|---|---|
| **Infrastructure** | Hardcoded paths at top of `.sh` | Where/how to run | `home_dir`, `scratch_dir`, partition, GPU count, container path |
| **Experiment** | Config YAML (referenced by `.sh`) | What to run | Model name, dataset, hyperparams, task description, episode index |

A training sbatch script should be short and linear. Before writing one, check the repo for existing slurm scripts and match their style and naming.

**Structure:** SBATCH directives at the top, a few path variables, then one `apptainer exec` (or equivalent container run) command that calls the application entry point. That's it.

**Let the application load its own config.** If the repo uses Hydra, PyTorch Lightning, or any config framework, pass the config name/path as a CLI argument. Do not parse YAML in bash, do not re-map config fields to shell variables, do not build long argument lists from shell-parsed values. The application already knows how to load its config.

**Storage layout:** Keep the repo clone on `$HOME` (small, code only). Keep heavy artifacts — container images, datasets, checkpoints, outputs, W&B caches — on scratch. Bind scratch paths into the container so training never writes large files to the home directory.

**Remote workspace hygiene:** Do not create ad hoc scripts, logs, manifests, or test files in the top level of a remote workspace, whether that is `$SCRATCH`, a VM home directory, or a cloud volume mount. The top level should contain project directories, not files like `passport_*.out`, `sign_*.sh`, or `test_extract.py`. Put temporary run artifacts under a run-scoped project directory such as `<remote_project_dir>/autohpc_runs/<run_id>/`; durable Slurm scripts belong in the target repo's `slurm/` directory.

**Only create scripts the user asked for.** If they say "training", create one training script. Do not also create eval, preflight, or "stage N" variants unless asked. Do not create scripts for things that should be run as direct `srun` commands.

**Train config files live with the repo's other configs** (e.g. `configurations/`), not in `slurm/`. The `slurm/` directory is only for sbatch scripts.

**Name scripts to match the repo's conventions.** Look at existing scripts in the repo. If there are none, ask the user or use a descriptive name like `<project>_train_<stage>_slurm.sh`.

**Resume support for walltime-limited jobs:** If the cluster enforces a max walltime (e.g. 1 day), the sbatch script should support resuming from a checkpoint. Define a `LOAD_CKPT_PATH` variable at the top of the script (empty by default) that appends a load argument to the training command when set. To resume, edit the variable in the script, commit, push, and submit — same git workflow as any other config change.

**Verify dataset paths on the cluster before submitting.** Datasets may not be where you expect — they might live under a different project's scratch, or not be uploaded yet. SSH in and `ls` the path. Do not assume local paths match remote ones.

**Bind files directly from scratch into the container.** If training needs a dataset on scratch, bind it with `--bind /scratch/.../dataset.h5:/mnt/dataset.h5`. Do not create symlink indirections, `/workspace/data/` wrappers, or copy data into the repo directory.

**Use `REPO_DIR` for config paths, not `SCRIPT_DIR`.** Slurm copies sbatch scripts to a spool directory before execution, so `$(dirname $0)` resolves to the spool path, not the repo. Always resolve config paths relative to the repo directory variable.

**Never overwrite `LD_LIBRARY_PATH` inside the container.** Apptainer's `--nv` flag injects host NVIDIA driver libraries via `LD_LIBRARY_PATH`. If you set `LD_LIBRARY_PATH=...` without appending `$LD_LIBRARY_PATH`, you remove the driver libs and get "Found no NVIDIA driver" errors. Always append: `export LD_LIBRARY_PATH=/your/paths:$LD_LIBRARY_PATH`.

**All config changes go through git.** When changing training parameters (batch size, learning rate, etc.), edit the config file in the repo, commit, push, pull on HPC, then submit. Do not use `--export` env var overrides or rsync individual files — that breaks the "push code, pull on HPC, submit" workflow and makes runs unreproducible.

### Sbatch Skeleton

Use existing repo scripts first. If writing one, keep it short:

```bash
#!/bin/bash
#SBATCH --job-name=<project>-<task>
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -e
module purge
module load <container_module>

repo_dir="/home/<project_code>/<username>/<project>"
scratch_dir="/scratch/<project_code>/<username>"
container="${scratch_dir}/<project>/container/<image>.sif"
config="${repo_dir}/<path/to/config.yaml>"
dataset="${scratch_dir}/<project>/<dataset>"

apptainer exec --nv \
  --pwd "${repo_dir}" \
  --bind "${scratch_dir}:${scratch_dir}" \
  "${container}" \
  python <entry_point> --config "${config}" --dataset "${dataset}"
```

## Quick Reference

| Goal | Command Template |
|---|---|
| Login | `ssh <ssh_alias>` |
| Submit | `sbatch <slurm_script>.sh` |
| Submit profile | `sbatch <slurm_script>.sh <profile>` only when the checked-in script documents that positional profile argument |
| Queue by user | `squeue -u <unix_user>` |
| Watch queue | `watch -n 1 squeue -u <unix_user>` |
| Tail job logs | `tail -f slurm-<job_id>.out` and `tail -f slurm-<job_id>.err` |
| Cancel all your jobs | `scancel -u "$(whoami)"` |
| Copy local -> remote | `rsync -avz -P <local_path> <ssh_alias>:<remote_path>` |
| Copy remote -> local | `rsync -avz -P <ssh_alias>:<remote_path> <local_path>` |
| GPU stats for job | `srun --jobid=<job_id> --overlap nvidia-smi -l 1` |
| Job accounting | `sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS,ExitCode` |
| Efficiency summary | `seff <job_id>` |
| W&B offline sync | `wandb sync <offline_run_dir>` |
| Interactive debug shell | `srun --gpus=1 --time=00:30:00 --pty /bin/bash` |
| Scratch usage | `du -sh <scratch_dir>` and `du -h --max-depth=1 <scratch_dir> \| sort -hr` |

## Observability Guidance

Queue status and GPU usage are not sufficient to verify training is healthy.
When monitoring a running job, check in this order:

1. **`slurm-<job_id>.out` — is the training loop advancing?** Tail the output log and look for step counts and loss values. If steps are advancing and loss is being logged, training is alive. If the last logged step was hours ago, training is stuck.
2. **`slurm-<job_id>.err` — any errors or warnings?** Check for disk full errors, NCCL timeouts, OOM messages, checkpoint I/O failures, or lock acquisition warnings. Errors here can indicate a job that's alive but not making progress.
3. **Disk usage — is there room for checkpoints?** A full filesystem silently deadlocks checkpoint writes. The training loop may continue computing steps but hang when the checkpoint thread blocks on disk I/O. Check usage with `du -sh <checkpoint_dir>` and compare against the filesystem's quota (see cluster profile for quota commands).
4. **GPU telemetry — is the GPU doing work?** `nvidia-smi` confirms the GPU is allocated and has processes, but doesn't distinguish productive training from a deadlocked process holding GPU memory. Only useful as a first sanity check, not as proof of progress.

A healthy training job shows recent step numbers, no errors, disk not near quota,
and GPU utilization >0%. Capture `sacct`/`seff` summaries after completion when
available.

W&B tracking is required for training runs. Prefer offline-first logging on
restricted clusters, then sync later. Never inline `WANDB_API_KEY`.

## Common Mistakes

- Parsing train config in bash (awk/sed/embedded Python) instead of letting the application load it
- Putting train config files in `slurm/` instead of with the repo's other configs
- Writing long sbatch scripts that re-map every config field to a shell variable
- Storing checkpoints/outputs/data on `$HOME` instead of scratch
- Mixing local and remote paths in one command
- Copying large datasets with `scp -r` when resumable `rsync -P` is needed
- Submitting without verifying dataset paths exist on the cluster
- Submitting without checking script/account/partition settings
- Creating indirection layers for container bind mounts (symlinks, wrapper dirs) instead of binding directly
- Forgetting `module load` for the container runtime (check cluster profile)
- Monitoring wrong user (`squeue -u`) due hardcoded shortname
- Using container paths not mounted in `apptainer exec --bind`
- Not checking disk usage before or during training — a full filesystem silently deadlocks checkpoint writes, which can hang the training loop for hours until walltime kills the job
- Running training without W&B tracking/sync and losing experiment visibility
