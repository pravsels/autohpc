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

## Core Pattern

The `slurm/` directory in the target repo is only for training and eval sbatch scripts. Do not create helper scripts, wrapper scripts, preflight scripts, or promotion scripts there.

The deployment workflow is simple:

1. Push code changes to GitHub.
2. Pull the repo on the HPC.
3. Upload the container image and any datasets to the HPC.
4. Submit the training job with `sbatch`.
5. Monitor, debug with `srun`, collect results.

Do not overengineer this. Do not create submission wrappers, promotion scripts, or multi-stage shell pipelines. Run SSH commands directly.

Run SSH commands yourself — do not ask the user to run them for you. Request `all` permissions so you can access the user's SSH config and certificates. Only fall back to asking the user if SSH fails after that (e.g. expired certificate).

Open an SSH ControlMaster connection at the start and reuse it for all subsequent commands:

```bash
export SSH_CTRL="/tmp/ssh-ctrl-%r@%h:%p"
ssh -fNM -o ControlPath="$SSH_CTRL" "$SSH_ALIAS"            # open once
ssh -o ControlPath="$SSH_CTRL" "$SSH_ALIAS" "<command>"      # reuse for each command
ssh -o ControlPath="$SSH_CTRL" -O exit "$SSH_ALIAS"          # close when done
```

1. Set `SSH_ALIAS`, `UNIX_USER`, `PROJECT_NAME`, `PROJECT_CODE`, `PROJECT_DIR`, `SCRATCH_DIR`.
2. Open ControlMaster connection to `SSH_ALIAS`.
3. Push code to GitHub, pull on HPC.
4. Upload container image and datasets to HPC scratch.
5. Submit training with `sbatch slurm/<training_script>.sh`. Use `srun` for debugging.
6. Pause for confirmation on high-impact actions (`scancel`, overwrite sync, destructive cleanup).
7. Never inline secrets; use hidden prompt input or secure env handling.
8. Enable observability: live logs, job accounting, and required W&B tracking.

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

**Only create scripts the user asked for.** If they say "training", create one training script. Do not also create eval, preflight, or "stage N" variants unless asked. Do not create scripts for things that should be run as direct `srun` commands.

**Train config files live with the repo's other configs** (e.g. `configurations/`), not in `slurm/`. The `slurm/` directory is only for sbatch scripts.

**Name scripts to match the repo's conventions.** Look at existing scripts in the repo. If there are none, ask the user or use a descriptive name like `<project>_train_<stage>_slurm.sh`.

**Resume support for walltime-limited jobs:** If the cluster enforces a max walltime (e.g. 1 day), the sbatch script should support resuming from a checkpoint path passed as an environment variable. Keep it simple — one optional `LOAD_CKPT_PATH` variable that appends a load argument to the training command.

**Verify dataset paths on the cluster before submitting.** Datasets may not be where you expect — they might live under a different project's scratch, or not be uploaded yet. SSH in and `ls` the path. Do not assume local paths match remote ones.

**Bind files directly from scratch into the container.** If training needs a dataset on scratch, bind it with `--bind /scratch/.../dataset.h5:/mnt/dataset.h5`. Do not create symlink indirections, `/workspace/data/` wrappers, or copy data into the repo directory.

**Use `REPO_DIR` for config paths, not `SCRIPT_DIR`.** Slurm copies sbatch scripts to a spool directory before execution, so `$(dirname $0)` resolves to the spool path, not the repo. Always resolve config paths relative to the repo directory variable.

**Never overwrite `LD_LIBRARY_PATH` inside the container.** Apptainer's `--nv` flag injects host NVIDIA driver libraries via `LD_LIBRARY_PATH`. If you set `LD_LIBRARY_PATH=...` without appending `$LD_LIBRARY_PATH`, you remove the driver libs and get "Found no NVIDIA driver" errors. Always append: `export LD_LIBRARY_PATH=/your/paths:$LD_LIBRARY_PATH`.

**All config changes go through git.** When changing training parameters (batch size, learning rate, etc.), edit the config file in the repo, commit, push, pull on HPC, then submit. Do not use `--export` env var overrides or rsync individual files — that breaks the "push code, pull on HPC, submit" workflow and makes runs unreproducible.

### Template

Base your sbatch scripts on this structure. Adapt paths, bind mounts, container runtime, and the training command to the target repo. Check `cluster-profiles/<cluster_name>.md` for cluster-specific details (path layout, modules, container runtime).

```bash
#!/bin/bash
#SBATCH --job-name=<project>-<task>
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0G
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -e

# Load cluster modules (check cluster profile for specifics).
# module purge
# module load <container_module>

# Paths — repo on home, everything heavy on scratch.
home_dir="/home/<project_code>/<username>"
scratch_dir="/scratch/<project_code>/<username>"
repo_dir="${home_dir}/<project>"
data_dir="${scratch_dir}/<project>"
container="${data_dir}/container/<image>.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"
WANDB_DIR="${data_dir}"
WANDB_CACHE_DIR="${data_dir}/wandb_cache"
WANDB_CONFIG_DIR="${data_dir}/wandb_config"

# Training config — use REPO_DIR, not SCRIPT_DIR (Slurm copies scripts to spool).
CONFIG_FILE="${repo_dir}/<path/to/config.yaml>"
DATASET_PATH="${data_dir}/<dataset_file>"
CHECKPOINT_DIR="${data_dir}/checkpoints"

mkdir -p "${HF_CACHE}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${CHECKPOINT_DIR}"

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "===================================="

TRAIN_CMD="python <entry_point> \
    --config ${CONFIG_FILE} \
    --dataset ${DATASET_PATH} \
    --checkpoint-dir ${CHECKPOINT_DIR}"

set +e
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${HF_CACHE}:/root/.cache/huggingface" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "${container}" \
    bash -c "export WANDB_DIR=${WANDB_DIR} WANDB_CACHE_DIR=${WANDB_CACHE_DIR} WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} && \
        ${TRAIN_CMD}"
EXIT_CODE=$?
set -e

end_time="$(date -Is --utc)"
echo ""
echo "===================================="
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
echo "Exit Code: ${EXIT_CODE}"
echo "===================================="

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Training failed with exit code ${EXIT_CODE}"
    echo "Check slurm-${SLURM_JOB_ID}.err for details"
    exit ${EXIT_CODE}
fi
```

## Quick Reference

| Goal | Command Template |
|---|---|
| Login | `ssh <ssh_alias>` |
| Submit | `sbatch <slurm_script>.sh` |
| Submit profile | `sbatch <slurm_script>.sh <profile>` |
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
| Interactive debug shell | `srun --nodes=1 --gres=gpu:1 --time=00:30:00 --pty /bin/bash` |
| Scratch usage | `du -sh <scratch_dir>` and `du -h --max-depth=1 <scratch_dir> \| sort -hr` |

## Implementation Example

```bash
export SSH_ALIAS="<your_cluster_alias>"
export UNIX_USER="${UNIX_USER:-$(whoami)}"
export PROJECT_NAME="<project_name>"
export PROJECT_CODE="<project_code>"
export PROJECT_DIR="$HOME/${PROJECT_NAME}"
export SCRATCH_DIR="/scratch/${PROJECT_CODE}/${UNIX_USER}/${PROJECT_NAME}"

# 1. Pull latest code on HPC
ssh "$SSH_ALIAS" "cd $PROJECT_DIR && git pull"

# 2. Upload container image (already built locally)
rsync -avP <image>.tar "$SSH_ALIAS:$SCRATCH_DIR/"

# 3. Submit training
ssh "$SSH_ALIAS" "cd $PROJECT_DIR && sbatch slurm/<training_script>.sh"

# 4. Monitor
ssh "$SSH_ALIAS" "squeue -u $UNIX_USER"
ssh "$SSH_ALIAS" "tail -f $PROJECT_DIR/slurm-<job_id>.out"
```

Private repo auth: do not embed PAT in URL.

## Observability Guidance

### Training health checks

**Queue status and GPU usage are not sufficient to verify training is healthy.** A job can show `RUNNING` in `squeue` and have active GPU memory in `nvidia-smi` while being completely stuck — deadlocked on I/O, hung on a collective, or spinning in an infinite retry loop. You must check the actual training output.

When monitoring a running job, check in this order:

1. **`slurm-<job_id>.out` — is the training loop advancing?** Tail the output log and look for step counts and loss values. If steps are advancing and loss is being logged, training is alive. If the last logged step was hours ago, training is stuck.
2. **`slurm-<job_id>.err` — any errors or warnings?** Check for disk full errors, NCCL timeouts, OOM messages, checkpoint I/O failures, or lock acquisition warnings. Errors here can indicate a job that's alive but not making progress.
3. **Disk usage — is there room for checkpoints?** A full filesystem silently deadlocks checkpoint writes. The training loop may continue computing steps but hang when the checkpoint thread blocks on disk I/O. Check usage with `du -sh <checkpoint_dir>` and compare against the filesystem's quota (see cluster profile for quota commands).
4. **GPU telemetry — is the GPU doing work?** `nvidia-smi` confirms the GPU is allocated and has processes, but doesn't distinguish productive training from a deadlocked process holding GPU memory. Only useful as a first sanity check, not as proof of progress.

A healthy training job shows: recent step numbers in `.out`, no errors in `.err`, disk not near quota, and GPU utilization >0%. All four must be true.

### Post-run accounting

- Capture `sacct`/`seff` summaries for memory, runtime, and exit status.
- W&B tracking is required for training runs.
- Prefer offline-first logging on restricted clusters, then sync later.
- Never inline `WANDB_API_KEY`; pass via secure environment setup.

## Red Flags - Stop and Re-check

- Commands contain hardcoded user handles, project IDs, or old job IDs
- Any secret appears inline (`PAT`, `WANDB_API_KEY`, tokenized git URL)
- Destructive commands run without confirmation (`scancel`, overwrite sync)
- Host/path assumptions are unverified (`~/...` vs `/scratch/...`)
- You are writing a shell script instead of running a command directly
- Submission requires env vars or flags on the `sbatch` command line
- Experiment parameters (model, dataset, hyperparams) live as shell vars in the `.sh` instead of a config YAML

## Common Mistakes

- Parsing train config in bash (awk/sed/embedded Python) instead of letting the application load it
- Putting train config files in `slurm/` instead of with the repo's other configs
- Creating scripts the user didn't ask for (eval, preflight, promotion, submission wrappers)
- Writing long sbatch scripts that re-map every config field to a shell variable
- Storing checkpoints/outputs/data on `$HOME` instead of scratch
- Creating submission wrappers, preflight scripts, or promotion scripts — just run commands directly
- Putting anything other than training/eval sbatch scripts in `slurm/`
- Mixing local and remote paths in one command
- Copying large datasets with `scp -r` when resumable `rsync -P` is needed
- Submitting without verifying dataset paths exist on the cluster
- Submitting without checking script/account/partition settings
- Creating indirection layers for container bind mounts (symlinks, wrapper dirs) instead of binding directly
- Resolving config paths from `SCRIPT_DIR` / `$(dirname $0)` — Slurm copies scripts to spool, so these point to the wrong place
- Overwriting `LD_LIBRARY_PATH` instead of appending — removes Apptainer's injected NVIDIA driver libs
- Using `--export` env overrides or rsync to change config on the cluster instead of committing and pushing through git
- Requiring env vars on the `sbatch` line — settings belong in the script or config YAML, not the submit command
- Forgetting `module load` for the container runtime (check cluster profile)
- Monitoring wrong user (`squeue -u`) due hardcoded shortname
- Using container paths not mounted in `apptainer exec --bind`
- Treating `squeue` RUNNING status or `nvidia-smi` GPU activity as proof that training is healthy — a job can be alive and holding GPU memory while deadlocked on I/O, stuck on a collective, or spinning without advancing steps; always check the `.out` log for advancing step counts
- Not checking disk usage before or during training — a full filesystem silently deadlocks checkpoint writes, which can hang the training loop for hours until walltime kills the job
- Running training without W&B tracking/sync and losing experiment visibility
