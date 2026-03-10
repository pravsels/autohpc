---
name: hpc-training-operations
description: Use when running ML training workflows on Slurm-based HPC clusters, including environment setup, data and image transfer, job submission, monitoring, debugging, and storage cleanup, especially when notes are ad-hoc, usernames differ, or commands may expose secrets.
---

# HPC Training Operations

## Overview

Use this when acting on cluster notes that may contain stale usernames, paths, job IDs, or secret-bearing commands.
Core principle: parameterize first, preflight second, execute staged commands third.

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

Slurm scripts go in `slurm/` in the target repo — only for repeatable jobs (training, eval). Do not write sbatch scripts for debugging or one-off operations. Use `srun` commands for those instead.

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
3. Preflight: host reachable, tools exist, paths/quota valid.
3. Run stages: setup -> transfer -> submit -> monitor -> debug -> cleanup.
4. Pause for confirmation on high-impact actions (`scancel`, overwrite sync, destructive cleanup).
5. Never inline secrets; use hidden prompt input or secure env handling.
6. If repo has `slurm/` scripts with profiles, submit from script path (not copied one-liners).
7. Enable observability: live logs, job accounting, and required W&B tracking.

```dot
digraph hpc_flow {
    "Start HPC workflow" [shape=ellipse];
    "Parameterize values" [shape=box];
    "Preflight passes?" [shape=diamond];
    "Run staged commands" [shape=box];
    "High-impact step?" [shape=diamond];
    "Ask confirmation / secure input" [shape=box];
    "Start HPC workflow" -> "Parameterize values";
    "Parameterize values" -> "Preflight passes?";
    "Preflight passes?" -> "Run staged commands" [label="yes"];
    "Preflight passes?" -> "Parameterize values" [label="no"];
    "Run staged commands" -> "High-impact step?";
    "High-impact step?" -> "Ask confirmation / secure input" [label="yes"];
}
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

Use this bootstrap before running setup/submit/monitor commands:

```bash
export SSH_ALIAS="<your_cluster_alias>"
export UNIX_USER="${UNIX_USER:-$(whoami)}"             # Or shortname.project if required
export PROJECT_NAME="<project_name>"
export PROJECT_CODE="<project_code>"
export PROJECT_DIR="$HOME/${PROJECT_NAME}"
export SCRATCH_DIR="/scratch/${PROJECT_CODE}/${UNIX_USER}/${PROJECT_NAME}"

ssh "$SSH_ALIAS" "hostname && command -v sbatch squeue scancel apptainer"
ssh "$SSH_ALIAS" "mkdir -p \"$PROJECT_DIR\" \"$SCRATCH_DIR\" && du -sh \"$SCRATCH_DIR\" || true"

# Optional repo-aware submit pattern
# ssh "$SSH_ALIAS" "cd \"$PROJECT_DIR\" && sbatch slurm/<script>.sh <profile>"
```

Private repo auth: do not embed PAT in URL.

## Observability Guidance

- Minimum visibility: queue status (`squeue`), job logs (`slurm-<job_id>.out/.err`), and GPU telemetry (`nvidia-smi`).
- Post-run visibility: capture `sacct`/`seff` summaries for memory, runtime, and exit status.
- W&B tracking is required for training runs.
- Prefer offline-first logging on restricted clusters, then sync later.
- Never inline `WANDB_API_KEY`; pass via secure environment setup.

## Rationalization Table

| Excuse | Reality |
|---|---|
| "Run my old notes exactly, no checks" | Stale paths and usernames break jobs; always preflight first. |
| "Hardcode my user/alias; it is faster" | Skills must be reusable; parameterize identity and paths. |
| "Paste token/API key inline for speed" | Inline secrets leak in history and logs; use env or secure prompts. |
| "Cancel everything now, we can recover later" | `scancel` is high impact; confirm scope before execution. |

## Red Flags - Stop and Re-check

- Commands contain hardcoded user handles, project IDs, or old job IDs
- Any secret appears inline (`PAT`, `WANDB_API_KEY`, tokenized git URL)
- Destructive commands run without confirmation (`scancel`, overwrite sync)
- Host/path assumptions are unverified (`~/...` vs `/scratch/...`)

All of these mean: pause, parameterize, preflight, then continue.

## Common Mistakes

- Mixing local and remote paths in one command
- Copying large datasets with `scp -r` when resumable `rsync -P` is needed
- Submitting without checking script/account/partition settings
- Running ad-hoc sbatch commands when repo `slurm/*.sh` already encodes the environment
- Monitoring wrong user (`squeue -u`) due hardcoded shortname
- Using container paths not mounted in `apptainer exec --bind`
- Relying only on queue state without tailing logs or collecting post-run accounting
- Running training without W&B tracking/sync and losing experiment visibility
