# Isambard Cluster Profile

## SSH Access

Isambard uses signed SSH certificates via a CLI tool called Clifton. Certificates are valid for **12 hours**.

## Authoritative Resources

- User documentation: [https://docs.isambard.ac.uk/](https://docs.isambard.ac.uk/)
- SSH and login: [https://docs.isambard.ac.uk/guides/login/](https://docs.isambard.ac.uk/guides/login/)

Before assuming SSH is unavailable, try the configured local SSH alias first:

```
ssh isambard "<command>"
```

The user may already have run:

```
clifton auth
```

`clifton auth` opens a browser for authentication and writes a certificate to `~/.ssh/`. If the certificate has expired, SSH will fail with `Permission denied (publickey)` or `Connection closed by remote host`.

If a current SSH attempt fails with those errors, do not keep retrying. Tell the user the exact command attempted, ask them to run `clifton auth`, then retry after they confirm.

The direct Isambard hostname form is:

```
ssh <PROJECT_ID>.aip2.isambard
```

## Hardware

- Grace Hopper (CPU+GPU) Superchip cluster — **arm64** architecture.
- Each GPU allocation gives 1 GH200 with 72 CPU cores and 115 GB Grace RAM.
- Container runtime: `apptainer`.

## Storage

- Home: `/home/<project_code>/<username>` — small quota, code only. Use for `git clone`.
- Scratch: `/scratch/<project_code>/<username>` — large (TBs), no backup. Use for containers, datasets, checkpoints, outputs, W&B caches.

Before uploading artifacts or submitting jobs, check both home and scratch.
Home can fill from accidental large files and then Slurm may fail before it can
write `.out` / `.err` logs. Scratch can fill from containers, datasets,
checkpoints, and W&B offline runs.

```bash
df -h /home/<project_code>/<username> /scratch/<project_code>/<username>
du -sh /home/<project_code>/<username> /scratch/<project_code>/<username>/<repo> 2>/dev/null || true
du -h --max-depth=1 /scratch/<project_code>/<username>/<repo> 2>/dev/null | sort -hr
```

If Slurm logs are written under the repo in home, verify home has enough free
space immediately before `sbatch`, or point `#SBATCH --output` and
`#SBATCH --error` at a scratch log directory.

## Modules

Include in sbatch scripts before running containers:

```bash
module purge
module load brics/apptainer-multi-node
```

## Slurm

- Docs: [https://docs.isambard.ac.uk/user-documentation/guides/slurm/](https://docs.isambard.ac.uk/user-documentation/guides/slurm/)
- You **must** specify GPU resources with `--gpus` or `--gpus-per-*`.
- Partition: `workq`.
- Max walltime: 1 day (`--time=1-00:00:00`). Jobs that hit the limit are killed — checkpoint frequently and support resume.

### Full-node GPU jobs

To use all 4 GPUs on a node, you **must** use `#SBATCH --mem=0G` (request all memory) and `#SBATCH --exclusive`. You cannot access all GPUs without requesting all the node's memory.

**Do not set `CUDA_VISIBLE_DEVICES` or `NVIDIA_VISIBLE_DEVICES`** in sbatch scripts. Slurm assigns devices automatically — manually overriding conflicts with the scheduler and can silently restrict which GPUs the job sees.

### Debugging (use `srun`, not sbatch scripts)

```bash
# Interactive shell on a compute node with 1 GPU
srun --gpus=1 --time=00:30:00 --pty /bin/bash --login

# Run a container interactively
srun --gpus=1 --time=00:30:00 apptainer shell --nv <sif_path>

# Attach to a running job for debugging
srun --ntasks=1 --gpus=1 --jobid=<job_id> --overlap --pty /bin/bash -l
```

### Training (sbatch scripts in `slurm/`)

```bash
sbatch slurm/<training_script>.sh
```

### Queue Start Estimates

Use Slurm's start-time estimator to see when queued jobs are expected to begin:

```bash
squeue --me --start
```

The `START_TIME` and `SCHEDNODES` fields are estimates and can change as
priority, reservations, and backfill opportunities change. This is especially
useful after submitting full-node GPU jobs that sit in `PD (Priority)`.

## GitHub Access

GitHub SSH keys don't work from Isambard. Clone repos using HTTPS with a personal access token (PAT). The user may have a token file on the cluster (e.g. `~/pat.txt`). Use `GIT_ASKPASS` to pass the token without exposing it in command history:

```bash
TOKEN="$(cat ~/pat.txt)"
ASKPASS="$(mktemp)"
cat > "$ASKPASS" <<'EOF'
#!/bin/sh
case "$1" in *Username*) echo "x-access-token";; *Password*) cat ~/pat.txt;; esac
EOF
chmod 700 "$ASKPASS"
GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 git clone https://github.com/<org>/<repo>.git
rm -f "$ASKPASS"
```

## Wandb Sync (inside Apptainer)

Do not install into the `.sif`; it is read-only. Install AutoHPC helpers into a
writable scratch environment that is mounted into the container, such as the
project venv:

```bash
srun --ntasks=1 --cpu-bind=cores apptainer exec \
  --bind /scratch/<project>/<user>:/scratch/<project>/<user> \
  --bind /home/<project>/<user>:/home/<project>/<user> \
  /scratch/<project>/<user>/<repo>/container/<sif_file> \
  bash -lc 'export UV_PROJECT_ENVIRONMENT=/scratch/<project>/<user>/<repo>/.venv
            uv pip install -e /home/<project>/<user>/autohpc/wandb-sync'
```

Then run sync inside the container/runtime environment:

```bash
srun --ntasks=1 --cpu-bind=cores apptainer exec \
  --bind /scratch/<project>/<user>:/scratch/<project>/<user> \
  --bind /home/<project>/<user>:/home/<project>/<user> \
  /scratch/<project>/<user>/<repo>/container/<sif_file> \
  bash -lc 'export UV_PROJECT_ENVIRONMENT=/scratch/<project>/<user>/<repo>/.venv
            autohpc-wandb-sync sync \
              --entity <wandb-entity> \
              --project <wandb-project> \
              --wandb-token-file /home/<project>/<user>/.wandb_token \
              --yes \
              /scratch/<project>/<user>/<repo>/wandb/offline-run-...'
```

If launching from outside the container runtime, use the explicit wrapper mode:

```bash
autohpc-wandb-sync launch \
  --ssh-host <PROJECT_ID>.aip2.isambard \
  --module brics/apptainer-multi-node \
  --entity <wandb-entity> \
  --project <wandb-project> \
  --offline-run-dir /scratch/<project>/<user>/<repo>/wandb/offline-run-... \
  --container /scratch/<project>/<user>/<repo>/container/<sif_file> \
  --bind /scratch/<project>/<user>:/scratch/<project>/<user> \
  --bind /home/<project>/<user>:/home/<project>/<user> \
  --wandb-token-file /home/<project>/<user>/.wandb_token \
  --dry-run
```

Always set both `--entity` and `--project`; do not rely on defaults embedded in
the offline run. Do not inline API keys in commands or logs; read them from a
private token file inside the remote shell.

## Notes

- Login nodes are assigned randomly; do not assume a persistent session.
- Do not run compute workloads on login nodes — use Slurm.
- `/tmp` is node-local and not shared between login and compute nodes. Write scripts and temp files to scratch or home.
- Architecture is arm64 — Docker images built for amd64 will not work. Check with `uname -m` on the cluster.
- `--nv` flag in Apptainer injects host NVIDIA libs via `LD_LIBRARY_PATH`. Never overwrite this variable inside the container — always append to it.
- Do not store secrets or credentials here.
