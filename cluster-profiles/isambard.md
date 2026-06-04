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

Do not run Python helpers or compute work on the login node. Run the AutoHPC
sync helper from the agent/local environment and let it SSH to Isambard to
launch only the `srun apptainer exec ... wandb sync` command remotely:

```bash
autohpc-wandb-sync \
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

After the dry run looks correct, remove `--dry-run` and add `--yes`. Always set
both `--entity` and `--project`; do not rely on defaults embedded in the
offline run. Do not inline API keys in commands or logs; read them from a
private token file inside the remote shell.

## Notes

- Login nodes are assigned randomly; do not assume a persistent session.
- Do not run compute workloads on login nodes — use Slurm.
- `/tmp` is node-local and not shared between login and compute nodes. Write scripts and temp files to scratch or home.
- Architecture is arm64 — Docker images built for amd64 will not work. Check with `uname -m` on the cluster.
- `--nv` flag in Apptainer injects host NVIDIA libs via `LD_LIBRARY_PATH`. Never overwrite this variable inside the container — always append to it.
- Do not store secrets or credentials here.
