# Isambard Cluster Profile

## Authoritative Resources

- User documentation: [https://docs.isambard.ac.uk/](https://docs.isambard.ac.uk/)
- SSH and login: [https://docs.isambard.ac.uk/guides/login/](https://docs.isambard.ac.uk/guides/login/)

## SSH Access

Isambard uses signed SSH certificates via a CLI tool called Clifton. Certificates are valid for **12 hours**.

Before any SSH operation (preflight, upload, job submission), the user must run:

```
clifton auth
```

This opens a browser for authentication and writes a certificate to `~/.ssh/`. If the certificate has expired, SSH will fail with `Permission denied (publickey)`.

After auth, connect with:

```
ssh <PROJECT_ID>.aip2.isambard
```

The agent cannot run `clifton auth` — if SSH fails, ask the user to run it and retry.

## Hardware

- Grace Hopper (CPU+GPU) Superchip cluster — **arm64** architecture.
- Each GPU allocation gives 1 GH200 with 72 CPU cores and 115 GB Grace RAM.
- Container runtime: `apptainer`.

## Slurm

- Docs: [https://docs.isambard.ac.uk/user-documentation/guides/slurm/](https://docs.isambard.ac.uk/user-documentation/guides/slurm/)
- You **must** specify GPU resources with `--gpus` or `--gpus-per-*`.
- Partition: `workq`.

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

## Notes

- Login nodes are assigned randomly; do not assume a persistent session.
- Do not run compute workloads on login nodes — use Slurm.
- Architecture is arm64 — Docker images built for amd64 will not work. Check with `uname -m` on the cluster.
- Do not store secrets or credentials here.
