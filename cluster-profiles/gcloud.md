# GCloud GPU Profile

## Prerequisites — do this before proceeding

Create the following as always-on agent rules in the **target repo**. These carry operational gotchas that must persist across sessions.

**GCloud GPU quota:** Before creating a GPU instance, always verify regional quota first. If quota is 0, the create command fails with a confusing error — don't debug the instance config, check quota.

**GCloud instance lifecycle:** These are ephemeral VMs, not persistent cluster
nodes. GPU instances are expensive, but never delete an instance without explicit user confirmation in the current conversation, even after checkpoints have been uploaded or downloaded — the user may intend to reuse the VM. Ask promptly once durable artifacts are safe.

## Authoritative Resources

- Compute Engine GPU docs: [https://cloud.google.com/compute/docs/gpus](https://cloud.google.com/compute/docs/gpus)
- GPU-optimized machine types: [https://cloud.google.com/compute/docs/accelerator-optimized-machines](https://cloud.google.com/compute/docs/accelerator-optimized-machines)
- Deep Learning VM images: [https://cloud.google.com/deep-learning-vm/docs/images](https://cloud.google.com/deep-learning-vm/docs/images)

## Hardware

- Architecture: **amd64** (x86_64).
- GPU type, machine type, region, and zone all depend on the user's project quota. Ask the user what they have allocated.

## How This Differs From Slurm Clusters

This is a single VM with Docker, not a multi-node Slurm cluster. Key differences:

- **No Slurm.** No `sbatch`, `squeue`, `srun`. Jobs run directly via `docker run --gpus all`.
- **No shared filesystem.** Everything lives on the boot disk.
- **Docker, not Apptainer.** Container runtime is Docker with the NVIDIA Container Toolkit.
- **Ephemeral.** Instances are created for a training run and deleted after. There is no persistent login node.

This means `hpc-training-operations/SKILL.md` (sbatch templates, Slurm monitoring) does **not** apply. The workflow is: create VM → SSH in → docker run → collect results → delete VM.

Skills that still apply:
- `hpc-container-promotion/SKILL.md` — Phase 1 (local Docker build/test) and Phase 3 (the Cloud VM path — clone and build on the VM, no export/upload).
- `hpc-run-tracking/SKILL.md` — run logs are still valuable, just replace Slurm job IDs with instance name/zone.
- `eval-tracking/SKILL.md` — same format, different provenance fields.
- `hpc-dataset-adaptation/SKILL.md` — still applies if the user's data format differs.

## Quota Verification

Always check quota before attempting to create an instance. Ask the user which GPU type and region they have quota for.

```bash
# Check quota for a specific region (filter by the GPU's quota metric name)
gcloud compute regions describe <region> \
    --format="list(quotas.filter(metric:NVIDIA_<GPU_TYPE>_GPUS))"

# Global GPU quota (all regions combined)
gcloud compute project-info describe \
    --format="list(quotas.filter(metric:GPUS_ALL_REGIONS))"
```

If quota is 0, the user needs to request a quota increase via the Cloud Console before proceeding. Do not retry instance creation with zero quota — it will never succeed.

Once you know the GPU type, find which zones support it:

```bash
gcloud compute accelerator-types list \
    --filter="name:<accelerator_name>" \
    --format="table(zone,name)"
```

Use these zones in the sniper loop below.

## Creating an Instance

GPU instances face frequent stockouts — zones can be exhausted for minutes to hours. Use a retry loop that cycles all compatible zones to grab the first available machine. Let this run in the background; it will break out when one succeeds.

```bash
while true; do
  for zone in <zone-1> <zone-2>; do
    echo "$(date -u '+%H:%M:%S') Attempting <machine_type> in $zone..."
    gcloud compute instances create <instance_name> \
      --machine-type=<machine_type> \
      --accelerator=type=<accelerator_name>,count=<gpu_count> \
      --zone="$zone" \
      --image-family=common-cu128-ubuntu-2204-nvidia-570 \
      --image-project=deeplearning-platform-release \
      --boot-disk-size=2000GB \
      --boot-disk-type=pd-ssd \
      --maintenance-policy=TERMINATE \
      --no-restart-on-failure \
      -q 2>&1 && echo "SUCCESS in $zone" && break 2
    echo "Stockout in $zone."
  done
  echo "All zones full. Retrying in 30s..."
  sleep 30
done
```

Fill in the zones from the quota verification step. Multiple zones in the same region often have independent capacity — cycling them improves chances.

If a smaller machine type is stocked out, a larger variant (more GPUs) sometimes has better availability. It costs more per hour but may be the only way to get capacity.

Notes:
- `--maintenance-policy=TERMINATE` and `--no-restart-on-failure` are **required** for GPU instances. GCP does not allow live migration of GPU VMs.
- 2TB PD-SSD boot disk houses the OS, datasets, container images, and checkpoints in one place. GCP will warn that the disk is larger than the image — this is expected and the DL VM handles auto-resizing.
- `common-cu128-ubuntu-2204-nvidia-570` provides CUDA 12.8 + NVIDIA driver 570 on Ubuntu 22.04. Check the [DL VM image list](https://cloud.google.com/deep-learning-vm/docs/images) for newer families.
- The `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` error sometimes suggests alternative zones in the response — the multi-zone loop already handles this.

## SSH Access

```bash
# Interactive
gcloud compute ssh <instance_name> --zone=<zone>

# Run a command remotely
gcloud compute ssh <instance_name> --zone=<zone> --command='<command>'
```

No certificates or special auth — `gcloud compute ssh` handles IAP tunneling and key management automatically.

**Docker group gotcha:** After `usermod -aG docker $USER`, interactive SSH sessions pick up the group via `newgrp docker`, but non-interactive `--command=` invocations do not. Wrap docker commands in `sg docker -c "..."` when running remotely:

```bash
gcloud compute ssh <instance_name> --zone=<zone> --command='
sg docker -c "docker logs training 2>&1 | tail -50"
'
```

## Environment Setup (First Boot)

The DL VM image includes NVIDIA drivers but may not have Docker configured for GPU passthrough. Run this once after creating the instance:

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Configure NVIDIA Container Toolkit for Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Add current user to docker group (avoids sudo for docker commands)
sudo usermod -aG docker $USER
newgrp docker
```

Verify GPU visibility through Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.0.1-base-ubuntu22.04 nvidia-smi
```

If this doesn't show the GPU, the nvidia-ctk step failed — rerun it.

## Storage

Everything lives on the boot disk. There is no separate scratch filesystem.

- Container images: built or pulled on the VM (stored in `/var/lib/docker`).
- Datasets: upload to the home directory or any path on the boot disk.
- Checkpoints/outputs: written to a bind-mounted directory on the boot disk.
- W&B cache: bind-mount a directory from the host into the container.

Before building/pulling images, uploading datasets, or starting training, check
boot disk usage. A full boot disk can break Docker pulls/builds, prevent logs
from being written, or fail checkpoint/W&B writes mid-run.

```bash
df -h / /var/lib/docker /path/to/outputs 2>/dev/null || df -h
du -sh /var/lib/docker /path/to/outputs /path/to/repo 2>/dev/null || true
du -h --max-depth=1 /path/to/outputs 2>/dev/null | sort -hr
docker system df 2>/dev/null || true
```

If the output path is close to full, clean up or resize before launching. Do
not rely on the nominal boot disk size; Docker layers, HF caches, datasets,
checkpoints, and W&B offline logs all share the same disk.

To upload files from your local machine:

```bash
gcloud compute scp /local/path <instance_name>:/remote/path --zone=<zone>
```

For large transfers, use `gcloud compute scp --recurse`:

```bash
gcloud compute scp --recurse /local/dir <instance_name>:/remote/path --zone=<zone>
```

## GitHub Access

For private repos, clone using HTTPS with a personal access token (PAT). The user may have a token file on the VM (e.g. `~/pat.txt`). Use `GIT_ASKPASS` to pass the token without exposing it in command history:

```bash
ASKPASS="$(mktemp)"
cat > "$ASKPASS" <<'EOF'
#!/bin/sh
case "$1" in *Username*) echo "x-access-token";; *Password*) cat ~/pat.txt;; esac
EOF
chmod 700 "$ASKPASS"
GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 git clone https://github.com/<org>/<repo>.git
rm -f "$ASKPASS"
```

Do not inline the PAT in clone URLs or scripts.

## Deploying Your Container

**Build on the VM, not locally.** Do not cross-build an amd64 image on an ARM Mac and upload it. Instead:

1. Clone the repo on the VM (it's amd64, so the build is native).
2. Run the repo's `docker/build_docker.sh` (or `docker build`) on the VM.
3. Run training directly — no export, no conversion, no upload.

This is faster than cross-building, avoids architecture mismatches, and skips uploading a multi-GB image. The container promotion skill's Phase 3 (export → convert → upload) does not apply here.

If the image is already published to a registry (Docker Hub, GHCR, etc.), `docker pull` on the VM works too.

```bash
# Option A: build on the VM
git clone https://github.com/<org>/<repo>.git
cd <repo>
./docker/build_docker.sh    # native amd64 build

# Option B: pull from registry
docker pull <registry>/<image>:<tag>
```

Verify the GPU is visible to the built image before starting training:

```bash
docker run --rm --gpus all <image>:<tag> nvidia-smi
```

## Running Training

No sbatch scripts. Run training directly with Docker.

**Always use `--shm-size=16g` (or higher).** Docker's default shared memory is 64MB, which crashes PyTorch DataLoader when `num_workers > 0`.

**Always set `-e PYTHONUNBUFFERED=1`.** Without it, Python buffers stdout and logs appear delayed or not at all in `docker logs`.

**Always persist logs to a file.** `docker logs` output is lost if the container crashes or is removed. Use `tee` to write to a bind-mounted path — this is the equivalent of Slurm's `.out`/`.err` files.

**Always check disk before launch.** The bind-mounted outputs path and Docker
storage must have enough free space for logs, checkpoints, W&B offline files,
and any temporary artifacts. Run the storage checks above immediately before
starting the training container.

```bash
docker run --detach --gpus all --name training \
    --shm-size=16g \
    -v /path/to/repo:/workspace/repo \
    -v /path/to/outputs:/workspace/outputs \
    -e PYTHONUNBUFFERED=1 \
    -e WANDB_MODE=offline \
    <image>:<tag> \
    bash -c "python <entry_point> --config <config_path> \
        2>&1 | tee /workspace/outputs/train.log"
```

Monitor with `docker logs -f training` or `tail -f` the log file on the host. The log file survives container restarts.

After starting a training run, create a run log per `hpc-run-tracking/SKILL.md`. Use instance name and zone as the `execution_id` instead of a Slurm job ID. After evaluating checkpoints, create eval logs per `eval-tracking/SKILL.md`.

## Debugging

```bash
# Interactive shell in a running container
docker exec -it training bash

# Check container exit code and state
docker inspect training --format='{{.State.ExitCode}} {{.State.Status}}'

# Check for OOM kills
dmesg | grep -i oom

# GPU health
nvidia-smi
```

## W&B Sync

W&B is typically not installed on the host. Run `wandb sync` inside the container:

```bash
docker run --rm --network=host \
    -e WANDB_API_KEY="$(cat ~/.wandb_key)" \
    -v /path/to/repo:/workspace/repo \
    <image>:<tag> \
    wandb sync /workspace/repo/wandb/<offline-run-dir>
```

Do not inline the API key in commands or scripts. Store it in a dotfile on the VM (e.g. `~/.wandb_key`).

## Checkpoint Integrity

On the VM holding the checkpoint, generate a manifest on the final publish
bundle immediately before upload:

```bash
uv pip install -e ../autohpc/checkpoint-integrity
manifest-checkpoint <checkpoint_or_publish_dir>
```

Include `CHECKPOINT_MANIFEST.json` in the upload. Run
`verify-checkpoint <downloaded_dir>` after download and before eval or
deployment. This verifies file bytes only; behavioral checks belong to the eval
and deployment protocols.

## Deleting the Instance

GPU instances are expensive, so ask about cleanup when work is done. Download or
upload results first, then ask the user whether to delete, stop/reuse, or keep
the VM alive. Do not infer permission from a successful upload, a completed
sync, or a prior general cleanup rule. The boot disk is destroyed with the
instance unless it was created separately.

```bash
# Download results before deleting
gcloud compute scp --recurse <instance_name>:/path/to/outputs ./local-outputs --zone=<zone>

# Check what's running
gcloud compute instances list

# Only after explicit user confirmation in this conversation.
gcloud compute instances delete <instance_name> --zone=<zone> --quiet
```

## Notes

- GPU VMs cannot be live-migrated. GCP will terminate them for host maintenance — training must support checkpoint/resume.
- The boot disk persists only as long as the instance exists. If you delete the instance, the boot disk is deleted too (unless you created it separately). Download any results before deleting.
- Never delete an instance, remove output directories, prune Docker layers, or delete disks without explicit user confirmation. Uploading checkpoints does not mean the VM is no longer needed.
- Code written for multi-GPU DDP (`torchrun`) may use bare `cuda` instead of `cuda:0` as the device. This causes errors on single-GPU runs — check and fix before submitting.
- Do not store secrets or credentials in this profile.
