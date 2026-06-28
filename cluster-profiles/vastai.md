# VastAI GPU Profile

## Prerequisites - do this before proceeding

Create the following as always-on agent rules in the **target repo**. These
carry operational gotchas that must persist across sessions.

**VastAI instance lifecycle:** VastAI instances are rented cloud machines. Always
check for existing instances before creating a new one. Never destroy an instance without explicit user confirmation in the current conversation, even after checkpoints have been uploaded or downloaded — the user may intend to reuse the machine. Idle GPU machines are expensive, so ask promptly once durable artifacts are safe.

**VastAI offers are ephemeral:** Offer IDs come from live marketplace search and
can disappear between UI inspection and CLI rental. Always re-query the CLI
immediately before creating an instance. Do not rely on stale UI IDs.

**VastAI secrets:** The API key should already live in the VastAI CLI config or a
private environment variable. Do not print, paste, or store API keys in profile
files, scripts, run logs, or command transcripts.

## Authoritative Resources

- CLI hello world: [https://docs.vast.ai/cli/hello-world](https://docs.vast.ai/cli/hello-world)
- CLI overview: [https://vast.ai/developers/cli](https://vast.ai/developers/cli)
- Instance creation API: [https://docs.vast.ai/api-reference/creating-instances-with-api](https://docs.vast.ai/api-reference/creating-instances-with-api)
- Template API: [https://docs.vast.ai/api-reference/creating-and-using-templates-with-api](https://docs.vast.ai/api-reference/creating-and-using-templates-with-api)

## Hardware

- Architecture: usually **amd64** (x86_64) for the high-end NVIDIA GPU hosts
  used by AutoHPC runs. Verify with `uname -m` after boot.
- GPU type, count, disk, network, location, and reliability depend on the live
  marketplace offer.
- For high-memory training runs, prefer offers with enough per-GPU VRAM, high
  reliability, and enough local disk for HF cache, datasets, checkpoints, and
  W&B offline logs.
- For cost-sensitive jobs that fit in consumer GPU memory, consumer GPU offers
  can be much cheaper than datacenter GPUs. Check per-GPU VRAM, network, and
  geolocation before choosing this path.

## How This Differs From Slurm Clusters

This is a rented Docker host, not a Slurm cluster. Key differences:

- **No Slurm.** No `sbatch`, `squeue`, or `srun`. Jobs run directly in Docker.
- **Docker runtime.** VastAI launches a Docker image as the instance container.
- **Marketplace capacity.** Offers appear and disappear; search immediately
  before rent.
- **Ephemeral local disk.** Treat instance disk as temporary unless using an
  explicit VastAI volume. Download artifacts before destroying the instance.

This means `hpc-training-operations/SKILL.md` Slurm submission templates do not
apply. The workflow is: search offers -> create instance -> SSH -> clone/build or
use the launched image -> run training -> sync/download artifacts -> destroy.

Skills that still apply:

- `hpc-container-promotion/SKILL.md` - Phase 1 local Docker validation and the
  cloud-VM pattern: build/pull Docker on the instance, no Apptainer conversion.
- `hpc-run-tracking/SKILL.md` - use instance ID/label as the execution ID.
- `hpc-dataset-adaptation/SKILL.md` - still applies when the user's dataset
  schema differs.
- `eval-tracking/SKILL.md` and `checkpoint-passport/SKILL.md` - unchanged after
  checkpoints exist.

## CLI Setup And Auth

Use the VastAI CLI for all lifecycle operations.

```bash
vastai show user --raw
vastai show instances-v1 --raw
```

`show user` verifies auth without exposing the API key. `show instances-v1`
checks whether a suitable instance already exists before renting another one.
With VastAI CLI 1.1.1, `show instances` still works but is deprecated.

If the CLI is missing, install it using the current VastAI docs. Do not install
or configure the CLI on a remote training host unless needed for that workflow.
Keep the API key in the CLI config, normally under the user's home directory, or
pass it through a private environment variable only for the command that needs
it.

## Searching Offers

Use raw JSON and parse it deliberately. Vast query syntax and raw output fields
are easy to misread, so validate the exact fields returned by the live CLI.

```bash
vastai search offers \
  'num_gpus>=1 reliability>0.98 rentable=true' \
  --raw
```

For high-memory multi-GPU hosts, adapt this pattern:

```bash
vastai search offers \
  'num_gpus=8 gpu_ram>=76 reliability>0.98 rentable=true verified=true' \
  --raw -o 'dph' --limit 20 |
python3 -c 'import json,sys
offers=json.load(sys.stdin)
print("count", len(offers))
for o in offers:
    print({k:o.get(k) for k in [
        "id","machine_id","gpu_name","num_gpus","gpu_ram","dph_total",
        "reliability","verified","inet_up","inet_down","disk_space",
        "cpu_cores","cpu_ram","cuda_vers","dlperf","geolocation"
    ]})'
```

Notes:

- `id` is the offer ID to pass to `vastai create instance`.
- Offer IDs are one-shot marketplace asks; re-search if creation fails.
- Exact-ID searches can be flaky or stale. If an ID from the UI or a previous
  search does not resolve, refresh the broader matching query and choose from
  the current raw JSON rather than debugging the old ID.
- Confirm `disk_space` and requested `--disk` are enough for the run. A cheap
  multi-GPU host with tiny disk may be unusable for checkpoint-heavy training.
- Prefer offers with high `reliability`, enough upload/download bandwidth, and
  sufficient local disk. Price alone is not the decision.
- If using `--direct` SSH, include `direct_port_count>=1` in searches when the
  field is available.

Example offer fields to inspect:

```text
id=<offer_id> gpu_name=<gpu_model> num_gpus=<count> gpu_ram=<per_gpu_mb>
dph_total=<total_hourly_cost> reliability=<score> disk_space=<available_gb>
inet_up=<mbps> inet_down=<mbps> geolocation=<region>
```

Do not hardcode offer IDs. They are examples of the field shape, not stable
resources.

## Creating An Instance

At minimum, `create instance` needs an offer ID and an image. For AutoHPC
training, default to renting with the target repo's already-built training image
from a registry. Prefer SSH direct mode unless the target repo has a known
connection requirement.

Choose the image deliberately before renting:

- **Default path:** build and push the target repo's training image to a registry
  that VastAI can pull, then create the instance with `--image <registry/image>`.
  This tests the real runtime from first boot, avoids repeated dependency setup
  on paid GPU time, and catches image/runtime compatibility before launching a
  long run.
- **Probe path:** launch a compatible base image, such as an NVIDIA PyTorch image,
  only for short CUDA or marketplace checks. This verifies GPU/CUDA access, but
  dependency compatibility is still unproven until the repo stack installs and a
  real smoke test runs.

VastAI cannot launch from a local Docker image on the agent machine. If the
desired image only exists locally, push it to a registry before using it in
`vastai create instance`.

```bash
vastai create instance <offer_id> \
  --image <registry>/<image>:<tag> \
  --disk 4000 \
  --ssh \
  --direct \
  --label <project>-<run> \
  --onstart-cmd "echo ready && nvidia-smi"
```

Use `--disk` large enough for the whole working set. For large training runs,
4TB is a reasonable starting point for repo, datasets, HF cache, checkpoints,
and W&B offline logs. If scheduling should fail rather than create a stopped
instance, add `--cancel-unavail`.

Account for total hourly cost, not only the displayed GPU base price. Large disk
allocations can materially increase the hourly rate. Compare the CLI's base GPU
price with `dph_total` after setting the requested disk size.

After create, capture the returned `new_contract` in the run log.

```bash
vastai show instances-v1 --raw
```

Record:

- instance ID / contract ID
- offer ID
- label
- image
- disk size
- GPU model/count
- hourly price
- location

## SSH Access

Use the CLI-provided SSH workflow. Query the instance after it enters a running
state and use the SSH command or connection info returned by VastAI.

```bash
vastai show instances-v1 --raw
vastai ssh-url <instance_id>
```

If `vastai ssh-url` is unavailable in the installed CLI version, inspect
`vastai --help` or the instance JSON for the current SSH command fields.

VastAI can report the contract as running before the container is online and
before SSH is injected. If SSH returns `connection refused` immediately after
creation, poll `show instances-v1` and retry until the instance/container status
is online before investigating SSH keys or auth.

Once connected, verify the host:

```bash
uname -m
nvidia-smi
pwd
df -h
```

For repeated agent operations, open an SSH ControlMaster connection and reuse it
instead of creating a fresh SSH handshake for every command:

```bash
export VAST_SSH_CTRL="/tmp/vast-ssh-%r@%h:%p"
ssh -fNM \
  -i ~/.ssh/id_ed25519 \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  -o ControlMaster=yes \
  -o ControlPath="$VAST_SSH_CTRL" \
  -o ControlPersist=30m \
  -p <port> root@<host>

ssh \
  -i ~/.ssh/id_ed25519 \
  -o IdentitiesOnly=yes \
  -o ControlPath="$VAST_SSH_CTRL" \
  -p <port> root@<host> '<command>'

ssh \
  -i ~/.ssh/id_ed25519 \
  -o IdentitiesOnly=yes \
  -o ControlPath="$VAST_SSH_CTRL" \
  -p <port> root@<host> -O exit
```

Keep SSH, `rsync`, and `scp` option syntax separate. `ssh` uses `-p <port>`, but
`scp` uses `-P <port>`. A common failure is reusing an `ssh` option string with
lowercase `-p` in `scp`, which makes `scp` connect to the wrong port and fail
with auth errors. Prefer `rsync -e "ssh <ssh_opts>"` for directories and define
an explicit `scp` option string when copying individual files.

## Storage

By default, treat instance disk as ephemeral local storage.

- Repo: clone into a stable workspace path inside the instance.
- Datasets: place on the rented disk or attach a VastAI volume if persistence is
  required.
- HF cache: put under the large disk path, not a tiny default cache path.
- Checkpoints/outputs/W&B: write to a run-scoped output directory.

Before building/pulling images, copying datasets, or starting training, inspect
disk space on the mounted workspace/output paths. VastAI disks are rented per
instance and can be smaller than expected if the offer or `--disk` request was
wrong.

```bash
df -h / /workspace /workspace/outputs 2>/dev/null || df -h
du -sh /workspace /workspace/outputs /workspace/.cache 2>/dev/null || true
du -h --max-depth=1 /workspace 2>/dev/null | sort -hr
docker system df 2>/dev/null || true
```

Stop before long training if the filesystem that holds logs, checkpoints, W&B
offline runs, or Docker layers is near full. Clean stale Docker layers, old
outputs, or failed-run caches only after confirming they are not the current
run's durable artifacts.

Before deleting the instance, download or upload all durable artifacts:

```bash
# Use the SSH/SCP details returned by VastAI for the instance.
rsync -avP <instance_ssh>:/path/to/outputs/ ./outputs/
```

If using VastAI volumes, document the volume ID, mount path, and whether the
volume should be retained or destroyed.

## GitHub Access

For private repos, clone using HTTPS with a personal access token file or another
secret-safe mechanism. Use `GIT_ASKPASS` to avoid putting tokens in shell
history:

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

Do not inline a PAT in clone URLs or scripts.

For Hugging Face, copy the token as a distinct step and verify it arrived before
running setup commands that depend on it:

```bash
ssh <vast_ssh> 'mkdir -p /workspace/secrets'
scp -i ~/.ssh/id_ed25519 -P <port> ~/.cache/huggingface/token \
  root@<host>:/workspace/secrets/hf_token
ssh <vast_ssh> 'test -s /workspace/secrets/hf_token && chmod 600 /workspace/secrets/hf_token'
```

Do not chain secret copy, repo sync, and dependency install into one long command.
If the secret copy fails, stop and fix transfer/auth before continuing.

## Deploying Your Container

VastAI already runs Docker. Do not build or upload Apptainer `.sif` files.
For real training runs, push the target Docker image to Docker Hub, GHCR, or
another registry that VastAI can pull, then rent the instance with that image.

Common paths:

1. Launch the target repo's published Docker image directly.
2. Launch a broad NVIDIA PyTorch image only for short probes, then destroy or
   recycle into the real image before long training.
3. Clone the repo and run its normal `docker/build_docker.sh` only if Docker is
   available inside the launched environment and the repo expects nested Docker.

Verify GPU visibility before training:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

This only proves that the launched image can see the GPUs. It does not prove the
target training stack is compatible. Run the target repo's install and a real
training smoke test before committing to a long run.

If the target repo uses its own Docker image and the VastAI instance already
launched into a container, avoid nested Docker unless you have confirmed Docker
inside Docker works on that instance. Prefer launching the desired image directly
with `--image`.

## Running Training

No sbatch scripts. Run the target repo's real training command directly inside
the VastAI instance/container.

Always:

- set `PYTHONUNBUFFERED=1` so logs stream promptly
- persist logs to a file with `tee`
- set W&B offline mode unless online logging has already been verified
- write outputs/checkpoints under a run-scoped directory on the large disk
- check `df -h` for the output/checkpoint/W&B filesystem before launching
- create or update a `run_logs/` entry in the target repo

Example:

```bash
mkdir -p /workspace/outputs/<run_id>
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

python <entry_point> --config <config_path> \
  2>&1 | tee /workspace/outputs/<run_id>/train.log
```

For distributed multi-GPU training, use the repo's normal launcher, for example
`torchrun`, `accelerate`, or the framework-specific command. Record the exact
command in the run log.

Monitor with:

```bash
nvidia-smi
tail -f /workspace/outputs/<run_id>/train.log
df -h
```

GPU allocation alone is not proof of progress. Check step counts, loss/metric
movement, errors, disk usage, and checkpoint writes.

## W&B Sync

Prefer the bounded AutoHPC helper when available in the training environment:

```bash
uv pip install -e ../autohpc/wandb-sync

autohpc-wandb-sync sync \
  --entity <wandb-entity> \
  --project <wandb-project> \
  --wandb-token-file ~/.wandb_token \
  --dry-run \
  /workspace/outputs/<run_id>/wandb/offline-run-...
```

After the dry run looks correct, remove `--dry-run` and add `--yes`.

If using plain `wandb sync`, still pass the destination deliberately and read the
token from a private file. Do not inline `WANDB_API_KEY` in reusable scripts or
logs.

## Checkpoint Passport And Download

Before downloading, uploading, evaluating, or handing off a checkpoint, follow
`checkpoint-passport/SKILL.md` in the training runtime:

```bash
uv pip install -e ../autohpc/checkpoint-passport
validate-checkpoint <ckpt_dir>
sign-checkpoint <ckpt_dir> --reason '<reason>'
validate-checkpoint <ckpt_dir> --require-signoff
```

Only move checkpoints after `MODEL_PASSPORT.json` and `SIGNOFF.json` are present
and validation passes.

## Destroying Instances

Always inspect and download/upload artifacts when finished, then ask the user
whether to destroy, stop/reuse, or keep the instance alive.

Do not infer permission from a successful upload, a completed sync, or a prior
general rule about cleaning up machines.

```bash
vastai show instances-v1 --raw

# Download artifacts using the instance SSH/SCP details.
rsync -avP <instance_ssh>:/workspace/outputs/<run_id>/ ./outputs/<run_id>/

# Only after explicit user confirmation in this conversation.
vastai destroy instance <instance_id> --yes
```

`vastai destroy instance <id> --yes` is confirmed in VastAI CLI 1.1.1. If a
future CLI version changes the lifecycle verb, inspect `vastai --help` and use
the current documented destroy/delete command. While waiting for user approval,
report the idle instance ID, hourly cost if known, and what artifacts have been
made durable.

## Notes

- VastAI hosts vary. Verify architecture, CUDA driver visibility, disk, network,
  and write paths on each new instance.
- Search comparisons for VRAM can be surprising. Use raw JSON and confirm the
  returned `gpu_ram` values before renting.
- Use labels that include project and run purpose; labels make cleanup safer.
- Prefer `--ssh --direct` for agent-operated sessions when direct ports are
  available.
- Never destroy an instance, delete outputs, prune Docker layers, or remove volumes without explicit user confirmation. Uploading checkpoints does not mean the machine is no longer needed.
- If an offer has excellent price but tiny disk, request a larger `--disk` and
  verify creation actually grants enough usable space.
- Do not store secrets or credentials in this profile.
