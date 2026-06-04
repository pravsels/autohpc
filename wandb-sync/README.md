# autohpc-wandb-sync

Bounded W&B offline sync helper for AutoHPC training runs.

This package exists so agents do not improvise ad hoc `wandb sync` commands.
It requires the destination W&B `--entity` and `--project`, reads the API key
from a token file, and defaults to running sync in the current container or
environment.

## Install

```bash
uv pip install -e ../autohpc/wandb-sync
```

Adjust the path if your target repo is not next to `autohpc`.

## Usage

Install the helper into the writable environment that already has `wandb`
available, usually a scratch venv mounted into the container. Then run:

```bash
autohpc-wandb-sync sync \
  --entity <wandb-entity> \
  --project <wandb-project> \
  --wandb-token-file ~/.wandb_token \
  --dry-run \
  /scratch/.../wandb/offline-run-...
```

Remove `--dry-run` and add `--yes` once the target and command are correct.

The tool also checks `~/.wandb_token` and `~/.wandb_key` when
`--wandb-token-file` is omitted. It never prints the raw token, and dry-run
output redacts the token-file path.

## Common Patterns

Launch the container explicitly only when you are outside the container runtime:

```bash
autohpc-wandb-sync launch \
  --ssh-host <ssh-alias-or-host> \
  --module <container-runtime-module> \
  --entity <wandb-entity> \
  --project <wandb-project> \
  --offline-run-dir /scratch/.../wandb/offline-run-... \
  --container /scratch/.../container/model.sif \
  --bind /scratch/...:/scratch/... \
  --bind /home/...:/home/... \
  --wandb-token-file /home/.../.wandb_token \
  --dry-run
```

Use Slurm allocation by default when running directly on a cluster compute-capable shell:

```bash
autohpc-wandb-sync ... --yes
```
