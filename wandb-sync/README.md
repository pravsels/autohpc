# autohpc-wandb-sync

Bounded W&B offline sync helper for AutoHPC training runs.

This package exists so agents do not improvise ad hoc `wandb sync` commands
on clusters. It requires the destination W&B `--entity` and `--project`, reads
the API key from a token file, and runs sync inside an Apptainer container.

## Install

```bash
uv pip install -e ../autohpc/wandb-sync
```

Adjust the path if your target repo is not next to `autohpc`.

## Usage

```bash
autohpc-wandb-sync \
  --entity <wandb-entity> \
  --project <wandb-project> \
  --offline-run-dir /scratch/.../wandb/offline-run-... \
  --container /scratch/.../container/model.sif \
  --bind /scratch/...:/scratch/... \
  --bind /home/...:/home/... \
  --wandb-token-file ~/.wandb_token \
  --dry-run
```

Remove `--dry-run` and add `--yes` once the target and command are correct.

The tool also checks `~/.wandb_token` and `~/.wandb_key` when
`--wandb-token-file` is omitted. It never prints the raw token, and dry-run
output redacts the token-file path.

## Common Patterns

Run from the agent/local environment and launch the sync on a remote cluster via SSH:

```bash
autohpc-wandb-sync \
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
