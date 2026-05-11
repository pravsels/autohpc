# checkpoint-passport

Post-train integrity + feeding-contract artifacts for trained robotics policy checkpoints. Generated immediately after a training run finishes (and `wandb sync` confirms the run is intact), **before** the checkpoint moves anywhere — HF upload, eval, copy to a robot, hand off. Every downstream consumer (including the eval harness itself) loads the checkpoint via the passport.

This skill ships **both** a SKILL.md (workflow guidance for the agent) and
a runnable Python package (`checkpoint_passport`).

**Two passport creation paths:**

- **Config-bearing checkpoints** (LeRobot, RAMEN): `generate-passport --config ...` writes `MODEL_PASSPORT.json` directly.
- **OpenPI checkpoints** (no static config): `extract-passport-seed openpi ...` then `assemble-passport ...` — the inference contract is runtime-constructed.

**Shared gates (both paths):** `validate-checkpoint`, `sign-checkpoint`, `check-publish-ready`, `publish-checkpoint upload/download`.

## Quick start (consumer)

```bash
uv pip install -e ../autohpc/checkpoint-passport

# Deployment gate — non-zero exit means do not load
validate-checkpoint <checkpoint_dir> --require-signoff
```

## Quick start (producer)

See `SKILL.md` for the full bounded workflow. The two paths converge at validate → sign → publish.

## Commands

| Command | Purpose |
|---------|---------|
| `generate-passport` | Static passport from config-bearing checkpoint |
| `extract-passport-seed` | Runtime seed from OpenPI (or future backends) |
| `assemble-passport` | Combine seed + checkpoint files into passport |
| `validate-checkpoint` | Run integrity checks |
| `sign-checkpoint` | Hash and write SIGNOFF.json |
| `check-publish-ready` | Pre-upload packaging gate |
| `publish-checkpoint` | Upload/download with validation gates |

## Why ship code from autohpc?

Other autohpc skills (eval-tracking, hpc-run-tracking, …) are docs-only —
they describe commands that target tools every cluster already has
(`docker`, `sbatch`, `squeue`). Checkpoint passports are the first artifact
that needs *bespoke* tooling: a schema, a validator, a signer. Putting that
tooling here keeps the schema and the skill that uses it in lockstep, and
lets any consuming repo treat passport validation as `pip install` + run.
