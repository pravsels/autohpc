# checkpoint-passport

Post-train integrity + feeding-contract artifacts for trained robotics policy checkpoints. Generated immediately after a training run finishes (and `wandb sync` confirms the run is intact), **before** the checkpoint moves anywhere — HF upload, eval, copy to a robot, hand off. Every downstream consumer (including the eval harness itself) loads the checkpoint via the passport.

This skill ships **both** a SKILL.md (workflow guidance for the agent) and
a runnable Python package (`checkpoint_passport`) with two CLIs:

- `validate-checkpoint` — run static + passport-driven integrity checks.
- `sign-checkpoint`     — compute file hashes and write `SIGNOFF.json`.

## Quick start (consumer)

From any sibling repo:

```bash
uv pip install -e ../autohpc/checkpoint-passport

validate-checkpoint <checkpoint_dir> --show-not-checked
sign-checkpoint <checkpoint_dir> --reason "passes all hard checks; one documented soft signal"
```

Or call the package directly without installing:

```bash
python -m checkpoint_passport.cli.validate <checkpoint_dir>
python -m checkpoint_passport.cli.sign     <checkpoint_dir>
```

## Skill

For the workflow that *produces* the `MODEL_PASSPORT.json` consumed by these
CLIs, see `SKILL.md` in this folder.

## Why ship code from autohpc?

Other autohpc skills (eval-tracking, hpc-run-tracking, …) are docs-only —
they describe commands that target tools every cluster already has
(`docker`, `sbatch`, `squeue`). Checkpoint passports are the first artifact
that needs *bespoke* tooling: a schema, a validator, a signer. Putting that
tooling here keeps the schema and the skill that uses it in lockstep, and
lets any consuming repo treat passport validation as `pip install` + run.
