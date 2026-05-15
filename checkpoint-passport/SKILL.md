---
name: checkpoint-passport
description: Use right after a training run finishes and `wandb sync` (or equivalent) confirms the run is intact, before the checkpoint moves anywhere — HF upload, copy to an eval box, copy to a robot, hand to a colleague. Produces and signs MODEL_PASSPORT.json + SIGNOFF.json so every downstream consumer (including the eval harness itself) can verify the checkpoint's feeding contract and integrity before loading it.
---

# Checkpoint Passport

This file is the canonical AutoHPC checkpoint passport workflow. If another skill
or alias such as `hpc-checkpoint-passport` gives conflicting phase-based
instructions, ignore it and follow this file.

## Overview

Two artifacts at the checkpoint root:

- **`MODEL_PASSPORT.json`** — chain-of-custody from sensor to action output. Records input contract, model identity, model internals, output spec, weight integrity, provenance, transform pipeline, reference test vectors, and known issues.
- **`SIGNOFF.json`** — sha256 of the passport plus every weight file, a verdict, and a one-line reason. The signer refuses to write a signoff when validation reports hard failures.

Together they let any consumer run `validate-checkpoint <ckpt_dir> --require-signoff` as a load-time gate. Non-zero exit means do not load.

**Where this fits:** post-train, pre-anything-else. The passport is generated immediately after training finishes, **before** the checkpoint is copied off the training filesystem. That includes HF uploads used purely as transport — by the time the checkpoint exists on HF, the passport must ride with it.

The skill ships a runnable Python package (`checkpoint_passport`). Read `checkpoint_passport/schema.py` once — it is the single source of truth for every field name and type.

## When to Use

- A training run just finished, `wandb sync` succeeded, and the checkpoint dir has no `MODEL_PASSPORT.json` / `SIGNOFF.json`.
- The user is about to upload / push / copy the checkpoint anywhere.
- A consuming repo wants to gate startup on `validate-checkpoint --require-signoff`.

Do not skip the passport for "experimental" or "internal-only" checkpoints. Do not defer it until "after eval looks good."

## Agent Algorithm

Follow this order. Later sections explain details, but this algorithm is the
canonical workflow.

1. **Preflight**
   - Read this file, not global/personal checkpoint passport skills.
   - Install/use `<autohpc>/checkpoint-passport` in the runtime environment.
   - Verify checkpoint path, training repo path, and target repo path if used.
   - If using a remote AutoHPC clone, verify it is on the intended commit/branch.
   - Create any temporary extraction/signing scripts and logs in a run-scoped
     remote work directory such as `<remote_project_dir>/autohpc_runs/<run_id>/`,
     not in the top level of `$SCRATCH`, a VM home directory, or a cloud volume
     mount.

2. **Classify the checkpoint**
   - If the checkpoint already ships a real passport-compatible `config.json`,
     use Path A.
   - Else if it is OpenPI, use Path B.
   - Else stop and ask; do not infer a new architecture workflow.

3. **Create `MODEL_PASSPORT.json`**
   - Path A: run `generate-passport` with `--config` and `--training-repo`.
   - Path B: run `extract-passport-seed openpi` inside the OpenPI runtime, with
     `--device` and real `--reference-dataset-path`, then run
     `assemble-passport`.
   - For OpenPI, never create or use a synthetic `config.json`.

4. **Validate**
   - Run `validate-checkpoint <ckpt_dir>`.
   - If hard failures exist, fix the passport/checkpoint or stop.
   - If only soft signals exist, record the accepted reason for signing.

5. **Sign**
   - Run `sign-checkpoint <ckpt_dir> --reason '<reason>'`.
   - Run/confirm validation with `--require-signoff`.

6. **Publish only after signing**
   - Create or update `README.md` and `TRAINING_LOG.md`; include W&B links when
     shareable synced runs exist.
   - Stage an explicit publish package with `publish-checkpoint stage`.
   - Run `check-publish-ready <publish_dir>`.
   - Report preflight: repo ID, revision, package mode, top-level contents,
     file count, total size, and W&B-link status.
   - Upload with `publish-checkpoint upload --publish-dir <publish_dir>`.

7. **Handoff**
   - If evaluating, follow `eval-tracking/SKILL.md`.
   - If deploying, follow `deployment-protocol/SKILL.md`.
   - If uploading another checkpoint, complete and verify the current upload
     before starting the next one.

## Two Valid Passport Creation Paths

There are exactly two ways to create a passport. Choose the one that matches the checkpoint.

### Path A: Config-Bearing Checkpoints (LeRobot, RAMEN, etc.)

These checkpoints ship a `config.json` that fully describes the input/output contract. One command produces `MODEL_PASSPORT.json`:

```bash
cd <autohpc>/checkpoint-passport

generate-passport <ckpt_dir> \
  --config <ckpt_dir>/config.json \
  --stats <ckpt_dir>/stats.json \
  --target-repo <deployment_repo_path> \
  --training-repo <training_code_repo_path> \
  --dataset user/dataset@abc123:lerobot.datasets.LeRobotDataset
```

`--config` and `--training-repo` are required for signable passports because validation requires `provenance.training_repo_commit`. `--stats` and `--dataset` are optional but strongly recommended. `--target-repo` is optional deployment debug context only. `--resolve-remote-revisions` is the only path that calls HF APIs; it is off by default.

After this, skip to [Validate](#validate).

### Path B: OpenPI Checkpoints (No Static Config)

OpenPI does not have a passport-compatible static config. Its inference contract is constructed at runtime by `cfg.data.create()`, transforms, norm stats, and adapter behavior. This path has two steps: extract a passport seed, then assemble the final passport.

Do not add a "Phase 0" that writes `config.json` for OpenPI. If a checkpoint is OpenPI and does not already ship a real passport-compatible config, go straight to `extract-passport-seed openpi`.

When passporting multiple checkpoints, do them one at a time. Before starting each checkpoint and after every major phase (seed extraction, assembly, validation, signing), report what is done, what is running now, and what remains.

#### Step 1: Extract Passport Seed

Run inside the **OpenPI runtime environment** (not a generic Python). Production
signing requires runtime enrichment and a real reference vector:

```bash
extract-passport-seed openpi \
  --checkpoint-dir <ckpt> \
  --out <ckpt>/PASSPORT_SEED.json \
  --openpi-config-name <name> \
  --default-prompt "<prompt>" \
  --resize-size 224 \
  --device cuda \
  --reference-dataset-path <dataset_path> \
  --reference-episode-index <episode> \
  --reference-start-frame <frame> \
  --reference-num-frames 10
```

**The reference test vector is required for signing.** Do not skip it. Do not substitute synthetic data for production passports. Do not write a separate dataset sampling script — the checked-in extractor command is the only valid path.

For MVP testing without model/dataset, `--dummy-reference-vector` generates synthetic data. This is for CI only — it does not satisfy production signing requirements.

#### Step 2: Assemble Passport

```bash
assemble-passport \
  --checkpoint-dir <ckpt> \
  --seed <ckpt>/PASSPORT_SEED.json \
  --out <ckpt>/MODEL_PASSPORT.json \
  --training-repo <training_code_repo_path> \
  --target-repo <deployment_repo_path>
```

The assembler reads the seed, hashes checkpoint files for `weight_integrity`,
populates provenance from git state, and writes `MODEL_PASSPORT.json`.

`--training-repo` is required for signable passports. `--target-repo` is optional debug context and must not be treated as a load gate.

Use `--skip-dir retain` (repeatable) to exclude training artifacts from hashing.

## Validate

Run the validator to check the passport:

```bash
validate-checkpoint <ckpt_dir>
validate-checkpoint <ckpt_dir> --show-not-checked
```

Three outcomes:

- **All hard checks pass, no soft signals** — proceed to Sign.
- **Soft signals only** — decide: is this a passport authoring bug (fix the passport), or a real but acceptable divergence (e.g. `state_dim_consistency` firing for rotation expansion)? Soft signals you accept must be documented in the signoff `--reason`.
- **Any hard fail** — fix the passport or the checkpoint and re-run. Do not proceed to signing with hard fails; the signer will refuse.

Optional flags:

| Flag | Purpose |
|------|---------|
| `--show-not-checked` | Include NOT_CHECKED rows (verbose forensic view) |
| `--target-repo <path>` | Optional deployment debug context; drift is reported as a soft signal, not a load gate |
| `--dataset-path <path>` | Enable input_contract_vs_dataset cross-check |
| `--require-signoff` | Missing SIGNOFF.json becomes hard fail |
| `--skip-section <category>` | Drop checks for a category while iterating |

**`--skip-section` is for iteration only.** Do not sign with skipped hard sections. Do not use `--skip-section signoff` with `--require-signoff` (the CLI rejects this combination). If `reference_test_vector` validation fails, fix the extraction — do not bypass validation.

### Common Soft Signals

- **`state_dim_consistency`** (model-facing 16D vs dataset-facing 13D) — expected for rotation expansion. Document in signoff reason.
- **`training_datasets_resolvable`** reporting unpinned commits — dataset repo IDs are required when known, but **dataset commit pins are optional provenance**. Missing commits are a soft signal, not a hard blocker. Do not add automatic HF API lookup to resolve them in the default flow.
- **`input_contract_vs_dataset`** NOT_CHECKED — the validator can't find the local HF dataset cache. Pass `--dataset-path` if available, or accept.

## Sign

```bash
sign-checkpoint <ckpt_dir> --reason '<one-liner explaining any soft signals>'
```

The signer:

1. Re-runs `validate-checkpoint` internally. Refuses to sign on hard failures (exits 1).
2. Requires `--reason` when soft signals are present.
3. Hashes `MODEL_PASSPORT.json` plus every file in `weight_integrity.weight_files[]`.
4. Writes `SIGNOFF.json` with verdict `pass` or `soft_signal`.
5. Re-validates with `--require-signoff` to confirm round-trip.

`--dry-run` prints the signoff without writing it.

**Do not sign with `--skip-section` active on any hard check.** If `reference_test_vector` is missing from the passport, go back to extraction and add `--reference-dataset-path`. If runtime enrichment is missing, go back and add `--device`. The signing gate exists to catch incomplete passports — do not route around it.

## Publish

Publish is staging-first. Do not upload the training checkpoint root directly.
Create a clean publish package containing only the files meant to move, inspect
the manifest/size, then upload that package directory.

Default to an inference package:

- `MODEL_PASSPORT.json`
- `SIGNOFF.json`
- `README.md`
- `TRAINING_LOG.md`
- `assets/`
- final params / inference weights only

Do not include intermediate step directories, `train_state/`, optimizer state,
W&B runs, caches, or full checkpoint roots unless the user explicitly asks for a
resume-training package.

Do not leave helper scripts or command logs in the top level of the remote
workspace while staging or signing. If you need a temporary script for
`srun`/`apptainer` or a cloud VM shell, write it under a run-scoped project
directory and remove that directory after the checkpoint is signed and uploaded.

Before staging, verify the tooling you will run is current. If using a remote
AutoHPC clone on a cluster, check its git commit/status or pull the intended
branch before running `publish-checkpoint`; do not assume the cluster clone has
the same code as the local repo.

Before upload, the model card must be complete enough for a downstream user to
understand the artifact. If a W&B run was synced and has a public/shareable URL,
include that clickable dashboard link in `README.md` and keep the training
dynamics in `TRAINING_LOG.md`.

Before uploading to HF:

```bash
publish-checkpoint stage \
  --out <publish_dir> \
  --file <ckpt>/MODEL_PASSPORT.json \
  --file <ckpt>/SIGNOFF.json \
  --file <ckpt>/README.md \
  --file <ckpt>/TRAINING_LOG.md \
  --dir <ckpt>/assets:assets \
  --dir <ckpt>/<step>/params:checkpoints/<step>/params

check-publish-ready <publish_dir>
```

This verifies `README.md`, `TRAINING_LOG.md`, `MODEL_PASSPORT.json`, and `SIGNOFF.json` are present, non-empty (for docs), valid JSON (for artifacts), and that the internal validator passes with `require_signoff=True`.

After staging and before `publish-checkpoint upload`, report a short preflight
summary and wait if anything is surprising:

- HF repo ID and revision
- publish directory path
- package mode: `params-only` or `full-train-state`
- top-level contents
- file count and total size from the stage manifest / `du -sh`
- confirmation that `README.md` includes W&B links when available

Package mode is inferred from what you staged: if the package contains final
params/inference weights and no `train_state`/optimizer state, report
`params-only`.

Never expose tokens in commands, logs, or status updates. Do not `cat` token
files or inline `hf_...` values in visible commands. Read tokens into environment
variables inside the remote shell, or use the authentication mechanism provided
by the cluster/user environment.

Upload and download use the bounded helpers:

```bash
publish-checkpoint upload \
  --publish-dir <publish_dir> \
  --repo-id <user-or-org>/<repo> \
  --revision main

publish-checkpoint download \
  --repo-id <user-or-org>/<repo> \
  --revision <sha-or-branch> \
  --out <local_dir>
```

Upload runs `check-publish-ready` and refuses on failure. Download runs `validate-checkpoint --require-signoff` after fetching and refuses to report success on failure.

If an upload/package mistake is found, stop parallel work on other checkpoints
until the upload is killed or corrected, the staged package is verified, and the
HF repo state is checked. Do not continue with another passport while a bad
upload may still be running.

**Do not write ad hoc `huggingface-cli upload`, `hf download`, `git lfs`, `rsync`, or Python upload scripts.** If HF auth, repo ID, revision, or output path is missing, stop and ask.

## Deployment Gate

From any consuming repo:

```bash
validate-checkpoint <ckpt_dir> --require-signoff
validate-checkpoint <ckpt_dir> --require-signoff --target-repo <deploy_repo>
```

Non-zero exit = do not load.

## Stop Gates

Stop and ask the user when:

- **Config-bearing checkpoint has no explicit config path.** Do not discover configs by searching the filesystem.
- **Architecture is unsupported.** The extractor will list supported backends and fail. Do not guess.
- **OpenPI config name is missing.** `--openpi-config-name` is required. Do not probe the registry.
- **OpenPI runtime imports fail.** The environment is wrong. Print the missing module and stop.
- **OpenPI runtime enrichment fails** because the env, checkpoint assets, or device are unavailable. Report the exact error.
- **Reference dataset path is missing** when signing requires `reference_test_vector`. Stop and ask for the path, episode index, and frame range.
- **Reference vector extraction fails** — cannot read frames, too few frames, missing image data. Do not substitute synthetic inputs.
- **Passport seed validation or assembly fails.** Report the validation errors.
- **Publish gate fails.** Report which files are missing or which validation checks failed.
- **HF auth, repo ID, revision, or local output path is missing.** Stop and ask.
- **Training-repo commit SHA is unknown.** Do not guess — a passport built from guesses defeats the purpose.
- **Passport tooling commit SHA is unknown.** Do not guess — the passport should record the autohpc commit that created it.

## Hard Rules

- Do not write a small config script to produce a fake `config.json` for OpenPI. Its contract is runtime-constructed.
- Do not discover environments by probing. If the right environment is not active, stop and ask.
- Do not read source code until you understand the inference path and then fill fields by judgment.
- Do not write a separate smoke-test script. Smoke testing is built into `extract-passport-seed openpi --device`.
- Do not write a separate reference vector extraction script. Reference vectors are built into `extract-passport-seed openpi --reference-dataset-path`.
- Do not write upload/download scripts for HF handoff. Use `publish-checkpoint`.
- Do not splice JSON with a small script to merge dynamic and static data. Use `assemble-passport`.
- Do not sign with skipped hard sections for production checkpoints.
- Do not treat missing `reference_test_vector` as acceptable — fix the extraction.
- Do not block signing on missing dataset commits when repo IDs are present. That is optional provenance.
