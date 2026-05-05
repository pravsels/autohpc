# Adversarial Inference Run Trial Log

## Run Header

- started_at: 2026-05-04T12:06:33Z
- executor: cursor-agent
- target_repo: alpha-robotics
- target_repo_commit: dc74ab1f507d833c0d122c53a5eaf9692b3c595e
- checkpoint_source: alpha-robotics/checkpoints/dit_block_tower_norm_fix
- clean_checkpoint_copy: /tmp/adv_trials/20260504T120633Z/clean/ckpt
- validator_version: checkpoint-passport @ autohpc (commit d4e5282)
- fresh_agent_channel: Hermes via Slack (Codex 5.3 model preferred)
- dataset_loader: `lerobot.datasets.LeRobotDataset` (NOT `datasets.load_dataset`)
- trials_plan: `docs/plans/2026-05-01-adversarial-inference-run-trials.md`

## Tooling Built During Trials

Tools built iteratively while running T1.x-T2.2, all in `checkpoint-passport/`:

| Tool | Entry point | Purpose | Commit |
|---|---|---|---|
| `generate-passport` | `checkpoint_passport/cli/generate.py` | Phase 1 static extraction — deterministic, no torch/GPU. Replaces hand-written passports. | 790fa80 |
| `replay-reference-vector` | `checkpoint_passport/cli/replay.py` | Replay golden test vector through adapter, compare output within tolerance. Catches T2.3-T2.8. | d4e5282 |
| `validate-checkpoint` | `checkpoint_passport/cli/validate.py` | Static passport checks. Pre-existing, hardened during trials. | various |
| `sign-checkpoint` | `checkpoint_passport/cli/sign.py` | Hash passport + weights, write SIGNOFF.json. Pre-existing. | pre-trials |

### Schema changes during trials

- `provenance.deployment_repo` + `deployment_repo_commit` — pins target deployment repo (907708d)
- `training_datasets[].loader_class` — records dataset loader class (907708d)
- `hf_revision` format validation — must be `^[0-9a-f]{7,40}$` commit SHA (cd96944)
- `reference_test_vector` externalized — numerical blobs stored as `.npy` files in `assets/`, passport stores path+hash only (d4e5282)

### Validator changes during trials

- `check_deployment_repo_commit` — hard-fail if target repo dirty or commit mismatch (907708d)
- `check_external_pretrained_assets_pinned` — now validates hf_revision SHA format (cd96944)
- `check_reference_test_vector_present` — promoted to **hard fail** (was soft signal) (d4e5282)
- `check_camera_identity` — promoted to **hard fail** (was soft signal) (d4e5282)

### Key learnings

1. **Deterministic tools > agent reasoning.** Agents bypass factory routing, see mismatches but don't flag them, and find creative workarounds for gates. Hard-coded script checks are non-circumventable.
2. **`generate-passport` prevents hand-authored errors.** Hermes recorded wrong class names, wrong hf_revisions, and missed fields when writing JSON by hand.
3. **`replay-reference-vector` catches all runtime semantic faults.** T2.3-T2.8 (temporal stack, delta/absolute, scaling, normalization, color order) are all caught by comparing golden output to actual output.
4. **Reference data should be real, not synthetic.** Real frames from ep0/frame0 exercise the full preprocessing pipeline and enable camera identity checks.
5. **`datasets.load_dataset()` misses video-encoded images** — must use `LeRobotDataset`.
6. **Codex 5.3 model follows "stop on hard failure" correctly**; default model does not.

## Baseline

- command: validate-checkpoint $TRIAL_ROOT/clean/ckpt --require-signoff --show-not-checked
- exit_code: 0
- hard_failures: none
- soft_signals: reference_test_vector, training_datasets_resolvable, camera_identity, state_dim_consistency
- not_checked_static: input_contract_vs_dataset
- notes: 23 passed, 4 soft signals, 1 not checked. Passport and signoff restored from .passport_backup.

## Trial Ledger

| Trial | Fault | Outcome | Caught by | Log |
|---|---|---|---|---|
| T1.1 | Tampered SIGNOFF.json | caught_static | signoff hash check | [T1.1](trials/T1.1.md) |
| T1.2 | Truncated weight file | caught_static | signoff hash check | [T1.2](trials/T1.2.md) |
| T1.3 | Missing manifest file | caught_static | signoff hash check | [T1.3](trials/T1.3.md) |
| T1.4 | Stale norm stats without re-signing | caught_static | signoff hash check | [T1.4](trials/T1.4.md) |
| T1.5 | Passport transformers constraint changed | caught_static | signoff hash check | [T1.5](trials/T1.5.md) |
| T1.6 | Passport action horizon changed | caught_static | signoff hash check | [T1.6](trials/T1.6.md) |
| T2.1 | Wrong model class (dirty deployment repo) | caught_static | deployment_repo_commit dirty check | [T2.1](trials/T2.1.md) |
| T2.2 | Null hf_revision + stale signoff | caught_static | signoff hash + hf_revision format | [T2.2](trials/T2.2.md) |
| T2.3 | Temporal stack off by one | pending | expected: replay-reference-vector | |
| T2.4 | Delta action treated as absolute | pending | expected: replay-reference-vector | |
| T2.5 | Absolute dims incorrectly delta-converted | pending | expected: replay-reference-vector | |
| T2.6 | Wrong action scaling | pending | expected: replay-reference-vector | |
| T2.7 | Image normalization omitted | pending | expected: replay-reference-vector | |
| T2.8 | Color order swap RGB/BGR | pending | expected: replay-reference-vector | |
| T2.9 | Reference test vector mismatch | pending | expected: signoff hash check | |
| T2.10 | Incorrect runtime dtype | pending | | |

## How to Continue (for a new agent)

### Prerequisites before running T2.3+

The T2.1 checkpoint at `/tmp/adv_trials/20260504T120633Z/trials/T2.1/ckpt` has the **old-format** `reference_test_vector` with inline arrays and synthetic data. Before running T2.3-T2.8, you need a checkpoint with the **new-format** reference vector (real data, .npy files). Either:

1. **Re-do Phase 2 on the clean checkpoint** — load ep0/frame0 from the training dataset, save reference frames as PNGs and state/output as .npy to `assets/`, populate the new passport fields, re-sign. See `checkpoint-passport/SKILL.md` Phase 2 instructions.
2. **Or create a fresh trial checkpoint** from `/tmp/adv_trials/20260504T120633Z/clean/ckpt` with the new reference vector format.

### Running a T2.3-T2.8 trial

1. Copy the clean signed checkpoint to a trial dir
2. Inject the fault (code change in the adapter, NOT in the checkpoint files)
3. Run `replay-reference-vector <ckpt> --adapter-module <mod> --adapter-class <cls>` in the alpha-robotics mamba env
4. Expect exit code 1 (mismatch) — the fault should cause the output to differ from the golden expected_output.npy
5. Record result in this file and create `trials/T2.X.md`

### Environment

- checkpoint-passport tools: `cd ~/Desktop/code/autohpc/checkpoint-passport && uv run <tool>`
- Model/adapter env: `mamba run -n alpha-robotics <command>`
- Target repo: `~/Desktop/code/alpha-robotics`
- Clean checkpoint: `/tmp/adv_trials/20260504T120633Z/clean/ckpt`

### Active fault injections (MUST REVERT before production use)

Check `~/Desktop/code/alpha-robotics` for any dirty state from previous trials:
```bash
cd ~/Desktop/code/alpha-robotics && git status --porcelain
git diff --name-only
```

All faults should be injected in trial-specific copies, not in the main repo. If the repo is dirty, `git stash` or revert before proceeding.
