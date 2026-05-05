# Adversarial Inference Run Trial Log

## Run Header

- started_at: 2026-05-04T12:06:33Z
- executor: cursor-agent
- target_repo: alpha-robotics
- target_repo_commit: (local)
- checkpoint_source: alpha-robotics/checkpoints/dit_block_tower_norm_fix
- clean_checkpoint_copy: /tmp/adv_trials/20260504T120633Z/clean/ckpt
- validator_version: validate-checkpoint (alpha-robotics env)
- deployment_protocol_revision: code-first verification (2026-05-04 evening)
- fresh_agent_channel: Hermes via Slack (Codex 5.3 model preferred)
- dataset_loader: `lerobot.datasets.LeRobotDataset` (NOT `datasets.load_dataset`)

## Baseline

- command: validate-checkpoint $TRIAL_ROOT/clean/ckpt --require-signoff --show-not-checked
- exit_code: 0
- hard_failures: none
- soft_signals: reference_test_vector, training_datasets_resolvable, camera_identity, state_dim_consistency
- not_checked_static: input_contract_vs_dataset
- notes: 23 passed, 4 soft signals, 1 not checked. Passport and signoff restored from .passport_backup.

## Trial Ledger

| Trial | Fault | Outcome | Element | Gap owner | Log |
|---|---|---|---|---|---|
| T1.1 | Tampered SIGNOFF.json | caught_static | 4 | none | [T1.1](trials/T1.1.md) |
| T1.2 | Truncated weight file | caught_static | 4 | none | [T1.2](trials/T1.2.md) |
| T1.3 | Missing manifest file | caught_static | 4 | none | [T1.3](trials/T1.3.md) |
| T1.4 | Stale norm stats without re-signing | caught_static | 4 | none | [T1.4](trials/T1.4.md) |
| T1.5 | Passport transformers constraint changed without re-signing | caught_static | 4 | none | [T1.5](trials/T1.5.md) |
| T1.6 | Passport action horizon changed without re-signing | caught_static | 4 | procedure | [T1.6](trials/T1.6.md) |
| T2.1 | Wrong model class (V2 with n_action_steps=12) | caught_static | provenance | none (tooling) | [T2.1](trials/T2.1.md) |
| T2.2 | Null hf_revision without re-signing | pending | 4 | — | [T2.2](trials/T2.2.md) |

## Handoff Notes (2026-05-04 evening)

### Current state
- **T2.1 is in-progress** with a new fault: `MultiTaskDiTPolicyV2` (n_action_steps=12)
  imported as `MultiTaskDiTPolicy` in the adapter. Awaiting Codex result.
- **T2.2 trial dir exists** at `/tmp/adv_trials/20260504T120633Z/trials/T2.2/ckpt`
  with hf_revision nulled. Ready to run after T2.1 completes.
- The deployment-protocol skill was rewritten to **code-first verification**
  (agents write assert-based scripts instead of filling tables). This is the
  version being tested in the current T2.1 run.

### Active fault injections (MUST REVERT before production use)
- `alpha-robotics/external/multitask_dit_policy/src/multitask_dit_policy/model/model.py`:
  `MultiTaskDiTPolicyV2` class added at end of file
- `alpha-robotics/missiontracker/adapters/multitask_dit_adapter.py`:
  line ~87 imports `MultiTaskDiTPolicyV2 as MultiTaskDiTPolicy`

### Key learnings so far
1. Agents bypass factory routing — load model directly instead of tracing entry point
2. Agents see mismatches but don't flag them without explicit assert instructions
3. `datasets.load_dataset()` misses video-encoded images; must use `LeRobotDataset`
4. Code-first verification (write asserts) >> prose-based ("compare and report")
5. Codex 5.3 model follows "stop on hard failure" correctly; default model does not

### To continue
1. Record T2.1 result when it comes back
2. Revert fault injection in adapter + model.py
3. Run T2.2 (already prepped, just needs prompt to Codex)
4. Continue with T2.3+ per the trials plan
