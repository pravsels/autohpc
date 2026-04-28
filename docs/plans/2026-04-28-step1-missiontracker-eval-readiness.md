# Step 1: MissionTracker Eval-Readiness Note

Output of Step 1 of `2026-04-27-checkpoint-lifecycle-roadmap.md`.

This was originally a code-reading verdict (Step 1). It has since been
extended with the runtime execution of Step 1.5 (build the DiT adapter and
verify it loads `dit_block_tower_norm_fix`). See "Step 1.5 execution log"
below for the latest state. The Step 1 inventory and the "Smallest backtest
plan" remain accurate as written for the parts that don't depend on Step 1.5.

## Status (handoff for next agent session)

- **Step 1**: `done` (inventory verdict: `needs_repair` for DiT, plan accepted).
- **Step 1.5**: `done` — DiT adapter loads `dit_block_tower_norm_fix`,
  schema (`action_dim=17, horizon=32, n_obs_steps=2, image_keys=[front,
  wrist], uses_text=True`) verified against config. Validation script:
  `alpha-robotics/scripts/verify_dit_adapter_load.py`.
- **Step 1.6**: `pending` — wire RAMEN normalize/unnormalize into
  `MultiTaskDiTAdapter.predict()` so backtest can actually run a forward
  pass on this checkpoint family. Currently `predict()` raises
  `NotImplementedError` for `stats_format="ramen"`.
- **Step 2**: effectively done — `dit_block_tower_norm_fix` is the natural
  trial checkpoint (only signed DiT checkpoint locally; passport already
  exists at `alpha-robotics/checkpoints/dit_block_tower_norm_fix/`).
- **Steps 3-12**: blocked on Step 1.6.

Resume work at "Step 1.5 execution log → Carried-forward debts" below.

## Location of this note

This is a roadmap follow-up note (assessment of an eval *backend*, not an eval
of a specific checkpoint), so it lives next to the roadmap in
`autohpc/docs/plans/`. Future per-checkpoint *eval logs* should still live in
the target repo's `eval_logs/` per `eval-tracking/SKILL.md` (e.g.
`alpha-robotics/eval_logs/`).

## Verdict

**`needs_repair`** — for the actually-targeted policy family
(`multitask_dit_policy`).

The original assessment below was written assuming the trial would use
ACT / RewACT / OpenPi (the architectures MissionTracker already supports).
On 2026-04-28 the user clarified that the lifecycle trial should use
`alpha-robotics/external/multitask_dit_policy` (DiT). MissionTracker has zero
adapter for that policy class:

- `detect_policy_architecture` recognises only `act` / `rewact` / `openpi` /
  `unknown`; `unknown` falls through to `_load_act_adapter`, which calls
  LeRobot's `PreTrainedConfig.from_pretrained`.
- DiT checkpoints are not LeRobot `PreTrainedConfig` artifacts — they are
  draccus-serialised `MultiTaskDiTConfig` + `model.safetensors` +
  `dataset_stats.json`, loaded via `MultiTaskDiTPolicy.load(checkpoint_dir)`
  from `multitask_dit_policy.model.model`. ACTAdapter would fail to decode
  the config.
- The DiT policy has a different feeding shape: temporal stacks of
  `n_obs_steps` frames, multi-camera images stacked into a single
  `(B, n_obs, num_cameras, C, H, W)` tensor before sampling, and external
  normalization via `dataset_stats.json` (or `ramen_stats.json`) packaged
  with the checkpoint.

**The repair is bounded, not catastrophic.** Everything in MissionTracker
*except* the model-loading layer remains directly reusable: the
`PolicyAdapter` interface, `PolicyObservation` / `PolicyOutput`, the runner,
sharding + cache, the metrics block (including `metric_unified_val_loss` for
cross-architecture comparison), multi-policy artefacts, and the artefact
format. The repair is essentially one new file:
`missiontracker/adapters/multitask_dit_adapter.py` (~200-400 LOC) plus
dispatch in `missiontracker/adapters/factory.py`, plus a deps/version-drift
strategy (see DiT-specific findings below).

**Human checkpoint required per roadmap line 458-459** (`needs_repair`
verdicts must be reviewed). See "Decision needed" below.

If the trial were instead being run on an ACT / RewACT / OpenPi checkpoint,
the verdict would be `usable_with_small_adapter` (autohpc-side wrapper for
passport gating + HF revision pinning, no MissionTracker code changes). That
analysis is preserved below in the "Inventory answers" section, since the
MissionTracker infrastructure under it stays the same regardless of which
policy is loaded.

## DiT-specific findings (2026-04-28)

### What exists

- **Source tree**: `alpha-robotics/external/multitask_dit_policy/` (uv-managed
  Python 3.12 project, own `pyproject.toml` and `uv.lock`).
- **Policy class**: `multitask_dit_policy.model.model.MultiTaskDiTPolicy`,
  loaded by static `MultiTaskDiTPolicy.load(checkpoint_dir)`. Checkpoint dir
  must contain `config.json` and `model.safetensors`. `dataset_stats.json`
  (or `ramen_stats.json`) is consumed separately by the inference wrapper.
- **Inference reference**:
  `external/multitask_dit_policy/src/multitask_dit_policy/examples/inference.py`
  — shows the full normalize → `policy.select_action()` → unnormalize loop.
- **Run logs**: `external/multitask_dit_policy/run_logs/` has 10 entries
  (block_tower, coffee_capsules, qwen_pooled variants, norm_fix, etc.).
- **At least one signed DiT checkpoint already exists**:
  `dit_block_tower_norm_fix` per
  `alpha-robotics/docs/notes/2026-04-23-passport-generation-friction.md`
  (23/24 hard checks PASS, 1 soft signal on state_dim 13 → 16 rot6d
  expansion, exit 0 with `--require-signoff`). Steps 2-3 of the roadmap
  are therefore already partially done for at least this one checkpoint.

### What MissionTracker would need

A new `MultiTaskDiTAdapter(PolicyAdapter)` that:

1. Loads the policy via `MultiTaskDiTPolicy.load(checkpoint_dir)`.
2. Loads `dataset_stats.json` from the checkpoint dir.
3. In `predict(obs: PolicyObservation)`:
   - Maintains a rolling window of `n_obs_steps` observations (the DiT policy
     expects temporal stacks; current ACT path feeds single frames).
   - Stacks per-camera images into a `(B, n_obs, num_cameras, C, H, W)`
     tensor under the `OBS_IMAGES` key (the DiT `_generate_actions` requires
     this; not done by `forward()` automatically — known footgun documented
     in the passport-friction note, line 57).
   - Calls `normalize_batch(...)` from `multitask_dit_policy.utils.utils`
     before, `unnormalize_batch(...)` after.
   - Returns `PolicyOutput(action=chunk[0], action_chunk=chunk)` — the DiT
     `predict_action_chunk(batch)` already returns the full chunk, so this
     maps cleanly onto MissionTracker's interface.
4. Implements `reset()` by calling `policy.reset()` (the DiT policy already
   has its own reset that clears `_queues`).
5. No reward output — `PolicyOutput.reward = None`, `lowest_bin_prob = None`.
   `RewardAnomalyCheck` should be auto-disabled in the config (`reward_check`
   defaults to enabled but produces no signal for non-RewACT).

Plus on the **factory** side:

- Add `multitask_dit` to the `PolicyArchitecture` literal in
  `missiontracker/adapters/factory.py`.
- Detection: easiest is `force_architecture="multitask_dit"` exposed via a
  config field; auto-detect could check for `dataset_stats.json` +
  `MultiTaskDiTConfig`-shaped `config.json` (e.g. presence of
  `observation_encoder.multimodal` or `model_objective` keys).

Plus a **dependency / version-drift** strategy. The passport-friction note
(line 55-56) documents that the DiT checkpoint format moves faster than any
pre-installed package will. Two options:

- **A: Bind-mount + PYTHONPATH**: clone `multitask_dit_policy` at the
  training-time commit recorded in the passport, prepend `src/` to
  `sys.path`, evict any pre-imported `multitask_dit_policy.*` modules from
  `sys.modules`. Robust but per-checkpoint setup.
- **B: Pinned uv env per checkpoint family**: install a known-good
  `multitask_dit_policy` revision into a sibling uv env that MissionTracker
  imports. Cheaper for a stable trial; same brittleness for older
  checkpoints.

For the first manual trial on `dit_block_tower_norm_fix`, option B (use the
current `external/multitask_dit_policy/` checkout in editable install) is
sufficient and matches what the passport-generation work already used.

### What stays unchanged in MissionTracker

- `PolicyAdapter` / `PolicyObservation` / `PolicyOutput` interface.
- `BacktestConfig` / `MultiPolicyBacktestConfig` schema (DiT just needs the
  same `policy_path` + `val_dataset_repo_ids` fields plus its own
  `force_architecture`).
- Sharded execution + crash-resilient cache.
- All metrics: `metric_policy_loss`, `metric_unified_val_loss`,
  `metric_time_coherence`, `metric_perturbation_coherence`.
- Multi-policy artefact for batch comparison (Step 9 of the roadmap).
- Anomaly detection layer (with the caveat that default thresholds are
  domain-specific and should be ignored on a new family until calibrated).

Net effect: the DiT adapter is a **localized, well-scoped repair** of one
file plus dispatch, not a structural change.

## Decision needed

Per roadmap line 458-459, this verdict needs human review before the trial
proceeds. The choices:

1. **Build the DiT adapter now**, then run Step 6 against
   `dit_block_tower_norm_fix` (or another DiT checkpoint). Adds ~half a day
   of MissionTracker work to Step 1, then unblocks Steps 4-9 fully.
2. **Switch the trial to a RewACT / OpenPi checkpoint** that MissionTracker
   already supports. Faster path through the lifecycle but trades away the
   user's stated target.
3. **Skip MissionTracker for now** and write a much thinner standalone
   evaluator (load DiT, run on N val episodes, compute MSE / coherence,
   write a parquet). Fastest to first eval; loses sharding, multi-policy
   comparison, anomaly detection.

Recommendation: **option 1**. The repair is bounded, the DiT checkpoint
already has a passport, and option 1 is the only path that lands the full
lifecycle on the user's actual target architecture. Option 3 looks cheap
but discards exactly the multi-policy / unified-val-loss infrastructure
that the ML-methodology critic identified as the strongest match for Step 9.

**Decision taken (2026-04-28):** option 1. See "Step 1.5 execution log" below.

---

## Step 1.5 execution log (2026-04-28)

### Outcome

`MultiTaskDiTAdapter` loads `dit_block_tower_norm_fix` and reports the
correct schema:

```
action_dim: 17  [OK, matches config.json action_feature]
action_horizon: 32  [OK, matches n_action_steps]
n_obs_steps: 2  [OK]
image_keys: ['observation.images.front', 'observation.images.wrist']  [OK]
uses_text: True  [OK]

Step 1.5 verification PASSED.
```

This is **load-only**. `predict()` is intentionally guarded — it raises
`NotImplementedError` when `stats_format != "lerobot"` (i.e., for any
RAMEN-format checkpoint, including this one). Wiring RAMEN through
`predict()` is Step 1.6.

### Environment

Created at the alpha-robotics repo level (env-per-repo, not env-per-subproject):

```
micromamba env name: alpha-robotics
location:            /home/user/micromamba/envs/alpha-robotics/
python:              3.12.13
key versions installed:
  torch:        2.10.0
  transformers: 5.4.0   <-- pinned, see "transformers drift" below
  diffusers:    0.35.2
  lerobot:      0.5.1
  safetensors:  0.7.0
  draccus:      0.10.0
  timm:         1.0.26
  multitask_dit_policy: editable from external/multitask_dit_policy
```

Setup commands (for re-creation):

```bash
cd /home/user/Desktop/code/alpha-robotics
/home/user/micromamba/micromamba create -n alpha-robotics -c conda-forge \
    python=3.12 pip -y
/home/user/micromamba/micromamba run -n alpha-robotics pip install \
    -e external/multitask_dit_policy
/home/user/micromamba/micromamba run -n alpha-robotics pip install \
    "transformers==5.4.0"
```

`missiontracker` is NOT pip-installable (no `pyproject.toml` / `setup.py`).
The verify script handles this with `sys.path.insert(0, REPO_ROOT)`. Any
future `python -m missiontracker.*` invocation needs the same treatment.

Run command:

```bash
cd /home/user/Desktop/code/alpha-robotics
unset PYTHONPATH   # the host shell exports /opt/ros/humble py3.10 paths
                   # which break py3.12 imports if not unset
/home/user/micromamba/micromamba run -n alpha-robotics \
    python scripts/verify_dit_adapter_load.py
```

### Files written

- `alpha-robotics/missiontracker/adapters/multitask_dit_adapter.py` —
  format-aware stats loading (lerobot vs ramen detection in
  `from_pretrained`); `predict()` guard that raises
  `NotImplementedError` for RAMEN until Step 1.6 lands the routing.
- `alpha-robotics/missiontracker/adapters/factory.py` — `multitask_dit`
  added to `PolicyArchitecture` literal; new `_load_multitask_dit_adapter`
  branch.
- `alpha-robotics/missiontracker/backtest/__init__.py` and
  `backtest/adapters/{__init__,multitask_dit_adapter}.py` — re-export
  shims so the new adapter is reachable via both `missiontracker.adapters`
  and `missiontracker.backtest.adapters`.
- `alpha-robotics/missiontracker/backtest/backtest.py` — explicit
  `multitask_dit` dispatch in `_create_backtest_with_datasets`; existing
  ACT-only checks (`needs_policy_small`, `reward_ae_check`,
  `enable_interpretability`) tightened from `architecture != "openpi"` to
  `architecture in ("act", "rewact")` so DiT correctly skips them.
- `alpha-robotics/scripts/verify_dit_adapter_load.py` — re-runnable Step 1.5
  verification script (asserts adapter schema matches the values printed in
  the integrity report). Pure check, no side effects.

### Two real failures hit during Step 1.5 (and how they were resolved)

#### 1. `transformers` API drift — `CLIPTextModel.text_model` wrapper removed

Symptom: `MultiTaskDiTPolicy.load(checkpoint)` failed with
`Missing key(s) in state_dict: "...text_encoder.text_encoder.embeddings..."`
plus `Unexpected key(s): "...text_encoder.text_encoder.text_model.embeddings..."`.

Diagnosis (the policy code did NOT change):

- Pickaxe across all branches of `multitask_dit_policy` for
  `CLIPTextModelWithProjection`: zero hits. The policy has always used
  plain `CLIPTextModel`.
- Probed both transformers versions:
  - `transformers==5.4.0`: `CLIPTextModel(...).state_dict()` keys start
    with `text_model.embeddings...` (still has the inner wrapper).
  - `transformers==5.6.2`: keys start with `embeddings...` (wrapper removed).
- Training happened 2026-04-17, after the policy's pin was bumped from
  `>=4.40` to `>=5.4.0,<6.0.0` (commit `0da2d05` on 2026-03-28). So
  training used some early 5.x; the wrapper was removed somewhere in
  5.4 → 5.6.

Resolution: pinned `transformers==5.4.0` in the env. Load works.

**Honest follow-up:** this pin lives only in the env, not in any
constraints file. If anything reinstalls transformers, load breaks again.
Two options for Step 1.6: (a) pin in a workspace-level constraints file
(simple, brittle), (b) add a state-dict key-remap shim that handles both
layouts (more durable). See "Carried-forward debts" below.

#### 2. Stats schema mismatch — `ramen_stats.json` vs `dataset_stats.json`

Symptom: after the transformers fix, model load succeeded but stats
parsing failed with `'str' object has no attribute 'items'` (the
top-level `format` key is a string, not a dict of stats).

Diagnosis: there are TWO on-disk stats schemas for MultiTaskDiT
checkpoints.

- LeRobot convention (`dataset_stats.json`):
  `{feature_key: {min: [...], max: [...], mean: [...], std: [...]}}`.
  Used by `multitask_dit_policy.utils.utils.normalize_batch /
  unnormalize_batch`.
- RAMEN convention (`ramen_stats.json`, what this checkpoint ships):
  `{format: "ramen_norm_stats", format_version: 1, metadata: {...},
    norm_mask: [...], state: {q02, q98}, action: {q02, q98}}`. Used by
  `multitask_dit_policy.utils.ramen_normalization.{load_ramen_stats,
    ramen_normalize_batch, ramen_unnormalize}`.

Resolution: adapter now detects format at load time (`stats_format
in {"lerobot", "ramen", "unknown"}`), parses both into the same
`Dict[str, Dict[str, torch.Tensor]]` shape, and `predict()` raises a
clear `NotImplementedError` when `stats_format == "ramen"`. Step 1.5
(load-only) passes; Step 1.6 needs to wire `ramen_normalize_batch` /
`ramen_unnormalize` into the predict path.

The RAMEN inference reference is **not** the example
`inference.py` in the multitask_dit_policy repo (which assumes lerobot
stats); it's `train.py` (which uses `ramen_normalize_batch` for the
training/eval forward) and `ramen_normalization.py` itself.

### Lifecycle validation — the passport already had the answer

The transformers-drift failure cost ~30 minutes of debugging today. The
existing `MODEL_PASSPORT.json` for `dit_block_tower_norm_fix` already
contains the exact pin needed:

```
model_identity.library_versions:
  python:       '3.12.3'
  torch:        '2.10.0+cu128'
  transformers: '5.4.0'           <-- exactly what we ended up needing
  diffusers:    '0.35.2'
  lerobot:      '0.5.0'
  safetensors:  '0.7.0'
  draccus:      '0.10.0'
  timm:         '1.0.26'
  multitask_dit_policy: '0.1.0'
  cuda_available: True
  cuda_version: '12.8'
provenance:
  training_repo:        'https://github.com/pravsels/multitask_dit_policy'
  training_repo_commit: 'af0a43a512841aa1f4d6bb2f93755e5358dca8cb'
```

So the lifecycle's design is **validated by today's failure**:

- **Schema:** correct. The passport has the right field
  (`model_identity.library_versions`).
- **Population:** correct. Whoever generated this passport on 2026-04-23
  captured the actual training-time versions.
- **Enforcement:** missing. Nothing read the passport before load was
  attempted. The integrity_report.txt already on disk even shows
  `external_pretrained_assets_pinned: ⬜ no passport loaded` — the
  integrity tool itself couldn't find the passport from the cwd it was
  invoked at.

The lifecycle gap revealed today is one specific thing: a **preflight
check that diffs `current_env` versions vs
`passport.model_identity.library_versions` and blocks on mismatch**. This
is roadmap Step 4's natural home. Estimated ~50 LOC.

Important: per manual-first principle, this preflight should NOT be
written before Step 6. The honest sequence is (i) finish Step 1.6 (RAMEN
predict path + transformers pin), (ii) run the manual Step 6 backtest
with the version-pinned env, (iii) record the friction in the eval log,
(iv) THEN write the preflight automating exactly this trail of evidence.

### Carried-forward debts (these gate Step 6)

1. **RAMEN `predict()` routing** — implement RAMEN-format normalize +
   unnormalize in `MultiTaskDiTAdapter.predict()` using
   `multitask_dit_policy.utils.ramen_normalization.{ramen_normalize_batch,
   ramen_unnormalize}`. Reference:
   `multitask_dit_policy/train.py` for how training uses these. ~50 LOC.
2. **`transformers` version pin** — currently env-resident only. Either
   add a workspace-level constraints file (`alpha-robotics/constraints.txt`
   or similar) referenced by env-creation docs, OR add a state-dict
   key-remap shim in the adapter that detects the layout and rewrites
   keys before `load_model_as_safetensor`. Recommend the latter — more
   durable, doesn't depend on env discipline.
3. **`PYTHONPATH` ROS leak** — the host shell exports
   `/opt/ros/humble/lib/python3.10/site-packages` which breaks all py3.12
   imports. The verify script guards against this; any future entry point
   (Modal Image, Docker run, backtest CLI) needs the same guard.
4. **Preflight env-vs-passport check** — out of scope for Step 1.6 per
   manual-first. Schedule for after the first Step 6 backtest runs to
   completion.

### What this unblocks

- Step 1.6 (RAMEN predict path) is now a small, well-scoped piece of
  work: ~50 LOC in one file, exact reference functions identified.
- Step 2 effectively done: `dit_block_tower_norm_fix` is the trial
  checkpoint.
- Step 3 (sign the checkpoint) was already done on 2026-04-23 per
  `alpha-robotics/docs/notes/2026-04-23-passport-generation-friction.md`
  and our own re-read of `SIGNOFF.json` (verdict `soft_signal`,
  documented `state_dim` rot6d expansion).
- Step 4 (preflight) gains a concrete first check to implement
  post-Step-6: `library_versions` diff.
- Step 6 (run the smallest backtest) blocked only on Step 1.6 + a
  config file. The "Smallest backtest plan" section below still applies
  with one substitution: `force_architecture="multitask_dit"` and a
  config that points at
  `alpha-robotics/checkpoints/dit_block_tower_norm_fix/checkpoints/29000/params`
  with stats at `.../assets/ramen_stats.json`.

---

## Original ACT/RewACT/OpenPi inventory (preserved for reference)

The remainder of this note documents MissionTracker's existing capability
against the policy types it already supports. This is still accurate and
still applies to the underlying infrastructure that the DiT adapter would
plug into.

## Inventory answers

### Q1 — Which policy types can it load today?

`missiontracker/adapters/factory.py`:

- **ACT** — LeRobot ACT policies via `ACTAdapter.from_pretrained` (uses
  LeRobot's `PreTrainedConfig.from_pretrained`, `make_policy`,
  `make_pre_post_processors`).
- **RewACT** — same loader as ACT, auto-detected at runtime by inspecting the
  loaded policy's `output_features` for `'reward'` / `'expected_value'` or
  presence of `reward_head` / `predict_reward`.
- **OpenPi (Pi0 / Pi0.5)** — separate `OpenPiAdapter.from_checkpoint` path,
  detected via path naming heuristics (`openpi`, `pi0`, `pi05`, etc.) or by
  the presence of `assets/*/norm_stats*.json` in the checkpoint dir. Requires
  `openpi_config_name` to be passed (e.g. `pi05_bin_pack_coffee_capsules_delta`).
- **`unknown`** falls through to the ACT loader.

All three architectures share the same `PolicyAdapter` interface
(`PolicyObservation` in, `PolicyOutput` out) so downstream metric / anomaly
code is architecture-agnostic.

### Q2 — What inputs does it expect?

The entry point is `examples/run_backtest.py <config.json>`. Required JSON
fields:

- `VAL_DATASET_REPO_IDS: List[str]` — HF dataset repo IDs (LeRobot format).
- `POLICY_PATHS: List[str]` — HF repo IDs or local paths (one or more; multi
  enables side-by-side comparison).

Optional JSON fields (auto-disable their checks if unset):

- `VAL_EPISODES: List[int] | null` — pin specific episodes.
- `POLICY_SMALL_PATH`, `SAE_MODEL_REPO_ID`, `CONTEXT_AE_REPO_ID` — for the
  attention-OOD anomaly check.
- `REWARD_AE_PATH`, `LATENT_SAFETY_WM_PATH`,
  `LATENT_SAFETY_CLASSIFIER_PATH`, `LATENT_SAFETY_STATS_PATH` — for the
  reward-AE / latent-safety anomaly checks.
- `LANGUAGE_INSTRUCTION_OVERRIDES` — ablation set.

Implicit requirements:

- The training dataset that the policy was trained on must be resolvable. If
  not provided explicitly, `ACTAdapter.from_pretrained` reads
  `TrainPipelineConfig.from_pretrained(policy_path)` and pulls
  `train_cfg.dataset.repo_id`, then loads it via
  `robocandywrapper.factory.make_dataset_without_config` to extract
  normalization stats.
- A hardcoded `key_rename_map` (`'action.pos' -> 'action'`,
  `'observation.state.pos' -> 'observation.state'`) is applied in both the
  data loader (`backtest/factory.py`) and the policy adapter
  (`adapters/act_adapter.py`). Datasets / policies using different keys would
  need this map adjusted.
- Heavy runtime deps: `lerobot`, `lerobot_policy_rewact`, `rewact_tools`,
  `robocandywrapper`, `physical_ai_interpretability`, plus
  `openpi`/`openpi-client` (and JAX) if OpenPi paths are used. CUDA GPU
  expected; defaults to `cuda` and falls back to CPU only if torch can't see
  one. See `examples/requirements.txt` and `examples/Dockerfile`.

### Q3 — What does it produce?

A self-contained "Artefact" — see `backtest/ARTEFACT_FORMAT.md`. Concretely:

- `metadata.json`, `config.json`, `summary.json` at the artefact root.
- `metrics.json` (single-policy) or `policies/<id>/metrics.json` (multi-policy)
  with mean / std / min / max / `mean_plus_2std` / p50 / p95 / p99 / p99.9
  for every metric.
- Per-episode parquet: `frames/`, `actions/` (predicted action chunks +
  ground truth), `anomalies/` (per-frame anomaly flags + raw metric values
  prefixed `metric_*`).
- `videos/<dataset>/episode_<N>/<camera_key>.mp4` (re-encoded, frame-aligned
  with parquet).
- Optional: `clusters/` (k-means / HDBSCAN / spectral on policy activations
  + proprio), `interpretability/` (attention + saliency maps as grayscale
  videos).
- The whole tree is compressed to `<name>.tar.gz` by default.
- Multi-policy artefacts share dataset / video / ground-truth data once and
  store predictions / anomalies per policy — directly useful for the
  roadmap's Step 9 batch comparison.

Provenance recorded in artefact:

- `policy_path` (HF repo ID or local path) — recorded verbatim per policy.
- `policy_architecture` (`act` / `rewact` / `openpi`) — recorded per policy.
- `val_dataset_repo_ids`, `val_episodes` — recorded.
- Backtest config hash via `MultiPolicyBacktestConfig.compute_hash()` (16-char
  sha256 prefix over the full config dict). Useful for shard cache reuse.
- **Not recorded**: HF revision sha (only the repo ID), passport sha, signoff
  sha, dataset commit sha. See "Gaps" below.

### Q4 — Can it run against a signed HF checkpoint snapshot without code changes?

**Yes for loading; no for snapshot pinning.** A signed HF checkpoint dir
satisfies LeRobot's `from_pretrained` contract — the extra
`MODEL_PASSPORT.json` / `SIGNOFF.json` files are ignored (LeRobot only reads
`config.json`, weights, and any norm stats it expects). MissionTracker passes
`policy_path` straight through to LeRobot, so passing a HF repo ID or a local
path to a signed checkpoint dir works without code changes.

What does NOT work without a code change:

- Pinning to an exact HF revision sha. `policy_path` accepts only the repo ID
  string; the revision used at load time is whatever HF's cache returns
  (latest by default). This needs to be solved on the autohpc / wrapper side
  (e.g. resolve the revision before passing the path) for the roadmap's "no
  ambiguity about which bytes the eval will consume" exit criterion (Step 4,
  line 580).

### Q5 — Does it validate MODEL_PASSPORT.json / SIGNOFF.json before loading?

**No.** Grep for `PASSPORT|SIGNOFF|validate.checkpoint|input_contract` across
the entire `missiontracker/` tree returns zero matches. As the roadmap
predicted (line 452: "Assume it does not until proven otherwise.").

This is not a MissionTracker defect — it predates the passport convention.
The fix lives on the autohpc side: a startup gate that runs
`validate-checkpoint <ckpt_dir> --require-signoff` before invoking the
backtest. See "Smallest backtest plan" below.

### Q6 — Does it feed the model per passport `input_contract`?

**No — it re-derives feeding from the training dataset's stats.**
`ACTAdapter.from_pretrained` (`adapters/act_adapter.py:50-101`) does:

1. Read the policy's `TrainPipelineConfig` to recover `train_dataset_repo_id`.
2. Load that training dataset via
   `make_dataset_without_config(train_dataset_repo_id, ...)`.
3. Apply a hardcoded `key_rename_map`.
4. Build `preprocessor, postprocessor` via LeRobot's
   `make_pre_post_processors(..., dataset_stats=train_dataset.meta.stats, ...)`.

The passport's `input_contract.images[]` (raw_shape, dtype, value_range,
color_order, channel_layout, encoder_resize) and
`input_contract.state.normalization` / `actions.normalization` are never
consulted. Feeding correctness is therefore conditional on the training
dataset still being available and unchanged at the same repo ID.

This is the **central methodological caveat** for the roadmap: as long as
the passport's `input_contract.training_datasets[]` and the dataset
MissionTracker actually loads agree on repo + revision + key layout, the
feeding is *consistent* with the passport but not *enforced* by it. Any
divergence (training dataset moved, schema changed, key rename map drifted)
will silently corrupt feeding. The eval log written in Step 7 should record
both sides explicitly so the inconsistency would be visible.

### Q7 — Other things worth recording

Strengths the roadmap can lean on:

- **Architecture-agnostic comparison via `metric_unified_val_loss`** —
  artefact format v2.1.0 explicitly added this for "ACT vs RewACT vs OpenPi"
  apples-to-apples comparison. Directly addresses the ML-methodology critic's
  concern that Step 9 batch comparison lacks a shared quantitative axis. Use
  `metric_unified_val_loss` (not `metric_policy_loss`) for any cross-arch
  batch.
- **Multi-policy artefact with shared validation data** — pin one dataset +
  episode list, run N checkpoints, get a single artefact. Matches the Step 9
  "small batch" intent without further glue.
- **Sharded crash-resilient execution** — each `(policy, dataset)` pair is
  split into ≤5000-frame shards; completed shards survive crashes and are
  reused on rerun. Removes a class of "the run died at hour 3" failure modes.
- **Per-metric statistics block** (mean, std, min/max, mean+2σ, p50/p95/p99/
  p99.9) is computed automatically and saved to `metrics.json`. Useful for
  the eval log without extra work.

Weaknesses worth flagging:

- The `joint_delta_check` and `reward_check` defaults in `BacktestConfig`
  hardcode anomaly thresholds (e.g. `policy_loss anomaly = 0.01`,
  `time_coherence = 0.02`). These are tuned for the current ARX5 / block-tower
  stacks; using them on a different domain without re-tuning will produce
  noisy "anomaly" verdicts. For first-pass triage of a new checkpoint family,
  treat the raw `metrics.json` numbers as ground truth and ignore
  `anomaly_counts` until thresholds are calibrated.
- It's an **open-loop** backtest — `runner.py` iterates dataset frames and
  feeds recorded states/images. There is no closed-loop rollout. Action
  predictions are compared to ground-truth actions; there is no notion of
  what would happen if the predicted action were actually executed. This is
  fine (and expected) for the roadmap's Stage 3 ("Offline Backtest"); just do
  not conflate it with Stage 5 ("Simulation").

## Smallest backtest plan

Once a checkpoint is selected in Step 2 and signed in Step 3, the smallest
end-to-end run looks like this. Treated as proposed; not yet executed by
this assessment.

1. **Resolve the exact HF revision** (autohpc side, before invoking
   MissionTracker). E.g.

   ```bash
   python -c 'import huggingface_hub as h; print(h.HfApi().model_info("<repo>").sha)'
   ```

   Record the sha in the trial note. Optionally `huggingface-cli download
   <repo> --revision <sha> --local-dir <pinned_snapshot_dir>` so the policy
   path becomes a local snapshot whose bytes are immutable for the run.

2. **Run the artifact gate** (passport-side; not in MissionTracker):

   ```bash
   validate-checkpoint <pinned_snapshot_dir> --require-signoff
   ```

   Non-zero exit → stop. Record passport sha and signoff sha.

3. **Author the smallest config** at
   `alpha-robotics/missiontracker/examples/configs/<trial>_smallest.json`:

   ```json
   {
     "VAL_DATASET_REPO_IDS": ["<owner>/<val_dataset>"],
     "VAL_EPISODES": [0],
     "POLICY_PATHS": ["<pinned_snapshot_dir_or_repo_id>"],
     "POLICY_SMALL_PATH": null,
     "SAE_MODEL_REPO_ID": null,
     "CONTEXT_AE_REPO_ID": null,
     "REWARD_AE_PATH": null,
     "LATENT_SAFETY_WM_PATH": null,
     "LATENT_SAFETY_CLASSIFIER_PATH": null,
     "LATENT_SAFETY_STATS_PATH": null
   }
   ```

   This auto-disables every check that needs an auxiliary model
   (`run_backtest.py:130-137`). Only `joint_delta_check`, `reward_check`
   (no-op for non-RewACT), and the always-on `ValidationMetrics` (policy
   loss, time coherence, perturbation coherence, unified val loss) will run.
   The `attention_ood_check` validate() rule is satisfied because both
   `policy_small_path` and `sae_model_repo_id` are null.

4. **Run the backtest**:

   ```bash
   cd /home/user/Desktop/code/alpha-robotics/missiontracker/examples
   bash quickstart.sh native configs/<trial>_smallest.json
   # OR, if Docker is preferred and CUDA is set up:
   bash quickstart.sh docker configs/<trial>_smallest.json
   ```

   Native path needs `lerobot_policy_rewact`, `rewact_tools`,
   `robocandywrapper`, `physical-ai-interpretability` installed (see
   `examples/requirements.txt`). Docker path builds a CUDA image from the
   detected driver version; one-time build cost.

5. **Output**: `outputs/artefacts/<dataset>_<policy>.tar.gz`. Read
   `metrics.json` and `summary.json` for the eval log; record:
   - `metric_unified_val_loss` (mean, std, p95) — primary number.
   - `metric_policy_loss` (mean, p95) — secondary, architecture-specific.
   - `metric_time_coherence`, `metric_perturbation_coherence` — stability.
   - `summary.total_anomalies` and `summary.anomaly_counts` — note that
     these are only meaningful if thresholds have been calibrated for the
     domain.

## Gaps to record in the trial / eval log

These are not blockers; they are caveats Step 7's eval log should make
visible:

1. **Passport-blind feeding**: MissionTracker derives normalization from
   training dataset stats, not from `MODEL_PASSPORT.json::input_contract`.
   The eval log should explicitly record both the passport's
   `input_contract.training_datasets[]` and the dataset MissionTracker
   actually loaded, so any divergence is visible. (Addresses the
   ML-methodology critic's contamination/feeding concerns.)
2. **No HF revision pinning inside MissionTracker**: the artefact records
   `policy_path` as the repo ID only. Pin the revision externally before
   invoking and record it in the eval log alongside the artefact path.
3. **No load-time signoff gate**: add `validate-checkpoint --require-signoff`
   as a startup gate at the autohpc-wrapper level before running the
   backtest. Eventually this can become a documented convention in
   `eval-tracking/SKILL.md` (per roadmap Step 10).
4. **Default anomaly thresholds are domain-specific**: do not treat
   `summary.total_anomalies` as quality evidence on a new checkpoint family
   until thresholds have been calibrated. Use raw `metric_*` distributions
   from `metrics.json`.
5. **Open-loop only**: this is a Stage 3 ("Offline Backtest") backend, not
   Stage 5 ("Simulation"). Do not conflate.

## Files inspected

- `alpha-robotics/missiontracker/__init__.py`,
  `missiontracker/missiontracker.py` (top-level)
- `missiontracker/examples/run_backtest.py` — entry point
- `missiontracker/examples/README.md`, `quickstart.sh`,
  `requirements.txt`
- `missiontracker/examples/configs/build_block_tower.json`,
  `build_block_tower_local.json`
- `missiontracker/backtest/__init__.py`, `factory.py`, `config.py`,
  `runner.py` (head), `ARTEFACT_FORMAT.md`
- `missiontracker/backtest/adapters/{factory,policy,act_adapter}.py`
  (re-export shims)
- `missiontracker/adapters/{factory,policy,act_adapter,openpi_adapter}.py`
  (canonical adapters)
- `autohpc/checkpoint-passport/SKILL.md` (head, for `input_contract` schema
  reference)

## Exit criteria check

From roadmap Step 1 (lines 468-471):

- [x] We know whether MissionTracker can be used for Step 4 — **yes for the
      data-loading / metrics / artefact infrastructure; no for DiT policy
      loading without a new adapter**. For ACT / RewACT / OpenPi, usable as
      described in the inventory above.
- [x] We know the smallest eval that can be run on one checkpoint — **for
      ACT-family, the config skeleton in "Smallest backtest plan" step 3.
      For DiT, the smallest eval requires the new adapter described in the
      DiT-specific findings before any backtest config matters**.

Step 1 is **complete**. The human decision (see "Decision needed" above)
was option 1 — build the DiT adapter. Step 1.5 has now also been executed
and passes load-only verification. See "Step 1.5 execution log" above for
state and follow-ups.

Step 1.5 exit criteria (informal — adopted during execution, in absence
of a roadmap-defined Step 1.5 contract):

- [x] Adapter source written and importable
      (`missiontracker.adapters.MultiTaskDiTAdapter`).
- [x] Factory dispatch routes `force_architecture="multitask_dit"` to the
      new adapter.
- [x] Adapter loads `dit_block_tower_norm_fix` end-to-end with no error.
- [x] Adapter-reported schema matches the values printed in the
      checkpoint's `integrity_report.txt` (action_dim=17, horizon=32,
      n_obs_steps=2, image_keys=[front,wrist], uses_text=True).
- [x] Verification is re-runnable
      (`alpha-robotics/scripts/verify_dit_adapter_load.py`, exits 0 on
      success, non-zero on schema mismatch).

Step 1.6 (RAMEN predict path + transformers pin/shim) is the smallest
remaining piece before Step 6 can execute. See "Step 1.5 execution log →
Carried-forward debts".
