---
name: checkpoint-passport
description: Use right after a training run finishes and `wandb sync` (or equivalent) confirms the run is intact, before the checkpoint moves anywhere — HF upload, copy to an eval box, copy to a robot, hand to a colleague. Produces and signs MODEL_PASSPORT.json + SIGNOFF.json so every downstream consumer (including the eval harness itself) can verify the checkpoint's feeding contract and integrity before loading it.
---

# Checkpoint Passport

## Overview

Two artifacts at the checkpoint root:

- **`MODEL_PASSPORT.json`** — a chain-of-custody document from sensor to action output. Every entity (sensor reading, image, state vector, action) and every transformation (resize, normalize, delta, clip, unnormalize) is recorded so it can be audited. The passport covers: input contract, model identity (including runtime constraints), model internals, output spec, weight integrity, provenance (including checkpoint lineage), the ordered transform pipeline, reference test vectors, normalization round-trip results, and known issues.
- **`SIGNOFF.json`** — sha256 of the passport plus every weight file the passport declares, plus a verdict and a one-line reason. Signed signoffs only ever carry verdict `pass` or `soft_signal`; the schema reserves `fail` for completeness, but the signer refuses to write a signoff at all when validation reports hard failures (it exits non-zero and prints the failures, leaving no signoff on disk).

Together they let any consumer (a robot, an eval harness, a teammate's repo) call `validate-checkpoint <ckpt_dir> --require-signoff` as a load-time gate. Non-zero exit means do not load this checkpoint.

**Where this fits:** post-train, pre-anything-else. The passport is generated immediately after the training run finishes and `wandb sync` confirms the run is intact, **before** the checkpoint is copied off the training filesystem. That includes HF uploads being used purely as a transport mechanism — by the time the checkpoint exists on HF, the passport must ride with it. The eval harness is itself a passport consumer (it reads `input_contract` to drive how the model is fed); evaling first and passporting later means an eval-feeding bug can silently corrupt your eval numbers.

The skill ships a runnable Python package (`checkpoint_passport`) with the schema, a `validate-checkpoint` CLI (most checks run on any host; `model_identity_resolvable` and the typed smoke buckets need the model's own container — see Phase 3), and a `sign-checkpoint` CLI that hashes everything and writes the signoff. Read `checkpoint_passport/schema.py` once — it is the single source of truth for every field name and type referenced below.

## When to Use

- A training run just finished, `wandb sync` succeeded, and the checkpoint dir has no `MODEL_PASSPORT.json` / `SIGNOFF.json`.
- The user is about to `huggingface-cli upload` / push the checkpoint to a registry.
- The user is about to `rsync` / `scp` a checkpoint to an eval box, a robot, or a colleague.
- A consuming repo's CI or eval harness wants to gate startup on `validate-checkpoint --require-signoff`.

Do not skip the passport for "experimental" or "internal-only" checkpoints, and do not defer it until "after eval looks good". The whole point is to detect drift between trainer and consumer *before* the consumer (including your own eval harness) acts on bad assumptions; "I know what's in this one" doesn't survive a week, and "eval will catch any feeding bug" is exactly what eval can't catch when it's the one doing the bad feeding.

## Authoritative References

- **Schema**: `checkpoint_passport/schema.py` in this folder. Every field, type, and convention. Read it before doing anything else.
- **Validator semantics**: `checkpoint_passport/kernel/*.py`. Each file is one section's checks; the docstrings explain pass / soft-signal / fail criteria.
- **CLIs**: `validate-checkpoint`, `sign-checkpoint` (installed by `pip install -e .` from this folder).

## Prerequisites (training-side metadata)

The single biggest pain in passport generation is recovering things that should have been captured at training time. The launcher script that submits the training job MUST drop these into the checkpoint directory before training starts:

- **Training-repo commit SHA** (`git rev-parse HEAD` in the model code repo at submit time). Without this you cannot reproduce the exact model class definitions.
- **Dataset commit SHAs / HF revisions** for every dataset used (one per `--dataset` arg).
- **Resolved config** — the merged hyperparameter config the trainer actually used, written as `config.json` in the checkpoint root (most frameworks already do this).
- **Run-log pointer** — either a sanitised local `TRAINING_LOG.md` or the W&B / MLflow URL.

If any of the above is missing on a checkpoint you've been asked to passport, **stop and ask the user** before guessing. A passport built from guesses defeats the purpose.

## Phase 1 — Static extraction (no torch, no GPU, no container)

Pure file inspection. You can do this from any host (with the caveat in the table's last row: once `model_identity.class_module` is filled, full validation needs Phase 2's container). Populate every field in the schema you can, leave dynamic-only ones as `null` for now. Iterate against `validate-checkpoint <ckpt> --show-not-checked` as you go — `--show-not-checked` reveals which checks couldn't run because their inputs were missing from the passport, which is the closest thing the validator has to a "what fields am I still missing?" view. Note that the validator reports check results, not a field-by-field schema completion linter; for the latter, `schema.py` itself is the canonical list.

**What to read, and what schema field it feeds:**

| Source | Schema target |
|---|---|
| `config.json` → `input_features.observation.images.*` | `input_contract.images[]` (one entry per camera; `raw_shape`, `dtype`, `value_range`, `color_order`, `channel_layout`) |
| `config.json` → `observation_encoder.resize_shape` + backbone identifier | `input_contract.images[].encoder_resize`; backbone repo + revision into `model_internals.pretrained_provenance[]` |
| `config.json` → `input_features.observation.state.shape` | `input_contract.state.total_dim` |
| `config.json` → `dataset_schema` (and `rot6d_slice` if present) | `input_contract.state.sub_keys` (per-sub-key model-facing breakdown — see convention below); rot6d expansion documented here too |
| `config.json` → `output_features.action.shape` + top-level `horizon` | `input_contract.actions.total_dim`, `.horizon` |
| `config.json` → `objective` block (scheduler, num_steps, `clip_sample`, `clip_sample_range`) and any `*_clip_value` post-processing fields | `output_spec.inference_parameters` (sampling config); also drives the smoke `range_check` acceptance bounds in Phase 2 |
| `config.json` → tokenizer / language fields | `input_contract.language` |
| `config.json` → top-level `n_obs_steps`, `control_rate_hz` | `input_contract.temporal` |
| Norm stats file (often `assets/<stack>_stats.json`, sometimes embedded in `config.json`) — keys + per-dim mean/std/quantiles + per-dim `norm_mask` | `input_contract.state.normalization`, `.actions.normalization`; `actions.norm_mask` and `actions.delta_dims` (structured `DeltaSpec` with `delta_mask: List[bool]` and `absolute_dims_reason`) |
| Training log / launcher metadata | `input_contract.training_datasets[]`, `provenance.*` |
| safetensors header (no weight load — `safe_open(..., framework='pt').keys()`) | `weight_integrity.weight_files[]` (path, sha256, size) |
| Config + class import path of training stack | `model_identity.class_name`, `.class_module` (note: once these are filled, `model_identity_resolvable` will hard-fail on any host that can't import the class — full validation moves to Phase 2's container) |
| Config + norm stats + dataset adapter code | `transform_pipeline[]` — ordered `TransformStep` list documenting the full data flow (sensor → action). Each step has `check_type: "static" \| "dynamic"` |
| Config + known bugs / environment issues | `known_issues[]` — `KnownIssue` entries with `id`, `severity`, `workaround`, `check_type` |
| Config → library versions + known breakage | `model_identity.runtime_constraints` — `required_versions`, `required_python`, `known_incompatible` |
| Camera metadata (serial numbers, USB paths, reference frames) | `input_contract.images[].camera_serial`, `.camera_usb_path`, `.reference_frame_hash`, `.reference_frame_path` |
| Checkpoint lineage (if fine-tuned) | `provenance.parent_checkpoint`, `.parent_description` |

**Conventions worth knowing now (full reasoning lives in the schema docstrings):**

- `state.sub_keys` describes the **model-facing** layout — i.e. what the forward pass sees, after any rotation expansion. For the raw dataset layout, look at `config.json::dataset_schema` if present. The two often disagree (e.g. a 6D rotation source expanded to 9D rot6d); the validator's `state_dim_consistency` soft signal will report the divergence — that's expected, document it in the passport.
- Norm-stats fingerprint: when stats are flat per-dim, populate `per_dim_q02` / `per_dim_q98`. When stats are per-timestep `(H, D)`, suffix the keys with `_at_t0` (the validator's helper takes row 0). The `_at_t0` convention is checked by the kernel.
- `provenance.run_log_path` accepts either a relative `.md` path under the checkpoint repo OR an `http(s)://` URI to an external dashboard (W&B, MLflow). Pick whichever is the canonical pointer for this run.
- `weight_integrity.weight_files[]` should list **every** file that ships with the checkpoint and is loaded at inference time, not just the safetensors. That includes `config.json`, norm stats, tokenizer files. The signer hashes everything in this list.
- `actions.delta_dims` is now a structured `DeltaSpec` with `delta_mask: List[bool]` (per-dim: `True` = delta, `False` = absolute) and `absolute_dims_reason` (e.g. "6D rotation (rot6d) passed through unchanged"). This replaces the free-text format from v0.1.
- `transform_pipeline` is a top-level ordered list of `TransformStep`. Populate one entry for each data transformation from raw sensor to model input and from model output to robot command. Example steps: resize, ImageNet normalize, rotation expansion, RAMEN normalize, delta computation, camera stacking, temporal stacking, text tokenization, RAMEN unnormalize, delta-to-absolute conversion.
- `runtime_constraints` on `model_identity` declares inference-time version requirements (contract), separate from `library_versions` (historical). Populate `required_versions` with pinned versions that **must** match (e.g. `transformers==5.4.0`), and `known_incompatible` with version ranges known to break.
- `known_issues` is a top-level list documenting environment/library/config bugs with workarounds and severity. Each issue has `check_type` to indicate whether code or an agent should verify it.

**Pretrained-backbone provenance:** for any external pretrained submodule (HF model, timm model, custom URL), record an entry in `model_internals.pretrained_provenance[]` with the framework's own identifier (HF revision sha for HF, full timm string like `vit_base_patch16_clip_224.openai` for timm, etc.). For HF specifically, do not leave `hf_revision: null` — `huggingface_hub.HfApi().model_info(repo).sha` returns the canonical revision in one call. Always pin.

## Phase 2 — Dynamic extraction (in the model's own container)

Load the model, walk it, run one forward pass on a synthetic calibration batch. This is the only phase that needs torch / GPU / the model package.

**Container contract** — these are non-negotiable:

1. **Run inside the model package's own container**, not a generic torch image. The container's torch / CUDA / dependency versions go straight into `model_identity.library_versions`.
2. **Bind-mount the training-repo source at the recorded commit SHA**, mounted read-only at e.g. `/repo:ro`, with `PYTHONPATH=/repo/src` (or wherever the package lives in that repo). Reason: the version installed in the container's image is almost always stale and will fail to load the checkpoint's config (draccus / pydantic / dataclasses all reject unknown fields). The launcher's commit SHA is the source of truth.
3. **Purge any pre-imported copy of the model package from `sys.modules`** before importing it from the bind-mount. Otherwise you'll silently use the stale installed version even after PYTHONPATH is set:
   ```python
   for name in list(sys.modules):
       if name == "<package>" or name.startswith("<package>."):
           del sys.modules[name]
   ```
4. **GPU access** (`docker run --gpus all ...`) — required. Forward pass on CPU is 10× slower for no benefit and the validator can't tell the difference.

**Calibration batch** — build it from `config.json` shapes. No hardcoding. For images use `torch.rand` (uniform `[0,1]`, matching most models' expected pre-normalization input range); for state use `torch.randn`; for language use the trained `default_prompt`. Use a small batch (1 or 2) — the smoke tests don't need volume, just signal.

**Inference call** — use the **public inference API** the deployment will use, not the lowest-level `forward()`. The two often differ in input preprocessing (image stacking across cameras, time-axis layout, attention mask construction). If you call `forward()` directly with your own preprocessing, you'll smoke-test a code path no real consumer ever runs. Read what the model's serving entry point (`select_action`, `predict`, `__call__`, etc.) does and reproduce its preprocessing in the calibration batch.

**What to populate from the dynamic run:**

- `model_identity.library_versions` — torch, python, cuda, the model package, all transitive deps you can list (`pkg_resources` or `importlib.metadata`).
- `model_internals.module_hierarchy` — walk `named_modules()`, build a tree (top 2 levels for structural identification; full tree is recoverable from `named_modules()`).
- `model_internals.parameters.summary` — totals (total_params, trainable, frozen, bytes, dtype_breakdown). Per-parameter list (`by_name`) was removed in v0.2; it's recoverable from `safe_open(path).metadata()`.
- `model_internals.buffers` — `named_buffers()`.
- `model_internals.state_dict` — `expected_keys_count`, `found_keys_count`, plus `missing_keys` / `unexpected_keys` from `model.load_state_dict(..., strict=False)`.
- `model_internals.numerical_health` — run forward twice with the same `torch.manual_seed(0)`, confirm no NaN/Inf, confirm `max_abs_diff == 0` between the two passes.
- `output_spec.smoke_results` — five typed buckets (`determinism`, `nan_inf`, `liveness`, `distribution`, `range_check`). Each is its own sub-dataclass in the schema with explicit fields — read `SmokeResults` in `schema.py` before populating. Each bucket needs `status: "pass" | "fail"`. The acceptance range for `range_check` should be derived from the model's own clip values (`clip_sample_range`, normalization clip values) — don't hardcode.
- `reference_test_vector` — golden input/output pair (state, prompt, image hashes, expected output, tolerance, torch seed). Run the model on this fixed input and record the output.
- `norm_round_trip_results` — for each normalization step in `transform_pipeline`, take a known input, normalize, unnormalize, verify recovery within tolerance. Records `max_abs_error`, `within_clip_bound`, `status`.

Splice the dynamic JSON into the static draft (a small Python script merging the two dicts is fine). The passport is now complete.

## Check Classification — Static vs Dynamic

Every checkable claim in the passport falls into one of two categories:

**Static checks (code runs, pass/fail, no judgment):**
- File hashes match signoff
- Library versions match `runtime_constraints.required_versions`
- State dict key count matches
- Config values match passport (action_dim, horizon, image shapes)
- Norm stats file sha matches passport fingerprint
- No NaN/Inf in weights
- Reference test vector reproduces expected output within tolerance
- Device paths / camera serials match expected mapping
- `observation_delta_indices` matches config
- Transform pipeline steps match code behavior
- Normalization round-trip invertibility

**Dynamic checks (agent runs, needs judgment or context):**
- Camera swap detection (visual comparison of live frame to reference)
- Soft signal triage (is state_dim 13→16 rot6d expansion intentional?)
- Environment contamination (unexpected PYTHONPATH entries)
- Output sanity on real data (are actions physically reasonable?)
- Cross-referencing passport against W&B training logs
- Deciding if a version mismatch is acceptable or blocking
- Validating that `physical_mounting` descriptions match physical setup

## Phase 3 — Validate and iterate

Run the validator inside the same container as Phase 2 (it needs to import the model class for `model_identity_resolvable`):

```
validate-checkpoint <ckpt_dir> --show-not-checked
```

Read the report. Three outcomes:

- **All hard checks pass, no soft signals** → proceed to Phase 4.
- **Soft signals only** → for each soft signal, decide: is this a passport authoring bug (fix the passport), or a real but acceptable divergence (e.g. `state_dim_consistency` firing for a model with rotation expansion)? Soft signals you accept must be documented in the signoff `--reason` flag in Phase 4.
- **Any hard fail** → fix the passport (or the checkpoint) and re-run. Do not proceed to Phase 4 with hard fails; the signer will refuse anyway.

Common soft signals and how to handle them:

- `state_dim_consistency` (model-facing 16D vs dataset-facing 13D, etc.) → expected for any model with a rotation conversion. Document in `state.sub_keys` and accept in signoff.
- `input_contract_vs_dataset` reporting `NOT_CHECKED` → the validator can't find the local HF dataset cache snapshot. Either pass `--dataset-path /local/dataset` if you have it, or accept (the cross-check just isn't running, no false claim of pass).
- `training_datasets_resolvable` soft-signalling on a `null` dataset commit → fix at training-launch time (capture the SHA), then regenerate. Don't ship without dataset commits if you can help it.

## Phase 4 — Sign

```
sign-checkpoint <ckpt_dir> --reason '<one-liner if any soft signals>'
```

The signer:

1. Re-runs `validate-checkpoint` internally. Refuses to sign if there are hard failures (exits 1, prints the failures).
2. Demands `--reason` when there are soft signals, and prints a paste-able auto-summary listing the check names so you have a starting point.
3. Hashes `MODEL_PASSPORT.json` plus every file listed in `passport.weight_integrity.weight_files[]`. Auto-discovery — you don't list weight files separately.
4. Writes `SIGNOFF.json` at the checkpoint root with a verdict derived from the validator (`ship_it` → `pass`, `human_look_here` → `soft_signal`).
5. Re-runs the validator with `--require-signoff` to confirm round-trip.

After signing, the deployment gate is one command anywhere:

```
validate-checkpoint <ckpt_dir> --require-signoff
```

Non-zero exit = do not ship.

## Quick Reference

| Step | Command |
|---|---|
| Install once per env | `uv pip install -e <autohpc>/checkpoint-passport` |
| Validation (host — most checks; `model_identity_resolvable` may fail if class isn't importable) | `validate-checkpoint <ckpt>` |
| Full validation (in model container) | `docker run --gpus all -v <repo>:/repo:ro -e PYTHONPATH=/repo/src <image> validate-checkpoint <ckpt>` |
| Show NOT_CHECKED rows (closest thing to a "what's missing in my passport" view) | `validate-checkpoint <ckpt> --show-not-checked` |
| Skip a section while iterating | `validate-checkpoint <ckpt> --skip-section <category>` |
| Sign (refuses on hard fails) | `sign-checkpoint <ckpt> --reason '<why soft signals OK>'` |
| Dry-run sign (print, don't write) | `sign-checkpoint <ckpt> --dry-run` |
| Deployment gate | `validate-checkpoint <ckpt> --require-signoff` |

## Common Mistakes

- **Skipping the passport because the checkpoint is "for internal use"** — the whole point is detecting drift, and "internal" use cases drift fastest because no one is paying attention.
- **Generating the passport in a generic torch container instead of the model's own container** — the recorded `library_versions` then describe an environment no consumer will ever match.
- **Calling `model.forward()` directly in the smoke test instead of the public inference method** — you smoke-test code no deployment uses; the real serving path's preprocessing bugs go undetected.
- **Not bind-mounting the training-time source** — the checkpoint's `config.json` will reference fields the container's installed package version doesn't know, and config decoding will fail.
- **Forgetting to purge `sys.modules` after setting PYTHONPATH** — Python silently keeps the pre-imported stale copy; you waste an hour debugging a "fix" that didn't take effect.
- **Hardcoding the smoke test's acceptance range** — derive from the model's own clip values; otherwise a config change that legitimately widens the action range will fail validation forever.
- **Listing only safetensors in `weight_integrity.weight_files[]`** — `config.json` and norm stats files are also required for inference. If they aren't hashed, a corrupted norm stats file will silently produce wrong actions and the signoff won't catch it.
- **Leaving `hf_revision: null` for a pretrained backbone** — one `HfApi.model_info(repo).sha` call away from a pinned revision; without it, `pip install` updates can silently swap the backbone.
- **Treating soft signals as "good enough, ship it"** — sign with `--reason` and explain *each* soft signal. The reason string is the only audit trail for an acceptance decision.
- **Editing the passport after signing without re-signing** — the passport sha in the signoff no longer matches; the next `validate-checkpoint --require-signoff` hard-fails. Always re-run `sign-checkpoint` after any passport edit.
- **Generating a passport without the training-repo commit SHA** — you'll guess wrong and produce a passport that documents the wrong code. Stop and ask the user instead.
- **Leaving `transform_pipeline` empty** — the pipeline is the chain-of-custody. Without it, the passport can't be audited end-to-end.
- **Leaving `runtime_constraints` empty when there are known version sensitivities** — `library_versions` is historical; if a specific version is *required* for correctness (e.g. transformers API drift), it must be in `runtime_constraints.required_versions`.
- **Not running the normalization round-trip check** — clipping-induced loss is expected and documented; what catches bugs is wrong stats / wrong function / wrong mask, and the round-trip surfaces all of these.
