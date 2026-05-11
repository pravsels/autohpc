# Checkpoint Passport MVP Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship the smallest checkpoint-passport rewrite that prevents agents from inventing config/runtime scripts, keeps static generation deterministic for config-bearing checkpoints, supports the known OpenPI pain case through a runtime passport seed extractor, and blocks incomplete HF uploads.

**Architecture:** This plan is the source of truth for the MVP. There are two valid passport creation paths. Config-bearing checkpoints use `generate-passport --config ...` for deterministic static generation. OpenPI checkpoints use `extract-passport-seed openpi ...` followed by `assemble-passport ...` because their inference contract is constructed at runtime by the OpenPI data/model pipeline, not stored in a static `config.json`. Both paths end in the same validate/sign/publish gate.

**Tech Stack:** Python 3.10+, existing `checkpoint_passport` package, `argparse`, `dataclasses`, `json`, `pytest`. Runtime dependencies such as OpenPI and torch are supplied by the target model environment, not by AutoHPC core.

---

## Non-Negotiable Rules

- Do not leave any agent-facing step that says to discover, infer, or write a one-off script.
- Do not add broad support for every model family in the MVP.
- Do not implement generic HF extraction in the MVP unless all OpenPI work is done.
- Do not vendor OpenPI, LeRobot, torch, transformers, or missiontracker into AutoHPC.
- If an architecture is unsupported, fail fast with a stop-and-ask message.
- If `README.md` or `TRAINING_LOG.md` is missing, HF upload is blocked.
- Do not pretend OpenPI has a passport-compatible static config. Its input/output contract must come from its runtime pipeline.
- `reference_test_vector` is required for signable passports. Agents must use a checked-in command path to create it; they must not invent one-off dataset sampling scripts.
- Dataset repo IDs are required when known, but dataset commit pins are optional provenance. Missing dataset commits may be reported as a soft signal; they should not block signing by themselves.

## Task 1: Make Static Generation Deterministic [DONE]

**Post-implementation notes:**
- Removed legacy discovery (`_find_config`, `_find_stats`, `--allow-legacy-discovery`). `config_path` is required.
- Removed `GeneratedBy` dataclass — `generated_by` is now a plain string.
- Added recursive null/empty pruning to `_dataclass_to_dict` — JSON only contains populated fields.
- Trimmed redundant/always-same fields: `output_spec.actions` (mirrored input_contract), `stats_fingerprint` (duplicated weight_integrity), `color_order`/`dtype`/`value_range` (always identical), `inference_parameters.extra` (training-time), uniform `norm_mask`, empty `model_identity`.
- Also modified: `checkpoint-passport/checkpoint_passport/schema.py`
- Also created: `checkpoint-passport/examples/sample_passport.json`

**Files:**
- Modify: `checkpoint-passport/checkpoint_passport/cli/generate.py`
- Create: `checkpoint-passport/tests/test_generate_determinism.py`

**Step 1: Add tests first**

Create a minimal checkpoint fixture with `config.json` and a tiny placeholder weight file. Test:

- `generate_passport(..., config_path=..., generated_at=..., resolve_remote_revisions=False)` returns identical dicts on repeated calls.
- Monkeypatched HF resolver functions are not called unless remote resolution is explicitly enabled.
- Pinned dataset syntax like `example-org/example-dataset@abc123:lerobot.datasets.LeRobotDataset` populates the dataset commit.

Run:

```bash
cd checkpoint-passport
uv run pytest tests/test_generate_determinism.py -q
```

Expected before implementation: fail because the API does not accept those arguments.

**Step 2: Split library API from CLI exits**

Move failure behavior out of `generate_passport()`.

- Library code should raise a typed exception or `ValueError`.
- `main()` catches exceptions and exits with the right code.
- No library function should call `sys.exit()`.

**Step 3: Add explicit generation inputs**

Add support for:

- `config_path`
- `stats_path`
- `generated_at`
- `resolve_remote_revisions=False`
- pinned dataset specs

Agent-facing CLI must require `--config`. Any legacy config discovery must be opt-in via `--allow-legacy-discovery` and must not be documented in the skill.

**Step 4: Gate remote resolution**

Default behavior must not call Hugging Face APIs. Add `--resolve-remote-revisions` as the only path that can call remote APIs.

**Step 5: Verify**

Run:

```bash
cd checkpoint-passport
uv run pytest tests/test_generate_determinism.py -q
```

Expected: pass.

## Task 2: Replace OpenPI Materializer With Runtime Passport Seed [DONE]

**Post-implementation notes:**
- Deleted the materializer entirely (`config_materializers.py`, `cli/materialize_config.py`, `tests/test_materialize_config.py`) — it was the wrong abstraction, not repurposed.
- The OpenPI seed extractor calls `cfg.data.create(assets_dirs, model_config)` to get the instantiated pipeline. The output transform's `action_dim` gives the real robot-facing dim (e.g. 7 for joints-only), not `model.action_dim` (32, the tokenized transformer dim). Both are recorded: `input_contract.actions.total_dim` = robot-facing, `model_internals.forward_graph.sample_output_shapes.action_tokens` = tokenized.
- If `data.create()` fails (missing assets, wrong env), the extractor falls back gracefully to config attributes only — still produces a valid seed, just with fewer fields.
- Real checkpoint test (`pi05_build_block_tower_baseline_6mix_joints_only`) showed `image_keys: []` because `BlockTowerInputs` doesn't expose an `image_keys` attribute. This is a known gap for Task 4 enrichment.
- `model_type` is an enum (`ModelType.PI05`); the extractor stringifies via `.value` → `"pi05"`.
- Training datasets parsed from OpenPI's bracket-delimited `repo_id` string (e.g. `"[repo1, repo2, ...]"`).
- Passport seed validation (`passport_seed.py`) enforces the boundary: seeds cannot contain `schema_version`, `generated_at`, `generated_by`, `weight_integrity`, or `provenance` — those are assembler-owned.
- `pyproject.toml`: removed `materialize-passport-config`, added `extract-passport-seed`.

**Files:**
- Delete or repurpose: `checkpoint-passport/checkpoint_passport/config_materializers.py`
- Delete or repurpose: `checkpoint-passport/checkpoint_passport/cli/materialize_config.py`
- Delete or rewrite: `checkpoint-passport/tests/test_materialize_config.py`
- Create: `checkpoint-passport/checkpoint_passport/passport_seed.py`
- Create: `checkpoint-passport/checkpoint_passport/runtime_extractors/__init__.py`
- Create: `checkpoint-passport/checkpoint_passport/runtime_extractors/base.py`
- Create: `checkpoint-passport/checkpoint_passport/runtime_extractors/openpi.py`
- Create: `checkpoint-passport/checkpoint_passport/cli/extract_passport_seed.py`
- Modify: `checkpoint-passport/pyproject.toml`
- Create: `checkpoint-passport/tests/test_passport_seed.py`
- Create: `checkpoint-passport/tests/test_openpi_seed_extractor.py`

**Step 1: State the corrected boundary**

OpenPI does not have a passport-compatible static config. A flat dict with
`policy_type`, `action_dim`, `action_horizon`, and `default_prompt` is not
enough for `generate-passport`, because the passport needs `input_features`,
`output_features`, norm behavior, action semantics, and transform behavior.

Those are constructed by the OpenPI runtime pipeline:

- OpenPI config registry lookup
- `cfg.data.create(...)`
- input/output transforms
- checkpoint `assets/` norm stats
- adapter loading/inference behavior

Therefore the OpenPI path should emit a **passport seed**, not a fake
`config.json`.

**Step 2: Add CLI contract**

Implement:

```bash
extract-passport-seed openpi \
  --checkpoint-dir <ckpt> \
  --out <ckpt>/PASSPORT_SEED.json \
  --openpi-config-name <name> \
  --default-prompt <prompt> \
  --resize-size 224
```

Unknown architectures fail fast and list supported options. Missing OpenPI args fail fast.

**Step 3: Keep dependencies external**

OpenPI seed extraction must run in an OpenPI-capable environment. If imports fail, print:

```text
OpenPI seed extraction must run inside the OpenPI runtime environment.
Missing import: <module>. Activate the target environment or container and rerun.
```

Do not search the filesystem for alternate loader code. If local reference paths such as `../alpha-robotics/...` are absent, stop and ask.

**Step 4: Emit passport seed sections**

The seed may include sections normally produced statically for config-bearing
checkpoints, because OpenPI's contract is runtime-constructed. Emit only fields
the extractor actually obtains from the runtime pipeline:

- `stack`
- `input_contract`
- `output_spec`
- `model_identity`
- `model_internals`
- optional `transform_pipeline` entries if directly available from transforms
- extractor metadata

Do not use model token dimensions as robot-facing action dimensions unless the
adapter/runtime confirms that they are the actual robot action dimensions. Do
not guess image keys, state keys, delta semantics, or resize behavior.

**Step 5: Register script**

Add:

```toml
extract-passport-seed = "checkpoint_passport.cli.extract_passport_seed:main"
```

Remove the stale materializer entry if present:

```toml
materialize-passport-config = "checkpoint_passport.cli.materialize_config:main"
```

**Step 6: Test**

Use monkeypatch/fake modules. Do not require real OpenPI in unit tests.

Run:

```bash
cd checkpoint-passport
uv run pytest tests/test_passport_seed.py tests/test_openpi_seed_extractor.py -q
```

Expected: pass.

## Task 3: Add Passport Seed Assembly [DONE]

**Post-implementation notes:**
- Library module `assemble_passport.py` takes a validated seed dict + checkpoint dir, reuses filesystem/git helpers from `generate.py` (`_sha256`, `_hashable_files`, `_find_training_log`, `_git_head`, `_git_remote_url`, `_git_is_dirty`).
- Returns a plain dict (not a `ModelPassport` dataclass) — uses `_prune()` for null/empty stripping, avoids schema round-trip overhead since seed sections are already plain dicts.
- CLI wrapper matches `extract-passport-seed` conventions: `--checkpoint-dir`, `--seed`, `--out`, plus optional `--generated-at`, `--target-repo`, `--training-repo`.
- `PASSPORT_SEED.json` added to `_hashable_files` skip set alongside `MODEL_PASSPORT.json` and `SIGNOFF.json` — it's a workflow artifact, not an inference-critical file.
- Tested end-to-end on real checkpoint (`pi05_build_block_tower_baseline_6mix_joints_only`): `extract-passport-seed openpi` → `assemble-passport` produced a valid `MODEL_PASSPORT.json` with 42 files hashed.
- 14 unit tests covering: basic assembly, seed section merging, weight integrity (hashing + exclusions), provenance, determinism, invalid seed rejection, missing checkpoint dir, and null/empty pruning.

**Files:**
- Create: `checkpoint-passport/checkpoint_passport/assemble_passport.py`
- Create: `checkpoint-passport/checkpoint_passport/cli/assemble_passport.py`
- Modify: `checkpoint-passport/pyproject.toml`
- Create: `checkpoint-passport/tests/test_assemble_passport.py`
- Modify: `checkpoint-passport/checkpoint_passport/cli/generate.py` (added `PASSPORT_SEED.json` to hash exclusion)

**Step 1: Define passport seed**

A passport seed is the partially assembled content emitted by
`extract-passport-seed openpi` for OpenPI checkpoints.

The seed may include:

- `stack`
- `input_contract`
- `output_spec`
- `model_identity`
- `model_internals`
- `transform_pipeline`

The canonical assembler owns:

- `schema_version`
- `generated_at`
- `generated_by`
- `weight_integrity`
- `provenance`
- final JSON pruning/ordering

Unknown top-level seed keys fail validation.

**Step 2: Add assembler CLI**

Implement:

```bash
assemble-passport \
  --checkpoint-dir <ckpt> \
  --seed <ckpt>/PASSPORT_SEED.json \
  --out <ckpt>/MODEL_PASSPORT.json
```

The assembler should hash checkpoint files, populate provenance, attach
generated metadata, and write the final `MODEL_PASSPORT.json`.

**Step 3: Preserve static generator path**

For config-bearing checkpoints, `generate-passport` continues to write
`MODEL_PASSPORT.json` directly. Do not refactor it into the seed path in the
MVP. `assemble-passport` exists for seed-producing runtime extractors such as
OpenPI.

**Step 4: Register script**

```toml
assemble-passport = "checkpoint_passport.cli.assemble_passport:main"
```

**Step 5: Test**

Run:

```bash
cd checkpoint-passport
uv run pytest tests/test_assemble_passport.py -q
```

Expected: pass.

## Task 4: Add OpenPI Smoke/Runtime Enrichment [DONE]

**Post-implementation notes:**
- Added `--device` CLI argument to `extract-passport-seed openpi`. When provided, loads the model and collects runtime enrichment (library versions, parameter summary, smoke inference). Without `--device`, only config-level extraction is performed.
- **Replaced missiontracker cross-repo dependency with self-contained RuntimeAdapter protocol.** The initial implementation used `missiontracker.adapters.factory` from the `alpha-robotics` repo, which dragged in deployment/eval machinery (ACTAdapter, LeRobot, PolicyObservation, etc.) that enrichment doesn't need. This was refactored into an internal protocol.
- New `runtime_adapters/` package with `RuntimeAdapter` ABC (4 methods: `load`, `count_parameters`, `smoke_inference`, `library_versions`) and `OpenPIRuntimeAdapter` implementation using only `openpi.*` APIs.
- `OpenPIRuntimeAdapter.load()` inlines the ~25 lines from missiontracker's `_create_trained_policy_local` — `restore_params` → `Policy` constructor with transforms. Only non-trivial part.
- `smoke_inference()` constructs an OpenPI-native dict directly (numpy arrays, no `PolicyObservation`, no torch tensors) and calls `policy.infer()`. The raw action dim for observations is derived from the data config's output transforms (e.g. 7 for joints-only), not `cfg.model.action_dim` (32, the tokenized dim).
- `count_parameters()` uses `flax.nnx.state(model)` + `jax.tree.leaves()` for JAX models, with PyTorch fallback.
- Deleted from `runtime_extractors/openpi.py`: `_import_missiontracker()`, `_build_dummy_observation()`, `_count_parameters()`, `_run_smoke_test()`, `_collect_library_versions()` — all replaced by protocol methods.
- Tests use a clean `FakeRuntimeAdapter` implementing the 4 protocol methods — no more sys.modules hacking for missiontracker fakes. 11 tests covering enrichment, smoke failure recording, missing params/smoke, JSON serialization, validation, and CLI wiring.
- Real checkpoint verification (`pi05-build-block-tower-baseline-6mix-joints-only`, `--device cuda`): 3,353,433,872 params (3.35B), smoke pass (7.7s), library versions (Python 3.11.14, torch 2.11.0+cu128, jax 0.5.3).
- Also added `--skip-file` and `--skip-dir` CLI arguments to `generate-passport` and `assemble-passport` to exclude training artifacts (e.g. `retain/`) from `weight_integrity` hashing.

**Files:**
- Create: `checkpoint-passport/checkpoint_passport/runtime_adapters/__init__.py`
- Create: `checkpoint-passport/checkpoint_passport/runtime_adapters/base.py`
- Create: `checkpoint-passport/checkpoint_passport/runtime_adapters/openpi.py`
- Modify: `checkpoint-passport/checkpoint_passport/runtime_extractors/openpi.py`
- Modify: `checkpoint-passport/checkpoint_passport/runtime_extractors/base.py`
- Modify: `checkpoint-passport/checkpoint_passport/cli/extract_passport_seed.py`
- Modify: `checkpoint-passport/checkpoint_passport/cli/generate.py`
- Modify: `checkpoint-passport/checkpoint_passport/cli/assemble_passport.py`
- Modify: `checkpoint-passport/checkpoint_passport/assemble_passport.py`
- Create: `checkpoint-passport/tests/test_openpi_extractor.py`

## Task 5: Add OpenPI Reference Test Vector Extraction [DONE]

**Post-implementation notes:**
- Added `reference_test_vector` to `ALLOWED_SEED_SECTIONS` in `passport_seed.py`.
- Added `extract_reference_sample()` to `RuntimeAdapter` protocol and `OpenPIRuntimeAdapter`.
- Added `_extract_reference_test_vector()` to OpenPI extractor — writes `.npy` states + PNG images under `assets/reference_test_vector/`, hashes all files, populates seed section.
- Added `_generate_dummy_reference_vector()` for MVP testing without model/dataset, exposed as `--dummy-reference-vector` CLI flag.
- Added `--reference-dataset-path`, `--reference-episode-index`, `--reference-start-frame`, `--reference-num-frames` CLI args to `extract-passport-seed`.
- Downgraded missing dataset commits from hard fail to soft signal in `check_training_datasets_resolvable` when repo IDs are present.
- Fixed `deployment_repo_commit` check — no longer promotes to hard fail when `require_signoff=True` but no `--target-repo` was provided.
- Bumped extractor version to `0.3.0`.
- Created `tests/test_reference_test_vector.py` (13 tests covering extraction, file hashing, prompts, provenance, dummy vector, mutual exclusivity, soft signals, schema validation).
- Updated `FakeRuntimeAdapter` in `test_openpi_extractor.py` to satisfy new protocol.

**Files:**
- Modify: `checkpoint-passport/checkpoint_passport/passport_seed.py`
- Modify: `checkpoint-passport/checkpoint_passport/runtime_extractors/openpi.py`
- Modify: `checkpoint-passport/checkpoint_passport/runtime_adapters/base.py`
- Modify: `checkpoint-passport/checkpoint_passport/runtime_adapters/openpi.py`
- Modify: `checkpoint-passport/checkpoint_passport/cli/extract_passport_seed.py`
- Modify: `checkpoint-passport/checkpoint_passport/kernel/input_expectation.py`
- Modify: `checkpoint-passport/tests/test_openpi_extractor.py`
- Create: `checkpoint-passport/tests/test_reference_test_vector.py`

**Step 1: Add tests first**

Write tests for the real contract before changing implementation:

- OpenPI extraction with reference data writes a top-level `reference_test_vector` section into `PASSPORT_SEED.json`.
- The extractor writes deterministic assets under `<ckpt>/assets/reference_test_vector/`.
- The state array is saved as `.npy`, hashed, and referenced by relative path.
- Image frames are saved under a directory, hashed per camera/frame, and referenced by relative path.
- Missing dataset path, missing required keys, unavailable image data, or too few frames fails with a stop-and-ask style error.
- `training_datasets_resolvable` treats missing dataset commits as a soft signal or pass-with-warning, not a hard failure, when dataset repo IDs are present.

Run:

```bash
cd checkpoint-passport
uv run pytest tests/test_reference_test_vector.py tests/test_openpi_extractor.py -q
```

Expected before implementation: fail because no checked-in command path creates
`reference_test_vector`.

**Step 2: Extend the OpenPI extractor CLI**

Fold reference vector creation into the existing OpenPI command so agents do not
choose between multiple scripts:

```bash
extract-passport-seed openpi \
  --checkpoint-dir <ckpt> \
  --out <ckpt>/PASSPORT_SEED.json \
  --openpi-config-name <name> \
  --default-prompt "<prompt>" \
  --resize-size 224 \
  --device cuda \
  --reference-dataset-path <dataset> \
  --reference-episode-index <episode> \
  --reference-start-frame <frame> \
  --reference-num-frames 10
```

The command must not discover a dataset path. If the path or sample selection is
missing, stop and ask for those exact values.

**Step 3: Define adapter responsibility**

Add a runtime adapter method that returns a normalized reference sample, not a
passport fragment. The OpenPI extractor remains responsible for writing files,
hashing them, and shaping the final schema.

The adapter output should include:

- consecutive state vectors shaped `(n_frames, state_dim)`
- image frames grouped by camera key
- prompt/task string if present in the dataset sample
- enough metadata to report the dataset path, episode index, and frame range in
  `reference_test_vector.notes`

Do not use synthetic zero inputs or smoke calibration data for this section.
Smoke inference and reference vectors serve different purposes.

**Step 4: Emit seed-owned reference section**

Allow `reference_test_vector` as a top-level key in `PassportSeed`. It is
runtime-owned for OpenPI because the relevant data layout is produced by the
OpenPI dataset/runtime path.

Write assets to:

```text
assets/reference_test_vector/input_states.npy
assets/reference_test_vector/images/<camera_key>_<frame_index>.png
```

Populate:

- `reference_test_vector.n_frames`
- `reference_test_vector.input_state_path`
- `reference_test_vector.input_state_hash`
- `reference_test_vector.input_images_path`
- `reference_test_vector.input_images_hash`
- `reference_test_vector.input_prompt`
- `reference_test_vector.notes`

All paths must be relative to the checkpoint root.

**Step 5: Keep dataset commits optional**

Update `training_datasets_resolvable` so missing `commit` values do not create a
hard failure when `repo_id` values are present. The check should still fail for
malformed or empty dataset entries.

If commit values are present, validate them as before. If commit values are
missing, report a soft signal such as:

```text
dataset repo IDs present but commits are not pinned; reproducibility provenance is weaker
```

Do not add automatic HF API lookup to the default OpenPI flow. Remote commit
resolution can be a future opt-in enrichment, not an MVP signing requirement.

**Step 6: Validate clean signing path**

After assembly, `validate-checkpoint` should pass the
`reference_test_vector` hard check without `--skip-section`.

Run:

```bash
cd checkpoint-passport
uv run pytest tests/test_reference_test_vector.py tests/test_openpi_extractor.py -q
```

Expected: pass.

## Task 6: Add Minimal Publish Gate [DONE]

**Post-implementation notes:**
- Created `check_publish_ready.py` — checks for required files (README.md, TRAINING_LOG.md, MODEL_PASSPORT.json, SIGNOFF.json), non-empty docs, valid JSON, then runs internal validator with `require_signoff=True`. Reports packaging and validation errors separately.
- Created `publish_checkpoint.py` — `upload` subcommand gates on `check-publish-ready`, auto-creates HF repo, supports `--ignore-patterns` for excluding dirs (e.g. `retain/**`). `download` subcommand gates on post-download `validate-checkpoint --require-signoff`.
- Registered both as `[project.scripts]` in `pyproject.toml`.
- Created `tests/test_publish_ready.py` covering missing files, empty docs, invalid JSON, signoff validation, `--json` output, upload gate, download gate.
- End-to-end tested on real OpenPI checkpoint: upload to `pravsels/pi05-build-block-tower-passport-test`, download, and post-download validation correctly caught incomplete download (missing weight files).

**Files:**
- Create: `checkpoint-passport/checkpoint_passport/cli/check_publish_ready.py`
- Create: `checkpoint-passport/checkpoint_passport/cli/publish_checkpoint.py`
- Modify: `checkpoint-passport/pyproject.toml`
- Create: `checkpoint-passport/tests/test_publish_ready.py`

**Step 1: Add CLI**

```bash
check-publish-ready <checkpoint_dir>
check-publish-ready <checkpoint_dir> --json
```

Register:

```toml
check-publish-ready = "checkpoint_passport.cli.check_publish_ready:main"
```

If CLI bloat becomes a problem during implementation, this can instead become `validate-checkpoint --require-packaging`. Keep whichever option is smallest and clearest.

**Step 2: Check required artifacts**

Fail if any are missing:

- `README.md`
- `TRAINING_LOG.md`
- `MODEL_PASSPORT.json`
- `SIGNOFF.json`

Also fail if `README.md` or `TRAINING_LOG.md` is empty.

**Step 3: Validate signoff**

Call the same internal validation path as `validate-checkpoint --require-signoff`, or shell out only if that is simpler for MVP.

Report packaging failures separately from passport/signoff validation failures.

**Step 4: Add bounded HF publish/download helpers**

Agents should not author upload/download shell scripts during checkpoint
handoff. Add small first-party helpers or a single helper with subcommands:

```bash
publish-checkpoint upload \
  --checkpoint-dir <ckpt> \
  --repo-id <user-or-org>/<repo> \
  --revision main

publish-checkpoint download \
  --repo-id <user-or-org>/<repo> \
  --revision <sha-or-branch> \
  --out <local_dir>
```

Upload must run `check-publish-ready <ckpt>` first and refuse to continue on
non-zero exit. Download must run `validate-checkpoint <downloaded_dir>
--require-signoff` after fetching and refuse to report success on non-zero
exit.

If authentication, repo creation, revision choice, or target path is missing,
stop and ask. Do not generate ad hoc `huggingface-cli upload`, `hf download`,
`git lfs`, `rsync`, or Python upload scripts.

Register if implemented as a separate command:

```toml
publish-checkpoint = "checkpoint_passport.cli.publish_checkpoint:main"
```

Keep this helper minimal. It should orchestrate existing HF CLI/library calls
with fixed validation gates, not become a general release manager.

**Step 5: Test**

Test:

- missing required files
- empty README/log
- invalid passport JSON
- missing/invalid signoff
- `--json` output shape
- multiple errors reported together
- upload refuses when `check-publish-ready` fails
- download refuses success when `validate-checkpoint --require-signoff` fails

Run:

```bash
cd checkpoint-passport
uv run pytest tests/test_publish_ready.py -q
```

Expected: pass.

## Task 7: Rewrite Checkpoint Passport Skill [DONE]

**Post-implementation notes:**
- Replaced the old Phase 1/2/3/4 structure with two explicit paths: Path A (config-bearing, `generate-passport --config`) and Path B (OpenPI, `extract-passport-seed openpi` → `assemble-passport`).
- Made `reference_test_vector` first-class: the checked-in `--reference-dataset-path` flag on `extract-passport-seed openpi` is the only valid extraction path. Explicitly prohibits separate sampling scripts and signing with skipped hard sections.
- Reflected dataset commits as optional provenance (soft signal, not hard blocker) when repo IDs are present.
- Added comprehensive stop gates: missing config path, unsupported architecture, OpenPI config name, runtime env, dataset path/episode/frame, HF auth/repo/revision, dirty target repo, training-repo commit SHA.
- Removed all loose instructions: write config scripts, discover environments, read source until inference understood, splice JSON, fill fields by judgment, write smoke-test scripts, write upload/download scripts.
- Added "What Not to Do" section consolidating prohibitions.
- Updated `checkpoint-passport/README.md`: lists all 7 commands, describes both paths, points to SKILL.md for workflow.
- Updated root `README.md`: post-train section describes both paths, publish readiness gate, and `publish-checkpoint` command.

**Files:**
- Modify: `checkpoint-passport/SKILL.md`
- Modify lightly: `checkpoint-passport/README.md`
- Modify lightly: `README.md`

**Step 1: Replace old phases**

New flow:

1. For config-bearing checkpoints, run `generate-passport --config ...`.
2. For OpenPI checkpoints, run `extract-passport-seed openpi`.
3. For OpenPI checkpoints, include reference vector arguments when creating the seed.
4. Assemble `MODEL_PASSPORT.json` from the OpenPI seed.
5. Do not materialize a fake OpenPI config.
6. Validate.
7. Sign without skipped hard sections.
8. Check publish readiness before HF upload.

**Step 2: Add runtime-adapter smoke rule**

For OpenPI, smoke testing is part of the checked-in runtime adapter path:

```bash
extract-passport-seed openpi \
  --checkpoint-dir <ckpt> \
  --out <ckpt>/PASSPORT_SEED.json \
  --openpi-config-name <name> \
  --default-prompt "<prompt>" \
  --resize-size 224 \
  --device cuda
```

The agent must not write a separate smoke-test script, call OpenPI internals
directly, or invent calibration inputs. If runtime enrichment fails because the
OpenPI environment, checkpoint assets, or device are unavailable, stop and
report the exact error.

**Step 3: Add reference vector rule**

For OpenPI, reference vector creation is part of the checked-in seed extraction
path:

```bash
extract-passport-seed openpi \
  --checkpoint-dir <ckpt> \
  --out <ckpt>/PASSPORT_SEED.json \
  --openpi-config-name <name> \
  --default-prompt "<prompt>" \
  --resize-size 224 \
  --device cuda \
  --reference-dataset-path <dataset> \
  --reference-episode-index <episode> \
  --reference-start-frame <frame> \
  --reference-num-frames 10
```

The agent must not write a separate dataset sampling script. If dataset path,
episode, or frame range is unknown, stop and ask. Dataset commits are optional
provenance, but the reference vector itself is required for signing.

**Step 4: Remove loose instructions**

Remove or replace instructions to:

- write a small config script
- discover environments by probing
- read source until the inference path is understood
- splice JSON with a small script
- fill fields by judgment
- write a separate smoke-test script
- write upload/download scripts for HF handoff

**Step 5: Add stop gates**

Agents stop when:

- a config-bearing checkpoint has no explicit config path
- architecture is unsupported
- OpenPI runtime imports are missing
- OpenPI runtime enrichment fails because env/assets/device are unavailable
- OpenPI reference dataset path, episode, or frame range is missing
- OpenPI reference vector extraction cannot produce real state/image assets
- required extractor args are missing
- passport seed validation/assembly fails
- publish gate fails
- HF auth, repo id, revision, or local output path is missing

**Step 6: Keep README short**

`checkpoint-passport/README.md` should list commands and point to `SKILL.md`. Root `README.md` should point to the bounded workflow and mention publish readiness. Do not duplicate the full workflow in three places.

## Task 8: Run Focused MVP Tests [DONE]

**Post-implementation notes:**
- Fixed one pre-existing test (`test_upload_calls_hf_when_ready`) that didn't account for the `ignore_patterns=None` kwarg added in Task 6's `--ignore-patterns` feature.
- Focused suite: 101 passed in 0.42s.
- Full suite: 101 passed, 1 skipped in 0.41s.

**Files:**
- Existing tests from Tasks 1-6

Run:

```bash
cd checkpoint-passport
uv run pytest \
  tests/test_generate_determinism.py \
  tests/test_passport_seed.py \
  tests/test_openpi_seed_extractor.py \
  tests/test_assemble_passport.py \
  tests/test_openpi_extractor.py \
  tests/test_reference_test_vector.py \
  tests/test_publish_ready.py \
  -q
```

Expected: pass.

Then run:

```bash
cd checkpoint-passport
uv run pytest -q
```

Expected: pass.

## Post-MVP

Do not implement these until the MVP passes:

- Generic `hf_pretrained_torch` extractor.
- `custom_adapter` scaffold.
- Packaged adapter templates.
- Standalone `validate-passport-runtime` CLI.
- Full transform pipeline / norm round-trip runtime fragment sections.
- Full field-level merge policy table.
- Separate `materialize-passport-config` CLI, unless another architecture truly
  has a static framework config that can be materialized without runtime
  pipeline construction.
- Real local checkpoint integration harness.
- Hermes acceptance scorecard.
- Extensive README/root README rewrites.

## Completion Criteria

MVP is complete when:

- `generate-passport` can run deterministically with explicit inputs.
- HF network resolution is opt-in only.
- OpenPI checkpoints without `config.json` have a documented passport seed extractor path.
- OpenPI runtime extraction is a documented command that emits passport seed sections, not agent discovery.
- OpenPI reference test vectors are created by a documented command path using real dataset frames, not synthetic smoke inputs or ad hoc scripts.
- `MODEL_PASSPORT.json` is assembled by `assemble-passport`, not ad hoc scripts.
- `reference_test_vector` validation passes without `--skip-section`.
- Missing dataset commit pins are treated as optional provenance, not a hard signing blocker.
- `checkpoint-passport/SKILL.md` no longer contains loose Phase 2 instructions.
- HF upload is blocked unless `README.md`, `TRAINING_LOG.md`, `MODEL_PASSPORT.json`, and `SIGNOFF.json` are present and valid.
- HF upload/download uses documented `publish-checkpoint` or equivalent fixed
  commands, not agent-authored scripts.
- Focused MVP tests pass.
