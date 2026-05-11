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

## Task 4: Add OpenPI Smoke/Runtime Enrichment [TODO]

**Files:**
- Modify: `checkpoint-passport/checkpoint_passport/runtime_extractors/openpi.py`
- Create: `checkpoint-passport/tests/test_openpi_extractor.py`

**Step 1: Extend the OpenPI seed extractor**

Use the same `extract-passport-seed openpi` CLI from Task 2. This task adds
runtime enrichment beyond static-ish contract extraction:

```bash
extract-passport-seed openpi \
  --checkpoint-dir <ckpt> \
  --out <ckpt>/PASSPORT_SEED.json \
  --device cuda \
  --openpi-config-name <name> \
  --default-prompt <prompt> \
  --resize-size 224
```

**Step 2: Fail fast for unsupported architectures**

Every unsupported architecture exits non-zero with:

```text
extractor not implemented for architecture <name>; stop and ask
```

Do not register `lerobot` or generic HF extractors in the MVP unless they have tests and working implementations.

**Step 3: Use the known OpenPI adapter path**

Use a documented path equivalent to:

```python
policy_info = load_policy_adapter(
    checkpoint_dir,
    device=device,
    force_architecture="openpi",
    openpi_config_name=config_name,
    default_prompt=default_prompt,
    resize_size=resize_size,
)
adapter = policy_info.adapter
```

If `missiontracker` or OpenPI imports fail, print a clear missing-runtime error and stop. Do not inspect unrelated deployment code.

**Step 4: Emit runtime enrichment**

Populate what can be safely extracted:

- extractor metadata
- resolved model class/name if available
- library versions for Python, torch/OpenPI/JAX if available
- parameter summary if available
- numerical health / smoke result if the adapter supports a bounded smoke call

Leave unknown fields unset. Do not guess.

**Step 5: Test**

Use fakes/monkeypatches for missing imports and happy-path shape. Do not require real OpenPI in unit tests.

Run:

```bash
cd checkpoint-passport
uv run pytest tests/test_openpi_seed_extractor.py tests/test_openpi_extractor.py -q
```

Expected: pass.

## Task 5: Add Minimal Publish Gate [TODO]

**Files:**
- Create: `checkpoint-passport/checkpoint_passport/cli/check_publish_ready.py`
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

**Step 4: Test**

Test:

- missing required files
- empty README/log
- invalid passport JSON
- missing/invalid signoff
- `--json` output shape
- multiple errors reported together

Run:

```bash
cd checkpoint-passport
uv run pytest tests/test_publish_ready.py -q
```

Expected: pass.

## Task 6: Rewrite Checkpoint Passport Skill [TODO]

**Files:**
- Modify: `checkpoint-passport/SKILL.md`
- Modify lightly: `checkpoint-passport/README.md`
- Modify lightly: `README.md`

**Step 1: Replace old phases**

New flow:

1. Materialize config when needed.
2. For config-bearing checkpoints, generate static passport deterministically.
3. For OpenPI checkpoints, run `extract-passport-seed openpi`.
4. Assemble `MODEL_PASSPORT.json` from the OpenPI seed.
5. Validate.
6. Sign.
7. Check publish readiness before HF upload.

**Step 2: Remove loose instructions**

Remove or replace instructions to:

- write a small config script
- discover environments by probing
- read source until the inference path is understood
- splice JSON with a small script
- fill fields by judgment

**Step 3: Add stop gates**

Agents stop when:

- a config-bearing checkpoint has no explicit config path
- architecture is unsupported
- OpenPI runtime imports are missing
- required extractor args are missing
- passport seed validation/assembly fails
- publish gate fails

**Step 4: Keep README short**

`checkpoint-passport/README.md` should list commands and point to `SKILL.md`. Root `README.md` should point to the bounded workflow and mention publish readiness. Do not duplicate the full workflow in three places.

## Task 7: Run Focused MVP Tests [TODO]

**Files:**
- Existing tests from Tasks 1-5

Run:

```bash
cd checkpoint-passport
uv run pytest \
  tests/test_generate_determinism.py \
  tests/test_passport_seed.py \
  tests/test_openpi_seed_extractor.py \
  tests/test_assemble_passport.py \
  tests/test_openpi_extractor.py \
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
- `MODEL_PASSPORT.json` is assembled by `assemble-passport`, not ad hoc scripts.
- `checkpoint-passport/SKILL.md` no longer contains loose Phase 2 instructions.
- HF upload is blocked unless `README.md`, `TRAINING_LOG.md`, `MODEL_PASSPORT.json`, and `SIGNOFF.json` are present and valid.
- Focused MVP tests pass.
