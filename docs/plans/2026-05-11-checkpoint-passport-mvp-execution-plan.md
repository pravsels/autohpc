# Checkpoint Passport MVP Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship the smallest checkpoint-passport rewrite that prevents agents from inventing config/runtime scripts, makes static generation deterministic, supports the known OpenPI pain case, and blocks incomplete HF uploads.

**Architecture:** This plan is the source of truth for the MVP. The MVP path is: explicit static generation -> OpenPI config materializer -> OpenPI runtime extractor -> merge -> validate/sign -> publish gate -> updated `SKILL.md`.

**Tech Stack:** Python 3.10+, existing `checkpoint_passport` package, `argparse`, `dataclasses`, `json`, `pytest`. Runtime dependencies such as OpenPI and torch are supplied by the target model environment, not by AutoHPC core.

---

## Non-Negotiable Rules

- Do not leave any agent-facing step that says to discover, infer, or write a one-off script.
- Do not add broad support for every model family in the MVP.
- Do not implement generic HF extraction in the MVP unless all OpenPI work is done.
- Do not vendor OpenPI, LeRobot, torch, transformers, or missiontracker into AutoHPC.
- If an architecture is unsupported, fail fast with a stop-and-ask message.
- If `README.md` or `TRAINING_LOG.md` is missing, HF upload is blocked.

## Task 1: Make Static Generation Deterministic [DONE]

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

## Task 2: Add OpenPI Config Materializer [TODO]

**Files:**
- Create: `checkpoint-passport/checkpoint_passport/config_materializers.py`
- Create: `checkpoint-passport/checkpoint_passport/cli/materialize_config.py`
- Modify: `checkpoint-passport/pyproject.toml`
- Create: `checkpoint-passport/tests/test_materialize_config.py`

**Step 1: Add CLI contract**

Implement:

```bash
materialize-passport-config openpi \
  --checkpoint-dir <ckpt> \
  --out <ckpt>/passport_config.json \
  --openpi-config-name <name> \
  --default-prompt <prompt> \
  --resize-size 224
```

Unknown architectures fail fast and list supported options. Missing OpenPI args fail fast.

**Step 2: Keep dependencies external**

OpenPI materialization must run in an OpenPI-capable environment. If imports fail, print:

```text
OpenPI materialization must run inside the OpenPI runtime environment.
Missing import: <module>. Activate the target environment or container and rerun.
```

Do not search the filesystem for alternate loader code. If local reference paths such as `../alpha-robotics/...` are absent, stop and ask.

**Step 3: Emit passport-facing config**

Extract what is available from OpenPI config/runtime:

- policy type / architecture
- config name
- default prompt
- action horizon
- action dimension if available
- image keys and resize size if available
- `use_delta_actions` if available
- checkpoint format: `model.safetensors` or Orbax `params`

Do not guess missing fields. Leave unknowns unset/null.

**Step 4: Register script**

Add:

```toml
materialize-passport-config = "checkpoint_passport.cli.materialize_config:main"
```

**Step 5: Test**

Use monkeypatch/fake modules. Do not require real OpenPI in unit tests.

Run:

```bash
cd checkpoint-passport
uv run pytest tests/test_materialize_config.py -q
```

Expected: pass.

## Task 3: Add Minimal Runtime Fragment And Merge [TODO]

**Files:**
- Create: `checkpoint-passport/checkpoint_passport/runtime_fragment.py`
- Create: `checkpoint-passport/checkpoint_passport/merge_runtime.py`
- Create: `checkpoint-passport/checkpoint_passport/cli/merge_runtime.py`
- Modify: `checkpoint-passport/pyproject.toml`
- Create: `checkpoint-passport/tests/test_runtime_fragment.py`
- Create: `checkpoint-passport/tests/test_merge_runtime.py`

**Step 1: Define minimal runtime fragment**

For MVP, include only:

- extractor metadata
- `model_identity`
- `model_internals`
- `output_spec.smoke_results`

Forbid runtime fragments from containing:

- `input_contract`
- `weight_integrity`
- `provenance`

Unknown top-level keys fail validation.

**Step 2: Validate inside merge**

Do not add a standalone `validate-passport-runtime` CLI in the MVP unless truly needed. `merge-passport-runtime` should load and validate the runtime fragment before merging.

**Step 3: Implement simple merge v1**

Rules:

- Protected static sections are never overwritten: `schema_version`, `generated_by`, `generated_at`, `weight_integrity`, `provenance`.
- Runtime fills missing/null runtime-owned fields.
- If runtime tries to change a non-null static value, exit non-zero and report the JSON path.
- Output JSON should be stable enough that repeated merges produce byte-identical output.

**Step 4: Add CLI**

```bash
merge-passport-runtime \
  --passport <ckpt>/MODEL_PASSPORT.json \
  --runtime <ckpt>/PASSPORT_RUNTIME.json \
  --out <ckpt>/MODEL_PASSPORT.json
```

Register:

```toml
merge-passport-runtime = "checkpoint_passport.cli.merge_runtime:main"
```

**Step 5: Test**

Run:

```bash
cd checkpoint-passport
uv run pytest tests/test_runtime_fragment.py tests/test_merge_runtime.py -q
```

Expected: pass.

## Task 4: Add OpenPI Runtime Extractor [TODO]

**Files:**
- Create: `checkpoint-passport/checkpoint_passport/runtime_extractors/__init__.py`
- Create: `checkpoint-passport/checkpoint_passport/runtime_extractors/base.py`
- Create: `checkpoint-passport/checkpoint_passport/runtime_extractors/openpi.py`
- Create: `checkpoint-passport/checkpoint_passport/cli/extract_runtime.py`
- Modify: `checkpoint-passport/pyproject.toml`
- Create: `checkpoint-passport/tests/test_extract_runtime_cli.py`
- Create: `checkpoint-passport/tests/test_openpi_extractor.py`

**Step 1: Add extractor CLI**

```bash
extract-passport-runtime openpi \
  --checkpoint-dir <ckpt> \
  --passport <ckpt>/MODEL_PASSPORT.json \
  --out <ckpt>/PASSPORT_RUNTIME.json \
  --device cuda \
  --openpi-config-name <name> \
  --default-prompt <prompt> \
  --resize-size 224
```

Register:

```toml
extract-passport-runtime = "checkpoint_passport.cli.extract_runtime:main"
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

**Step 4: Emit minimal runtime fragment**

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
uv run pytest tests/test_extract_runtime_cli.py tests/test_openpi_extractor.py -q
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
2. Generate static passport deterministically.
3. Run runtime extractor.
4. Merge runtime fragment.
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

- no config and no materializer exists
- architecture is unsupported
- OpenPI runtime imports are missing
- required extractor args are missing
- runtime fragment validation/merge fails
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
  tests/test_materialize_config.py \
  tests/test_runtime_fragment.py \
  tests/test_merge_runtime.py \
  tests/test_extract_runtime_cli.py \
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
- Real local checkpoint integration harness.
- Hermes acceptance scorecard.
- Extensive README/root README rewrites.

## Completion Criteria

MVP is complete when:

- `generate-passport` can run deterministically with explicit inputs.
- HF network resolution is opt-in only.
- OpenPI checkpoints without `config.json` have a documented materializer path.
- OpenPI runtime extraction is a documented command, not agent discovery.
- Runtime data is merged by `merge-passport-runtime`, not ad hoc scripts.
- `checkpoint-passport/SKILL.md` no longer contains loose Phase 2 instructions.
- HF upload is blocked unless `README.md`, `TRAINING_LOG.md`, `MODEL_PASSPORT.json`, and `SIGNOFF.json` are present and valid.
- Focused MVP tests pass.
