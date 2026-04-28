# Passport v0.2 Schema Revision

## The goal

The passport is a **chain-of-custody document** from sensor to action
output. Every entity in the chain (sensor reading, image, state vector,
action) and every transformation (resize, normalize, delta, clip,
unnormalize) should be recorded so it can be audited. Code is also a
transformation, audited via library versions and commit hashes. Checks
against the passport fall into two categories: **static checks**
(code-based, deterministic, automated) and **dynamic checks**
(agent-driven, requiring judgment or runtime context).

## The full chain (sensor to robot execution)

Every arrow is a place where something can go wrong silently.

```
PHYSICAL WORLD
  1. Camera sensor (front) → raw frames at /dev/videoX
  2. Camera sensor (wrist) → raw frames at /dev/videoY
  3. Robot joints → raw state reading (joint positions, gripper)
  4. Task instruction → text string

DATASET PIPELINE (data collection time)
  5. Raw frames → recorded to dataset (resolution, FPS, color space)
  6. Raw state → recorded to dataset (key names, dimensions)
  7. Raw actions → recorded to dataset (what was the human/policy doing)
  8. Multiple datasets mixed → combined training set (6 datasets, sampling weights)

TRAINING PIPELINE (training time)
  9. Dataset loaded via LeRobot (repo + commit + version)
  10. Key rename map applied (dataset keys → model keys)
  11. Sub-features assembled (observation.state + observation.eef_6d_pose → flat vector)
  12. Rotation conversion (RPY 6D → rot6d 9D, state goes 13→16)
  13. Delta actions computed (action − state on norm_mask'd dims)
  14. RAMEN stats computed (q02/q98 percentiles per timestep per dim)
  15. RAMEN normalize state (percentile clamp, rot6d dims exempt)
  16. RAMEN normalize delta actions (same, per-timestep stats)
  17. ImageNet normalize images (mean/std per channel)
  18. Images resized to 224x224 (CLIP encoder input)
  19. Cameras stacked into OBS_IMAGES tensor (ordered: front, wrist)
  20. Temporal stacking (n_obs_steps=2, delta_indices=[-1, 0])
  21. Text tokenized via CLIP tokenizer (max 77 tokens)
  22. Model forward pass (diffusion, 100 train steps, epsilon prediction)
  23. Loss computed, gradients, optimizer step
  24. Checkpoint saved (config.json + model.safetensors + ramen_stats.json)

INFERENCE PIPELINE (deploy/eval time)
  25. Load checkpoint (config.json → draccus deserialize → policy class)
  26. Load stats (ramen_stats.json — detect format: ramen vs lerobot)
  27. Library versions must match training (transformers==5.4.0!)
  28. Live camera frame → same resize as training?
  29. Live state reading → same key mapping as training?
  30. Same rotation conversion?
  31. Same normalization (RAMEN with same stats file)?
  32. Same image normalization (ImageNet)?
  33. Same camera stacking order?
  34. Same temporal stacking?
  35. Same text tokenization?
  36. Model forward (diffusion, 20 inference steps, DDIM, clip_sample_range=1.5)
  37. RAMEN unnormalize action output
  38. Delta-to-absolute conversion (add current state back)
  39. Action sent to robot controller

ROBOT EXECUTION
  40. Controller interprets action (joint positions? velocities? deltas?)
  41. Safety limits applied?
  42. Execution at what rate? (control_rate_hz)
```

## Current problems with v0.1

- 76% of the passport (130KB) is `parameters.by_name` — a per-parameter
  dump that doesn't serve the chain-of-custody goal and is recoverable
  from the safetensors file header.
- The transform pipeline (the actual chain) is scattered across
  `input_contract.images[].normalization`,
  `input_contract.state.normalization`,
  `input_contract.actions.delta_dims`, `output_spec.post_processing` —
  you can't read it top to bottom.
- No distinction between what code should check vs what an agent should
  check.
- Missing links: camera identity for swap detection, runtime version
  constraints, temporal frame spacing, reference test vector, checkpoint
  lineage, normalization round-trip verification, known issues.

## Changes

### 1. TRIM — remove what doesn't serve the chain

- **Delete `parameters.by_name`** (130KB). Keep `parameters.summary`
  (total_params, trainable, frozen, bytes, dtype_breakdown). The
  per-parameter list is recoverable from `safe_open(path).metadata()`
  without loading the model, and the safetensors file is already hashed
  in the signoff.

- **Slim `module_hierarchy`** to top 2 levels. The full nested tree
  (3.2KB, all submodules) is recoverable from `named_modules()`. Top 2
  levels (backbone, heads, encoder) is enough for structural
  identification.

### 2. STRENGTHEN — make the chain explicit

- **Add `transform_pipeline`** — a new top-level section. An ordered
  list of steps that describes the full data flow:

```python
@dataclass
class TransformStep:
    order: int                              # position in the pipeline
    name: str                               # e.g. "resize_images", "ramen_normalize_state"
    applies_to: str                         # "images.front", "images.wrist", "state", "action", "all_images"
    operation: str                          # "resize", "imagenet_normalize", "ramen_normalize", "delta", "stack_cameras", ...
    direction: str                          # "input" (pre-model) or "output" (post-model)
    parameters: Dict[str, Any]             # operation-specific params (target_size, mean/std, clip_value, etc.)
    check_type: str                         # "static" or "dynamic"
    check_description: Optional[str]        # what to verify and how
```

  Example pipeline for dit_block_tower_norm_fix:

  1. resize images to 224x224 (static: compare config resize_shape)
  2. ImageNet normalize images (static: verify mean/std constants)
  3. rot6d expand state 13D→16D (static: verify norm_mask + rot6d_slice)
  4. RAMEN normalize state (static: verify stats file hash + q02/q98 fingerprint)
  5. RAMEN normalize action as delta (static: verify delta_dims + stats)
  6. stack cameras into OBS_IMAGES tensor (static: verify key order)
  7. temporal stack n_obs_steps=2 frames (static: verify observation_delta_indices)
  8. CLIP encode text prompt (static: verify tokenizer + model)
  9. diffusion forward pass (static: verify inference params)
  10. RAMEN unnormalize action output (static: verify inverse of step 5)
  11. delta-to-absolute conversion (static: verify dims match step 5)

- **Add `runtime_constraints`** to `ModelIdentity` — versions that are
  *required* at inference, not just observed at training. Today's
  `library_versions` is a historical record; `runtime_constraints` is a
  contract:

```python
@dataclass
class RuntimeConstraints:
    required_versions: Dict[str, str]       # {"transformers": "==5.4.0", "torch": ">=2.10.0"}
    required_python: Optional[str]          # ">=3.12,<3.13"
    known_incompatible: List[str]           # ["transformers>=5.5.0 (CLIP key layout change)"]
```

  Static check: compare installed versions against `required_versions`
  before loading.

- **Surface `observation_delta_indices`** in `TemporalSpec` — already
  exists as `delta_timestamps` but not populated in this passport. For
  this DiT checkpoint it's `[-1, 0]` (previous frame + current frame).
  Without this, feeding frames at wrong spacing silently corrupts.

- **Make `delta_dims` structured**, not free text. Currently it's
  `"dims 0-9 and 16 are RAMEN-normalized deltas; dims 10-15 are
  absolute 6D rotation..."`. Replace with:

```python
@dataclass
class DeltaSpec:
    delta_mask: List[bool]                  # per-dim: True = delta, False = absolute
    absolute_dims_reason: Optional[str]     # e.g. "6D rotation (rot6d) passed through unchanged"
```

### 3. ADD — new links in the chain

- **`camera_identity`** on `ImageSpec` — for camera swap detection:

```python
camera_serial: Optional[str]                # hardware serial number
camera_usb_path: Optional[str]              # USB bus/port topology (survives replugs)
reference_frame_hash: Optional[str]         # sha256 of a reference image from this camera
reference_frame_path: Optional[str]         # path to the reference image
```

  Static check (code): device path + serial match expected.
  Dynamic check (agent): compare live frame against reference — "does
  this view look like a front camera or a wrist camera?"

- **`reference_test_vector`** — a golden input/output pair for pipeline
  verification:

```python
@dataclass
class ReferenceTestVector:
    input_state: List[float]                # fixed state vector
    input_prompt: str                       # fixed text prompt
    input_images_hash: Dict[str, str]       # {cam_key: sha256 of fixed input image}
    input_images_path: Optional[str]        # path to stored reference images
    expected_output: List[List[float]]      # (horizon, action_dim) expected action chunk
    tolerance: float                        # max absolute diff for pass
    torch_seed: int                         # RNG seed used
    notes: Optional[str]
```

  Static check: load model, run reference input, compare output within
  tolerance.

- **`normalization_round_trip`** — verify normalize/unnormalize is
  invertible. For each normalize step in the transform pipeline, take a
  known input (from the reference test vector or a training dataset
  sample), normalize it, unnormalize it, and verify the original is
  recovered within tolerance. This catches wrong stats file, wrong norm
  function, wrong mask, or wrong clip value — independently of the model
  forward pass. Clipping-induced loss is expected and documented via
  `clip_value`; the check records max round-trip error and whether it's
  within the clipping bound.

```python
@dataclass
class NormRoundTripResult:
    step_name: str                          # which transform_pipeline step
    max_abs_error: float                    # max |unnorm(norm(x)) - x|
    within_clip_bound: bool                 # True if error <= expected clipping loss
    input_source: str                       # "reference_test_vector" | "training_dataset_sample"
    status: str                             # "pass" | "fail"
```

  Static check: run round-trip on each normalization step, verify pass.

- **`checkpoint_lineage`** in `Provenance`:

```python
parent_checkpoint: Optional[str]            # passport hash of parent if fine-tuned
parent_description: Optional[str]           # e.g. "pretrained DiT base, 50K steps on coffee_capsules"
```

- **`known_issues`** — top-level section:

```python
@dataclass
class KnownIssue:
    id: str                                 # short identifier
    severity: str                           # "critical" | "warning" | "info"
    description: str                        # what goes wrong
    workaround: Optional[str]               # how to avoid it
    check_type: str                         # "static" or "dynamic"
```

  Examples for this checkpoint:

  - `transformers_drift`: transformers>=5.5.0 breaks CLIP key layout
    (static: version check)
  - `ros_pythonpath_leak`: /opt/ros PYTHONPATH breaks py3.12 imports
    (static: env check)
  - `ramen_stats_format`: checkpoint uses ramen_stats.json not
    dataset_stats.json (static: file presence check)

- **`training_augmentations`** — `ImageSpec.augmentations_in_training`
  already exists in the schema but is empty in this passport. Should be
  populated (e.g. `["random_crop", "color_jitter"]` or `[]` if none).

### 4. CLASSIFY — static vs dynamic checks

Each checkable claim in the passport should be tagged. Rather than
adding a field to every dataclass, the cleaner approach is a **check
registry** in the SKILL.md and validator:

**Static checks (code runs, pass/fail, no judgment):**

- File hashes match signoff
- Library versions match `runtime_constraints.required_versions`
- State dict key count matches
- Config values match passport (action_dim, horizon, image shapes)
- Norm stats file sha matches passport fingerprint
- No NaN/Inf in weights
- Reference test vector reproduces expected output
- Normalization round-trip is invertible within tolerance
- Device paths / camera serials match expected mapping
- `observation_delta_indices` matches config
- Transform pipeline steps match code behavior

**Dynamic checks (agent runs, needs judgment or context):**

- Camera swap detection (visual comparison of live frame to reference)
- Soft signal triage (is state_dim 13→16 rot6d expansion intentional?)
- Environment contamination (unexpected PYTHONPATH entries)
- Output sanity on real data (are actions physically reasonable?)
- Cross-referencing passport against W&B training logs
- Deciding if a version mismatch is acceptable or blocking
- Validating that `physical_mounting` descriptions match physical setup

### 5. Schema version

- Replace `SCHEMA_VERSION` `"0.1"` with `"0.2"` outright. No backward
  compat needed — no one is using v0.1 in production yet.
- `SUPPORTED_PASSPORT_VERSIONS = {"0.2"}` (drop `"0.1"`)
- Same for signoff schema.

## Size impact

- Current passport: ~170KB
- After trim (`by_name` removed, hierarchy slimmed): ~40KB
- After additions (transform_pipeline, reference_test_vector,
  camera_identity, known_issues, runtime_constraints): ~45-50KB
- Net: **~70% smaller**, more readable, more auditable

## Files to change

- `autohpc/checkpoint-passport/checkpoint_passport/schema.py` — add new
  dataclasses, remove `ParameterEntry` / `by_name`, add `TransformStep`,
  `RuntimeConstraints`, `ReferenceTestVector`, `KnownIssue`, `DeltaSpec`,
  camera fields on `ImageSpec`.
- `autohpc/checkpoint-passport/SKILL.md` — update guidance for what to
  populate, add static vs dynamic check classification.
- Regenerate `dit_block_tower_norm_fix/MODEL_PASSPORT.json` against new
  schema.
- Re-sign after regeneration.
- Update validator kernels for new checks.
