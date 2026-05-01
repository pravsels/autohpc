# Adversarial Inference Run - Hybrid Preflight Plan (Setup)

Output of Step 12 of `2026-04-27-checkpoint-lifecycle-roadmap.md`.

This plan is part 1 of 2. It covers the framing, all reference material, the
deployment-protocol skill augmentations (Phase 1), and the trial harness setup
(Phase 2). The actual adversarial trials (Phase 3) and the synthesis writeup
(Phase 4) live in
[`2026-05-01-adversarial-inference-run-trials.md`](2026-05-01-adversarial-inference-run-trials.md).
Finish this plan first, then move to the trials plan.

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to execute this
> plan task-by-task. Use `deployment-protocol/SKILL.md` for preflight work and
> `checkpoint-passport/SKILL.md` for passport validation/signoff semantics.

**Goal:** Red-team the signed-checkpoint deployment path by injecting one fault
at a time and measuring whether the deterministic validator, the deployment
preflight procedure, or the agent's semantic inspection catches it.

**Architecture:** The run treats deployment readiness as a hybrid preflight:
static validator kernels catch deterministic artifact/schema/runtime failures;
the deployment protocol catches chain-of-custody failures in the target runner
and live or replay source; the agent records any missed fault as a concrete gap
with an owner. The output is not a new validator implementation. The output is
an evidence-backed map of what the current system guarantees, what only the
agent procedure can catch, and what must be added later.

**Tech stack:** `checkpoint-passport` schema v0.2, `validate-checkpoint`,
`sign-checkpoint`, `deployment-protocol/SKILL.md`, a signed checkpoint such as
`alpha-robotics/checkpoints/dit_block_tower_norm_fix/`, the target repo's real
inference container/runner, and Hermes or an equivalent fresh agent reached by
Slack.

---

## Status (handoff for next agent session)

- **Plan document:** `done`. This file.
- **Roadmap cross-link:** `done`. Step 12 in
  `2026-04-27-checkpoint-lifecycle-roadmap.md` points here.
- **Phase 1 (skill augmentations):** `done`. Updated
  `deployment-protocol/SKILL.md` from the patch-ready prose in this plan
  (sections A-D in "Reference: Deployment-Protocol Augmentation Prose") and
  added the explicit terminal hard-validator gate exposed by T1.1.
- **Phase 2 (trial harness):** `done`. Clean trial workspace:
  `/tmp/adv_trials/20260501T125545Z`; baseline validator report:
  `/tmp/adv_trials/20260501T125545Z/reports/baseline_validate.txt`; durable
  trial log: `/tmp/adv_trials/20260501T125545Z/TRIAL_LOG.md`.
- **Phase 3 (adversarial trials):** tracked in
  [`2026-05-01-adversarial-inference-run-trials.md`](2026-05-01-adversarial-inference-run-trials.md).
- **Phase 4 (synthesis):** tracked in
  [`2026-05-01-adversarial-inference-run-trials.md`](2026-05-01-adversarial-inference-run-trials.md).

Setup is complete. Resume in the trials plan at Phase 3 unless the user asks to
rebuild the trial harness.

## Roadmap Linkage

This is the execution plan for Step 12, "Adversarial Inference Run", in
`2026-04-27-checkpoint-lifecycle-roadmap.md`.

The roadmap defined eight faults. This plan covers all eight and expands them
into thirty focused trials across four tiers (T1: 6, T2: 10, T3: 5, T4: 9).

| Roadmap fault | Trial(s) in this plan |
|---|---|
| Wrong library version | T1.5 |
| Tampered `SIGNOFF.json` | T1.1 |
| Missing a file in passport manifest | T1.3 |
| Truncated or corrupted weight file | T1.2 |
| Stale or mismatched normalization stats | T1.4, T4.2 |
| Wrong action scaling or delta-vs-absolute mismatch | T2.4, T2.6 |
| Swapped or missing camera input | T3.1, T3.3, T3.5 |
| Incorrect dtype, float16 vs bfloat16 | T2.10 |

The roadmap's exit criteria map to the outputs here:

- "Every fault caught or logged as gap" maps to the trial ledger in Phase 3.
- "Passport updated" maps to the validator/schema backlog in Phase 4.
- "Inference protocol updated" maps to Phase 1 and the procedure backlog in
  Phase 4.

## Scope

In scope:

- Write and apply deployment-protocol augmentations that make the hybrid
  validator/procedure split explicit.
- Build a trial harness around a clean signed checkpoint copy.
- Run one-fault-at-a-time adversarial trials against a fresh agent.
- Record exactly where each fault is caught: validator, chain element, or not
  caught.
- Produce prioritized follow-up backlogs.

Out of scope:

- Implement new validator kernels.
- Implement a `VALIDATION_REPORT.md` writer.
- Change checkpoint schema fields.
- Modify model code except temporary fault injection in isolated trial copies.
- Actually send actions to hardware during preflight.

## Success Criteria

The run is successful when every trial has one of these outcomes:

- `caught_static`: `validate-checkpoint` or signoff verification rejects it.
- `caught_preflight`: the deployment protocol catches it at a named chain
  element.
- `caught_agent`: the fresh agent catches it through semantic reasoning that
  is not yet encoded in the protocol.
- `missed_gap`: the agent would have proceeded, and the gap is recorded with
  the exact validator kernel, schema field, or procedure change needed.
- `not_run`: the trial could not be run, with a concrete blocker and next
  action.

Do not mark a trial successful just because the fault looks obvious to the
executor. The measurement is whether a fresh agent following the standard
materials catches it without a hint.

## Core Idea: Hybrid Preflight

The checkpoint passport is a contract, not a replacement for deployment
preflight. The validator can deterministically check files, schema fields,
hashes, version constraints, typed smoke results, and artifact consistency.
It cannot reliably decide whether a front camera is visually swapped with a
wrist camera, whether a physical mount description matches the rig, whether a
runner's local adapter is semantically equivalent, or whether an action payload
would be unsafe in a specific live context.

This plan therefore separates checks into four buckets:

- `code-check`: deterministic and should be owned by `validate-checkpoint`.
- `agent-check`: semantic, contextual, or live-rig dependent.
- `hybrid`: code can narrow the search space, but agent procedure must finish
  the comparison.
- `drop`: not meaningful for deployment preflight or not worth checking.

The adversarial run is a calibration exercise for that split. A missed static
fault becomes validator-kernel backlog. A missed semantic fault becomes
deployment-protocol backlog. A gap caused by missing contract data becomes
schema or passport-generation backlog.

## Reference: Chain Elements

Use the ten deployment-protocol chain elements for every trial:

1. source identity
2. raw source sample
3. source-to-passport bindings
4. checkpoint identity and integrity
5. runtime model load path and model internals
6. preprocessing and transformation steps
7. final model input contract
8. model output shape and value sanity
9. output unnormalization and post-processing
10. would-be emission behavior

Each trial records the first chain element at which the fresh agent stopped or
should have stopped. Static validator failures usually map to element 4 or 5.
Input binding and camera failures usually map to elements 1-3. Transform and
normalization failures usually map to elements 6-9.

## Reference: Schema-Driven Coverage Matrix

This matrix is the canonical field-by-field intent for passport v0.2 coverage.
It does not assert what `validate-checkpoint` already implements today. It says
what the adversarial run expects a mature system to guarantee.

Legend:

- `code-check`: deterministic validator or generated report.
- `agent-check`: deployment protocol / semantic inspection.
- `hybrid`: validator checks structure or metadata; agent checks live meaning.
- `drop`: documentation only for this run.

Important: classifications describe **mature target intent**, not what
`validate-checkpoint` implements today. Several `code-check` rows below have
no kernel yet (e.g. `runtime_constraints.required_python`,
`runtime_constraints.known_incompatible`, `parameters.summary.dtype_breakdown`,
`weight_integrity.manifest_hash`, `transform_pipeline[].*`,
`reference_test_vector.*`, `norm_round_trip_results[]`, output_spec
inference parameters, smoke result numerical fields). For each, the
adversarial trial is what proves the gap; closing the gap goes to the P1
backlog. Executors should never assume a `code-check` row already runs;
confirm via the Phase 2.3 baseline output and the per-element validator
support notes in section B.

| Schema path | Classification | Expected check or rationale |
|---|---|---|
| `schema_version` | code-check | Must equal supported passport schema, currently `0.2`. |
| `generated_by.tool` | drop | Useful provenance; not a deployment gate unless unknown tooling is later banned. |
| `generated_by.version` | drop | Useful for debugging generator drift. |
| `generated_by.agent` | drop | Audit provenance only. |
| `generated_at` | code-check | Parse as ISO timestamp; stale age may become soft signal later. |
| `stack` | hybrid | Validator checks presence/known family; agent confirms runner stack matches. |
| `input_contract.images[].key` | hybrid | Validator checks uniqueness; agent confirms source binding uses it. |
| `input_contract.images[].aliases` | hybrid | Validator checks type; agent confirms any remap is explicit. |
| `input_contract.images[].raw_shape` | hybrid | Validator checks shape format; agent compares raw sample shape. |
| `input_contract.images[].encoder_resize` | hybrid | Validator compares config; agent traces runner resize path. |
| `input_contract.images[].crop` | hybrid | Validator checks declared params; agent confirms runner uses same crop. |
| `input_contract.images[].color_order` | hybrid | Validator checks known values; agent checks raw sample and transform path. |
| `input_contract.images[].channel_layout` | hybrid | Validator checks known values; agent checks final tensor layout. |
| `input_contract.images[].dtype` | hybrid | Validator checks declared dtype; agent checks raw and final observed dtype. |
| `input_contract.images[].value_range` | hybrid | Validator checks valid numeric range; agent samples live/replay values. |
| `input_contract.images[].normalization` | hybrid | Validator checks constants/fingerprint; agent traces runner normalization. |
| `input_contract.images[].augmentations_in_training` | drop | Training context; not a deployment gate unless later tied to robustness. |
| `input_contract.images[].physical_mounting` | agent-check | Human/agent visual and rig-context comparison. |
| `input_contract.images[].camera_serial` | hybrid | Validator can compare device metadata if exposed; agent confirms binding. |
| `input_contract.images[].camera_usb_path` | hybrid | Validator can compare device path; agent handles replay/no-device cases. |
| `input_contract.images[].reference_frame_hash` | hybrid | Validator can verify stored reference; agent visually compares live view. |
| `input_contract.images[].reference_frame_path` | hybrid | Validator checks path/hash; agent opens sample if available. |
| `input_contract.state.total_dim` | code-check | Compare config, final model input, norm stats dimensions. |
| `input_contract.state.sub_keys[]` | hybrid | Validator checks dims sum; agent confirms live source units/frames. |
| `input_contract.state.normalization` | hybrid | Validator checks stats fingerprint; agent traces normalize call. |
| `input_contract.actions.total_dim` | code-check | Compare config, output spec, final model output. |
| `input_contract.actions.horizon` | code-check | Compare config and output spec. |
| `input_contract.actions.sub_keys[]` | hybrid | Validator checks dims sum; agent confirms robot/controller semantics. |
| `input_contract.actions.norm_mask` | code-check | Compare length to action dim and norm stats. |
| `input_contract.actions.delta_dims.delta_mask` | hybrid | Validator checks length/consistency; agent checks post-processing semantics. |
| `input_contract.actions.delta_dims.absolute_dims_reason` | agent-check | Reasoning note; agent confirms it matches architecture intent. |
| `input_contract.actions.normalization` | hybrid | Validator checks stats fingerprint; agent traces output unnormalize. |
| `input_contract.language.tokenizer_class` | code-check | Compare resolved tokenizer/processor class. |
| `input_contract.language.tokenizer_version` | code-check | Compare installed package or pinned tokenizer asset. |
| `input_contract.language.max_sequence_length` | code-check | Compare config and tokenizer behavior. |
| `input_contract.language.default_prompt` | agent-check | Agent confirms prompt used in dry run is intended. |
| `input_contract.language.training_prompts` | drop | Context unless task prompt mismatch is suspected. |
| `input_contract.temporal.n_obs_steps` | hybrid | Validator checks config; agent verifies runner history assembly. |
| `input_contract.temporal.observation_delta_indices` | hybrid | Validator checks config; agent verifies sample spacing. |
| `input_contract.temporal.delta_timestamps` | hybrid | Validator checks shape; agent verifies live/replay timing. |
| `input_contract.temporal.control_rate_hz` | hybrid | Validator checks declared rate; agent checks runner/emission rate. |
| `input_contract.training_datasets[].repo` | code-check | Format and optional resolvability. |
| `input_contract.training_datasets[].commit` | code-check | Must be pinned where possible; soft signal if missing. |
| `input_contract.training_datasets[].version` | drop | Context for humans. |
| `input_contract.training_datasets[].num_episodes` | drop | Context unless cross-checking dataset snapshot. |
| `input_contract.training_datasets[].total_frames` | drop | Context unless cross-checking dataset snapshot. |
| `input_contract.training_datasets[].episode_filter` | agent-check | Agent checks eval/replay source did not ignore intended subset. |
| `input_contract.training_datasets[].sampling_weight` | drop | Training context. |
| `input_contract.training_datasets[].key_rename_map` | hybrid | Validator can compare against config; agent checks live/replay remap. |
| `input_contract.training_datasets[].delta_timestamps_at_training` | hybrid | Validator checks shape; agent compares to runner history. |
| `input_contract.training_datasets[].contributes_to_norm_stats` | hybrid | Validator checks stats provenance if present; agent notes mixed datasets. |
| `model_identity.class_name` | code-check | Runtime loaded class must match. |
| `model_identity.class_module` | code-check | Import path must resolve in target runtime. |
| `model_identity.config_architectures[]` | code-check | Compare config architecture declarations. |
| `model_identity.resolved_via` | drop | Debug context. |
| `model_identity.resolved_class_name` | code-check | Loaded class must match expected resolution. |
| `model_identity.library_versions` | drop | Historical record; not by itself a gate. |
| `model_identity.runtime_constraints.required_versions` | code-check | Installed versions must satisfy constraints. |
| `model_identity.runtime_constraints.required_python` | code-check | Runtime Python must satisfy constraint. |
| `model_identity.runtime_constraints.known_incompatible` | code-check | Installed runtime must not match known bad ranges. |
| `model_identity.python_version` | drop | Historical generation context. |
| `model_identity.cuda_version` | drop | Historical generation context unless runtime constraint added. |
| `model_internals.module_hierarchy` | code-check | Loaded structure should match enough to catch wrong class/checkpoint. |
| `model_internals.parameters.summary.total_params` | code-check | Compare loaded model parameter count. |
| `model_internals.parameters.summary.trainable_params` | drop | Training context. |
| `model_internals.parameters.summary.frozen_params` | drop | Training context. |
| `model_internals.parameters.summary.total_bytes` | code-check | Compare loaded state size within tolerance. |
| `model_internals.parameters.summary.dtype_breakdown` | code-check | Catch float16/bfloat16 drift. |
| `model_internals.buffers[]` | code-check | Compare buffer names/shapes/dtypes. |
| `model_internals.state_dict.expected_keys_count` | code-check | Compare loaded state dict. |
| `model_internals.state_dict.found_keys_count` | code-check | Compare checkpoint state dict. |
| `model_internals.state_dict.missing_keys[]` | code-check | Must remain empty unless explicitly accepted. |
| `model_internals.state_dict.unexpected_keys[]` | code-check | Must remain empty unless explicitly accepted. |
| `model_internals.pretrained_provenance[].submodule` | code-check | Named submodule should exist. |
| `model_internals.pretrained_provenance[].source` | code-check | Known source type. |
| `model_internals.pretrained_provenance[].timm_string` | code-check | Resolve or compare if timm asset used. |
| `model_internals.pretrained_provenance[].hf_revision` | code-check | Must be pinned for HF assets. |
| `model_internals.pretrained_provenance[].frozen_in_training` | drop | Training context. |
| `model_internals.pretrained_provenance[].lr_multiplier` | drop | Training context. |
| `model_internals.quantization.scheme` | code-check | Runtime dtype/quantization should match. |
| `model_internals.quantization.per_tensor_scales` | code-check | If quantized, scales must exist and match. |
| `model_internals.forward_graph.forward_signature` | code-check | Compare public inference signature where feasible. |
| `model_internals.forward_graph.expected_input_keys` | hybrid | Validator checks keys; agent captures actual final model input. |
| `model_internals.forward_graph.sample_input_shapes` | hybrid | Validator checks shapes; agent captures actual final model input. |
| `model_internals.forward_graph.sample_output_shapes` | code-check | Compare dry-run output shapes. |
| `model_internals.forward_graph.flops_estimate` | drop | Deployment planning only. |
| `model_internals.forward_graph.peak_memory_inference_b1_bytes` | drop | Deployment planning only. |
| `model_internals.numerical_health.determinism` | code-check | Bucket must pass or be explained. |
| `model_internals.numerical_health.no_nan_inf` | code-check | Bucket must pass. |
| `model_internals.numerical_health.dropout_in_eval` | code-check | Bucket should pass for deterministic eval. |
| `model_internals.numerical_health.bn_running_stats_present` | code-check | Bucket should pass or be not applicable. |
| `output_spec.actions.layout` | hybrid | Validator checks consistency; agent confirms controller semantics. |
| `output_spec.actions.sub_keys` | hybrid | Validator checks dims; agent confirms would-be payload semantics. |
| `output_spec.actions.horizon` | code-check | Compare input action horizon and dry-run output. |
| `output_spec.actions.control_rate_hz` | hybrid | Validator checks numeric; agent checks runner emission cadence. |
| `output_spec.actions.action_latency_budget_ms` | agent-check | Agent compares dry-run timing and deployment tolerance. |
| `output_spec.auxiliary_outputs.reward_head` | drop | Not part of action preflight unless used by runner. |
| `output_spec.auxiliary_outputs.value_head` | drop | Not part of action preflight unless used by runner. |
| `output_spec.auxiliary_outputs.latents_exposed` | drop | Debug capability only. |
| `output_spec.auxiliary_outputs.attention_maps_exposed` | drop | Debug capability only. |
| `output_spec.inference_parameters.type` | code-check | Compare runner inference mode. |
| `output_spec.inference_parameters.num_inference_steps` | code-check | Compare runner sampling config. |
| `output_spec.inference_parameters.scheduler` | code-check | Compare runner scheduler. |
| `output_spec.inference_parameters.prediction_type` | code-check | Compare runner/model config. |
| `output_spec.inference_parameters.clip_sample` | code-check | Compare runner/model config. |
| `output_spec.inference_parameters.clip_sample_range` | code-check | Compare runner/model config and range check. |
| `output_spec.inference_parameters.chunk_aggregation` | hybrid | Validator checks config; agent confirms runner behavior. |
| `output_spec.inference_parameters.chunks_executed_per_inference` | hybrid | Validator checks config; agent confirms emission behavior. |
| `output_spec.inference_parameters.extra` | hybrid | Executor classifies each architecture-specific key. |
| `output_spec.post_processing.unnormalize` | hybrid | Validator checks declared behavior; agent traces output transform. |
| `output_spec.post_processing.delta_to_absolute` | hybrid | Validator checks shape/mask; agent confirms payload semantics. |
| `output_spec.post_processing.action_smoothing` | hybrid | Validator checks params; agent confirms runner applies it. |
| `output_spec.post_processing.action_clamping` | hybrid | Validator checks params; agent confirms runner applies it. |
| `output_spec.smoke_results.calibration_batch_source` | drop | Context only. |
| `output_spec.smoke_results.calibration_batch_size` | code-check | Type/range sanity. |
| `output_spec.smoke_results.determinism` | code-check | Must pass or block signing. |
| `output_spec.smoke_results.nan_inf` | code-check | Must pass. |
| `output_spec.smoke_results.liveness` | code-check | Must pass or soft signal. |
| `output_spec.smoke_results.distribution` | code-check | Must pass or soft signal with rationale. |
| `output_spec.smoke_results.range_check` | code-check | Must pass expected range. |
| `weight_integrity.weight_files[]` | code-check | Every listed file must exist and hash/size match. |
| `weight_integrity.manifest_hash` | code-check | Recompute rollup if present. |
| `provenance.run_log_path` | hybrid | Validator checks format; agent confirms relevant run evidence. |
| `provenance.training_repo` | code-check | Format and optional reachability. |
| `provenance.training_repo_commit` | code-check | Must be full valid SHA where possible. |
| `provenance.config_snapshot_path` | code-check | Path must exist or be explainable. |
| `provenance.merged_config_sha256` | code-check | Recompute if config snapshot is present. |
| `provenance.parent_checkpoint` | hybrid | Validator checks hash format; agent confirms lineage meaning. |
| `provenance.parent_description` | agent-check | Human-readable lineage note. |
| `transform_pipeline[].order` | code-check | Orders must be unique and contiguous. |
| `transform_pipeline[].name` | code-check | Required stable identifier. |
| `transform_pipeline[].applies_to` | hybrid | Validator checks schema path; agent maps it to source/runner stage. |
| `transform_pipeline[].operation` | hybrid | Validator checks known op; agent traces actual code path. |
| `transform_pipeline[].direction` | code-check | Must be `input` or `output`. |
| `transform_pipeline[].parameters` | hybrid | Validator checks known params; agent confirms runner uses them. |
| `transform_pipeline[].check_type` | code-check | Must be `static` or `dynamic`. |
| `transform_pipeline[].check_description` | agent-check | Agent follows it where dynamic. |
| `reference_test_vector.input_state` | code-check | Used for golden forward check if present. |
| `reference_test_vector.input_prompt` | code-check | Used for golden forward check if present. |
| `reference_test_vector.input_images_hash` | code-check | Verify reference images if present. |
| `reference_test_vector.input_images_path` | code-check | Path must resolve if golden check is required. |
| `reference_test_vector.expected_output` | code-check | Compare output within tolerance. |
| `reference_test_vector.tolerance` | code-check | Must be finite and justified. |
| `reference_test_vector.torch_seed` | code-check | Use for deterministic replay. |
| `reference_test_vector.notes` | drop | Context only. |
| `norm_round_trip_results[]` | code-check | Every normalization transform should have a passing result. |
| `known_issues[].id` | code-check | Stable unique identifier. |
| `known_issues[].severity` | hybrid | Validator gates critical known issues; agent interprets warnings. |
| `known_issues[].description` | agent-check | Agent uses it to inspect the environment. |
| `known_issues[].workaround` | agent-check | Agent applies or records workaround. |
| `known_issues[].check_type` | code-check | Must be `static` or `dynamic`. |
| `extra_sections` | drop | Forward compatibility; executor must classify if relied on. |
| `SIGNOFF.schema_version` | code-check | Must equal supported signoff schema. |
| `SIGNOFF.signed_at` | code-check | Parse as ISO timestamp. |
| `SIGNOFF.signed_by.tool` | drop | Provenance. |
| `SIGNOFF.signed_by.version` | drop | Provenance. |
| `SIGNOFF.artifacts[]` | code-check | Every artifact hash must match bytes on disk. |
| `SIGNOFF.verdict` | code-check | Must be `pass` or accepted `soft_signal`; never accept `fail`. |
| `SIGNOFF.verdict_reason` | hybrid | Required for soft signal; agent checks it explains the risk. |

## Reference: Deployment-Protocol Augmentation Prose

Phase 1 copies these sections into `deployment-protocol/SKILL.md`. Keep the
wording close to this text unless the surrounding document requires small
transitions.

### A. Add hybrid preflight framing near the overview

Add as a new section after `## Core Pattern` (currently the section ending
around line 50 of `deployment-protocol/SKILL.md`) and before
`## Preflight Rubric`. Title the new section `## Hybrid Preflight (Validator + Agent)`:

```markdown
Preflight is hybrid. `validate-checkpoint` owns deterministic artifact and
schema checks: signoff hashes, required files, runtime version constraints,
state/action dimensions, dtype summaries, smoke-test buckets, reference test
vectors, and normalization round-trip results. The deployment protocol owns
the contextual chain checks that code cannot fully decide: which live or replay
source is bound to each passport key, whether a camera view semantically
matches the declared mount, whether a target runner's adapter actually applies
the declared transforms, and whether the final would-be command has the
expected robot semantics.

Do not blur those responsibilities. Run the validator first and record its
coverage, then run the ten-element chain audit to cover the semantic and
runtime gaps. If a mismatch could have been caught deterministically, record it
as validator backlog. If it requires live context or judgment, record it as
procedure backlog.
```

### B. Add per-element validator/agent split reminders

Add these bullets to the relevant chain elements:

```markdown
#### Element 1: Source identity

Validator support: none beyond passport fields such as camera serials or
reference-frame paths. Agent responsibility: identify the actual live device,
topic, replay dataset, log, sample ID, timestamp, and any local binding layer.

#### Element 2: Raw source sample

Validator support: expected shapes, dtypes, value ranges, and camera metadata
from the passport. Agent responsibility: inspect at least one raw sample before
preprocessing and compare it to those expectations.

#### Element 3: Source-to-passport bindings

Validator support: declared image keys, state sub-keys, rename maps, temporal
indices, and camera identity fields. Agent responsibility: prove each observed
source maps to exactly one passport input and that any remap layer is explicit.

#### Element 4: Checkpoint identity and integrity

Validator support: this is the strongest static stage. Run
`validate-checkpoint <ckpt> --require-signoff` and record hard failures, soft
signals, and not-checked rows. Agent responsibility: confirm the target runner
is pointed at the same checkpoint bundle the validator checked.

#### Element 5: Runtime model load path and model internals

Validator support: class/module resolution, required package versions, Python
constraint, dtype breakdown, state-dict completeness, and smoke-test buckets.
Agent responsibility: confirm the deployment runner did not bypass this loader,
silently cast dtype, or fall back to another class path.

#### Element 6: Preprocessing and transformation steps

Validator support: transform-pipeline declarations, norm stats fingerprints,
reference test vector, and normalization round-trip results. Agent
responsibility: trace the actual runner path from raw sample to model-facing
tensors and name every adapter or transform layer.

#### Element 7: Final model input contract

Validator support: expected keys, shapes, dtypes, temporal axes, and value
ranges. Agent responsibility: capture what the model actually received in the
target runner and compare it against the passport, not just against config.

#### Element 8: Model output shape and value sanity

Validator support: smoke-test liveness, distribution, range, determinism, and
NaN/Inf buckets. Agent responsibility: run one no-emit dry run through the
target path and record the actual output shape, dtype, range, and obvious
sanity.

#### Element 9: Output unnormalization and post-processing

Validator support: declared post-processing, action dims, delta mask, norm
mask, normalization round-trip, and expected range. Agent responsibility:
confirm the target runner applies the same unnormalize, clipping, smoothing,
and delta-to-absolute behavior before command formation.

#### Element 10: Would-be emission behavior

Validator support: action shape, horizon, control rate, and latency fields.
Agent responsibility: prove the run is no-emit, name the publish/actuation path
that would have been used, and confirm the payload semantics match the robot
controller.
```

### C. Add validator coverage map requirement

Add this section after the quick-reference table:

```markdown
## Validator Coverage Map

When `validate-checkpoint` can produce a coverage report, include it in the
preflight artifact. Until then, create a manual coverage note with three lists:

- `covered_static`: validator checks that ran and passed or failed.
- `not_checked_static`: schema paths that should be deterministic but were not
  checked by the current validator.
- `requires_agent`: schema paths or chain claims that require live context,
  semantic judgment, or target-runner inspection.

The preflight verdict must not imply that `not_checked_static` paths were
validated. If an adversarial trial exposes a missed deterministic fault, add a
validator-kernel backlog item naming the exact schema path and failure mode.
```

### D. Add audit artifact tweaks

Add these fields to the `Run Header / Provenance` list:

```markdown
- validator command and exit code
- validator report path, if generated
- validator hard failures, soft signals, and not-checked rows
- explicit statement of whether each trial fault was caught by validator,
  deployment protocol, agent judgment, or missed
```

Add this to `Remaining Gaps / Next Actions`:

```markdown
Separate gaps into:

- validator gaps: deterministic checks that should become code
- schema/passport gaps: missing contract fields needed for a future check
- procedure gaps: agent/deployment instructions that need to become clearer
```

## Reference: Validator Coverage-Output Spec

This is not implemented in this plan. It is the target design for later P2
work so trial findings can be expressed mechanically.

`validate-checkpoint <ckpt> --coverage-report VALIDATION_REPORT.md` should
write a markdown report with these sections:

```markdown
# Validation Report

## Summary

- checkpoint:
- passport:
- signoff:
- command:
- exit_code:
- verdict: `pass` | `soft_signal` | `fail`

## Hard Failures

| Check | Schema path | Reason | Evidence |
|---|---|---|---|

## Soft Signals

| Check | Schema path | Reason | Operator action |
|---|---|---|---|

## Covered Static Checks

| Schema path | Check name | Result | Evidence |
|---|---|---|---|

## Not Checked Static Claims

| Schema path | Intended check | Missing implementation or missing input |
|---|---|---|

## Requires Agent / Dynamic Context

| Schema path or chain element | Why code cannot decide alone | Procedure reference |
|---|---|---|
```

Minimal JSON shape for the same data:

```json
{
  "checkpoint": "<ckpt_dir>",
  "command": "validate-checkpoint <ckpt_dir> --require-signoff",
  "exit_code": 0,
  "verdict": "pass",
  "hard_failures": [],
  "soft_signals": [],
  "covered_static": [
    {
      "schema_path": "signoff.artifacts[].sha256",
      "check_name": "weight_files_match_signoff",
      "result": "pass",
      "evidence": "all listed weight files matched recorded sha256"
    },
    {
      "schema_path": "MODEL_PASSPORT.json",
      "check_name": "passport_checksum_valid",
      "result": "pass",
      "evidence": "passport sha256 matches signoff"
    }
  ],
  "not_checked_static": [
    {
      "schema_path": "model_internals.parameters.summary.dtype_breakdown",
      "intended_check": "compare loaded dtype summary to passport",
      "reason": "kernel not implemented"
    }
  ],
  "requires_agent": [
    {
      "schema_path": "input_contract.images[].physical_mounting",
      "reason": "requires visual/rig-context judgment",
      "procedure": "deployment-protocol element 1-3"
    }
  ]
}
```

Rules:

- The report must distinguish "field absent" from "kernel absent".
- `not_checked_static` is not a failure by itself, but it weakens any preflight
  claim and feeds P2 backlog.
- `requires_agent` is not technical debt by itself. It becomes procedure debt
  only if the deployment protocol does not tell the agent how to inspect it.
- Reports should be written next to the trial log under `/tmp/adv_trials/` or
  committed only if the target repo convention stores preflight artifacts.

## Phase 1: Apply Skill Augmentations

Purpose: make the deployment protocol explicitly hybrid before adversarial
trials begin. This prevents the executor from treating checkpoint validation
as complete preflight.

### Phase 1.1: Inspect current deployment protocol

Files:

- Read: `deployment-protocol/SKILL.md`
- Modify: `deployment-protocol/SKILL.md`

Steps:

1. Read the overview, ten chain elements, quick reference, and audit artifact
   sections.
2. Identify the nearest insertion point for sections A-D above.
3. Confirm no existing language already covers the same split.

Expected result: the executor knows exactly where to place each augmentation.

### Phase 1.2: Add hybrid framing

Steps:

1. Insert section A near the overview after the current core-principle text.
2. Keep the tone consistent with the existing skill.
3. Do not mention this plan in the skill; the skill should stand alone.

Verification:

```bash
rg -n "Preflight is hybrid|not_checked_static|validator gaps" deployment-protocol/SKILL.md
```

Expected: all three phrases appear.

### Phase 1.3: Add per-element split reminders

Steps:

1. Add the section B reminders under the matching element descriptions.
2. Keep each reminder short enough that the original procedure remains easy to
   follow.
3. Preserve the ten-element order.

Verification:

```bash
rg -n "Validator support|Agent responsibility" deployment-protocol/SKILL.md
```

Expected: at least one validator/agent reminder appears for each chain element.

### Phase 1.4: Add coverage map and audit tweaks

Steps:

1. Add section C after the quick reference.
2. Add section D fields to the audit artifact instructions.
3. Add the three-way gap split to remaining gaps.

Verification:

```bash
rg -n "Validator Coverage Map|covered_static|schema/passport gaps" deployment-protocol/SKILL.md
```

Expected: all terms appear in coherent context.

### Phase 1.5: Review and commit

Steps:

1. Read the edited skill top-to-bottom.
2. Confirm it still says checkpoint-only validation is not deployment
   preflight.
3. Commit if the user asked for commits in this execution session.

Suggested commit message:

```bash
git commit -m "clarify hybrid deployment preflight"
```

If not committing, leave the changes staged or unstaged according to the user's
workflow preference.

## Phase 2: Set Up Trial Harness

Purpose: isolate every fault injection so the clean checkpoint can be restored
without ambiguity.

### Phase 2.1: Choose checkpoint and target runner

Default checkpoint:

```bash
CKPT_SRC="alpha-robotics/checkpoints/dit_block_tower_norm_fix"
TARGET_REPO="alpha-robotics"
```

If this checkpoint is unavailable, choose another signed checkpoint with:

- `MODEL_PASSPORT.json`
- `SIGNOFF.json`
- all files listed in `weight_integrity.weight_files[]`
- a known target runner or replay preflight path

Record the selection at the top of the trial log.

### Phase 2.2: Create isolated workspace

Commands:

```bash
TRIAL_ROOT="/tmp/adv_trials/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$TRIAL_ROOT"/{clean,trials,reports,logs}
rsync -a "$CKPT_SRC"/ "$TRIAL_ROOT/clean/ckpt/"
```

State note for `dit_block_tower_norm_fix` specifically: previous testing
deleted `MODEL_PASSPORT.json` and `SIGNOFF.json` from the source tree; the
backup lives at `$CKPT_SRC/.passport_backup/`. Restore them in the clean copy
before the baseline validator runs:

```bash
if [ -d "$CKPT_SRC/.passport_backup" ]; then
  cp "$CKPT_SRC/.passport_backup/MODEL_PASSPORT.json" "$TRIAL_ROOT/clean/ckpt/"
  cp "$CKPT_SRC/.passport_backup/SIGNOFF.json"        "$TRIAL_ROOT/clean/ckpt/"
fi
```

If a different signed checkpoint is used, this restoration step is unnecessary.

Rules:

- Never inject faults into the source checkpoint.
- Every trial gets its own copy under `$TRIAL_ROOT/trials/<trial_id>/ckpt`.
- Keep raw validator output under `$TRIAL_ROOT/reports/<trial_id>/`.
- Keep Hermes/fresh-agent transcript excerpts under
  `$TRIAL_ROOT/logs/<trial_id>.md`.

### Phase 2.3: Run baseline validator

Command:

```bash
validate-checkpoint "$TRIAL_ROOT/clean/ckpt" --require-signoff --show-not-checked \
  2>&1 | tee "$TRIAL_ROOT/reports/baseline_validate.txt"
```

Expected:

- Exit code `0`, or only known soft signals already accepted in signoff.
- No unexplained hard failure.

If baseline fails, stop. Do not run adversarial trials against an already
broken checkpoint.

### Phase 2.4: Create trial log skeleton

Create `$TRIAL_ROOT/TRIAL_LOG.md`:

```markdown
# Adversarial Inference Run Trial Log

## Run Header

- started_at:
- executor:
- target_repo:
- target_repo_commit:
- checkpoint_source:
- clean_checkpoint_copy:
- validator_version:
- deployment_protocol_revision:
- fresh_agent_channel:

## Baseline

- command:
- exit_code:
- hard_failures:
- soft_signals:
- not_checked_static:
- notes:

## Trial Ledger

| Trial | Fault | Expected catch | Actual outcome | First chain element | Gap owner |
|---|---|---|---|---|---|
```

Append the per-trial template after each trial.

### Phase 2.5: Fresh-agent prompt wrapper

Use this wrapper for Hermes or the equivalent fresh agent. Replace the bracketed
values per trial, but do not reveal the injected fault.

```text
We need a deployment preflight for a signed checkpoint.

Target repo: <target_repo>
Checkpoint path: <trial_ckpt_dir>
Mode: replay preflight unless live rig is explicitly available.
Replay/source data: <source_path_or_description>

Please follow the standard AutoHPC deployment protocol from scratch. Read the
checkpoint passport and signoff, run the validator gate, inspect the target
repo's real inference path, and write the ten-element chain audit. Stop if any
hard gate fails. Do not send actions to hardware; use no-emit/dry-run behavior
only.

Report:
- final verdict: PASS (live), PASS (replay, provisional), FAIL, or NOT RUN
- first failing chain element if any
- validator command and result
- what evidence you inspected
- any gaps you could not check
```

## Phases 3 and 4: Adversarial Trials and Synthesis

These phases move to a dedicated execution plan to keep this file focused on
setup. Once Phase 2 above is complete (clean checkpoint copied, baseline
validator green, trial log skeleton in place, fresh-agent prompt wrapper
ready), continue from
[`2026-05-01-adversarial-inference-run-trials.md`](2026-05-01-adversarial-inference-run-trials.md).

That plan contains:

- the per-trial setup, validator, and recording template,
- 30 trials across 4 tiers (artifact contract, code-gap, agent-only,
  hybrid boundary),
- the synthesis steps that turn trial outcomes into a final verdict and
  three ranked follow-up backlogs.

Track Phase 3 and Phase 4 status from inside the trials plan, not here.

## Backlog Priorities

### P1: Validator Kernels Needed Before Strong Static Claims

- Signoff artifact hash coverage must include every inference-critical file:
  config, weights, norm stats, tokenizer/processor assets, and any declared
  external aux file.
- Runtime constraints must be checked against the environment that loads the
  model, not just the environment that generated the passport.
- Dtype breakdown and runtime load dtype must be compared to catch float16 vs
  bfloat16 drift.
- Transform pipeline checks should cover ordered steps, direction, known
  operations, and selected params for resize, normalization, temporal stacking,
  inference steps, and post-processing.
- Reference test vectors should run where present and fail on mismatch.
- Normalization round-trip results should be required for every declared
  normalization step or explicitly listed as not checked.

### P2: Coverage Reporting

- `validate-checkpoint` should produce `covered_static`,
  `not_checked_static`, and `requires_agent` sections.
- Coverage report should preserve schema paths so backlogs can be precise.
- Soft signals should carry operator-action text and signoff reason links.
- The report should make it impossible to imply that not-checked static claims
  passed.

### P3: Schema or Passport Generation Improvements

- Populate camera serials, USB paths, reference frames, and mounting notes when
  real deployment cameras are known.
- Populate `observation_delta_indices` and temporal source expectations for
  every temporal policy.
- Populate `delta_mask` with one boolean per action dim and a reason for every
  absolute segment.
- Populate runtime constraints only when they are true requirements, not merely
  historical observations.
- Include all inference-critical external paths in either signed artifacts or a
  clearly marked dynamic dependency list.

### P4: Deployment Procedure Improvements

- Require target repo dirty status and relevant local diff paths in every
  preflight artifact.
- Require raw sample value summaries before preprocessing.
- Require final model input capture after preprocessing.
- Require explicit no-emit proof before any dry-run forward pass.
- Require external aux artifact path comparison when runner config overrides
  checkpoint-local paths.

### P5: Later Convenience

- Add a helper to scaffold adversarial trial directories and trial-log
  templates.
- Add a helper to summarize validator output into the trial ledger.
- Add a small markdown linter or checklist for deployment audit artifacts.
- Add canned Hermes prompt templates once the first adversarial run proves the
  format.

## Handoff Prompt for Executing Phase 1

Use this prompt in a fresh session if continuing later:

```text
Read ../autohpc/README.md, then read
docs/plans/2026-05-01-adversarial-inference-run-hybrid-preflight.md.
Resume at Phase 1. Apply the deployment-protocol augmentations exactly enough
to preserve the existing skill style. Do not run adversarial trials yet. After
editing deployment-protocol/SKILL.md, verify the key phrases with rg and report
the diff.
```

