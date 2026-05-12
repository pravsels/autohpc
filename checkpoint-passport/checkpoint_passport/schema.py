"""Schema for MODEL_PASSPORT.json and SIGNOFF.json — single source of truth."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "0.2"
SIGNOFF_SCHEMA_VERSION = "0.2"

# Anything not in this set lands in ModelPassport.extra_sections (forward compat).
KNOWN_PASSPORT_SECTIONS = {
    "input_contract",
    "model_identity",
    "model_internals",
    "output_spec",
    "weight_integrity",
    "provenance",
    "transform_pipeline",
    "known_issues",
}


# ── input_contract ──────────────────────────────────────────────────────
# What the model expects to receive at inference time.
# The upstream agent populates this by reading the training config + dataset.
# The downstream validator cross-checks it against checkpoint files and
# (optionally) a local copy of the training dataset.


# Describes a single image input the model expects (there's one per camera).
# key is the dataset column name, e.g. "observation.images.front".
@dataclass
class ImageSpec:
    key: str
    aliases: List[str] = field(default_factory=list)       # equivalent dataset keys
    raw_shape: Optional[List[int]] = None                  # CHW before encoder
    encoder_resize: Optional[List[int]] = None             # HW the vision encoder sees
    crop: Optional[Dict[str, Any]] = None
    color_order: Optional[str] = None                      # "RGB" | "BGR"
    channel_layout: Optional[str] = None                   # "CHW" | "HWC"
    dtype: Optional[str] = None
    value_range: Optional[List[float]] = None              # e.g. [0.0, 1.0]
    normalization: Optional[Dict[str, Any]] = None         # type + mean/std if applicable
    augmentations_in_training: List[str] = field(default_factory=list)
    physical_mounting: Optional[str] = None                # free text, e.g. "wrist cam"
    camera_serial: Optional[str] = None                    # hardware serial number
    camera_usb_path: Optional[str] = None                  # USB bus/port topology
    reference_frame_hash: Optional[str] = None             # sha256 of a reference image
    reference_frame_path: Optional[str] = None             # path to the reference image


# Aggregate state vector (joint positions, gripper, etc.).
# sub_keys breaks out per-component dims so kernel checks can cross-check
# against norm stats and config independently.
@dataclass
class StateSpec:
    """Aggregate state input -- total dim plus per-sub-key breakdown.

    Convention: `total_dim` and `sub_keys` describe the **model-facing**
    state layout, i.e. what the forward pass actually sees after any
    rotation / normalization conversions performed by the dataset adapter.
    For the dataset-side raw layout, see `config.json`'s
    `dataset_schema.state`. They differ for any policy that expands a
    compact rotation source (e.g. axis-angle, 3 dims) into a learned-rotation
    representation (e.g. rot6d, 6 dims) -- the model sees more dims than the
    dataset emits. Picking model-facing here keeps `sub_keys` consistent
    with `input_features.observation.state.shape` in `config.json`.
    """
    total_dim: Optional[int] = None
    sub_keys: List[Dict[str, Any]] = field(default_factory=list)  # [{name, dim, ...}]
    normalization: Optional[Dict[str, Any]] = None                # type + mean/std if applicable


# Structured representation of which action dims are deltas vs absolute.
@dataclass
class DeltaSpec:
    delta_mask: List[bool] = field(default_factory=list)   # per-dim: True = delta, False = absolute
    absolute_dims_reason: Optional[str] = None             # e.g. "6D rotation (rot6d) passed through unchanged"


# Action representation the model was trained on.
# delta_dims + normalization together let kernel checks catch the
# delta-vs-absolute bug (norm stats spanning +/-2.5 when mode is delta).
@dataclass
class ActionSpec:
    """Action output -- horizon, dims, sub-keys, norm mask, normalization."""
    total_dim: Optional[int] = None
    horizon: Optional[int] = None                                 # action chunk length
    sub_keys: List[Dict[str, Any]] = field(default_factory=list)  # [{name, dim, ...}]
    norm_mask: Optional[List[bool]] = None                        # per-dim: True = normalized
    delta_dims: Optional[DeltaSpec] = None                        # which dims are delta vs absolute
    normalization: Optional[Dict[str, Any]] = None                # type + mean/std if applicable


# Language conditioning (for language-conditioned policies like OpenPI/RT-2).
# Tokenizer class is checked against the loaded processor by model_internals.
@dataclass
class LanguageSpec:
    tokenizer_class: Optional[str] = None
    tokenizer_version: Optional[str] = None
    max_sequence_length: Optional[int] = None
    default_prompt: Optional[str] = None               # used when no task instruction given
    training_prompts: Optional[Dict[str, Any]] = None  # task → prompt mapping seen in training


# Temporal contract: how many observation frames the model needs and at what rate.
# delta_timestamps is the LeRobot-style per-key timestamp offsets dict.
@dataclass
class TemporalSpec:
    n_obs_steps: Optional[int] = None             # observation history length
    observation_delta_indices: Optional[List[int]] = None  # e.g. [-1, 0] for prev + current frame
    delta_timestamps: Optional[Any] = None        # {key: [offsets]} or flat list
    control_rate_hz: Optional[float] = None       # expected inference frequency


# One dataset that contributed to the model's training inputs.
# Lives under input_contract (not provenance) because the dataset defines the
# input distribution — key names, rename maps, norm stats contributions.
# input_expectation checks use repo+commit to resolve and cross-check on disk.
@dataclass
class TrainingDatasetSpec:
    """One dataset that contributed to the model's training inputs."""
    repo: Optional[str] = None                    # HuggingFace repo id
    commit: Optional[str] = None                  # pinned commit hash
    version: Optional[str] = None                 # e.g. "v2.1", "v3.0"
    loader_class: Optional[str] = None            # e.g. "lerobot.datasets.LeRobotDataset"
    num_episodes: Optional[int] = None
    total_frames: Optional[int] = None
    episode_filter: Optional[str] = None          # expression if subset was used
    sampling_weight: Optional[float] = None       # for multi-dataset mixing
    key_rename_map: Dict[str, str] = field(default_factory=dict)  # dataset key → model key
    delta_timestamps_at_training: Optional[Dict[str, Any]] = None
    contributes_to_norm_stats: Optional[bool] = None


# Top-level input contract: everything the model expects to receive.
# Each sub-spec is optional — a vision-only policy has no state; a non-language
# policy has no language block.  Kernel checks skip absent sections gracefully.
@dataclass
class InputContract:
    images: List[ImageSpec] = field(default_factory=list)
    state: Optional[StateSpec] = None
    actions: Optional[ActionSpec] = None             # action representation used in training
    language: Optional[LanguageSpec] = None
    temporal: Optional[TemporalSpec] = None
    training_datasets: List[TrainingDatasetSpec] = field(default_factory=list)


# ── model_identity ──────────────────────────────────────────────────────


# Versions required at inference time (contract), vs library_versions which
# records what was observed at training time (history).
@dataclass
class RuntimeConstraints:
    required_versions: Dict[str, str] = field(default_factory=dict)  # {"transformers": "==5.4.0"}
    required_python: Optional[str] = None                            # ">=3.12,<3.13"
    known_incompatible: List[str] = field(default_factory=list)      # ["transformers>=5.5.0 (CLIP key change)"]


# Identity of the model class the checkpoint was trained with.
# Catches silent class-swap bugs where config.json declares one architecture but
# the loader silently resolves to a different one (e.g. version mismatch in
# transformers mapping a newer arch to an older fallback).
@dataclass
class ModelIdentity:
    class_name: Optional[str] = None              # canonical class name from the training repo
    class_module: Optional[str] = None            # importable module path
    config_architectures: List[str] = field(default_factory=list)  # from config.json "architectures"
    resolved_via: Optional[str] = None            # "direct_import" | "transformers.AutoModel"
    resolved_class_name: Optional[str] = None     # what the loader actually instantiated
    library_versions: Dict[str, str] = field(default_factory=dict)  # torch/transformers/etc. (training-time)
    runtime_constraints: RuntimeConstraints = field(default_factory=RuntimeConstraints)
    python_version: Optional[str] = None
    cuda_version: Optional[str] = None


# ── model_internals ─────────────────────────────────────────────────────


# Non-parameter persistent tensor (e.g. running mean/var in batch norm).
@dataclass
class BufferEntry:
    name: str
    shape: List[int]
    dtype: str


# High-level parameter budget.  dtype_breakdown maps dtype strings to
# param counts, useful for catching mixed-precision surprises.
@dataclass
class ParametersSummary:
    total_params: Optional[int] = None
    trainable_params: Optional[int] = None
    frozen_params: Optional[int] = None
    total_bytes: Optional[int] = None
    dtype_breakdown: Dict[str, int] = field(default_factory=dict)  # {"float32": N, ...}


@dataclass
class ParametersBlock:
    summary: ParametersSummary = field(default_factory=ParametersSummary)


# Result of load_state_dict(strict=False): which keys were expected vs found.
# missing_keys and unexpected_keys are the primary signals for weight
# completeness — missing keys mean randomly-initialized layers.
@dataclass
class StateDictBlock:
    expected_keys_count: Optional[int] = None
    found_keys_count: Optional[int] = None
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)


# An external pretrained component embedded in the model (e.g. a vision
# encoder from timm or a language backbone from HuggingFace).
# source_revision pins the exact version; the kernel check flags unpinned assets.
@dataclass
class PretrainedAsset:
    submodule: str                                # dotted path within the model
    source: Optional[str] = None                  # "timm" | "huggingface" | ...
    source_identifier: Optional[str] = None       # source-specific model name (e.g. timm string, HF repo id)
    source_revision: Optional[str] = None         # pinned version (commit SHA for HF, null when source_identifier is sufficient)
    frozen_in_training: Optional[bool] = None
    lr_multiplier: Optional[float] = None


@dataclass
class QuantizationBlock:
    scheme: str = "none"                          # none | int8 | int4 | fp16 | bf16
    per_tensor_scales: Optional[Any] = None


# Records the model's forward() contract: what keys it expects, what shapes
# go in and come out, and resource estimates for deployment planning.
@dataclass
class ForwardGraph:
    forward_signature: Optional[str] = None       # string repr of the function signature
    expected_input_keys: List[str] = field(default_factory=list)
    sample_input_shapes: Dict[str, List[int]] = field(default_factory=dict)
    sample_output_shapes: Dict[str, List[int]] = field(default_factory=dict)
    flops_estimate: Optional[str] = None
    peak_memory_inference_b1_bytes: Optional[int] = None  # batch=1 inference


# Results of numerical health checks run by the upstream agent on a
# calibration batch.  Each field is a dict with at least a "passed" key.
@dataclass
class NumericalHealth:
    determinism: Dict[str, Any] = field(default_factory=dict)            # same input twice → same output
    no_nan_inf: Dict[str, Any] = field(default_factory=dict)             # no NaN/Inf in forward pass
    dropout_in_eval: Dict[str, Any] = field(default_factory=dict)        # all dropout disabled in eval mode
    bn_running_stats_present: Dict[str, Any] = field(default_factory=dict)  # batch norm has tracked stats


# Top-level: everything about the model's structure and health that can be
# determined by loading it once and running a calibration forward pass.
@dataclass
class ModelInternals:
    module_hierarchy: List[Dict[str, Any]] = field(default_factory=list)  # nested module tree
    parameters: ParametersBlock = field(default_factory=ParametersBlock)
    buffers: List[BufferEntry] = field(default_factory=list)
    state_dict: StateDictBlock = field(default_factory=StateDictBlock)
    pretrained_provenance: List[PretrainedAsset] = field(default_factory=list)
    quantization: QuantizationBlock = field(default_factory=QuantizationBlock)
    forward_graph: ForwardGraph = field(default_factory=ForwardGraph)
    numerical_health: NumericalHealth = field(default_factory=NumericalHealth)


# ── output_spec ─────────────────────────────────────────────────────────


# What the model produces at inference time.
# The kernel cross-checks horizon and sub_keys against input_contract.actions
# to make sure the output matches what was trained on.
@dataclass
class OutputActions:
    layout: Optional[str] = None                # e.g. "mirrors input_contract.actions"
    sub_keys: Optional[Any] = None              # list of sub-key dicts, or a "see ..." ref string
    horizon: Optional[int] = None
    control_rate_hz: Optional[float] = None     # expected rate at deploy time
    action_latency_budget_ms: Optional[float] = None  # max allowed inference latency


# Optional secondary outputs beyond the primary action prediction.
@dataclass
class AuxiliaryOutputs:
    reward_head: Optional[Dict[str, Any]] = None
    value_head: Optional[Dict[str, Any]] = None
    latents_exposed: bool = False               # intermediate representations accessible
    attention_maps_exposed: bool = False         # attention weights accessible


# How to run inference for this model type.  Diffusion policies need
# num_inference_steps and a scheduler; regression policies need neither.
# extra catches architecture-specific knobs without schema changes.
@dataclass
class InferenceParameters:
    type: Optional[str] = None                  # "diffusion" | "regression" | ...
    num_inference_steps: Optional[int] = None
    scheduler: Optional[str] = None
    prediction_type: Optional[str] = None
    clip_sample: Optional[bool] = None
    clip_sample_range: Optional[float] = None
    chunk_aggregation: Optional[str] = None     # how overlapping action chunks are combined
    chunks_executed_per_inference: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# Transforms applied to raw model output before sending to the robot.
@dataclass
class PostProcessing:
    unnormalize: Optional[bool] = None
    delta_to_absolute: Optional[Dict[str, Any]] = None   # conversion params if model outputs deltas
    action_smoothing: Optional[Dict[str, Any]] = None
    action_clamping: Optional[Dict[str, Any]] = None


# Per-bucket smoke-test result schemas. Each bucket has a small set of
# documented fields plus a free-form `details` dict for upstream-specific
# extras. The kernel's `_bucket_verdict()` reads the typed fields directly
# (status / boolean flags); anything in `details` is documentation only.
#
# Keeping these as proper dataclasses (rather than `Dict[str, Any]`) makes
# the contract discoverable from the schema alone, instead of requiring
# skill authors to read the kernel helper to know what to populate.
@dataclass
class DeterminismResult:
    """Two forward passes on the same input + RNG state produced the same output."""
    status: Optional[str] = None                # "pass" | "fail"
    max_abs_diff: Optional[float] = None        # max |out_1 - out_2|; expect 0 or near-0
    method: Optional[str] = None                # how the check was performed
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NanInfResult:
    """Forward pass produced no NaN or Inf values."""
    status: Optional[str] = None                # "pass" | "fail"
    n_nan: Optional[int] = None
    n_inf: Optional[int] = None
    samples_checked: Optional[int] = None       # total scalar values inspected
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LivenessResult:
    """Outputs aren't stuck/collapsed (e.g. all-zeros, constant)."""
    status: Optional[str] = None                # "pass" | "fail"
    std: Optional[float] = None                 # output std; should exceed `criterion` threshold
    mean: Optional[float] = None
    criterion: Optional[str] = None             # human-readable acceptance rule
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributionResult:
    """Output distribution stats; optionally compared against training stats."""
    status: Optional[str] = None                # "pass" | "fail"  (optional)
    ratio_in_acceptable_range: Optional[bool] = None  # primary verdict if status absent
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    per_dim_mean: List[float] = field(default_factory=list)
    per_dim_std: List[float] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RangeCheckResult:
    """Outputs lie within the expected numeric range (e.g. clipped action range)."""
    status: Optional[str] = None                # "pass" | "fail"  (optional)
    in_expected_range: Optional[bool] = None    # primary verdict if status absent
    expected_range: Optional[List[float]] = None  # [lo, hi]
    actual_min: Optional[float] = None
    actual_max: Optional[float] = None
    rationale: Optional[str] = None             # why these bounds were picked
    details: Dict[str, Any] = field(default_factory=dict)


# Results from the upstream agent's smoke tests on a calibration batch.
# Each bucket is a typed sub-dataclass.  The kernel aggregates these into
# a single pass/fail/soft-signal per bucket.
@dataclass
class SmokeResults:
    calibration_batch_source: Optional[str] = None  # dataset the batch came from
    calibration_batch_size: Optional[int] = None
    determinism: Optional[DeterminismResult] = None
    nan_inf: Optional[NanInfResult] = None
    liveness: Optional[LivenessResult] = None
    distribution: Optional[DistributionResult] = None
    range_check: Optional[RangeCheckResult] = None


# Top-level output section: what comes out, how to run inference, and what
# the upstream smoke tests found.
@dataclass
class OutputSpec:
    actions: Optional[OutputActions] = None
    auxiliary_outputs: AuxiliaryOutputs = field(default_factory=AuxiliaryOutputs)
    inference_parameters: Optional[InferenceParameters] = None
    post_processing: Optional[PostProcessing] = None
    smoke_results: Optional[SmokeResults] = None


# ── weight_integrity ────────────────────────────────────────────────────


# Per-file hash recorded by the upstream agent after writing the checkpoint.
# The signoff layer recomputes these at validation time to detect tampering
# or incomplete downloads.
@dataclass
class WeightFileEntry:
    path: str                                    # relative to checkpoint root
    sha256: str
    size_bytes: int


@dataclass
class WeightIntegrity:
    weight_files: List[WeightFileEntry] = field(default_factory=list)
    manifest_hash: Optional[str] = None          # rollup hash over all weight files


# ── provenance ──────────────────────────────────────────────────────────


# Pointers back to the training run that produced this checkpoint.
# These are for traceability, not for validation — the kernel checks only
# verify format (commit is a valid hash, path doesn't escape the root).
@dataclass
class Provenance:
    run_log_path: Optional[str] = None            # relative .md path OR https:// URI
    training_repo: Optional[str] = None           # git remote URL
    training_repo_commit: Optional[str] = None    # full 40-char sha
    passport_creation_repo: Optional[str] = None  # autohpc/tooling repo used to create the passport
    passport_creation_repo_commit: Optional[str] = None  # tooling commit used to create the passport
    config_snapshot_path: Optional[str] = None    # path to saved training config
    merged_config_sha256: Optional[str] = None    # hash of the resolved training config
    parent_checkpoint: Optional[str] = None       # passport hash of parent if fine-tuned
    parent_description: Optional[str] = None      # e.g. "pretrained DiT base, 50K steps on ..."
    deployment_repo: Optional[str] = None         # git remote URL of target deployment repo
    deployment_repo_commit: Optional[str] = None  # optional debug pointer; not a load gate


# ── transform_pipeline ──────────────────────────────────────────────────


# One step in the ordered data-flow pipeline from sensor to action output.
# The full pipeline is the chain-of-custody: every transform the data
# undergoes between raw sensor reading and robot command.
@dataclass
class TransformStep:
    order: int = 0                                # position in the pipeline
    name: str = ""                                # e.g. "resize_images", "ramen_normalize_state"
    applies_to: str = ""                          # "images.front", "state", "action", "all_images"
    operation: str = ""                           # "resize", "imagenet_normalize", "ramen_normalize", ...
    direction: str = "input"                      # "input" (pre-model) or "output" (post-model)
    parameters: Dict[str, Any] = field(default_factory=dict)  # operation-specific params
    check_type: str = "static"                    # "static" or "dynamic"
    check_description: Optional[str] = None       # what to verify and how


# ── reference_test_vector ──────────────────────────────────────────────


# Real dataset frames saved alongside the checkpoint for static
# verification: camera identity, state shape/range, normalization sanity,
# and hash integrity after transfer.
@dataclass
class ReferenceTestVector:
    n_frames: int = 10                                   # how many consecutive frames are stored
    input_state_path: Optional[str] = None               # relative to ckpt root, .npy shape (n_frames, state_dim)
    input_state_hash: Optional[str] = None               # sha256 of the .npy file
    input_images_path: Optional[str] = None               # dir containing {cam}_{frame:03d}.png
    input_images_hash: Dict[str, List[str]] = field(default_factory=dict)  # {cam_key: [sha256 per frame]}
    input_prompt: str = ""
    notes: Optional[str] = None


# ── normalization round-trip ───────────────────────────────────────────


# Result of normalize → unnormalize on a known input, verifying invertibility.
@dataclass
class NormRoundTripResult:
    step_name: str = ""                           # which transform_pipeline step
    max_abs_error: float = 0.0
    within_clip_bound: bool = True
    input_source: str = ""                        # "reference_test_vector" | "training_dataset_sample"
    status: str = "pass"                          # "pass" | "fail"


# ── known_issues ───────────────────────────────────────────────────────


@dataclass
class KnownIssue:
    id: str = ""
    severity: str = "warning"                     # "critical" | "warning" | "info"
    description: str = ""
    workaround: Optional[str] = None
    check_type: str = "static"                    # "static" or "dynamic"


# ── Top-level passport ──────────────────────────────────────────────────


# The complete MODEL_PASSPORT.json at the checkpoint root.
# Six sections covering the full lifecycle: what goes in, what the model is,
# what's inside it, what comes out, whether the files are intact, and where
# it came from.  extra_sections preserves unknown keys from newer schema
# versions so a v0.1 reader can round-trip a v0.2 passport without data loss.
@dataclass
class ModelPassport:
    """Complete v0.2 passport at the checkpoint root."""

    schema_version: str = SCHEMA_VERSION
    generated_by: Optional[str] = None            # tool name, e.g. "generate-passport"
    generated_at: Optional[str] = None            # ISO 8601 timestamp
    stack: Optional[str] = None                   # e.g. "lerobot", "openpi"

    input_contract: InputContract = field(default_factory=InputContract)
    model_identity: ModelIdentity = field(default_factory=ModelIdentity)
    model_internals: ModelInternals = field(default_factory=ModelInternals)
    output_spec: OutputSpec = field(default_factory=OutputSpec)
    weight_integrity: WeightIntegrity = field(default_factory=WeightIntegrity)
    provenance: Provenance = field(default_factory=Provenance)
    transform_pipeline: List[TransformStep] = field(default_factory=list)
    reference_test_vector: Optional[ReferenceTestVector] = None
    norm_round_trip_results: List[NormRoundTripResult] = field(default_factory=list)
    known_issues: List[KnownIssue] = field(default_factory=list)

    extra_sections: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = _dataclass_to_dict(self)
        # Inline extra_sections at the top level (don't nest under a key)
        extras = d.pop("extra_sections", {})
        d.update(extras)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelPassport:
        known_top = {
            "schema_version", "generated_by", "generated_at", "stack",
            "input_contract", "model_identity", "model_internals",
            "output_spec", "weight_integrity", "provenance",
            "transform_pipeline", "reference_test_vector",
            "norm_round_trip_results", "known_issues",
        }
        extras = {k: v for k, v in data.items() if k not in known_top}

        mi_data = data.get("model_identity")
        model_identity = _model_identity_from_dict(mi_data)

        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            generated_by=data.get("generated_by"),
            generated_at=data.get("generated_at"),
            stack=data.get("stack"),
            input_contract=_input_contract_from_dict(data.get("input_contract")),
            model_identity=model_identity,
            model_internals=_model_internals_from_dict(data.get("model_internals")),
            output_spec=_output_spec_from_dict(data.get("output_spec")),
            weight_integrity=_weight_integrity_from_dict(data.get("weight_integrity")),
            provenance=_dict_to_dataclass(Provenance, data.get("provenance")),
            transform_pipeline=[
                _dict_to_dataclass(TransformStep, s)
                for s in data.get("transform_pipeline", []) or []
            ],
            reference_test_vector=(
                _dict_to_dataclass(ReferenceTestVector, data["reference_test_vector"])
                if data.get("reference_test_vector") else None
            ),
            norm_round_trip_results=[
                _dict_to_dataclass(NormRoundTripResult, r)
                for r in data.get("norm_round_trip_results", []) or []
            ],
            known_issues=[
                _dict_to_dataclass(KnownIssue, ki)
                for ki in data.get("known_issues", []) or []
            ],
            extra_sections=extras,
        )


# ── SIGNOFF.json ────────────────────────────────────────────────────────


# One file covered by the signoff: path relative to checkpoint root + its hash.
# The validator recomputes sha256 on disk and compares — any mismatch is a
# hard fail (file was modified or corrupted after signing).
@dataclass
class SignoffArtifact:
    path: str                                    # relative to checkpoint root
    sha256: str


# What produced the signoff (typically the upstream agent's passport tool).
@dataclass
class SignoffSigner:
    tool: Optional[str] = None
    version: Optional[str] = None


# SIGNOFF.json: the upstream agent's attestation that the passport and weights
# are consistent.  Written after self-validation passes.  The validator treats
# checksum mismatches as hard security failures regardless of other checks.
@dataclass
class Signoff:
    schema_version: str = SIGNOFF_SCHEMA_VERSION
    signed_at: Optional[str] = None              # ISO 8601 timestamp
    signed_by: SignoffSigner = field(default_factory=SignoffSigner)
    artifacts: List[SignoffArtifact] = field(default_factory=list)
    verdict: Optional[str] = None                # "pass" | "fail" | "soft_signal"
    verdict_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Signoff:
        return cls(
            schema_version=data.get("schema_version", SIGNOFF_SCHEMA_VERSION),
            signed_at=data.get("signed_at"),
            signed_by=_dict_to_dataclass(SignoffSigner, data.get("signed_by")),
            artifacts=[_dict_to_dataclass(SignoffArtifact, a)
                       for a in data.get("artifacts", []) or []],
            verdict=data.get("verdict"),
            verdict_reason=data.get("verdict_reason"),
        )


# ── Helpers ─────────────────────────────────────────────────────────────


def _prune(value: Any) -> Any:
    """Recursively strip None, empty lists, empty dicts, and False booleans
    that are just dataclass defaults.  Preserves meaningful zeros and
    False values inside non-empty structures."""
    if isinstance(value, dict):
        pruned = {}
        for k, v in value.items():
            cleaned = _prune(v)
            if cleaned is not None:
                pruned[k] = cleaned
        return pruned if pruned else None
    if isinstance(value, list):
        pruned = [_prune(item) for item in value]
        pruned = [item for item in pruned if item is not None]
        return pruned if pruned else None
    if value is None:
        return None
    if value is False:
        return None
    return value


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """asdict with recursive pruning of null/empty/default values.

    from_dict handles missing keys by filling defaults, so stripping
    empties is lossless — it just makes the JSON readable.
    """
    raw = asdict(obj)
    pruned = _prune(raw)
    return pruned if isinstance(pruned, dict) else {}


def _dict_to_dataclass(dc_cls, data):
    """Build a dataclass from a dict, ignoring keys the dataclass doesn't know."""
    if data is None:
        return dc_cls()
    if not isinstance(data, dict):
        return dc_cls()
    field_names = {f.name for f in fields(dc_cls)}
    kwargs = {k: v for k, v in data.items() if k in field_names}
    return dc_cls(**kwargs)


def _action_spec_from_dict(data) -> Optional[ActionSpec]:
    if not data or not isinstance(data, dict):
        return None
    spec = _dict_to_dataclass(ActionSpec, data)
    dd = data.get("delta_dims")
    if dd and isinstance(dd, dict):
        spec.delta_dims = _dict_to_dataclass(DeltaSpec, dd)
    elif dd and isinstance(dd, str):
        spec.delta_dims = DeltaSpec(absolute_dims_reason=dd)
    return spec


def _input_contract_from_dict(data) -> InputContract:
    if data is None:
        return InputContract()
    return InputContract(
        images=[_dict_to_dataclass(ImageSpec, i) for i in data.get("images", []) or []],
        state=_dict_to_dataclass(StateSpec, data.get("state")) if data.get("state") else None,
        actions=_action_spec_from_dict(data.get("actions")),
        language=_dict_to_dataclass(LanguageSpec, data.get("language")) if data.get("language") else None,
        temporal=_dict_to_dataclass(TemporalSpec, data.get("temporal")) if data.get("temporal") else None,
        training_datasets=[
            _dict_to_dataclass(TrainingDatasetSpec, d)
            for d in data.get("training_datasets", []) or []
        ],
    )


def _model_identity_from_dict(data) -> ModelIdentity:
    if data is None:
        return ModelIdentity()
    mi = _dict_to_dataclass(ModelIdentity, data)
    rc_data = data.get("runtime_constraints")
    if rc_data and isinstance(rc_data, dict):
        mi.runtime_constraints = _dict_to_dataclass(RuntimeConstraints, rc_data)
    return mi


def _pretrained_asset_from_dict(data) -> PretrainedAsset:
    """Load a PretrainedAsset with backward compat for old field names."""
    if not data or not isinstance(data, dict):
        return PretrainedAsset(submodule="")
    d = dict(data)
    if "timm_string" in d and "source_identifier" not in d:
        d["source_identifier"] = d.pop("timm_string")
    elif "timm_string" in d:
        d.pop("timm_string")
    if "hf_revision" in d and "source_revision" not in d:
        d["source_revision"] = d.pop("hf_revision")
    elif "hf_revision" in d:
        d.pop("hf_revision")
    return _dict_to_dataclass(PretrainedAsset, d)


def _model_internals_from_dict(data) -> ModelInternals:
    if data is None:
        return ModelInternals()

    params_data = data.get("parameters", {}) or {}
    parameters = ParametersBlock(
        summary=_dict_to_dataclass(ParametersSummary, params_data.get("summary")),
    )

    return ModelInternals(
        module_hierarchy=data.get("module_hierarchy", []) or [],
        parameters=parameters,
        buffers=[_dict_to_dataclass(BufferEntry, b) for b in data.get("buffers", []) or []],
        state_dict=_dict_to_dataclass(StateDictBlock, data.get("state_dict")),
        pretrained_provenance=[
            _pretrained_asset_from_dict(a)
            for a in data.get("pretrained_provenance", []) or []
        ],
        quantization=_dict_to_dataclass(QuantizationBlock, data.get("quantization")),
        forward_graph=_dict_to_dataclass(ForwardGraph, data.get("forward_graph")),
        numerical_health=_dict_to_dataclass(NumericalHealth, data.get("numerical_health")),
    )


def _output_spec_from_dict(data) -> OutputSpec:
    if data is None:
        return OutputSpec()
    return OutputSpec(
        actions=_dict_to_dataclass(OutputActions, data.get("actions")) if data.get("actions") else None,
        auxiliary_outputs=_dict_to_dataclass(AuxiliaryOutputs, data.get("auxiliary_outputs")),
        inference_parameters=_dict_to_dataclass(InferenceParameters, data.get("inference_parameters")) if data.get("inference_parameters") else None,
        post_processing=_dict_to_dataclass(PostProcessing, data.get("post_processing")) if data.get("post_processing") else None,
        smoke_results=_smoke_results_from_dict(data.get("smoke_results")),
    )


def _smoke_results_from_dict(data) -> Optional[SmokeResults]:
    """Convert each bucket dict into its typed sub-dataclass.

    Unknown keys in a bucket are silently dropped (same convention as the
    rest of the loader); callers should put upstream-specific extras into
    the bucket's `details: Dict[str, Any]` field if they want them
    preserved on round-trip.
    """
    if not data:
        return None
    return SmokeResults(
        calibration_batch_source=data.get("calibration_batch_source"),
        calibration_batch_size=data.get("calibration_batch_size"),
        determinism=_dict_to_dataclass(DeterminismResult, data.get("determinism")) if data.get("determinism") else None,
        nan_inf=_dict_to_dataclass(NanInfResult, data.get("nan_inf")) if data.get("nan_inf") else None,
        liveness=_dict_to_dataclass(LivenessResult, data.get("liveness")) if data.get("liveness") else None,
        distribution=_dict_to_dataclass(DistributionResult, data.get("distribution")) if data.get("distribution") else None,
        range_check=_dict_to_dataclass(RangeCheckResult, data.get("range_check")) if data.get("range_check") else None,
    )


def _weight_integrity_from_dict(data) -> WeightIntegrity:
    if data is None:
        return WeightIntegrity()
    return WeightIntegrity(
        weight_files=[_dict_to_dataclass(WeightFileEntry, w) for w in data.get("weight_files", []) or []],
        manifest_hash=data.get("manifest_hash"),
    )


# ── Schema version compatibility table ──────────────────────────────────


SUPPORTED_PASSPORT_VERSIONS = {"0.2"}
SUPPORTED_SIGNOFF_VERSIONS = {"0.2"}


def is_passport_version_supported(version: str) -> bool:
    return version in SUPPORTED_PASSPORT_VERSIONS


def is_signoff_version_supported(version: str) -> bool:
    return version in SUPPORTED_SIGNOFF_VERSIONS
