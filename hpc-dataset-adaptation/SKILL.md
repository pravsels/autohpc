---
name: hpc-dataset-adaptation
description: Use when the user's dataset format differs from what the target repo's data loaders expect, to adapt the code (not the data) for compatibility.
---

# HPC Dataset Adaptation

## Overview

Use when the target repo's loader does not read the user's dataset directly.
Adapt code to read the data as-is; do not bulk-convert or copy large datasets
unless the user explicitly asks.

## When to Use

- User data has a different schema, layout, or file format than the repo expects.
- Training works on demo/repo data but not on the user's data.
- The dataset is too large to casually copy or rewrite.

Skip this if the data already matches the repo's expected format.

## Agent Algorithm

1. **Preflight**
   - Confirm the repo's normal training smoke test works with expected/demo data.
   - Identify the user's dataset path and the container/runtime used for training.

2. **Inspect actual data inside the container**
   - Use native libraries for the format to record keys, shapes, dtypes, lengths,
     modalities, state/action layout, and timestamps.
   - Save or paste a small schema summary; do not copy or rewrite the dataset.

3. **Inspect expected loader contract**
   - Read the repo's dataset classes/configs.
   - Identify expected keys, shapes, dtype, normalization, windowing, and
     language/action/state mappings.

4. **Map the gap**
   - Write the mapping from user fields to expected fields.
   - If required data is missing or ambiguous, stop and ask.

5. **Implement adapter/loader**
   - Add code in the target repo that reads the user's format directly and emits
     the repo's expected sample structure.
   - Register it through the repo's normal config mechanism.

6. **Verify**
   - Instantiate the loader inside the container and fetch samples.
   - Assert keys, shapes, and dtypes.
   - Run a small training smoke test with the real entry point.
   - Record evidence before proceeding to full training.

## Quick Reference

| Goal | Approach |
|---|---|
| Inspect dataset schema | Use native Python library inside the container |
| Find expected loader | Read dataset classes and config files |
| Validate adapter | Instantiate dataset, fetch sample, assert keys/shapes/dtypes |
| Smoke test training | Same entry point with small batch/few steps |

## Stop Gates

- Dataset path unavailable inside the container.
- Required modality/key is missing or semantically ambiguous.
- Loader sample does not match expected shape/dtype.
- Training smoke test fails.

## Common Mistakes

- Converting/copying large datasets instead of writing a loader.
- Inspecting data on the host instead of inside the container.
- Loading whole trajectories into memory for large datasets.
- Forgetting to register the loader/config.
- Testing the loader but skipping a training step.
