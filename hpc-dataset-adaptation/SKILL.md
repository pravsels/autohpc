---
name: hpc-dataset-adaptation
description: Use when the user's dataset format differs from what the target repo's data loaders expect, to adapt the code (not the data) for compatibility.
---

# HPC Dataset Adaptation

## Overview

Use this when the user has their own dataset and the target repo's existing loaders don't read it directly.
Core principle: adapt the code to read the data as-is. Do **not** convert or copy large datasets into a different format.

## When to Use

- User's data has a different schema, layout, or file format than what the repo expects
- User's data is large enough that copying or converting is impractical
- Training smoke tests pass with repo demo data but need to work with user data too

## When to Skip

- User is training on the repo's own provided data
- User's data already matches the repo's expected format

## Core Pattern

1. **Inspect the user's dataset inside the container.** Use the container's installed tools (e.g. `h5py`, `numpy`, `pandas`) to examine file structure, keys, shapes, and dtypes. Do not install inspection tools on the host.
2. **Inspect the repo's existing data loaders.** Read the dataset classes and configs to understand the expected schema — what keys, shapes, and structure the training code consumes.
3. **Map the gap.** Identify which fields in the user's data correspond to which expected inputs, what's missing, and what needs renaming or reshaping.
4. **Write a loader or adapter in the target repo.** Create a new dataset class that reads the user's format directly and outputs the data structures the training code expects. Register it alongside existing dataset options.
5. **Validate the loader independently.** Instantiate the new dataset class inside the container and verify it returns correctly shaped and typed outputs before running training.
6. **Smoke test training end-to-end.** Run training inside the container (`--gpus all`, small batch, few steps) with the user's actual data. The same entry point and config structure as the repo's standard training path.

## Quick Reference

| Goal | Approach |
|---|---|
| Inspect dataset schema | Use the format's native Python library inside the container to walk structure, keys, shapes, dtypes |
| Find existing loaders | Search for dataset classes in the target repo (`Dataset`, `DataLoader`, config files) |
| Validate new loader | Instantiate dataset, fetch one sample, print keys and shapes |
| Smoke test training | Same training command as Phase 1 but pointing at user's data path |

## Common Mistakes

- Converting or copying large datasets instead of writing a loader
- Inspecting data on the host instead of inside the container
- Writing a loader that loads entire episodes/trajectories into memory at once for large datasets
- Forgetting to register the new dataset class and config so training can select it
- Testing the loader in isolation but skipping an actual training step with it
