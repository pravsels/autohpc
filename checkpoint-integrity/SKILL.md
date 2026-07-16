---
name: checkpoint-integrity
description: Use immediately before moving or publishing a checkpoint, and after downloading it, to generate or verify a lightweight SHA-256 file manifest.
---

# Checkpoint Integrity

## Overview

This is a narrow file-integrity check, not a model-quality or inference-contract
gate. It hashes the checkpoint bundle without loading the model. Evaluation and
deployment audits own semantic correctness.

## Agent Algorithm

1. Assemble the exact files to publish or move. Include weights, model config,
   processors, normalization assets, and useful README/training records.
2. Exclude optimizer state, caches, transient W&B staging files, and symlink
   aliases unless the user explicitly wants them.
3. Generate the manifest immediately before transfer:

   ```bash
   uv pip install -e ../autohpc/checkpoint-integrity
   manifest-checkpoint <checkpoint_or_publish_dir>
   ```

4. Upload or copy the bundle, including `CHECKPOINT_MANIFEST.json`.
5. After download, and before eval or deployment, run:

   ```bash
   verify-checkpoint <downloaded_dir>
   ```

6. Stop if a declared file is missing or its size/hash changed. Extra files are
   warnings by default because registries may add metadata such as
   `.gitattributes`; use `--strict` when exact contents are required.

## Boundaries

- The manifest proves that declared bytes did not change.
- It does not approve a checkpoint, validate a config, document tensor
  contracts, run inference, or predict deployment safety.
- Generate a fresh manifest after any intentional file change.
- Do not hand-author hashes in a README.
- Symlinks are skipped and reported; publish resolved files instead.
