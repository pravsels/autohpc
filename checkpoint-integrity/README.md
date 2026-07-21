# checkpoint-integrity

Fast, framework-independent integrity checks for checkpoint bundles.

`manifest-checkpoint` hashes the files already present in a checkpoint or
publish directory and writes `CHECKPOINT_MANIFEST.json`. It does not import the
model, inspect model semantics, run calibration data, or block upload on
anything beyond readable files.

```bash
uv pip install -e ../autohpc/checkpoint-integrity

manifest-checkpoint <checkpoint_or_publish_dir>
verify-checkpoint <checkpoint_or_download_dir>
```

Run `manifest-checkpoint` immediately before upload. Run `verify-checkpoint`
after download and before eval or deployment. By default verification checks
every declared file and warns about extra files, which accommodates
registry-added metadata such as `.gitattributes`. Use `--strict` when the
downloaded directory must contain exactly the manifested files.

Symlinks are skipped and reported when generating a manifest. Publish the
resolved checkpoint files rather than symlink aliases such as `last/`.
