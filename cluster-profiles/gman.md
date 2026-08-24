# GMAN (givemeanode)

On-demand H100 / CPU nodes for agents. Not Slurm. Not a laptop substitute.

**Manual:** [llms.txt](https://givemeanode.com/llms.txt) (also `gman://docs/llms.txt`). MCP: `https://mcp.givemeanode.com`.

Do not store secrets in this file.

## Heuristics

- **Mission first.** `open_mission(name)` once, pass `mission` on create / run / import / export / stop. Hand the mission URL to the human immediately.
- **Name things.** Unique node names (`cra-busybox-smoke-<who>`). No product-codename leaks in new names. `list_nodes` first; only touch rows with `same_mission`.
- **Secrets, never argv.** `gman secret create --material-stdin`. `run_command` env: `{VAR: {secret: "name"}}`. Command strings are logged.
- **Pull, don’t upload.** Clone the git repo + submodules with a GitHub secret. HF datasets via `connection` + `import_data`, not `write_file` of a tree.
- **CPU then GPU.** `cpu-2` / `cpu-8` for clone, `uv sync`, dataset import. `h100-1` (80 GB) for CRA smoke. Do not wake an H100 for unit tests.
- **Disk.** `~` parks across `stop_node`. `/scratch` is wiped on stop. Cache weights under `~` if you will stop and wake.
- **Stop when idle.** Nodes bill while running or in grace. `stop_node` the moment work is done. Confirm before `delete_node`.
- **Size to the cgroup.** Use `$GMN_CPU_LIMIT` / `$GMN_MEMORY_LIMIT_BYTES`, not host `nproc`/`free`.
- **Jobs for fire-and-forget.** `submit_job` / `submit_jobs` for completion-shaped work. Interactive smoke stays on `create_node` + detached `run_command`.

## CRA smoke (alpha-robotics)

Target repo: `experimental/cra`. Hook: `scripts/train.py` (Candy loader, no submodule edits). Dataset: `villekuosmanen/busybox_push_green_button` (LeRobot v3). Embodiment: `so100_3rgb`.

- Image: `cuda-12.9`. Official smoke is H100 80 GB.
- From `external/cra` after `uv sync --frozen --extra training`, plus Candy/LeRobot (training extra does not ship `lerobot`).
- Do not use the 5090 workstation (robot inference).
- Last 1-step failure: NVIDIA loader wants `meta/episodes.jsonl` (v2.1-only). Adapter is the fix; install real Candy before the next GPU run.

## Skills that still apply

- `hpc-run-tracking/SKILL.md` — use mission + node name as the execution ID.
- `hpc-model-dataset-audit/SKILL.md` — after 1-step works, before a real finetune.
- Slurm templates in `hpc-training-operations/SKILL.md` do **not** apply.
