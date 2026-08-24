# GMAN (givemeanode)

On-demand H100 / CPU nodes for agents.

**Manual:** [llms.txt](https://givemeanode.com/llms.txt). Cursor: add `"givemeanode": { "url": "https://mcp.givemeanode.com" }` to `~/.cursor/mcp.json` ([docs](https://cursor.com/docs/mcp)); OAuth on first connect.

Do not store secrets in this file.

## Heuristics

- **Mission first.** `open_mission(name)` once, pass `mission` on create / run / import / export / stop. Hand the mission URL to the human immediately.
- **Name things.** Unique node names. `list_nodes` first; only touch rows with `same_mission`.
- **Secrets, never argv.** `gman secret create --material-stdin`. `run_command` env: `{VAR: {secret: "name"}}`. Command strings are logged.
- **Pull, don’t upload.** Clone the git repo + submodules with a GitHub secret. HF datasets via `connection` + `import_data`, not `write_file` of a tree.
- **CPU then GPU.** `cpu-2` / `cpu-8` for clone, install, dataset import. GPU only for the actual train/eval. Do not wake a GPU for unit tests.
- **Disk.** `~` parks across `stop_node`. `/scratch` is wiped on stop. Cache weights under `~` if you will stop and wake.
- **Stop when idle.** Nodes bill while running or in grace. `stop_node` the moment work is done. Confirm before `delete_node`.
- **Size to the cgroup.** Use `$GMN_CPU_LIMIT` / `$GMN_MEMORY_LIMIT_BYTES`, not host `nproc`/`free`.
- **Jobs for fire-and-forget.** `submit_job` / `submit_jobs` for completion-shaped work. Interactive smoke stays on `create_node` + detached `run_command`.

## Skills that still apply

- `hpc-run-tracking/SKILL.md` — use mission + node name as the execution ID.
- `hpc-model-dataset-audit/SKILL.md` — before a real training run.
