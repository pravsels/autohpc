from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


class WandbSyncError(Exception):
    """Raised when a W&B sync command cannot be built or completed safely."""


@dataclass(frozen=True)
class WandbSyncConfig:
    offline_run_dir: Path
    container: Path
    entity: str | None = None
    project: str | None = None
    token_file: Path | None = None
    binds: list[Path] | None = None
    use_srun: bool = True
    srun_args: list[str] | None = None
    apptainer_args: list[str] | None = None
    wandb_args: list[str] | None = None
    ssh_host: str | None = None
    modules: list[str] | None = None


@dataclass(frozen=True)
class WandbSyncResult:
    returncode: int
    url: str | None
    stdout: str
    stderr: str


Runner = Callable[[Sequence[str]], tuple[int, str, str]]

_TOKEN_FILENAMES = (".wandb_token", ".wandb_key")
_WANDB_URL_RE = re.compile(r"https://wandb\.ai/[^\s)>\"]+")


def find_token_file(home: Path | None = None) -> Path | None:
    root = Path.home() if home is None else home
    for name in _TOKEN_FILENAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def validate_config(cfg: WandbSyncConfig) -> None:
    if not cfg.entity:
        raise WandbSyncError(
            "missing --entity: ask the user which W&B entity/team should receive this run"
        )
    if not cfg.project:
        raise WandbSyncError(
            "missing --project: ask the user which W&B project name should receive this run"
        )
    if cfg.token_file is None:
        raise WandbSyncError(
            "missing W&B token file: pass --wandb-token-file or create ~/.wandb_token"
        )
    if not cfg.token_file.is_file():
        raise WandbSyncError(f"W&B token file not found: {cfg.token_file}")
    if not cfg.offline_run_dir.exists():
        raise WandbSyncError(f"offline run dir not found: {cfg.offline_run_dir}")
    if not cfg.container.is_file():
        raise WandbSyncError(f"container not found: {cfg.container}")


def build_wandb_sync_command(cfg: WandbSyncConfig) -> list[str]:
    validate_config(cfg)

    command = _build_cluster_command(cfg)
    if cfg.ssh_host:
        remote_parts = []
        for module_name in cfg.modules or []:
            remote_parts.append("module load " + shlex.quote(module_name))
        remote_parts.append(render_command(command))
        return ["ssh", cfg.ssh_host, "; ".join(remote_parts)]
    return command


def _build_cluster_command(cfg: WandbSyncConfig) -> list[str]:
    command: list[str] = []
    if cfg.use_srun:
        command.extend(["srun", *(cfg.srun_args or ["--ntasks=1", "--cpu-bind=cores"])])

    command.extend(["apptainer", "exec"])
    for bind in cfg.binds or []:
        command.extend(["--bind", str(bind)])
    command.extend(cfg.apptainer_args or [])
    command.extend([str(cfg.container), "bash", "-lc", _build_inner_script(cfg)])
    return command


def render_command(command: Sequence[str], token_file: Path | None = None) -> str:
    token_text = str(token_file) if token_file is not None else None
    rendered = []
    for part in command:
        redacted = part.replace(token_text, "<wandb-token-file>") if token_text else part
        rendered.append(shlex.quote(redacted))
    return " ".join(rendered)


def sync_wandb(
    cfg: WandbSyncConfig,
    *,
    runner: Runner | None = None,
) -> WandbSyncResult:
    command = build_wandb_sync_command(cfg)
    run = runner or _subprocess_runner
    returncode, stdout, stderr = run(command)
    url = extract_wandb_url(stdout + "\n" + stderr)
    if returncode != 0:
        raise WandbSyncError(
            f"wandb sync failed with exit code {returncode}\n{stderr.strip()}"
        )
    return WandbSyncResult(returncode=returncode, url=url, stdout=stdout, stderr=stderr)


def extract_wandb_url(text: str) -> str | None:
    matches = _WANDB_URL_RE.findall(text)
    return matches[-1] if matches else None


def coerce_paths(values: Iterable[str] | None) -> list[Path]:
    return [Path(value) for value in values or []]


def _build_inner_script(cfg: WandbSyncConfig) -> str:
    assert cfg.token_file is not None
    parts = [
        'export WANDB_API_KEY="$(cat ' + shlex.quote(str(cfg.token_file)) + ')"',
        "wandb sync",
        "--entity",
        shlex.quote(cfg.entity or ""),
        "--project",
        shlex.quote(cfg.project or ""),
        *(shlex.quote(arg) for arg in cfg.wandb_args or []),
        shlex.quote(str(cfg.offline_run_dir)),
    ]
    return " ".join(parts)


def _subprocess_runner(command: Sequence[str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        list(command),
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr
