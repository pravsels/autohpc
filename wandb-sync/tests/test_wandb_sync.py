from __future__ import annotations

from pathlib import Path

import pytest

from autohpc_wandb_sync.sync import (
    InnerSyncConfig,
    LaunchSyncConfig,
    WandbSyncError,
    build_inner_sync_command,
    build_launch_command,
    find_token_file,
    render_command,
    run_inner_sync,
)


def test_builds_inner_wandb_sync_command_with_required_destination(tmp_path):
    token = tmp_path / ".wandb_token"
    token.write_text("wandb-secret\n")
    offline = tmp_path / "wandb" / "offline-run-20260604_083000-abcd"
    offline.mkdir(parents=True)

    cfg = InnerSyncConfig(
        offline_run_dir=offline,
        entity="alpha-robotics",
        project="so101-stacking-rings",
        token_file=token,
    )

    command = build_inner_sync_command(cfg)

    assert command[:2] == ["bash", "-lc"]
    script = command[-1]
    assert 'WANDB_API_KEY="$(cat ' in script
    assert "&& wandb sync" in script
    assert "--entity alpha-robotics" in script
    assert "--project so101-stacking-rings" in script
    assert str(offline) in script


def test_requires_entity_and_project(tmp_path):
    token = tmp_path / ".wandb_token"
    token.write_text("secret")
    offline = tmp_path / "offline-run"
    offline.mkdir()

    with pytest.raises(WandbSyncError, match="missing --entity"):
        build_inner_sync_command(InnerSyncConfig(
            offline_run_dir=offline,
            project="project",
            token_file=token,
        ))

    with pytest.raises(WandbSyncError, match="missing --project"):
        build_inner_sync_command(InnerSyncConfig(
            offline_run_dir=offline,
            entity="entity",
            token_file=token,
        ))


def test_finds_common_token_file_names(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    token = home / ".wandb_token"
    token.write_text("secret\n")

    assert find_token_file(home) == token


def test_rendered_dry_run_redacts_token_file_path(tmp_path):
    token = tmp_path / ".wandb_token"
    token.write_text("wandb-secret\n")
    offline = tmp_path / "offline-run"
    offline.mkdir()

    cfg = InnerSyncConfig(
        offline_run_dir=offline,
        entity="entity",
        project="project",
        token_file=token,
    )

    rendered = render_command(build_inner_sync_command(cfg), token)

    assert str(token) not in rendered
    assert "wandb-secret" not in rendered
    assert "<wandb-token-file>" in rendered


def test_sync_parses_wandb_url_from_output(tmp_path):
    token = tmp_path / ".wandb_token"
    token.write_text("secret")
    offline = tmp_path / "offline-run"
    offline.mkdir()
    cfg = InnerSyncConfig(
        offline_run_dir=offline,
        entity="entity",
        project="project",
        token_file=token,
    )

    seen = {}

    def fake_runner(command):
        seen["command"] = command
        return 0, "View project at https://wandb.ai/entity/project/runs/abcd\n", ""

    result = run_inner_sync(cfg, runner=fake_runner)

    assert result.url == "https://wandb.ai/entity/project/runs/abcd"
    assert seen["command"][0:2] == ["bash", "-lc"]


def test_launch_mode_runs_inner_sync_inside_apptainer(tmp_path):
    token = tmp_path / ".wandb_token"
    token.write_text("secret")
    offline = tmp_path / "offline-run"
    offline.mkdir()
    container = tmp_path / "container.sif"
    container.write_text("sif")
    cfg = LaunchSyncConfig(
        offline_run_dir=offline,
        container=container,
        entity="entity",
        project="project",
        token_file=token,
        binds=[tmp_path],
        ssh_host="u6kr.aip2.isambard",
        modules=["brics/apptainer-multi-node"],
    )

    command = build_launch_command(cfg)

    assert command[:2] == ["ssh", "u6kr.aip2.isambard"]
    remote_script = command[-1]
    assert "module load brics/apptainer-multi-node" in remote_script
    assert "srun --ntasks=1 --cpu-bind=cores apptainer exec" in remote_script
    assert "autohpc-wandb-sync sync --entity entity --project project" in remote_script


def test_launch_mode_allows_paths_that_exist_only_remotely():
    cfg = LaunchSyncConfig(
        offline_run_dir=Path("/scratch/u6kr/user/project/wandb/offline-run-remote"),
        container=Path("/scratch/u6kr/user/project/container/openpi.sif"),
        entity="entity",
        project="project",
        token_file=Path("/home/u6kr/user/.wandb_token"),
        binds=[Path("/scratch/u6kr/user:/scratch/u6kr/user")],
        ssh_host="u6kr.aip2.isambard",
    )

    command = build_launch_command(cfg)

    assert command[:2] == ["ssh", "u6kr.aip2.isambard"]
    assert "/scratch/u6kr/user/project/wandb/offline-run-remote" in command[-1]
