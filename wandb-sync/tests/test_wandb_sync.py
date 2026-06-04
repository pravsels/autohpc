from __future__ import annotations

import pytest

from autohpc_wandb_sync.sync import (
    WandbSyncConfig,
    WandbSyncError,
    build_wandb_sync_command,
    find_token_file,
    render_command,
    sync_wandb,
)


def test_builds_srun_apptainer_command_with_required_destination(tmp_path):
    token = tmp_path / ".wandb_token"
    token.write_text("wandb-secret\n")
    offline = tmp_path / "wandb" / "offline-run-20260604_083000-abcd"
    offline.mkdir(parents=True)
    container = tmp_path / "container.sif"
    container.write_text("sif")

    cfg = WandbSyncConfig(
        offline_run_dir=offline,
        container=container,
        entity="alpha-robotics",
        project="so101-stacking-rings",
        token_file=token,
        binds=[tmp_path],
    )

    command = build_wandb_sync_command(cfg)

    assert command[:6] == [
        "srun",
        "--ntasks=1",
        "--cpu-bind=cores",
        "apptainer",
        "exec",
        "--bind",
    ]
    assert str(tmp_path) in command
    assert str(container) in command
    script = command[-1]
    assert 'WANDB_API_KEY="$(cat ' in script
    assert "wandb sync" in script
    assert "--entity alpha-robotics" in script
    assert "--project so101-stacking-rings" in script
    assert str(offline) in script


def test_requires_entity_and_project(tmp_path):
    token = tmp_path / ".wandb_token"
    token.write_text("secret")
    offline = tmp_path / "offline-run"
    offline.mkdir()
    container = tmp_path / "container.sif"
    container.write_text("sif")

    with pytest.raises(WandbSyncError, match="missing --entity"):
        build_wandb_sync_command(WandbSyncConfig(
            offline_run_dir=offline,
            container=container,
            project="project",
            token_file=token,
        ))

    with pytest.raises(WandbSyncError, match="missing --project"):
        build_wandb_sync_command(WandbSyncConfig(
            offline_run_dir=offline,
            container=container,
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
    container = tmp_path / "container.sif"
    container.write_text("sif")

    cfg = WandbSyncConfig(
        offline_run_dir=offline,
        container=container,
        entity="entity",
        project="project",
        token_file=token,
    )

    rendered = render_command(build_wandb_sync_command(cfg), token)

    assert str(token) not in rendered
    assert "wandb-secret" not in rendered
    assert "<wandb-token-file>" in rendered


def test_sync_parses_wandb_url_from_output(tmp_path):
    token = tmp_path / ".wandb_token"
    token.write_text("secret")
    offline = tmp_path / "offline-run"
    offline.mkdir()
    container = tmp_path / "container.sif"
    container.write_text("sif")
    cfg = WandbSyncConfig(
        offline_run_dir=offline,
        container=container,
        entity="entity",
        project="project",
        token_file=token,
        use_srun=False,
    )

    seen = {}

    def fake_runner(command):
        seen["command"] = command
        return 0, "View project at https://wandb.ai/entity/project/runs/abcd\n", ""

    result = sync_wandb(cfg, runner=fake_runner)

    assert result.url == "https://wandb.ai/entity/project/runs/abcd"
    assert seen["command"][0:2] == ["apptainer", "exec"]
