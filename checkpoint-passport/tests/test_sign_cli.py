from __future__ import annotations

import sys
import types
from pathlib import Path

from checkpoint_passport import PASSPORT_FILENAME
from checkpoint_passport.cli.sign import main


def test_sign_checkpoint_forwards_target_repo_to_validation(
    tmp_path: Path,
    monkeypatch,
):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / PASSPORT_FILENAME).write_text("{}")
    target_repo = tmp_path / "deploy"
    target_repo.mkdir()
    calls = []

    passport = types.SimpleNamespace(
        weight_integrity=types.SimpleNamespace(weight_files=[]),
    )
    result = types.SimpleNamespace(
        has_failures=False,
        verdict="ship_it",
        observations=[],
        passport_load=types.SimpleNamespace(passport=passport),
    )

    def fake_self_validate_passport(checkpoint_dir, **kwargs):
        calls.append((checkpoint_dir, kwargs))
        return result

    monkeypatch.setattr(
        "checkpoint_passport.cli.sign.self_validate_passport",
        fake_self_validate_passport,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sign-checkpoint",
            str(ckpt),
            "--target-repo",
            str(target_repo),
        ],
    )

    main()

    assert len(calls) == 2
    assert calls[0][1]["target_repo"] == target_repo
    assert calls[0][1].get("require_signoff") is None
    assert calls[1][1]["target_repo"] == target_repo
    assert calls[1][1]["require_signoff"] is True
