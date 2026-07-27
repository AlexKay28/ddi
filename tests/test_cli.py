"""Smoke tests for the CLI wiring (issue #2)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ddi import cli


@pytest.fixture(autouse=True)
def _reset_runners():
    saved = dict(cli._RUNNERS)
    cli._RUNNERS.clear()
    yield
    cli._RUNNERS.clear()
    cli._RUNNERS.update(saved)


def test_load_config_hashes_bytes(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("task: foo\nk: 1\n")
    config, h1 = cli.load_config(path)
    _, h2 = cli.load_config(path)
    assert config == {"task": "foo", "k": 1}
    assert h1 == h2
    assert len(h1) == 16


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="not a mapping"):
        cli.load_config(path)


def test_dispatch_requires_task() -> None:
    with pytest.raises(ValueError, match="missing required key: task"):
        cli.dispatch({}, "abc123", 0)


def test_dispatch_unknown_task() -> None:
    with pytest.raises(NotImplementedError, match="no runner registered"):
        cli.dispatch({"task": "nope"}, "abc123", 0)


def test_register_and_dispatch() -> None:
    calls: list[tuple[dict, str, int]] = []

    @cli.register_runner("smoke")
    def _runner(config, config_hash, seed):
        calls.append((config, config_hash, seed))

    cli.dispatch({"task": "smoke", "x": 1}, "hash", 42)
    assert calls == [({"task": "smoke", "x": 1}, "hash", 42)]


def test_duplicate_registration_rejected() -> None:
    cli.register_runner("dup")(lambda *_: None)
    with pytest.raises(ValueError, match="already registered"):
        cli.register_runner("dup")(lambda *_: None)


def test_main_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(textwrap.dedent("""\
        task: smoke_e2e
        payload: 7
    """))

    seen: dict = {}

    @cli.register_runner("smoke_e2e")
    def _runner(config, config_hash, seed):
        seen["config"] = config
        seen["hash"] = config_hash
        seen["seed"] = seed

    rc = cli.main(["--config", str(path), "--seed", "13"])
    assert rc == 0
    assert seen["config"] == {"task": "smoke_e2e", "payload": 7}
    assert seen["seed"] == 13
    assert len(seen["hash"]) == 16
