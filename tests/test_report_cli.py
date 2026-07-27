"""Smoke tests for ddi/report_cli.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from ddi import report, report_cli


@pytest.fixture(autouse=True)
def _snapshot_renderers():
    saved = dict(report._RENDERERS)
    yield
    report._RENDERERS.clear()
    report._RENDERERS.update(saved)


def test_list_prints_registered_reports(capsys, tmp_path):
    # Add a test renderer so --list is not empty even if the registry is cleared.
    report.register("T_smoke")(lambda store: "hello")
    rc = report_cli.main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    assert "T_smoke" in out


def test_name_dispatches_table_renderer(capsys, tmp_path):
    report.register("T_smoke")(lambda store: "the smoke table\n")
    fake_store = tmp_path / "runs.jsonl"
    fake_store.touch()
    rc = report_cli.main(["--name", "T_smoke", "--output", str(tmp_path / "unused.txt")])
    # --output is accepted but tables don't write; still exit 0 and print to stdout
    assert rc == 0
    assert "the smoke table" in capsys.readouterr().out


def test_name_dispatches_figure_renderer(tmp_path):
    output = tmp_path / "F_smoke.pdf"

    @report.register("F_smoke")
    def _fake_figure(store: Path, out: Path) -> None:
        out.write_bytes(b"pdf-bytes")

    rc = report_cli.main(["--name", "F_smoke", "--output", str(output)])
    assert rc == 0
    assert output.read_bytes() == b"pdf-bytes"


def test_missing_name_without_list_errors():
    with pytest.raises(SystemExit):
        report_cli.main([])


def test_unknown_name_raises():
    with pytest.raises(KeyError, match="unknown report"):
        report_cli.main(["--name", "T_does_not_exist"])
