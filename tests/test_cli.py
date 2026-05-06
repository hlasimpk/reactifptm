"""Integration tests for the reactifptm CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from reactifptm.cli import build_parser, main


def test_parser_defaults() -> None:
    p = build_parser()
    args = p.parse_args(["scores.json", "model.pdb"])
    assert args.input == "scores.json"
    assert args.structure == "model.pdb"
    assert args.threshold == 8.0
    assert args.output is None


def test_parser_threshold_and_output() -> None:
    p = build_parser()
    args = p.parse_args(["s.json", "m.pdb", "-t", "5.5", "-o", "out.json"])
    assert args.threshold == 5.5
    assert args.output == "out.json"


def test_parser_requires_both_inputs() -> None:
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["only-one-arg.json"])


def test_main_prints_scores(
    sample_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    scores, pdb = sample_pair
    rc = main([str(scores), str(pdb)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "reactifpTM:" in out
    assert "pairwise reactifpTM (directional)" in out
    assert "pairwise reactifpTM (max of both directions)" in out
    assert "A-B:" in out
    assert "B-A:" in out


def test_main_writes_output_file(
    sample_pair: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scores, pdb = sample_pair
    out = tmp_path / "results.json"
    rc = main([str(scores), str(pdb), "-o", str(out)])
    assert rc == 0
    assert out.exists()

    written = json.loads(out.read_text())
    assert "actifptm" in written
    assert "pairwise_actifptm" in written
    assert "pairwise_actifptm_max" in written


def test_main_threshold_changes_score(
    sample_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Different contact thresholds should generally produce different scores."""
    scores, pdb = sample_pair
    main([str(scores), str(pdb), "-t", "4.0"])
    tight = capsys.readouterr().out
    main([str(scores), str(pdb), "-t", "12.0"])
    loose = capsys.readouterr().out
    # At minimum, the printed text differs (scores or pairwise mix)
    assert tight != loose


def test_console_script_installed() -> None:
    """The `reactifptm` console script should be importable via -m as well."""
    result = subprocess.run(
        [sys.executable, "-m", "reactifptm.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "reactifPTM" in result.stdout or "reactifptm" in result.stdout
