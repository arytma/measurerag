"""The release's figures (scripts/figures.py): every value they draw is in its source, and
drawing them again gives the same bytes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import number_audit  # noqa: E402


def test_every_value_a_figure_draws_is_read_again_from_its_source():
    figures, problems = number_audit.audit_figures(REPO)
    assert problems == []
    data = {k: v for k, v in figures.items() if k != "pipeline"}
    assert data and all(v["values_traced"] > 0 for v in data.values())
    assert figures["pipeline"]["values_traced"] == 0           # the diagram draws no data


def test_the_figures_are_drawn_again_byte_for_byte():
    p = subprocess.run([sys.executable, str(REPO / "scripts/figures.py"), "--check"], cwd=REPO, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
