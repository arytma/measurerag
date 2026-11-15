"""scripts/fill.py --all and --check against a fake repository (--root)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
FILL = REPO / "scripts" / "fill.py"

TOKENS = {"em": "0.412", "n": "200", "gstar": "0.46 [0.30, 0.61]"}
TEMPLATES = {
    "README.md": "# Demo\n\nEM was {{em}} on {{n}} questions.\n",
    "docs/CROSSOVER.md": "# Crossover\n\ng* = {{ gstar }} (n = {{n}}).\n",
}


def header(rel: str) -> str:
    return (f"<!-- Generated from templates/{rel} and results/stats.json by scripts/fill.py. "
            "Edit the template, not this file. -->\n")


def make_repo(root: Path, templates: dict[str, str] = TEMPLATES, tokens: dict[str, str] = TOKENS) -> Path:
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "docs.yaml").write_text(
        yaml.safe_dump({"stats": "results/stats.json", "templates": "templates", "documents": sorted(templates)}),
        encoding="utf-8")
    (root / "results").mkdir()
    (root / "results" / "stats.json").write_text(json.dumps({"source": "x", "tokens": tokens}), encoding="utf-8")
    for rel, text in templates.items():
        p = root / "templates" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def fill(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(FILL), *args, "--root", str(root)],
                          capture_output=True, text=True, timeout=60)


def test_all_writes_generated_files_with_the_header(tmp_path):
    root = make_repo(tmp_path)
    p = fill(root, "--all")
    assert p.returncode == 0, p.stderr
    assert (root / "README.md").read_text(encoding="utf-8") == header("README.md") + "# Demo\n\nEM was 0.412 on 200 questions.\n"
    assert (root / "docs" / "CROSSOVER.md").read_text(encoding="utf-8") == (
        header("docs/CROSSOVER.md") + "# Crossover\n\ng* = 0.46 [0.30, 0.61] (n = 200).\n")
    assert "fill: wrote README.md" in p.stdout and "fill: wrote docs/CROSSOVER.md" in p.stdout


def test_all_is_idempotent(tmp_path):
    root = make_repo(tmp_path)
    assert fill(root, "--all").returncode == 0
    p = fill(root, "--all")
    assert p.returncode == 0 and "wrote" not in p.stdout


def test_check_passes_right_after_all(tmp_path):
    root = make_repo(tmp_path)
    assert fill(root, "--all").returncode == 0
    p = fill(root, "--check")
    assert p.returncode == 0, p.stderr
    assert p.stderr == ""


def test_check_fails_before_anything_was_generated(tmp_path):
    root = make_repo(tmp_path)
    p = fill(root, "--check")
    assert p.returncode == 1
    assert not (root / "README.md").exists()  # --check never writes


def test_check_fails_after_a_generated_file_is_edited(tmp_path):
    root = make_repo(tmp_path)
    assert fill(root, "--all").returncode == 0
    readme = root / "README.md"
    edited = readme.read_text(encoding="utf-8").replace("0.412", "0.413")
    readme.write_text(edited, encoding="utf-8")
    p = fill(root, "--check")
    assert p.returncode == 1
    assert "README.md is stale or edited by hand" in p.stderr
    assert "CROSSOVER" not in p.stderr
    assert readme.read_text(encoding="utf-8") == edited  # --check leaves it alone
    # --all repairs it, and only it
    p = fill(root, "--all")
    assert p.returncode == 0 and p.stdout.strip() == "fill: wrote README.md"
    assert fill(root, "--check").returncode == 0


def test_check_fails_after_the_stats_change(tmp_path):
    root = make_repo(tmp_path)
    assert fill(root, "--all").returncode == 0
    (root / "results" / "stats.json").write_text(json.dumps({"tokens": {**TOKENS, "em": "0.500"}}), encoding="utf-8")
    p = fill(root, "--check")
    assert p.returncode == 1 and "README.md is stale" in p.stderr


def test_all_fails_and_writes_nothing_when_a_token_is_missing(tmp_path):
    templates = {**TEMPLATES, "docs/CROSSOVER.md": TEMPLATES["docs/CROSSOVER.md"] + "And {{nowhere}}.\n"}
    root = make_repo(tmp_path, templates=templates)
    p = fill(root, "--all")
    assert p.returncode == 1
    assert "templates/docs/CROSSOVER.md: 1 token(s) missing from stats: nowhere" in p.stderr
    # not even the template that could be filled is written
    assert not (root / "README.md").exists()
    assert not (root / "docs" / "CROSSOVER.md").exists()
    assert fill(root, "--check").returncode == 1


def test_all_does_not_overwrite_when_a_token_goes_missing(tmp_path):
    root = make_repo(tmp_path)
    assert fill(root, "--all").returncode == 0
    before = (root / "README.md").read_text(encoding="utf-8")
    (root / "results" / "stats.json").write_text(json.dumps({"tokens": {"n": "200", "gstar": "x"}}), encoding="utf-8")
    p = fill(root, "--all")
    assert p.returncode == 1 and "em" in p.stderr
    assert (root / "README.md").read_text(encoding="utf-8") == before
