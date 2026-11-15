"""The environment check that exists because the first unattended run segfaulted.

Three wheels (torch, faiss-cpu, scikit-learn) each vendor a libomp.dylib. Two of
them loaded into one process crashed the Czech retrieval stage four times inside
__kmp_fork_barrier. These tests work on a fake site-packages tree so they are
fast and give the same answer on a machine that happens to be healthy.
"""

from __future__ import annotations

import platform

import pytest

from measurerag import openmp


def _fake_tree(root, packages, canonical_bytes=b"OMP"):
    """Build a site-packages-shaped directory with one libomp per package."""
    paths = {}
    for pkg, rel in packages.items():
        p = root / pkg / rel / "libomp.dylib"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(canonical_bytes + pkg.encode())
        paths[pkg] = p
    return paths


LAYOUT = {"torch": "lib", "faiss": ".dylibs", "sklearn": ".dylibs"}


def test_finds_every_vendored_copy(tmp_path):
    _fake_tree(tmp_path, LAYOUT)
    found = openmp.find_runtimes(tmp_path)
    assert {r.package for r in found} == {"torch", "faiss", "sklearn"}


def test_three_separate_files_are_three_images(tmp_path):
    _fake_tree(tmp_path, LAYOUT)
    assert len(openmp.distinct_images(openmp.find_runtimes(tmp_path))) == 3


def test_conflict_is_reported_on_macos(tmp_path, monkeypatch):
    _fake_tree(tmp_path, LAYOUT)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    assert openmp.check(tmp_path).conflicted


def test_not_flagged_off_darwin(tmp_path, monkeypatch):
    # the vendored-copy collision is a dyld behaviour; don't cry wolf on Linux
    _fake_tree(tmp_path, LAYOUT)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert not openmp.check(tmp_path).conflicted


def test_unify_dry_run_changes_nothing(tmp_path):
    paths = _fake_tree(tmp_path, LAYOUT)
    actions = openmp.unify(tmp_path, apply=False)
    assert len(actions) == 2                       # faiss and sklearn move
    assert not any(p.is_symlink() for p in paths.values())


def test_unify_collapses_to_one_image_and_keeps_backups(tmp_path):
    paths = _fake_tree(tmp_path, LAYOUT)
    openmp.unify(tmp_path, apply=True)

    assert len(openmp.distinct_images(openmp.find_runtimes(tmp_path))) == 1
    assert not paths["torch"].is_symlink()          # torch stays the real file
    for pkg in ("faiss", "sklearn"):
        assert paths[pkg].is_symlink()
        assert paths[pkg].resolve() == paths["torch"]
        backup = paths[pkg].with_suffix(".dylib.orig-backup")
        assert backup.exists(), "the replaced copy must be recoverable"


def test_unify_is_idempotent(tmp_path):
    openmp_paths = _fake_tree(tmp_path, LAYOUT)
    openmp.unify(tmp_path, apply=True)
    assert openmp.unify(tmp_path, apply=True) == []
    assert openmp_paths["faiss"].resolve() == openmp_paths["torch"]


def test_backups_are_not_mistaken_for_runtimes(tmp_path):
    _fake_tree(tmp_path, LAYOUT)
    openmp.unify(tmp_path, apply=True)
    assert all(not r.path.name.endswith(".orig-backup") for r in openmp.find_runtimes(tmp_path))


def test_preflight_refuses_and_says_how_to_fix(tmp_path, monkeypatch):
    _fake_tree(tmp_path, LAYOUT)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    with pytest.raises(openmp.OpenMPConflict) as exc:
        openmp.preflight(tmp_path)
    assert "doctor --fix" in str(exc.value)


def test_preflight_passes_once_repaired(tmp_path, monkeypatch):
    _fake_tree(tmp_path, LAYOUT)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    openmp.unify(tmp_path, apply=True)
    assert not openmp.preflight(tmp_path).conflicted


def test_this_machine_is_safe_to_measure_on():
    """Not a unit test: a standing guard on the environment the results came from."""
    report = openmp.check()
    assert not report.conflicted, (
        "this environment would segfault mid-run; run 'measurerag doctor --fix'. "
        f"{report.as_dict()}"
    )
