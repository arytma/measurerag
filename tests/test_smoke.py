"""The end-to-end guarantee: `measurerag smoke` works offline.

This is the test that protects the README's promise of a first run in seconds. It runs the real
runner, real subprocess isolation, real outputs - only the data is the bundled
fixture. Nothing here touches the network.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    from measurerag.config import load_config
    from measurerag.runner import run

    out = tmp_path_factory.mktemp("smoke")
    cfg = load_config(REPO / "configs" / "smoke.yaml")
    return run(cfg, config_path="configs/smoke.yaml", out_dir=out, log=lambda _: None)


def test_every_configuration_completes(smoke_run):
    failed = [r for r in smoke_run["results"] if r.get("status") != "ok"]
    assert not failed, [f.get("error") for f in failed]


def test_all_the_promised_files_are_written(smoke_run):
    out = Path(smoke_run["out_dir"])
    for name in (
        "results.csv",
        "results.json",
        "results_table.txt",
        "manifest.json",
        "curve_quality_vs_memory.png",
        "curve_quality_vs_latency.png",
    ):
        assert (out / name).exists(), name
        assert (out / name).stat().st_size > 0, name


def test_the_exact_slice_used_is_saved_beside_the_results(smoke_run):
    slice_dir = Path(smoke_run["out_dir"]) / "slice"
    for name in ("corpus.jsonl", "queries.jsonl", "qrels.jsonl", "meta.json"):
        assert (slice_dir / name).exists()


def test_metrics_are_in_range_and_not_degenerate(smoke_run):
    for r in smoke_run["results"]:
        n = r["metrics"]["ndcg@10"]
        assert 0.0 <= n <= 1.0
        # a fixture where everything scores 1.0 teaches a newcomer nothing
        assert n < 1.0
        assert r["metrics"]["recall@100"] >= r["metrics"]["recall@10"]


def test_every_configuration_gets_a_verdict_for_every_budget(smoke_run):
    n_budgets = len(smoke_run["budgets"])
    for r in smoke_run["results"]:
        assert len(r["verdicts"]) == n_budgets


def test_cost_is_actually_measured_not_left_at_zero(smoke_run):
    for r in smoke_run["results"]:
        m = r["measurement"]
        assert m["peak_rss_gb"] > 0
        assert m["retrieval_ms_p95"] >= m["retrieval_ms_p50"] >= 0


def test_configurations_are_isolated_from_each_other(smoke_run):
    # in one shared process the later configurations would inherit the earlier
    # peak; each here should be near its own true footprint, not monotonic
    peaks = [r["measurement"]["peak_rss_gb"] for r in smoke_run["results"]]
    assert min(peaks) < 1.0


def test_the_manifest_fingerprints_the_data_and_the_outputs(smoke_run):
    manifest = smoke_run["manifest"]
    assert len(manifest["config_sha256"]) == 64
    assert len(manifest["results_stable_sha256"]) == 64
    ds = manifest["datasets"][0]
    for key in ("corpus_sha256", "queries_sha256", "qrels_sha256"):
        assert len(ds[key]) == 64
    assert "results.csv" in manifest["files"]


def test_the_machine_is_recorded_with_the_numbers(smoke_run):
    machine = smoke_run["manifest"]["machine"]
    assert machine["ram_gb"] > 0
    assert machine["chip"]
    assert machine["os_version"]


def test_the_table_prints_a_verdict_column_per_budget(smoke_run):
    table = smoke_run["table"]
    for b in smoke_run["budgets"]:
        assert b["name"] in table


def test_the_dataset_notes_travel_into_the_results(smoke_run):
    # a reader must not be able to reach a number without its caveats
    notes = smoke_run["datasets"][0]["notes"]
    assert any("synthetic" in n.lower() for n in notes)


def test_smoke_is_reproducible_within_a_session(tmp_path_factory):
    from measurerag.config import load_config
    from measurerag.manifest import compare_manifests
    from measurerag.runner import run

    cfg = load_config(REPO / "configs" / "smoke.yaml")
    a = run(cfg, out_dir=tmp_path_factory.mktemp("a"), log=lambda _: None)
    b = run(load_config(REPO / "configs" / "smoke.yaml"),
            out_dir=tmp_path_factory.mktemp("b"), log=lambda _: None)
    cmp = compare_manifests(a["manifest"], b["manifest"])
    assert cmp["metrics_identical"], cmp["differences"][:5]


def test_the_cli_smoke_command_exits_zero_with_no_network(tmp_path):
    env = dict(os.environ)
    # HF_HUB_OFFLINE makes any accidental download attempt fail loudly instead
    # of quietly working on a machine that happens to have a connection
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "measurerag.cli", "smoke", "--out", str(tmp_path / "smoke")],
        cwd=REPO, capture_output=True, text=True, timeout=600, env=env,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "smoke: OK" in proc.stdout
