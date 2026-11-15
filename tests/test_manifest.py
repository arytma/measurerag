"""Manifest hashing - the reproducibility claim rests on these."""

from __future__ import annotations


from measurerag.hashing import sha256_bytes, sha256_json, sha256_records
from measurerag.manifest import compare_manifests, strip_volatile


def test_json_hash_ignores_key_order():
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_json_hash_distinguishes_values():
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})


def test_json_hash_distinguishes_types():
    assert sha256_json({"a": 1}) != sha256_json({"a": "1"})


def test_record_hash_is_order_sensitive():
    # a corpus in a different order is a different corpus: it changes tie-breaks
    a = [{"id": "1"}, {"id": "2"}]
    assert sha256_records(a) != sha256_records(list(reversed(a)))


def test_record_hash_handles_unicode_stably():
    assert sha256_records([{"t": "Příliš"}]) == sha256_records([{"t": "Příliš"}])


def test_strip_volatile_removes_timings_at_any_depth():
    obj = {"metrics": {"ndcg@10": 0.5}, "a": {"b": {"wall_seconds": 12, "keep": 1}}}
    assert strip_volatile(obj) == {"metrics": {"ndcg@10": 0.5}, "a": {"b": {"keep": 1}}}


def test_strip_volatile_walks_lists():
    obj = {"results": [{"peak_rss_gb": 1.0, "metrics": {"x": 1}}]}
    assert strip_volatile(obj) == {"results": [{"metrics": {"x": 1}}]}


def _manifest(ndcg: float, seconds: float) -> dict:
    stable = [{"config_id": "bm25", "metrics": {"ndcg@10": ndcg}}]
    return {
        "results_stable": stable,
        "results_stable_sha256": sha256_json(stable),
        "wall_seconds": seconds,
        "created_utc": f"2026-01-0{int(seconds) % 9 + 1}",
        "config": {"seed": 1},
    }


def test_two_runs_differing_only_in_timing_compare_identical():
    cmp = compare_manifests(_manifest(0.5, 1.0), _manifest(0.5, 9.0))
    assert cmp["identical_ignoring_timings"]
    assert cmp["metrics_identical"]
    assert cmp["differences"] == []


def test_two_runs_differing_in_a_metric_do_not():
    cmp = compare_manifests(_manifest(0.5, 1.0), _manifest(0.6, 1.0))
    assert not cmp["metrics_identical"]
    assert not cmp["identical_ignoring_timings"]
    assert any("ndcg@10" in d for d in cmp["differences"])


def test_bytes_hash_is_sha256():
    assert sha256_bytes(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
