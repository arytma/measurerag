"""Measurement hygiene (Rule 6) without running anything heavy: the calibration
probe's drift, the contention sampler's verdict, the queue's order of work, and
the headline sentence for a dataset with and without a root."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import calibration_probe as cp  # noqa: E402
import sweep_queue as nq  # noqa: E402
import stats_tokens as st  # noqa: E402


# -- calibration probe -----------------------------------------------------------
def test_drift_is_relative_to_the_baseline():
    assert cp.drift(1.10, 1.00) == pytest.approx(0.10)
    assert cp.drift(0.95, 1.00) == pytest.approx(-0.05)
    assert cp.drift(3.0, 2.0) == pytest.approx(0.5)
    assert cp.drift(2.0, 2.0) == 0.0


def test_drift_sign_says_slower_or_faster():
    assert cp.drift(1.2, 1.0) > 0 > cp.drift(0.8, 1.0)


def test_probe_constants_match_the_brief():
    assert cp.N == 20 and cp.TOLERANCE == 0.10
    assert (cp.GEN, cp.K, cp.MAX_TOKENS) == ("qwen2.5-1.5b", 5, 64)


# -- contention sampler ------------------------------------------------------------
def sampler_with(n: int, n_contended: int) -> nq.Sampler:
    s = nq.Sampler(root_pid=1)
    s.samples = [
        {"max": 90.0, "max_comm": f"hog{i % 10}", "sum": 120.0 + i, "contended": True} if i < n_contended
        else {"max": 20.0, "max_comm": "idle", "sum": 40.0, "contended": False}
        for i in range(n)
    ]
    return s


@pytest.mark.parametrize("n, n_contended, contaminated", [
    (0, 0, False),
    (2, 2, False),       # below the floor of 3 even when every sample is contended
    (10, 2, False), (10, 3, True),
    (60, 2, False), (60, 3, True),     # 5% of 60 = 3
    (100, 4, False), (100, 5, True),   # 5% of 100 = 5
    (101, 5, False), (101, 6, True),   # ceil(5.05) = 6
    (200, 9, False), (200, 10, True),
])
def test_contaminated_at_max_of_three_and_five_percent(n, n_contended, contaminated):
    s = sampler_with(n, n_contended).summary()
    assert s["n_samples"] == n and s["n_contended"] == n_contended
    assert s["contaminated"] is contaminated
    assert contaminated == (n_contended >= max(3, math.ceil(0.05 * n)))


def test_summary_reports_the_worst_sample_and_contending_commands():
    s = sampler_with(40, 12).summary()
    assert s["worst_foreign"] == {"max": 90.0, "max_comm": "hog1", "sum": 131.0}
    assert s["contended_comms"] == [f"hog{i}" for i in range(8)]  # sorted, unique, at most 8


def test_summary_of_no_samples():
    s = nq.Sampler(root_pid=1).summary()
    assert s == {"n_samples": 0, "n_contended": 0, "contaminated": False, "worst_foreign": None,
                 "contended_comms": []}


def test_sampler_counts_only_processes_outside_the_harness(monkeypatch):
    """run() with a fake `ps`: the harness tree, the queue itself and `ps` never count."""
    me, harness, child, grandchild = os.getpid(), 50_000, 50_001, 50_002
    own = [(me, 1, 100.0, 5.0, "python"), (harness, me, 400.0, 10.0, "measurerag"),
           (child, harness, 300.0, 10.0, "python3.12"), (grandchild, child, 300.0, 10.0, "llama"),
           (60_000, 1, 99.0, 1.0, "ps")]
    snapshots = [
        own + [(70_000, 1, 85.0, 1.0, "mds_stores")],                                   # one hog >= 80%
        own + [(70_001, 1, 50.0, 1.0, "a"), (70_002, 1, 60.0, 1.0, "b"), (70_003, 1, 45.0, 1.0, "c")],  # sum >= 150%
        own + [(70_004, 1, 79.9, 1.0, "a"), (70_005, 1, 70.0, 1.0, "b")],               # 149.9%: quiet
    ]
    s = nq.Sampler(harness)
    calls = iter(range(len(snapshots)))

    def fake_ps():
        i = next(calls)
        if i == len(snapshots) - 1:
            s.stop.set()
        return snapshots[i]

    monkeypatch.setattr(nq, "ps_table", fake_ps)
    monkeypatch.setattr(nq, "SAMPLE_S", 0)
    s.run()  # synchronously, not as a thread
    assert [x["contended"] for x in s.samples] == [True, True, False]
    assert [x["max_comm"] for x in s.samples] == ["mds_stores", "b", "a"]
    assert [x["sum"] for x in s.samples] == [85.0, 155.0, 149.9]
    assert s.summary()["contaminated"] is False  # 2 contended < 3
    assert not s.stuck


# -- the queue's order of work -------------------------------------------------------
def _is_bge_m3(stage: dict) -> bool:
    if stage["kind"] == "timed":
        return stage["config_id"].startswith("bge-m3+")
    return stage["spec"].split(":")[0] == "bge-m3"


def _is_nq(stage: dict) -> bool:
    return stage.get("config") == nq.NQ or stage.get("dataset") == "nq"


def test_stage_ids_are_unique_and_steps_never_go_back():
    ids = [s["id"] for s in nq.STAGES]
    assert len(ids) == len(set(ids))
    steps = [s["step"] for s in nq.STAGES]
    assert steps == sorted(steps)


def test_hotpotqa_k5_comes_before_nq():
    # bge-m3 is step 5, after both natural sweeps, so it is outside this rule
    first_nq = next(i for i, s in enumerate(nq.STAGES) if _is_nq(s))
    k5 = [i for i, s in enumerate(nq.STAGES) if s.get("config") == nq.K5 and not _is_bge_m3(s)]
    assert len(k5) == 8  # 4 retrievers x 2 generators
    assert max(k5) < first_nq


def test_every_non_bge_m3_stage_comes_before_the_first_bge_m3_stage():
    first = next(i for i, s in enumerate(nq.STAGES) if _is_bge_m3(s))
    assert all(_is_bge_m3(s) for s in nq.STAGES[first:])
    assert sum(not _is_bge_m3(s) for s in nq.STAGES) == first


def test_a_ranking_stage_comes_after_the_batch_whose_index_it_reuses():
    for i, s in enumerate(nq.STAGES):
        if s["kind"] != "ranking":
            continue
        builder = next(j for j, t in enumerate(nq.STAGES)
                       if t["kind"] == "timed" and t["config"] == s["reuse_cfg"] and t["config_id"] == s["reuse_cid"])
        assert builder < i, s["id"]


# -- the headline sentence -------------------------------------------------------------
LEVELS_G = [0.912, 0.731, 0.552, 0.384, 0.196, 0.0]   # p = 0 has no retriever step


def result_with(raw_iso: dict, per_gb_iso: dict | None = None) -> dict:
    levels = [{"p": round(0.2 * i, 1), "g_any": {"point": g},
               "ret_minus_gen": None if i == 0 else {"point": 0.01, "lo": -0.05, "hi": 0.07, "established": False}}
              for i, g in enumerate(LEVELS_G)]
    per_unit = {} if per_gb_iso is None else {"per_gb|bm25->e5-small": {"isotonic": per_gb_iso}}
    return {"config": {"priced_step": ["bm25", "e5-small"]},
            "datasets": {"hotpotqa": {"dial": {"levels": levels, "roots": {"raw": {"isotonic": raw_iso},
                                                                            "per_unit": per_unit}}},
                         "czech": {"status": "no slice"}}}


def no_root(above: int, below: int) -> dict:
    return {"point": None, "lo": 0.0, "hi": 0.73, "n_boot": 200, "n_no_root": above + below,
            "no_root_above": above, "no_root_below": below}


def test_headline_without_a_root_censored_above():
    t = st.headline(result_with(no_root(150, 30)), {})
    assert t["headline.hotpotqa.gstar_raw"] == (
        "no crossing inside g = 0.00 to 0.73 (censored above 0.73; 180 of 200 resamples have no root)")
    assert t["headline.hotpotqa.gstar_gb"] == "not measured"
    assert not any(k.startswith("headline.czech") for k in t)


def test_headline_without_a_root_censored_below():
    t = st.headline(result_with(no_root(10, 170)), {})
    assert t["headline.hotpotqa.gstar_raw"] == (
        "no crossing inside g = 0.00 to 0.73 (censored below 0.00; 180 of 200 resamples have no root)")


def test_headline_without_a_root_ties_go_above():
    t = st.headline(result_with(no_root(5, 5)), {})
    assert "(censored above 0.73;" in t["headline.hotpotqa.gstar_raw"]


def test_headline_with_a_root():
    root = {"point": 0.456, "lo": 0.301, "hi": 0.612, "n_boot": 200, "n_no_root": 3,
            "no_root_above": 2, "no_root_below": 1}
    t = st.headline(result_with(root, per_gb_iso=dict(root, point=0.5, lo=0.25, hi=0.75)), {})
    # the censored count travels with the root
    assert t["headline.hotpotqa.gstar_raw"] == "g* = 0.46 [0.30, 0.61]; 3 of 200 resamples have no root"
    assert t["headline.hotpotqa.gstar_gb"] == "g* = 0.50 [0.25, 0.75]; 3 of 200 resamples have no root"


def test_headline_says_when_a_crossing_rests_on_the_no_gold_level():
    root = {"point": 0.1, "lo": 0.0, "hi": 0.78, "n_boot": 200, "n_no_root": 0,
            "no_root_above": 0, "no_root_below": 0}
    res = result_with(root)
    lv = res["datasets"]["hotpotqa"]["dial"]["levels"]
    for x in lv[1:]:
        x["ret_minus_gen"] = {"point": -0.03, "lo": -0.08, "hi": 0.02, "established": False}
    lv[-1]["g_any"]["point"] = 0.0                       # the no-gold level
    lv[-1]["ret_minus_gen"] = {"point": 0.02, "lo": -0.02, "hi": 0.06, "established": False}
    t = st.headline(res, {})
    assert t["headline.hotpotqa.gstar_raw"].endswith("it rests on the no-gold level alone")


def test_headline_skips_a_dataset_without_a_dial():
    assert st.headline({"config": {"priced_step": ["bm25", "e5-small"]},
                        "datasets": {"nq": {"status": "no data"}}}, {}) == {}
