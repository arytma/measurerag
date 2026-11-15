"""The specification: "a second run of crossover.py from cached generations must reproduce
crossover.json byte for byte".

A synthetic slice (40 questions, 400 passages, one or two gold passages each), a
retrieval cache for one dense-like retriever and a generation cache holding an
answer for every prompt the dial needs are written to tmp_path; then
scripts/crossover.py runs on them as a subprocess, several times.

crossover.json depends on the fixture only. stats.json also carries the
stats_tokens families that read the live repository (results/, configs/,
scripts/), which a running benchmark may rewrite; the runs are therefore
repeated (at most three times) until those inputs stayed unchanged across them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from measurerag.crossover.dial import dial_context, dial_uniforms, gold_set
from measurerag.crossover.generate import gen_key
from measurerag.crossover.retrieval import cache_path
from measurerag.datasets.base import Dataset, Doc
from measurerag.hashing import sha256_json
from measurerag.metrics import evaluate_run, exact_match
from measurerag.rag.generator import build_prompt

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "crossover.py"

NAME = "hotpotqa"            # stats_tokens.design() requires cfg["datasets"]["hotpotqa"]
RID = "dense-x"
BASE, BIG = "qwen2.5-1.5b", "qwen2.5-3b"
SEED, MAX_TOKENS, K, N_BOOT = 20260101, 64, 5, 200
LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_Q, N_DOCS, TOP = 40, 400, 100
# P(correct answer | some gold in context), P(correct answer | no gold in context)
P_CORRECT = {BASE: (0.55, 0.10), BIG: (0.80, 0.30)}

# files in the live repository that scripts/stats_tokens.py reads (or that are the code)
LIVE_INPUTS = ["results/*/manifest.json", "results/*/results.json", "results/*/predictions_*.json",
               "results/crossover/*.json", "results/crossover/calibration/*",
               "results/crossover/natural/*/*.json", "results/crossover/superseded/*/*.json",
               "configs/crossover/*.yaml", "scripts/*.py", "src/measurerag/**/*.py"]


def live_fingerprint() -> list[tuple[str, int, int]]:
    out = []
    for pat in LIVE_INPUTS:
        for p in sorted(REPO.glob(pat)):
            if p.is_file():
                st = p.stat()
                out.append((p.relative_to(REPO).as_posix(), st.st_size, st.st_mtime_ns))
    return out


def build_fixture(root: Path) -> dict:
    g = np.random.default_rng(7)
    ids = [f"d{i:03d}" for i in range(N_DOCS)]
    docs = [Doc(doc_id=d, title=f"Title {i}", text=f"Passage {i} on topic {i % 17}, part {i % 5}.")
            for i, d in enumerate(ids)]
    queries, qrels, answers, run, scores = {}, {}, {}, {}, {}
    for i in range(N_Q):
        q = f"q{i:02d}"
        queries[q] = f"Which answer belongs to question {i}?"
        answers[q] = [f"answer {i}"]
        n_gold = 2 if i % 3 == 0 else 1
        gold = [int(x) for x in g.choice(N_DOCS, size=n_gold, replace=False)]
        qrels[q] = {ids[j]: 1 for j in gold}
        others = [int(j) for j in g.permutation(N_DOCS) if int(j) not in gold][: TOP - n_gold]
        ranking = [ids[j] for j in others]
        for j in gold:  # most gold near the top, some below rank k
            pos = int(g.integers(0, 4)) if g.random() < 0.7 else int(g.integers(K, 30))
            ranking.insert(pos, ids[j])
        run[q] = ranking
        scores[q] = [round(1.0 - r / TOP, 6) for r in range(TOP)]

    slice_dir = root / "slice"
    Dataset(name="synthetic", language="en", docs=docs, queries=queries, qrels=qrels, answers=answers,
            license="CC0", source="tests/test_crossover_determinism.py").save(slice_dir)
    ds = Dataset.load_slice(slice_dir)  # build prompts from exactly what crossover.py will load

    cache_dir = root / "cache"
    cache_dir.mkdir()
    cache_path(cache_dir, RID).write_text(json.dumps({
        "retriever_id": RID, "retriever": {"kind": "dense", "model": "fixture", "device": "cpu"},
        "declaration": {"kind": "dense", "model": "fixture"},
        "dataset": ds.name, "corpus_sha256": ds.corpus_hash(), "queries_sha256": ds.query_hash(),
        "top_k": TOP, "build": {"build_seconds": 0.0, "index_bytes": 0}, "wall_seconds": 0.0,
        "metrics": {k: evaluate_run(run, ds.qrels, ks=[10, 100], context_k=k) for k in (3, 5, 10)},
        "run": run, "scores": scores, "run_sha256": sha256_json(run),
    }, indent=1, sort_keys=True), encoding="utf-8")

    # one answer per distinct prompt, the way scripts/dial_generate.py would cache it
    by_id = {d.doc_id: d for d in ds.docs}
    rows: dict[str, dict[str, dict]] = {BASE: {}, BIG: {}}
    expected = {"g_any": {}, "em": {}}
    for p in LEVELS:
        g_any, em = [], {BASE: [], BIG: []}
        for q in sorted(ds.queries):
            gold = gold_set(ds.qrels[q])
            ctx = dial_context(run[q], gold, K, p, dial_uniforms(NAME, q, gold))
            prompt = build_prompt(ds.queries[q], [by_id[d].indexable() for d in ctx if d in by_id], language=ds.language)
            has_gold = bool(gold & set(ctx))
            g_any.append(int(has_gold))
            for gname in (BASE, BIG):
                key = gen_key(gname, MAX_TOKENS, SEED, prompt)
                u = int(key[:12], 16) / 16 ** 12
                text = ds.answers[q][0] if u < P_CORRECT[gname][0 if has_gold else 1] else "unanswerable"
                rows[gname][key] = {"key": key, "text": text, "generator": gname,
                                    "max_tokens": MAX_TOKENS, "seed": SEED}
                em[gname].append(exact_match(rows[gname][key]["text"], ds.answers[q]))
        expected["g_any"][p] = float(np.mean(g_any))
        expected["em"][p] = {gname: float(np.mean(v)) for gname, v in em.items()}
    for gname, r in rows.items():
        with open(cache_dir / f"gen_{gname}.jsonl", "w", encoding="utf-8") as fh:
            for key in sorted(r):
                fh.write(json.dumps(r[key], sort_keys=True) + "\n")
    expected["cache_rows"] = {gname: len(r) for gname, r in rows.items()}

    cfg = {
        "seed": SEED, "n_boot": N_BOOT, "max_tokens": MAX_TOKENS, "generation_cap": 30000,
        "agreement_datasets": [NAME], "base_generator": BASE, "big_generator": BIG,
        "priced_step": ["bm25", RID],
        "datasets": {NAME: {"slice": str(slice_dir), "cache": str(cache_dir),
                            "dial": {"retriever": RID, "k": K, "levels": LEVELS},
                            "ladder": [RID], "harness": []}},
    }
    cfg_path = root / "crossover.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return {"cfg": cfg_path, "expected": expected}


def crossover(cfg: Path, out: Path, stats: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    stats.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(cfg), "--out", str(out), "--stats", str(stats)],
        cwd=out.parent, capture_output=True, text=True, timeout=120,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    assert p.returncode == 0, p.stdout + p.stderr


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("crossover")
    fx = build_fixture(root)
    for attempt in range(3):
        work = root / f"attempt{attempt}"
        before = live_fingerprint()
        crossover(fx["cfg"], work / "a.json", work / "a_stats.json")
        crossover(fx["cfg"], work / "b.json", work / "b_stats.json")
        # the same --out/--stats twice, the first pair set aside
        crossover(fx["cfg"], work / "same" / "crossover.json", work / "same" / "stats.json")
        (work / "first").mkdir()
        for nm in ("crossover.json", "stats.json"):
            shutil.copy(work / "same" / nm, work / "first" / nm)
        crossover(fx["cfg"], work / "same" / "crossover.json", work / "same" / "stats.json")
        if live_fingerprint() == before:
            return {"work": work, **fx}
    pytest.fail("the live repository inputs of stats_tokens kept changing during the runs")


def test_crossover_json_is_byte_identical_across_runs(runs):
    w = runs["work"]
    a = (w / "a.json").read_bytes()
    assert a == (w / "b.json").read_bytes()
    assert a == (w / "same" / "crossover.json").read_bytes() == (w / "first" / "crossover.json").read_bytes()


def test_stats_json_is_byte_identical_for_the_same_out_path(runs):
    w = runs["work"]
    assert (w / "first" / "stats.json").read_bytes() == (w / "same" / "stats.json").read_bytes()


def test_stats_json_differs_between_out_paths_only_in_source(runs):
    w = runs["work"]
    a = json.loads((w / "a_stats.json").read_text(encoding="utf-8"))
    b = json.loads((w / "b_stats.json").read_text(encoding="utf-8"))
    assert a["source"] == str(w / "a.json") and b["source"] == str(w / "b.json")
    assert {**a, "source": None} == {**b, "source": None}


def test_dial_section_has_the_configured_levels(runs):
    res = json.loads((runs["work"] / "a.json").read_text(encoding="utf-8"))
    d = res["datasets"][NAME]
    assert d["dial_missing_answers"] == {BASE: 0, BIG: 0}
    assert d["n_questions"] == N_Q and d["partial"] is False
    assert "status" not in d and "dial_status" not in d
    dial = d["dial"]
    assert dial["retriever"] == RID and dial["k"] == K
    assert [lv["p"] for lv in dial["levels"]] == LEVELS
    assert dial["levels"][0]["delta_ret"] is None and dial["levels"][0]["ret_minus_gen"] is None
    assert all(lv["delta_ret"] is not None for lv in dial["levels"][1:])
    assert [r["p"] for r in dial["price_table"]] == LEVELS[1:]
    assert dial["roots"]["raw"]["isotonic"]["n_boot"] == N_BOOT


def test_dial_reproduces_the_fixture(runs):
    """Every level's g and EM equal what the fixture planted, and the levels are nested."""
    res = json.loads((runs["work"] / "a.json").read_text(encoding="utf-8"))
    exp = runs["expected"]
    levels = res["datasets"][NAME]["dial"]["levels"]
    g = [lv["g_any"]["point"] for lv in levels]
    assert g == pytest.approx([exp["g_any"][p] for p in LEVELS], abs=1e-6)
    assert g[-1] == 0.0 and 0.0 < g[0] < 1.0
    assert all(x >= y for x, y in zip(g, g[1:]))
    for lv in levels:
        for gname in (BASE, BIG):
            assert lv["em"][gname]["point"] == pytest.approx(exp["em"][lv["p"]][gname], abs=1e-6)
    # the curves are not degenerate: the big generator helps somewhere, and gold matters
    assert any(lv["delta_gen"]["point"] != 0 for lv in levels)
    assert levels[0]["em"][BASE]["point"] > levels[-1]["em"][BASE]["point"]
    assert res["datasets"][NAME]["generations"]["dial_cache_rows"] == exp["cache_rows"]


def test_stats_tokens_come_from_the_fixture(runs):
    tok = json.loads((runs["work"] / "a_stats.json").read_text(encoding="utf-8"))["tokens"]
    assert tok["n_boot"] == str(N_BOOT)
    assert tok[f"{NAME}.n"] == str(N_Q)
    assert tok[f"{NAME}.dial_retriever"] == RID
    assert tok[f"{NAME}.status"] == "complete"
    assert tok[f"data.{NAME}.docs"] == f"{N_DOCS:,}"
    assert tok["gens.dial_cache"] == f"{sum(runs['expected']['cache_rows'].values()):,}"
    assert tok[f"headline.{NAME}.gstar_raw"].startswith(("g* = ", "no crossing inside g = "))
    assert tok[f"headline.{NAME}.gstar_gb"] == "not measured"
