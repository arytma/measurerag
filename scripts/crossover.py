"""Locate the retriever/generator crossover, with intervals. -> results/crossover.json

    uv run python scripts/crossover.py                  # everything that is complete
    uv run python scripts/crossover.py --allow-partial  # a progress look, never committed

Inputs are caches only - the retrieval rankings, the dial's generation cache and
the harness's prediction files - so a second run reproduces crossover.json byte
for byte (checked by scripts/repro_check.py and tests/test_crossover_determinism.py). Definitions are in docs/CROSSOVER.md.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from measurerag.crossover.dial import dial_context, dial_uniforms, gold_counts, gold_set  # noqa: E402
from measurerag.crossover.generate import GenCache, gen_key  # noqa: E402
from measurerag.crossover.retrieval import cache_path, load_run  # noqa: E402
from measurerag.czech_store import dial_inputs  # noqa: E402
from measurerag.crossover.stats import (  # noqa: E402
    bootstrap_counts,
    break_even_gain,
    established,
    interval,
    linear_fit,
    resampled_means,
    root_with_interval,
)
from measurerag.datasets.base import Dataset  # noqa: E402
from measurerag.datasets.nq import contains_answer  # noqa: E402
from measurerag.metrics import exact_match, normalize_answer, token_f1  # noqa: E402
from measurerag.rag.generator import build_prompt  # noqa: E402

GEN_SHORT = {"qwen2.5-1.5b": "1.5B", "qwen2.5-3b": "3B", "ministral-3-3b": "3B", "ministral-3-8b": "8B"}


def clean(obj):
    """Round floats to 6 places and turn NaN into null, recursively."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else round(obj, 6)
    if isinstance(obj, (np.floating,)):
        return clean(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


class Rows:
    """Named per-question outcome vectors, resampled together."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.names: list[str] = []
        self.data: list[np.ndarray] = []

    def add(self, name: str, values) -> None:
        v = np.asarray(values, dtype=np.int64)
        assert v.shape == (self.n,), (name, v.shape)
        self.names.append(name)
        self.data.append(v)

    def has(self, name: str) -> bool:
        return name in self.names

    def resample(self, counts: np.ndarray) -> None:
        x = np.vstack(self.data)
        self.point = {nm: float(x[i].mean()) for i, nm in enumerate(self.names)}
        boot = resampled_means(x, counts)
        self.boot = {nm: boot[:, i] for i, nm in enumerate(self.names)}


def ci(point: float, boot: np.ndarray) -> dict:
    lo, hi = interval(boot)
    return {"point": point, "lo": lo, "hi": hi, "established": established(lo, hi)}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_harness(dirs: list[str], ds: Dataset, corpus_sha: str, peaks: dict | None = None) -> dict:
    """`peaks`, {(retriever, generator): GB}, replaces those rows' peak RSS (the estimate without the double-counted
    mapped pages, mmap_estimate below); without it every row is read as measured."""
    found: dict[tuple[str, str, int], dict] = {}
    for d in dirs:
        rj = REPO / d / "results.json"
        if not rj.exists():
            continue
        payload = json.loads(rj.read_text(encoding="utf-8"))
        if payload["datasets"][0]["corpus_sha256"] != corpus_sha:
            raise SystemExit(f"{d}: corpus hash differs from the slice; refusing to mix slices")
        k = int(payload["config"]["retrieval"]["context_k"])
        for r in payload["results"]:
            if r.get("status") != "ok" or r.get("rag_status") != "ok":
                continue
            key = (r["retriever_id"], r["generator"]["key"], k)
            if key in found:
                continue
            pf = REPO / d / f"predictions_{r['config_id'].replace('/', '_')}.json"
            if not pf.exists():
                continue
            m = r["measurement"]
            found[key] = {
                "preds": json.loads(pf.read_text(encoding="utf-8")),
                "source": d,
                "max_tokens": int(payload["config"]["rag"]["max_tokens"]),
                "n_answered": int(r["answers"]["n_answered"]),
                "harness_gold_in_context": r["metrics"].get(f"gold_in_context@{k}"),
                "budget": {
                    "peak_rss_gb": (peaks or {}).get((r["retriever_id"], r["generator"]["key"]), m["peak_rss_gb"]),
                    "index_bytes": m["index_bytes"],
                    "model_bytes": m["model_bytes"],
                    "retrieval_ms_p50": m["retrieval_ms_p50"],
                    "e2e_s_p50": m["e2e_s_p50"],
                    "e2e_s_p95": m["e2e_s_p95"],
                },
            }
    return found


def analyse_dataset(name: str, dcfg: dict, cfg: dict, allow_partial: bool, peaks: dict | None = None) -> dict:
    base, big = cfg["base_generator"], cfg["big_generator"]
    gens = (base, big)
    seed, mt = int(cfg["seed"]), int(cfg["max_tokens"])
    dial = dcfg["dial"]
    drid, dk, levels = dial["retriever"], int(dial["k"]), [float(p) for p in dial["levels"]]
    dial_text: dict = {}
    leak: dict = {}
    missing = {g: 0 for g in gens}
    if dcfg.get("records"):
        # the Czech block reads the records that ship, so a reader can recompute it
        rec = dial_inputs(REPO / dcfg["records"], dcfg["set"], dcfg["ladder"], levels, dcfg.get("leak_field", "leak"), dcfg.get("qids"), dcfg.get("em_field", "em"))
        ds = Dataset(name=dcfg.get("dataset_name", rec.name), language="cs", docs=[], queries={}, qrels={}, answers={}, license="", source="")
        corpus_sha = rec.corpus_sha256
        qids = sorted(rec.qids)
        out: dict = {"dataset": ds.name, "corpus_sha256": corpus_sha, "queries_sha256": rec.queries_sha256}
        runs = {rid: {} for rid in rec.retrieval}
        out["retrieval"] = rec.retrieval
        caches = None
        distinct = {g: len({(q, tuple(c)) for (_, q, h), c in rec.context.items() if h == g}) for g in gens}
        complete = set(qids)
        for p in levels:
            for q in qids:
                gold = rec.gold[q]
                leak[(p, q)] = rec.leak.get((p, q), 0)
                for g in gens:
                    if (p, q, g) not in rec.em:
                        missing[g] += 1
                        complete.discard(q)
                        dial_text[(p, q, g)] = (None, 0, 0)
                        continue
                    ga, gl = gold_counts(rec.context[(p, q, g)], gold)
                    dial_text[(p, q, g)] = (None, ga, gl)
        em_of = lambda p, q, g: rec.em[(p, q, g)]         # noqa: E731
        f1_of = lambda p, q, g: rec.f1[(p, q, g)]         # noqa: E731
    else:
        rec = None
        ds = Dataset.load_slice(REPO / dcfg["slice"])
        corpus_sha = ds.corpus_hash()
        cache_dir = REPO / dcfg["cache"]
        by_id = {d.doc_id: d for d in ds.docs}
        qids = sorted(q for q in ds.answers if gold_set(ds.qrels.get(q, {})))
        out = {"dataset": ds.name, "corpus_sha256": corpus_sha, "queries_sha256": ds.query_hash()}

        runs = {}
        for rid in dcfg["ladder"]:
            if cache_path(cache_dir, rid).exists():
                r = load_run(cache_dir, rid)
                if r["corpus_sha256"] != corpus_sha:
                    raise SystemExit(f"{name}/{rid}: retrieval cache is for another corpus")
                runs[rid] = r
        out["retrieval"] = {
            rid: {"ndcg@10": r["metrics"]["5"]["ndcg@10"] if "5" in r["metrics"] else r["metrics"][5]["ndcg@10"],
                  "run_sha256": r["run_sha256"]}
            for rid, r in runs.items()
        }
        caches = {g: GenCache(cache_dir / f"gen_{g}.jsonl") for g in gens}
        distinct = None
        complete = set(qids)
        em_of = lambda p, q, g: exact_match(dial_text[(p, q, g)][0], ds.answers[q])      # noqa: E731
        f1_of = lambda p, q, g: token_f1(dial_text[(p, q, g)][0], ds.answers[q])         # noqa: E731

    # ---- dial ------------------------------------------------------------
    if rec is None and drid in runs:
        run = runs[drid]["run"]
        for p in levels:
            for q in qids:
                gold = gold_set(ds.qrels[q])
                ctx = dial_context(run[q], gold, dk, p, dial_uniforms(name, q, gold))
                prompt = build_prompt(ds.queries[q], [by_id[d].indexable() for d in ctx if d in by_id], language=ds.language)
                ga, gl = gold_counts(ctx, gold)
                # yes/no answers are not extractable from a passage; skip them
                ans = [a for a in ds.answers[q] if normalize_answer(a) not in ("yes", "no")]
                leak[(p, q)] = int(bool(ans) and any(
                    contains_answer(by_id[d].indexable()[:1200], ans) for d in ctx if d in by_id and d not in gold))
                for g in gens:
                    row = caches[g].get(gen_key(g, mt, seed, prompt))
                    if row is None:
                        missing[g] += 1
                        complete.discard(q)
                    dial_text[(p, q, g)] = (row["text"] if row else None, ga, gl)
    dial_ok = drid in runs and (not any(missing.values()) or allow_partial)
    out["dial_missing_answers"] = missing
    if drid in runs and any(missing.values()) and not allow_partial:
        out["dial_status"] = "incomplete"
    use = sorted(complete) if (allow_partial and any(missing.values())) else qids
    n = len(use)
    out["n_questions"] = n
    out["partial"] = bool(allow_partial and any(missing.values()))

    rows = Rows(n)
    if dial_ok and n:
        for p in levels:
            rows.add(f"dial|g_any|{p}", [dial_text[(p, q, base)][1] for q in use])
            rows.add(f"dial|g_all|{p}", [dial_text[(p, q, base)][2] for q in use])
            rows.add(f"dial|leak|{p}", [leak[(p, q)] for q in use])
            for g in gens:
                rows.add(f"dial|em|{g}|{p}", [em_of(p, q, g) for q in use])

    # ---- natural sweep, from the harness ----------------------------------
    harness = load_harness(dcfg.get("harness", []), ds, corpus_sha, peaks)
    ks = sorted({k for (_, _, k) in harness})
    natural_f1: dict = {}
    for (rid, g, k), h in sorted(harness.items()):
        if rid not in runs:
            continue
        if any(q not in h["preds"] for q in use):
            continue
        rows.add(f"nat|em|{rid}|{g}|{k}", [exact_match(h["preds"][q], ds.answers[q]) for q in use])
        natural_f1[(rid, g, k)] = float(np.mean([token_f1(h["preds"][q], ds.answers[q]) for q in use]))
        for kind, idx in (("g_any", 0), ("g_all", 1)):
            nm = f"nat|{kind}|{rid}|{k}"
            if not rows.has(nm):
                rows.add(nm, [gold_counts(runs[rid]["run"][q][:k], gold_set(ds.qrels[q]))[idx] for q in use])
    if not rows.names or n == 0:
        out["status"] = "no data"
        return out

    counts = bootstrap_counts(n, int(cfg["n_boot"]), "crossover", name, n)
    rows.resample(counts)
    P, B = rows.point, rows.boot

    # ---- costs: the harness at the dial's k ---------------------------------
    def budget(rid, g, k=dk):
        h = harness.get((rid, g, k))
        return h["budget"] if h else None

    costs: dict = {"memory_unit": "GB of peak resident set size (whole process: retriever, index, generator)",
                   "latency_unit": "ms of added end-to-end p50 latency", "k": dk}
    gen_at = drid
    bb, bg = budget(drid, base), budget(drid, big)
    if not (bb and bg):
        # the generator step's memory cost barely depends on the retriever in front
        # of it (run 1: +1.81 / +1.82 / +1.77 GB), so price it at the priced
        # step's upper rung until the dial retriever's own harness run exists
        gen_at = cfg["priced_step"][1]
        bb, bg = budget(gen_at, base), budget(gen_at, big)
    # "ms" is the harness's end-to-end p50, which starts after retrieval (prompt
    # build + generation); "ms_total" adds the retrieval p50, so it prices the
    # whole wait a user sees (a sum of p50s)
    costs["generator_step"] = (
        {"from": base, "to": big, "at_retriever": gen_at, "fallback": gen_at != drid,
         "gb": bg["peak_rss_gb"] - bb["peak_rss_gb"], "ms": 1000 * (bg["e2e_s_p50"] - bb["e2e_s_p50"]),
         "ms_total": 1000 * (bg["e2e_s_p50"] - bb["e2e_s_p50"]) + bg["retrieval_ms_p50"] - bb["retrieval_ms_p50"]}
        if bb and bg else None
    )
    steps = []
    ladder = [r for r in dcfg["ladder"]]
    for lo, hi in zip(ladder, ladder[1:]):
        a, b = budget(lo, base), budget(hi, base)
        if a and b:
            steps.append({
                "from": lo, "to": hi, "gb": b["peak_rss_gb"] - a["peak_rss_gb"],
                "ms": 1000 * (b["e2e_s_p50"] - a["e2e_s_p50"]),
                "retrieval_ms": b["retrieval_ms_p50"] - a["retrieval_ms_p50"],
                "ms_total": 1000 * (b["e2e_s_p50"] - a["e2e_s_p50"]) + b["retrieval_ms_p50"] - a["retrieval_ms_p50"],
            })
    costs["retriever_steps"] = steps
    priced = next((s for s in steps if [s["from"], s["to"]] == list(cfg["priced_step"])), None)
    costs["priced_retriever_step"] = priced
    out["costs"] = costs
    out["budget"] = {f"{rid}+{GEN_SHORT.get(g, g)}@k{k}": {**h["budget"], "source": h["source"], "max_tokens": h["max_tokens"]}
                     for (rid, g, k), h in sorted(harness.items())}

    # ---- the dial analysis --------------------------------------------------
    if dial_ok and n:
        L = len(levels)
        gpt = np.array([P[f"dial|g_any|{p}"] for p in levels])
        gall_pt = np.array([P[f"dial|g_all|{p}"] for p in levels])
        gb_ = np.column_stack([B[f"dial|g_any|{p}"] for p in levels])
        gall_b = np.column_stack([B[f"dial|g_all|{p}"] for p in levels])
        em = {g: (np.array([P[f"dial|em|{g}|{p}"] for p in levels]),
                  np.column_stack([B[f"dial|em|{g}|{p}"] for p in levels])) for g in gens}
        dgen_pt, dgen_b = em[big][0] - em[base][0], em[big][1] - em[base][1]
        dret_pt = np.full(L, np.nan)
        dret_b = np.full((counts.shape[0], L), np.nan)
        dret_pt[1:] = em[base][0][:-1] - em[base][0][1:]
        dret_b[:, 1:] = em[base][1][:, :-1] - em[base][1][:, 1:]

        table = []
        for i, p in enumerate(levels):
            f1s = {}
            for g in gens:
                f1s[g] = float(np.mean([f1_of(p, q, g) for q in use]))
            table.append({
                "p": p,
                "g_any": ci(gpt[i], gb_[:, i]), "g_all": ci(gall_pt[i], gall_b[:, i]),
                "em": {g: ci(em[g][0][i], em[g][1][:, i]) for g in gens},
                "f1": f1s,
                "answer_in_nongold": P[f"dial|leak|{p}"],
                "delta_gen": ci(dgen_pt[i], dgen_b[:, i]),
                "delta_ret": ci(dret_pt[i], dret_b[:, i]) if i > 0 else None,
                "ret_minus_gen": ci(dret_pt[i] - dgen_pt[i], dret_b[:, i] - dgen_b[:, i]) if i > 0 else None,
            })

        def roots(x_pt, x_b, d_pt, d_b) -> dict:
            iso = root_with_interval(x_pt, d_pt, x_b, d_b, "isotonic").as_dict()
            lin = root_with_interval(x_pt, d_pt, x_b, d_b, "linear").as_dict()
            a, b = linear_fit(x_pt, d_pt)

            def inside(v, s):
                return v is not None and not math.isnan(v) and s["lo"] <= v <= s["hi"]

            return {
                "isotonic": iso, "linear": {**lin, "intercept": a, "slope": b},
                "agree": bool(inside(iso["point"], lin) and inside(lin["point"], iso)),
                "points": {"g": x_pt, "d": d_pt},
            }

        sl = slice(1, L)
        raw_d_pt, raw_d_b = dret_pt[sl] - dgen_pt[sl], dret_b[:, sl] - dgen_b[:, sl]
        res = {"raw": roots(gpt[sl], gb_[:, sl], raw_d_pt, raw_d_b)}
        res["raw_g_all"] = roots(gall_pt[sl], gall_b[:, sl], raw_d_pt, raw_d_b)
        # step-size sensitivity: a retriever step worth two dial levels
        s2 = slice(2, L)
        d2_pt = (em[base][0][:-2] - em[base][0][2:]) - dgen_pt[s2]
        d2_b = (em[base][1][:, :-2] - em[base][1][:, 2:]) - dgen_b[:, s2]
        res["raw_two_level_step"] = roots(gpt[s2], gb_[:, s2], d2_pt, d2_b)

        cg = costs["generator_step"]
        per_unit = {}
        for s in steps:
            # "ms_component": the retriever step priced at the retrieval p50 it adds,
            # the generator step at the end-to-end p50 it adds
            for unit, ret_key, gen_key_ in (("gb", "gb", "gb"), ("ms_total", "ms_total", "ms_total"),
                                            ("ms", "ms", "ms"), ("ms_component", "retrieval_ms", "ms")):
                tag = f"per_{unit}|{s['from']}->{s['to']}"
                if cg is None or cg[gen_key_] <= 0 or s[ret_key] <= 0:
                    per_unit[tag] = {"undefined": True, "retriever_cost": s[ret_key],
                                     "generator_cost": cg[gen_key_] if cg else None,
                                     "reason": "a move that does not add cost has no per-unit price"}
                    continue
                d_pt = dret_pt[sl] / s[ret_key] - dgen_pt[sl] / cg[gen_key_]
                d_b = dret_b[:, sl] / s[ret_key] - dgen_b[:, sl] / cg[gen_key_]
                per_unit[tag] = {"retriever_cost": s[ret_key], "generator_cost": cg[gen_key_],
                                 **roots(gpt[sl], gb_[:, sl], d_pt, d_b)}
        res["per_unit"] = per_unit
        # the generator step priced in front of every rung instead: how much
        # does the per-GB crossover move with the price?
        by_rung = {}
        if priced and priced["gb"] > 0:
            for rid in dcfg["ladder"]:
                a, b = budget(rid, base), budget(rid, big)
                if not (a and b):
                    continue
                gb = b["peak_rss_gb"] - a["peak_rss_gb"]
                if gb <= 0:
                    by_rung[rid] = {"generator_cost": gb, "undefined": True}
                    continue
                d_pt = dret_pt[sl] / priced["gb"] - dgen_pt[sl] / gb
                d_b = dret_b[:, sl] / priced["gb"] - dgen_b[:, sl] / gb
                by_rung[rid] = {"generator_cost": gb, **roots(gpt[sl], gb_[:, sl], d_pt, d_b)}
        res["per_gb_by_generator_rung"] = by_rung

        # break-even gold gain: how much g a retriever upgrade must add, at each
        # operating point, to match the generator upgrade (raw and per GB)
        order_g = np.argsort(gpt, kind="stable")
        be = []
        for i, p in enumerate(levels):
            entry = {"p": p, "g_any": gpt[i], "headroom": float(gpt.max() - gpt[i])}
            ratios = [("raw", 1.0)]
            if priced and cg and priced["gb"] > 0 and cg["gb"] > 0:
                ratios.append(("per_gb", priced["gb"] / cg["gb"]))
            for tag, ratio in ratios:
                pt = break_even_gain(gpt[order_g], em[base][0][order_g], gpt[i], dgen_pt[i] * ratio)
                bt = np.empty(counts.shape[0])
                cens = 0
                for j in range(counts.shape[0]):
                    oj = np.argsort(gb_[j], kind="stable")
                    v = break_even_gain(gb_[j][oj], em[base][1][j][oj], gb_[j][i], dgen_b[j, i] * ratio)
                    if np.isnan(v):
                        cens += 1
                        v = float(gb_[j].max() - gb_[j][i])
                    bt[j] = v
                lo, hi = interval(bt)
                entry[tag] = {"point": pt, "lo": lo, "hi": hi, "n_censored": cens, "target_ratio": ratio}
            be.append(entry)
        res["break_even_gold_gain"] = be

        # price table: each move on each side of g*
        gstar = res["raw"]["isotonic"]["point"]
        price = []
        for i, p in enumerate(levels):
            if i == 0:
                continue
            row = {"p": p, "g_any": gpt[i], "side": (None if gstar is None or math.isnan(gstar)
                                                    else ("below g*" if gpt[i] < gstar else "above g*")),
                   "delta_ret": dret_pt[i], "delta_gen": dgen_pt[i]}
            if priced and cg:
                row.update({
                    "ret_per_gb": dret_pt[i] / priced["gb"] if priced["gb"] > 0 else None,
                    "gen_per_gb": dgen_pt[i] / cg["gb"] if cg["gb"] > 0 else None,
                    "ret_per_ms": dret_pt[i] / priced["ms"] if priced["ms"] > 0 else None,
                    "ret_per_ms_component": dret_pt[i] / priced["retrieval_ms"] if priced["retrieval_ms"] > 0 else None,
                    "ret_per_ms_total": dret_pt[i] / priced["ms_total"] if priced["ms_total"] > 0 else None,
                    "gen_per_ms_total": dgen_pt[i] / cg["ms_total"] if cg["ms_total"] > 0 else None,
                    "gen_per_ms": dgen_pt[i] / cg["ms"] if cg["ms"] > 0 else None,
                })
            price.append(row)

        out["dial"] = {"retriever": drid, "k": dk, "levels": table, "roots": res, "price_table": price,
                       "retriever_step_definition": "one dial level (p -> p - 0.2), base generator"}

        # dial curves for the band checks (resampled), sorted by g ascending
        order = np.argsort(gpt, kind="stable")
        curve = {"g": (gpt[order], gb_[:, order]),
                 "em_base": (em[base][0][order], em[base][1][:, order]),
                 "delta_gen": (dgen_pt[order], dgen_b[:, order])}
    else:
        curve = None

    # ---- natural sweep and the dial band -----------------------------------
    def interp_boot(xq_pt, xq_b, key):
        gx_pt, gx_b = curve["g"]
        y_pt, y_b = curve[key]
        pt = float(np.interp(xq_pt, gx_pt, y_pt))
        bt = np.array([np.interp(xq_b[j], gx_b[j], y_b[j]) for j in range(gx_b.shape[0])])
        return pt, bt

    natural = {}
    for k in ks:
        present = [r for r in ladder if all(rows.has(f"nat|em|{r}|{g}|{k}") for g in gens)]
        if not present:
            continue
        cfgs, steps_k, gen_checks = [], [], []
        for r in present:
            gk = f"nat|g_any|{r}|{k}"
            h_gic = harness[(r, base, k)]["harness_gold_in_context"]
            cfgs.append({
                "retriever": r, "k": k, "g_any": ci(P[gk], B[gk]),
                "g_all": P[f"nat|g_all|{r}|{k}"],
                "harness_gold_in_context": h_gic,
                "g_matches_harness": h_gic is None or abs(h_gic - P[gk]) < 1e-9 or out["partial"],
                "em": {g: ci(P[f"nat|em|{r}|{g}|{k}"], B[f"nat|em|{r}|{g}|{k}"]) for g in gens},
                "f1": {g: natural_f1[(r, g, k)] for g in gens},
                "max_tokens": {g: harness[(r, g, k)]["max_tokens"] for g in gens},
                "source": {g: harness[(r, g, k)]["source"] for g in gens},
            })
            dg_pt = P[f"nat|em|{r}|{big}|{k}"] - P[f"nat|em|{r}|{base}|{k}"]
            dg_b = B[f"nat|em|{r}|{big}|{k}"] - B[f"nat|em|{r}|{base}|{k}"]
            chk = {"retriever": r, "g_any": P[gk], "delta_gen": ci(dg_pt, dg_b),
                   # the dial's own point is "compared with itself" only when the dial and the sweep answered on one machine
                   "by_construction": bool(r == drid and k == dk and cfg.get("dial_same_machine", True))}
            if curve is not None:
                lo_g, hi_g = float(curve["g"][0].min()), float(curve["g"][0].max())
                pr_pt, pr_b = interp_boot(P[gk], B[gk], "delta_gen")
                band = interval(pr_b)
                chk.update({"dial_band": {"point": pr_pt, "lo": band[0], "hi": band[1]},
                            "inside_band": bool(band[0] <= dg_pt <= band[1]),
                            "extrapolated": bool(not lo_g <= P[gk] <= hi_g),
                            "difference": ci(dg_pt - pr_pt, dg_b - pr_b),
                            "consistent": not established(*interval(dg_b - pr_b)),
                            "cross_k": k != dk})
            gen_checks.append(chk)
        for lo, hi in zip(present, present[1:]):
            glo, ghi = f"nat|g_any|{lo}|{k}", f"nat|g_any|{hi}|{k}"
            d_pt = P[f"nat|em|{hi}|{base}|{k}"] - P[f"nat|em|{lo}|{base}|{k}"]
            d_b = B[f"nat|em|{hi}|{base}|{k}"] - B[f"nat|em|{lo}|{base}|{k}"]
            gl_pt = P[f"nat|em|{lo}|{big}|{k}"] - P[f"nat|em|{lo}|{base}|{k}"]
            gl_b = B[f"nat|em|{lo}|{big}|{k}"] - B[f"nat|em|{lo}|{base}|{k}"]
            st = {"from": lo, "to": hi, "g_from": P[glo], "g_to": P[ghi], "delta_ret": ci(d_pt, d_b),
                  "delta_gen_at_from": ci(gl_pt, gl_b), "gen_minus_ret": ci(gl_pt - d_pt, gl_b - d_b)}
            if curve is not None:
                lo_g, hi_g = float(curve["g"][0].min()), float(curve["g"][0].max())
                a_pt, a_b = interp_boot(P[glo], B[glo], "em_base")
                b_pt, b_b = interp_boot(P[ghi], B[ghi], "em_base")
                band = interval(b_b - a_b)
                st.update({"dial_predicted": {"point": b_pt - a_pt, "lo": band[0], "hi": band[1]},
                           "inside_band": bool(band[0] <= d_pt <= band[1]),
                           "extrapolated": bool(not (lo_g <= P[glo] <= hi_g and lo_g <= P[ghi] <= hi_g)),
                           "difference": ci(d_pt - (b_pt - a_pt), d_b - (b_b - a_b)),
                           "consistent": not established(*interval(d_b - (b_b - a_b))),
                           "cross_k": k != dk})
            steps_k.append(st)
        natural[str(k)] = {"configs": cfgs, "retriever_steps": steps_k, "generator_checks": gen_checks}
    out["natural"] = natural
    out["generations"] = {
        "dial_cache_rows": distinct if caches is None else {g: len(caches[g].entries) for g in gens},
        "harness": {f"{rid}+{GEN_SHORT.get(g, g)}@k{k}": {"n": h["n_answered"], "source": h["source"]}
                    for (rid, g, k), h in sorted(harness.items())},
    }
    return out


# --------------------------------------------------------------------------
# results/stats.json: every number the prose quotes, pre-formatted
# --------------------------------------------------------------------------
def f(v, nd=3, signed=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def yn(b):
    return "yes" if b else "no"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# v0.4.2: a family's tokens say when an interval's end is a censored edge, not an estimate (Qwen2.5's keep their text)
_MARK = {"on": False}


def censored_side(r) -> str:
    return "upper" if r.get("no_root_above", 0) >= r.get("no_root_below", 0) else "lower"


def interval_text(r, nd=2):
    if not r or r.get("point") is None:
        return "no crossing in range"
    cens = f"; {r['n_no_root']} of {r['n_boot']} resamples censored" if r.get("n_no_root") else ""
    edge = f", the {censored_side(r)} end censored" if _MARK["on"] and r.get("interval_censored") else ""
    return f"{f(r['point'], nd)} [{f(r['lo'], nd)}, {f(r['hi'], nd)}]{edge}{cens}"


def build_tokens(result: dict, cfg: dict) -> dict:
    _MARK["on"] = bool(cfg.get("mark_censored"))
    t: dict[str, str] = {}
    base, big = cfg["base_generator"], cfg["big_generator"]
    priced = "->".join(cfg["priced_step"])
    t["n_boot"] = str(cfg["n_boot"])
    t["max_tokens"] = str(cfg["max_tokens"])
    t["priced_step"] = " → ".join(cfg["priced_step"])
    total = 0
    for name, d in result["datasets"].items():
        p = f"{name}."
        t[p + "status"] = d.get("status") or d.get("dial_status") or ("complete" if "dial" in d else "not run")
        if "dial" not in d:
            continue
        dial = d["dial"]
        t[p + "n"] = str(d["n_questions"])
        t[p + "dial_retriever"] = dial["retriever"]
        t[p + "k"] = str(dial["k"])
        roots = dial["roots"]
        for tag, key in (("raw", "raw"), ("gall", "raw_g_all"), ("step2", "raw_two_level_step")):
            r = roots[key]
            t[p + f"gstar_{tag}"] = interval_text(r["isotonic"])
            t[p + f"gstar_{tag}_linear"] = interval_text(r["linear"])
            t[p + f"gstar_{tag}_agree"] = yn(r["agree"])
            t[p + f"gstar_{tag}_point"] = f(r["isotonic"]["point"], 2)
        for unit in ("gb", "ms_total", "ms", "ms_component"):
            r = roots["per_unit"].get(f"per_{unit}|{priced}")
            if r is None or r.get("undefined"):
                why = "undefined (the priced move does not add cost)" if r else "not measured"
                t[p + f"gstar_{unit}"] = t[p + f"gstar_{unit}_linear"] = why
                t[p + f"gstar_{unit}_agree"] = "n/a"
            else:
                t[p + f"gstar_{unit}"] = interval_text(r["isotonic"])
                t[p + f"gstar_{unit}_linear"] = interval_text(r["linear"])
                t[p + f"gstar_{unit}_agree"] = yn(r["agree"])
        lv_rows = [lv for lv in dial["levels"] if lv.get("ret_minus_gen")]
        gs = [lv["g_any"]["point"] for lv in lv_rows]
        n_ret = sum(lv["ret_minus_gen"]["point"] > 0 for lv in lv_rows)
        n_est_ret = sum(lv["ret_minus_gen"]["established"] and lv["ret_minus_gen"]["point"] > 0 for lv in lv_rows)
        n_est_gen = sum(lv["ret_minus_gen"]["established"] and lv["ret_minus_gen"]["point"] < 0 for lv in lv_rows)
        t[p + "g_range"] = f"{f(min(gs), 2)} to {f(max(gs), 2)}"
        t[p + "g_top"] = f(dial["levels"][0]["g_any"]["point"], 3)
        t[p + "n_levels"] = str(len(lv_rows))
        t[p + "levels_ret_larger"] = f"{n_ret} of {len(lv_rows)}"
        t[p + "levels_ret_established"] = f"{n_est_ret} of {len(lv_rows)}"
        t[p + "levels_gen_established"] = f"{n_est_gen} of {len(lv_rows)}"
        iso = roots["raw"]["isotonic"]
        est = n_est_ret + n_est_gen
        n = len(lv_rows)
        # a level with no gold (g = 0) is degenerate: the dial cannot remove what is
        # not there, and it is also where leaked answers dominate
        est_lv = [lv for lv in lv_rows if lv["ret_minus_gen"]["established"]]
        gold_lv = [lv for lv in lv_rows if lv["g_any"]["point"] > 0]
        est_gold = [lv for lv in est_lv if lv["g_any"]["point"] > 0]
        t[p + "established_with_gold"] = f"{len(est_gold)} of {len(gold_lv)}"
        if est == 0:
            est_txt = "no level's difference is established"
        elif not est_gold:
            est_txt = (f"the difference is established at {est} of {n} levels, and only at the no-gold level "
                       "(g = 0), so nowhere that gold is present")
        else:
            est_txt = f"the difference is established at {est} of {n} levels ({len(est_gold)} with gold present)"
        t[p + "established_text"] = est_txt
        ret_lv = [lv for lv in lv_rows if lv["ret_minus_gen"]["point"] > 0]
        gen_lv = [lv for lv in lv_rows if lv["ret_minus_gen"]["point"] < 0]
        t[p + "lean_points"] = "retriever" if len(ret_lv) >= len(gen_lv) else "generator"
        t[p + "levels_gen_larger"] = f"{len(gen_lv)} of {n}"
        if iso["point"] is None:
            lean = "retriever" if iso.get("no_root_above", 0) >= iso.get("no_root_below", 0) else "generator"
            t[p + "lean"] = lean
            t[p + "verdict"] = (f"no crossing between g = {f(min(gs), 2)} and {f(max(gs), 2)}: the {lean} step is the larger "
                                f"point estimate at {len(ret_lv) if lean == 'retriever' else len(gen_lv)} of {n} "
                                f"dial levels, but {est_txt}")
        else:
            t[p + "lean"] = "crosses"
            minority = ret_lv if len(ret_lv) <= len(gen_lv) else gen_lv
            who = "retriever" if minority is ret_lv else "generator"
            rests = (f"; the {who} step leads only at the no-gold level (g = 0), so the crossing rests on that "
                     "level alone" if minority and all(lv["g_any"]["point"] == 0 for lv in minority) else "")
            t[p + "verdict"] = (f"g* = {interval_text(iso)} (isotonic), {interval_text(roots['raw']['linear'])} (linear); "
                                f"{est_txt}{rests}")
        leaks = [lv.get("answer_in_nongold") for lv in dial["levels"]]
        t[p + "leak_no_gold"] = f"{100 * leaks[-1]:.0f}%" if leaks[-1] is not None else "n/a"
        t[p + "leak_max"] = f"{100 * max(x for x in leaks if x is not None):.0f}%" if any(x is not None for x in leaks) else "n/a"
        g_noctx = dial["levels"][-1]["em"]
        t[p + "em_no_gold_base"] = f(g_noctx[base]["point"])
        t[p + "em_no_gold_big"] = f(g_noctx[big]["point"])
        t[p + "has_root"] = yn(iso["point"] is not None)
        berows = []
        for e in roots["break_even_gold_gain"]:
            def cell(v):
                if not v:
                    return "n/a"
                if v["point"] is None:
                    return (f"> headroom ({f(e['headroom'])}) [{f(v['lo'])}, {f(v['hi'])}]"
                            + (f"; {v['n_censored']} censored" if _MARK["on"] and v.get("n_censored") else ""))
                hi = "> headroom" if _MARK["on"] and v["n_censored"] > 0.025 * cfg["n_boot"] else f(v["hi"])
                return f"{f(v['point'])} [{f(v['lo'])}, {hi}]" + (f"; {v['n_censored']} censored" if v["n_censored"] else "")
            berows.append([f(e["p"], 1), f(e["g_any"]), f(e["headroom"]), cell(e.get("raw")), cell(e.get("per_gb"))])
        c = d["costs"]
        cg, cr = c.get("generator_step"), c.get("priced_retriever_step")
        # the per-GB column is a bar for one upgrade of the priced step's memory, not a rate; the headline family's
        # tables say so in the header (the frozen tables keep theirs)
        per_gb = (f"gold gain needed, for an upgrade costing {f(cr['gb'], 2)} GB like the priced step [95%]"
                  if _MARK["on"] and cr else "per GB [95%]")
        t[p + "table_break_even"] = md_table(
            ["p", "operating point g", "headroom to dial top", "gold gain needed to match the generator step, raw [95%]", per_gb], berows)
        t[p + "cost_gen_gb"] = f(cg["gb"], 2) if cg else "n/a"
        t[p + "cost_gen_ms"] = f(cg["ms"], 0) if cg else "n/a"
        t[p + "cost_ret_gb"] = f(cr["gb"], 2) if cr else "n/a"
        t[p + "cost_ret_ms"] = f(cr["ms"], 0) if cr else "n/a"
        t[p + "cost_ret_retrieval_ms"] = f(cr["retrieval_ms"], 1) if cr else "n/a"
        t[p + "cost_ret_ms_total"] = f(cr["ms_total"], 0) if cr else "n/a"
        t[p + "cost_gen_ms_total"] = f(cg["ms_total"], 0) if cg else "n/a"

        rows = []
        for lv in dial["levels"]:
            dr = lv["delta_ret"]
            rows.append([
                f(lv["p"], 1), f"{f(lv['g_any']['point'])}", f(lv["g_all"]["point"]),
                f(lv["em"][base]["point"]), f(lv["em"][big]["point"]),
                f"{f(lv['delta_gen']['point'], signed=True)} [{f(lv['delta_gen']['lo'], signed=True)}, {f(lv['delta_gen']['hi'], signed=True)}]",
                "n/a" if dr is None else f"{f(dr['point'], signed=True)} [{f(dr['lo'], signed=True)}, {f(dr['hi'], signed=True)}]",
                "n/a" if dr is None else (
                    f"{f(lv['ret_minus_gen']['point'], signed=True)} [{f(lv['ret_minus_gen']['lo'], signed=True)}, {f(lv['ret_minus_gen']['hi'], signed=True)}]"
                    + (" established" if lv["ret_minus_gen"]["established"] else " not established")),
                "n/a" if lv.get("answer_in_nongold") is None else f"{100 * lv['answer_in_nongold']:.0f}%",
            ])
        t[p + "table_dial"] = md_table(
            ["p", "g any", "g all", f"EM {GEN_SHORT[base]}", f"EM {GEN_SHORT[big]}", "Δ_gen [95%]", "Δ_ret [95%]", "Δ_ret − Δ_gen [95%]",
             "answer in a non-gold passage"], rows)
        # v0.4.2: every other family's dial, read the same way, each its own table and ranges
        ci = lambda x: f"{f(x['point'], signed=True)} [{f(x['lo'], signed=True)}, {f(x['hi'], signed=True)}]"  # noqa: E731
        fam_label = {"gemma-4": "Gemma 4", "gemma-3n": "Gemma 3n", "ministral-3": "Ministral 3", "qwen2.5": "Qwen2.5"}
        size_label = {"gemma-4-e2b": "E2B", "gemma-4-e4b": "E4B", "gemma-3n-e2b": "E2B", "gemma-3n-e4b": "E4B", "ministral-3-3b": "3B", "ministral-3-8b": "8B",
                      "qwen2.5-1.5b": "1.5B", "qwen2.5-3b": "3B"}
        for fam, fb in (d.get("dial_families") or {}).items():
            fd = (fb or {}).get("dial") or {}
            if not fd.get("levels"):
                continue
            a, b = fb["generators"]
            fp = f"{p}dial_family.{fam.replace('-', '_').replace('.', '_')}."
            frows = []
            for lv in sorted(fd["levels"], key=lambda x: -x["p"]):
                dr = lv.get("delta_ret")
                frows.append([f(lv["p"], 1), f(lv["g_any"]["point"]), f(lv["em"][a]["point"]), f(lv["em"][b]["point"]), ci(lv["delta_gen"]),
                              "n/a" if dr is None else ci(dr)])
            t[fp + "table"] = md_table(["p", "g any", f"EM {size_label.get(a, a)}", f"EM {size_label.get(b, b)}", "Δ_gen [95%]", "Δ_ret [95%]"], frows)
            gens = [lv["delta_gen"] for lv in fd["levels"]]
            t[fp + "label"] = fam_label.get(fam, fam)
            t[fp + "gen_step_range"] = f"{f(min(x['point'] for x in gens), signed=True)} to {f(max(x['point'] for x in gens), signed=True)}"
            t[fp + "gen_step_established_above"] = f"{sum(x['established'] and x['point'] > 0 for x in gens)} of {len(gens)}"
            t[fp + "gen_step_established_below"] = f"{sum(x['established'] and x['point'] < 0 for x in gens)} of {len(gens)}"
            top = max(fd["levels"], key=lambda x: x["g_any"]["point"])
            t[fp + "em_full_gold"] = f"{f(top['em'][a]['point'])} and {f(top['em'][b]['point'])}"

        prow = []
        for r in dial["price_table"]:
            def per_s(key):
                return f(None if r.get(key) is None else 1000 * r[key], 3, True)
            prow.append([f(r["p"], 1), f(r["g_any"], 3), r["side"] or "n/a",
                         f(r["delta_ret"], signed=True), f(r["delta_gen"], signed=True),
                         f(r.get("ret_per_gb"), 3, True), f(r.get("gen_per_gb"), 3, True),
                         per_s("ret_per_ms_total"), per_s("gen_per_ms_total")])
        t[p + "table_price"] = md_table(
            ["p", "g", "side", "Δ_ret", "Δ_gen", "Δ_ret / GB", "Δ_gen / GB",
             "Δ_ret / s of added latency", "Δ_gen / s of added latency"], prow)

        unit_label = {"per_gb": "per GB of peak RSS", "per_ms_total": "per ms of added latency (retrieval + generation)",
                      "per_ms": "per ms of generation-stage latency only",
                      "per_ms_component": "retriever per ms of retrieval latency, generator per ms of generation"}
        sens = []
        for tag, r in sorted(roots["per_unit"].items()):
            unit, step = tag.split("|", 1)
            label = f"{unit_label.get(unit, unit)}: {step.replace('->', ' → ')}"
            if r.get("undefined"):
                sens.append([label, f(r.get("retriever_cost"), 2), f(r.get("generator_cost"), 2), "undefined", "n/a"])
            else:
                sens.append([label, f(r["retriever_cost"], 2), f(r["generator_cost"], 2),
                             interval_text(r["isotonic"]), interval_text(r["linear"])])
        t[p + "table_price_sensitivity"] = md_table(["pricing", "retriever cost", "generator cost", "g* isotonic", "g* linear"], sens)
        rr = []
        for rid, r in (roots.get("per_gb_by_generator_rung") or {}).items():
            if r.get("undefined"):
                rr.append([rid, f(r["generator_cost"], 2), "undefined", "n/a"])
            else:
                rr.append([rid, f(r["generator_cost"], 2), interval_text(r["isotonic"]), interval_text(r["linear"])])
        t[p + "table_per_gb_by_gen_rung"] = md_table(
            ["generator step priced in front of", "its GB", "per-GB g* isotonic [95%]", "per-GB g* linear [95%]"],
            rr) if rr else "_not measured yet._"

        # the generator step's cost at every rung: peak RSS is one maximum over the
        # whole process (index build included), so the difference of two
        # maxima varies with the retriever in front; the dial prices it at one rung
        gk = dial["k"]
        grows = []
        for rid in cfg["datasets"][name]["ladder"]:
            a = d["budget"].get(f"{rid}+{GEN_SHORT[base]}@k{gk}")
            b = d["budget"].get(f"{rid}+{GEN_SHORT[big]}@k{gk}")
            if a and b:
                grows.append([rid + (" (priced)" if rid == d["costs"]["generator_step"]["at_retriever"] else "")
                              if d["costs"].get("generator_step") else rid,
                              f(b["peak_rss_gb"] - a["peak_rss_gb"], 2),
                              f(1000 * (b["e2e_s_p50"] - a["e2e_s_p50"]) + b["retrieval_ms_p50"] - a["retrieval_ms_p50"], 0)])
        t[p + "table_gen_step_by_rung"] = (md_table(["retriever in front", f"{GEN_SHORT[base]} → {GEN_SHORT[big]}: GB of peak RSS",
                                                     f"{GEN_SHORT[base]} → {GEN_SHORT[big]}: ms of added latency"], grows)
                                           if grows else "_not measured yet._")
        brow = []
        for key, b in d["budget"].items():
            brow.append([key, f(b["peak_rss_gb"], 2), f(b["index_bytes"] / 1e6, 1), f(b["model_bytes"] / 1e9, 2),
                         f(b["e2e_s_p50"], 2), f(b["e2e_s_p95"], 2), str(b["max_tokens"]), b["source"]])
        t[p + "table_budget"] = md_table(["configuration", "peak RSS GB", "index MB", "model GB", "e2e p50 s", "e2e p95 s", "max_tokens", "source"], brow)

        ks = sorted(d.get("natural", {}), key=int)
        t[p + "natural_ks"] = ", ".join(ks) if ks else "none"
        for k in ["3", "5", "10"]:
            nat = d.get("natural", {}).get(k)
            if not nat:
                t[p + f"table_natural_k{k}"] = f"_k = {k} was not run._"
                t[p + f"table_check_ret_k{k}"] = t[p + f"table_check_gen_k{k}"] = f"_k = {k} was not run._"
                t[p + f"natural_k{k}_inside"] = t[p + f"natural_k{k}_consistent"] = "none (not measured yet)"
                t[p + f"natural_k{k}_inconsistent"] = "none (not measured yet)"
                t[p + f"table_steps_k{k}"] = f"_k = {k} was not run._"
                continue
            nrows = [[c["retriever"], f(c["g_any"]["point"]), f(c["g_all"]), f(c["em"][base]["point"]),
                      f(c["em"][big]["point"]), f(c["f1"][base]), f(c["f1"][big])] for c in nat["configs"]]
            t[p + f"table_natural_k{k}"] = md_table(["retriever", "g any", "g all", f"EM {GEN_SHORT[base]}", f"EM {GEN_SHORT[big]}",
                                                     f"F1 {GEN_SHORT[base]}", f"F1 {GEN_SHORT[big]}"], nrows)
            srows = []
            for s in nat["retriever_steps"]:
                dp = s.get("dial_predicted", {})
                srows.append([f"{s['from']} → {s['to']}", f"{f(s['g_from'])} → {f(s['g_to'])}",
                              f"{f(s['delta_ret']['point'], signed=True)} [{f(s['delta_ret']['lo'], signed=True)}, {f(s['delta_ret']['hi'], signed=True)}]",
                              f"{f(dp.get('point'), signed=True)} [{f(dp.get('lo'), signed=True)}, {f(dp.get('hi'), signed=True)}]",
                              yn(s.get("inside_band")), yn(s.get("consistent")),
                              yn(s.get("extrapolated"))])
            t[p + f"table_check_ret_k{k}"] = md_table(
                ["step", "g", "Δ_ret measured [95%]", "dial predicts [95%]", "inside band", "difference interval includes 0", "extrapolated"], srows)
            grows = []
            for c in nat["generator_checks"]:
                band = c.get("dial_band", {})
                grows.append([c["retriever"], f(c["g_any"]),
                              f"{f(c['delta_gen']['point'], signed=True)} [{f(c['delta_gen']['lo'], signed=True)}, {f(c['delta_gen']['hi'], signed=True)}]",
                              f"{f(band.get('point'), signed=True)} [{f(band.get('lo'), signed=True)}, {f(band.get('hi'), signed=True)}]",
                              yn(c.get("inside_band")), yn(c.get("consistent")), yn(c.get("extrapolated"))])
            t[p + f"table_check_gen_k{k}"] = md_table(
                ["retriever", "g", "Δ_gen measured [95%]", "dial band [95%]", "inside band", "difference interval includes 0", "extrapolated"], grows)
            checks = [x for x in nat["retriever_steps"] + nat["generator_checks"] if not x.get("by_construction")]
            t[p + f"natural_k{k}_inside"] = f"{sum(bool(x.get('inside_band')) for x in checks)} of {len(checks)}"
            t[p + f"natural_k{k}_consistent"] = f"{sum(bool(x.get('consistent')) for x in checks)} of {len(checks)}"
            bad = []
            for x in checks:
                if x.get("consistent", True):
                    continue
                dd = x["difference"]
                what = (f"{x['from']} → {x['to']} (g {f(x['g_from'])} → {f(x['g_to'])})" if "from" in x
                        else f"the generator step at {x['retriever']} (g {f(x['g_any'])})")
                bad.append(f"{what}: measured minus predicted {f(dd['point'], signed=True)} "
                           f"[{f(dd['lo'], signed=True)}, {f(dd['hi'], signed=True)}]")
            t[p + f"natural_k{k}_inconsistent"] = "; ".join(bad) if bad else "none"
            # v0.4.2: where the dial and the sweep answered on two machines, the dial retriever's own check measures the machine
            own = [x for x in nat["generator_checks"] if x["retriever"] == dial["retriever"] and int(k) == dial["k"]
                   and "difference" in x and not x.get("by_construction")]
            if own:
                dd = own[0]["difference"]
                t[p + f"natural_k{k}_dial_own"] = f"{f(dd['point'], signed=True)} [{f(dd['lo'], signed=True)}, {f(dd['hi'], signed=True)}]"
            srows = []
            for st in nat["retriever_steps"]:
                if "gen_minus_ret" not in st:
                    continue
                gm, dr, dg = st["gen_minus_ret"], st["delta_ret"], st["delta_gen_at_from"]
                srows.append([f"{st['from']} → {st['to']}", f"{f(st['g_from'])} → {f(st['g_to'])}",
                              f"{f(dr['point'], signed=True)} [{f(dr['lo'], signed=True)}, {f(dr['hi'], signed=True)}]",
                              f"{f(dg['point'], signed=True)} [{f(dg['lo'], signed=True)}, {f(dg['hi'], signed=True)}]",
                              f"{f(gm['point'], signed=True)} [{f(gm['lo'], signed=True)}, {f(gm['hi'], signed=True)}]"
                              + (" established" if gm["established"] else " not established")])
                key = f"{p}step.{st['from']}->{st['to']}.k{k}"
                t[key + ".ret"] = f"{f(dr['point'], signed=True)} [{f(dr['lo'], signed=True)}, {f(dr['hi'], signed=True)}]"
                t[key + ".gen"] = f"{f(dg['point'], signed=True)} [{f(dg['lo'], signed=True)}, {f(dg['hi'], signed=True)}]"
                t[key + ".gen_minus_ret"] = (f"{f(gm['point'], signed=True)} [{f(gm['lo'], signed=True)}, "
                                             f"{f(gm['hi'], signed=True)}]")
                t[key + ".g"] = f"{f(st['g_from'])} → {f(st['g_to'])}"
                t[key + ".g_change"] = f(st["g_to"] - st["g_from"], signed=True)
                t[key + ".larger"] = ("the generator upgrade" if gm["point"] > 0 else "the retriever upgrade") + \
                    (", established" if gm["established"] else ", not established")
            t[p + f"table_steps_k{k}"] = md_table(
                ["retriever step", "g", "Δ_ret [95%]", "Δ_gen at the lower rung [95%]",
                 "Δ_gen − Δ_ret [95%]"], srows) if srows else "_none._"

        g = d["generations"]
        n_dial = sum(g["dial_cache_rows"].values())
        n_h = sum(v["n"] for v in g["harness"].values() if v["source"].startswith("results/crossover"))
        t[p + "gens_dial"] = str(n_dial)
        t[p + "gens_harness_this_run"] = str(n_h)
        t[p + "gens_harness_reused"] = str(sum(v["n"] for v in g["harness"].values() if not v["source"].startswith("results/crossover")))
        total += n_dial + n_h
    t["gens_total_this_run"] = str(total)
    rows = []
    for nm, a in result.get("agreement_all", {}).items():
        r = a["root"]
        rows.append([nm, f"{f(a['g_min'], 2)} to {f(a['g_max'], 2)}",
                     interval_text(r) if r.get("point") is not None else f"censored {a['censored_side']} (no crossing)"])
    t["table_agreement"] = md_table(["dataset", "dial g range", "g* (isotonic) [95%]"], rows) if rows else "_no dataset analysed_"
    leans = {nm: t.get(f"{nm}.lean") for nm in result.get("agreement_all", {})}
    if len(leans) < 2:
        t["agreement_verdict"] = "only one dataset analysed; no agreement can be stated"
    else:
        def describe(nm):
            lp = t.get(f"{nm}.lean_points")
            cnt = t.get(f"{nm}.levels_ret_larger") if lp == "retriever" else t.get(f"{nm}.levels_gen_larger")
            tail = (f", and the isotonic fit gives g* = {t.get(f'{nm}.gstar_raw')}" if leans[nm] == "crosses"
                    else ", and the fitted curves never cross")
            return f"on {nm} the {lp} step is the larger point estimate at {cnt} dial levels{tail}"
        est = {nm: t.get(f"{nm}.established_with_gold", "0 of 0") for nm in leans}
        none_est = all(v.split()[0] == "0" for v in est.values())
        t["agreement_verdict"] = ("; ".join(describe(nm) for nm in leans) + ". The gap between the two moves is "
                                  "established at " + " and ".join(f"{v} {nm}" for nm, v in est.items())
                                  + " levels that contain gold"
                                  + (", so the data do not locate a common g*" if none_est else ""))
    both = {nm for nm, v in leans.items() if v == "crosses"}
    if len(leans) < 2:
        t["headline.nq_agrees"] = "not testable with one dataset"
    elif len(both) == 2:
        a = result.get("agreement", {}).get("raw", {})
        t["headline.nq_agrees"] = "yes, the g* intervals overlap" if a.get("intervals_overlap") else \
            "no, the g* intervals do not overlap"
        if _MARK["on"] and a.get("intervals_overlap"):
            t["headline.nq_agrees"] = ("not shown to differ: the g* intervals overlap, but both are wide and censored at an end, "
                                       "so their overlap cannot show that the two datasets agree")
    else:
        lp = {nm: t.get(f"{nm}.lean_points") for nm in leans}
        t["headline.nq_agrees"] = ("not testable: there is no common g* to compare"
                                   + (f" (the two datasets lean opposite ways: "
                                      + ", ".join(f"{nm} towards the {v}" for nm, v in lp.items()) + ")"
                                      if len(set(lp.values())) > 1 else ""))
    # The specification's test: do the HotpotQA and NQ g* intervals overlap?
    for tag in ("raw", "per_gb"):
        a = result.get("agreement", {}).get(tag)
        if a and a.get("both_have_roots"):
            t[f"agree_{tag}"] = yn(a["intervals_overlap"])
        elif a:
            no = [nm for nm in ("hotpotqa", "nq") if not has_root(a[nm])]
            one = len(no) == 1
            t[f"agree_{tag}"] = ("not applicable: " + " and ".join(no)
                                 + (" has no crossing inside its dial range, so it has no g* interval to overlap"
                                    if one else " have no crossing inside their dial ranges, so there are no g* "
                                    "intervals to overlap")
                                 + (" (every resample is censored at its own dial range's edge, so the roots say nothing "
                                    "about agreement)" if _MARK["on"] else f" (the resampled roots, censored ones included, "
                                    f"{'do' if a['intervals_overlap'] else 'do not'} overlap)"))
        else:
            t[f"agree_{tag}"] = "not measured"
    return t


def has_root(r: dict | None) -> bool:
    p = (r or {}).get("point")
    return p is not None and not (isinstance(p, float) and math.isnan(p))


def agreements(ds: dict, cfg: dict) -> tuple[dict, dict]:
    """(agreement_all, agreement): each agreement dataset's crossing, and the cross-dataset test."""
    agreement_all: dict = {}
    # the cross-dataset question is HotpotQA against NQ; Czech is reported on its own
    agree_on = cfg.get("agreement_datasets", ["hotpotqa", "nq"])
    for nm, d in ds.items():
        if nm not in agree_on:
            continue
        r = d.get("dial", {}).get("roots", {}).get("raw", {}).get("isotonic")
        if r:
            lv = [x for x in d["dial"]["levels"] if x.get("ret_minus_gen")]
            agreement_all[nm] = {
                "root": r, "g_min": min(x["g_any"]["point"] for x in lv), "g_max": max(x["g_any"]["point"] for x in lv),
                "censored_side": None if r["point"] == r["point"] and r["point"] is not None and not (isinstance(r["point"], float) and math.isnan(r["point"]))
                else ("above" if r["no_root_above"] >= r["no_root_below"] else "below"),
            }
    h, q = ds.get("hotpotqa", {}), ds.get("nq", {})
    agreement = {}
    for tag in ("raw", "per_gb"):
        def pick(d):
            roots = d.get("dial", {}).get("roots", {})
            if tag == "raw":
                return roots.get("raw", {}).get("isotonic")
            key = f"per_gb|{'->'.join(cfg['priced_step'])}"
            return roots.get("per_unit", {}).get(key, {}).get("isotonic")
        a, b = pick(h), pick(q)
        if a and b:
            # The specification's test needs a g* in both; a dataset whose point estimate
            # never crosses has only a censored interval of resampled roots (NaN
            # here, null once cleaned), and an overlap with that is not agreement
            agreement[tag] = {"hotpotqa": a, "nq": b, "both_have_roots": has_root(a) and has_root(b),
                              "intervals_overlap": bool(a["lo"] <= b["hi"] and b["lo"] <= a["hi"])}
    return agreement_all, agreement


# --------------------------------------------------------------------------
# peak RSS without the double-counted mapped pages: an estimate, beside the measured
# --------------------------------------------------------------------------
# llama-cpp-python maps the GGUF by default, and on the CPU llama.cpp also copies most weights into its own buffers,
# so a row's peak RSS counts that part of the weights twice. scripts/mmap_probe.py measured by how much, per model;
# every row of the family's timed sweeps gets an estimated peak, and everything the documents derive from peak RSS is
# computed again from the estimates by the code above. crossover.json and every measured token stay as they are.
MEASURED = "as measured, model file memory-mapped (library default)"
ESTIMATE = "estimate without the double-counted mapped pages (probe; see LIMITATIONS)"


def md_parse(md: str) -> tuple[list[str], list[list[str]]]:
    lines = md.split("\n")
    cells = [ln[2:-2].split(" | ") for ln in lines if ln.startswith("| ")]
    return cells[0], cells[1:]


def md_merge(measured: str, estimate: str) -> str:
    """One table from two of the same shape: a column whose cells all agree stays one column; a column that differs
    becomes two, the measured and the estimate, each labelled."""
    hm, rm = md_parse(measured)
    he, re_ = md_parse(estimate)
    assert hm == he and len(rm) == len(re_) and all(a[0] == b[0] for a, b in zip(rm, re_)), "the two tables differ in shape"
    cols = []
    for j, h in enumerate(hm):
        a, b = [r[j] for r in rm], [r[j] for r in re_]
        cols += [(h, a)] if a == b else [(f"{h}, {MEASURED}", a), (f"{h}, {ESTIMATE}", b)]
    return md_table([c[0] for c in cols], [list(x) for x in zip(*[c[1] for c in cols])])


def sensitivity_merge(measured: str, estimate: str, gb_label: str = "per GB of peak RSS") -> str:
    """The price sensitivity: the per-GB rows twice, measured then the estimate, each labelled; the per-ms rows once."""
    h, rm = md_parse(measured)
    _, re_ = md_parse(estimate)
    gb_m = [[f"{gb_label}, {MEASURED}{r[0][len(gb_label):]}"] + r[1:] for r in rm if r[0].startswith(gb_label + ":")]
    gb_e = [[f"{gb_label}, {ESTIMATE}{r[0][len(gb_label):]}"] + r[1:] for r in re_ if r[0].startswith(gb_label + ":")]
    rest_m = [r for r in rm if not r[0].startswith(gb_label + ":")]
    assert rest_m == [r for r in re_ if not r[0].startswith(gb_label + ":")], "a per-ms row moved with peak RSS"
    return md_table(h, gb_m + gb_e + rest_m)


def guarded_estimate(measured: float, surplus: float, bound: float) -> tuple[float, float]:
    """(measured minus surplus, the estimate used). The estimate assumes the row peaked while its generator was loaded;
    `bound` bounds the index build, which runs first, from above. Below it, the row's true peak lies between
    measured minus surplus and min(measured, bound), and the higher end is used."""
    est = measured - surplus
    return est, (est if est > bound else min(measured, max(bound, est)))


def mmap_estimate(cfg: dict, tokens: dict, measured: dict) -> tuple[dict | None, dict]:
    """(the record for results/crossover/mmap_estimate.json, the en_m3.est.* tokens), or (None, {}) without the
    config's mmap_estimate block. `measured` is the family's measured analysis, each dataset as crossover.json holds it."""
    ecfg = cfg.get("mmap_estimate")
    if not ecfg:
        return None, {}
    from mmap_probe import surplus as probe_surplus
    from stats_tokens import budget_tables, read_jsonl
    from measurerag.budget import Budget, Measurement, check_all

    probe_path = REPO / ecfg["probe"]
    probe = probe_surplus(probe_path)
    lines = read_jsonl(probe_path)
    fam = ecfg["family"]
    hf = cfg["headline_families"][fam]
    p = hf["prefix"]
    e = p + "est."
    fb, fg = hf["generators"]
    fcfg = {**cfg, "base_generator": fb, "big_generator": fg, "dial_same_machine": hf.get("dial_same_machine", True)}

    # every row: measured, minus its generator's surplus, against the guard's bound
    rows, peaks, sweeps = [], {}, {}
    for name, harness in hf["harness"].items():
        d = harness[0]
        sweeps[f"{name}_k5"] = d
        payload = json.loads((REPO / d / "results.json").read_text(encoding="utf-8"))
        bound_rows = {x["retriever_id"]: x for x in json.loads((REPO / ecfg["bound"][name] / "results.json").read_text(
            encoding="utf-8"))["results"] if x.get("status") == "ok" and x["generator"]["key"] == ecfg["bound_generator"]}
        peaks[name] = {}
        for x in payload["results"]:
            if x.get("status") != "ok" or x.get("rag_status") != "ok":
                continue
            gen, rid = x["generator"]["key"], x["retriever_id"]
            m = x["measurement"]["peak_rss_gb"]
            bound = bound_rows[rid]["measurement"]["peak_rss_gb"]
            est, used = guarded_estimate(m, probe[gen]["surplus_gb"], bound)
            peaks[name][(rid, gen)] = used
            rows.append({"sweep": d, "config_id": x["config_id"], "generator": gen, "measured_peak_rss_gb": m,
                         "surplus_gb": probe[gen]["surplus_gb"], "measured_minus_surplus_gb": est,
                         "bound_config": bound_rows[rid]["config_id"], "bound_peak_rss_gb": bound,
                         "above_bound": est > bound, "estimate_gb": used, "budgets": payload["budgets"],
                         "measurement": x["measurement"]})

    # the family's analysis again, on the estimated peaks
    fds = {name: analyse_dataset(name, {**cfg["datasets"][name], "harness": hf["harness"][name]}, fcfg, True, peaks[name])
           for name in hf["harness"]}
    fall, fagree = agreements(fds, fcfg)
    # read back as crossover.json is, keys sorted, so that every table has the measured one's row order
    est = json.loads(json.dumps(clean({"datasets": fds, "agreement_all": fall, "agreement": fagree}), sort_keys=True, allow_nan=False))
    sub = {"config": {**fcfg, "mark_censored": True}, **est}
    tok = build_tokens(sub, {**fcfg, "mark_censored": True})
    from stats_tokens import headline

    tok.update(headline(sub, tok, mark_censored=True))
    _MARK["on"] = False

    t: dict[str, str] = {}
    for name in hf["harness"]:
        q = f"{name}."
        for k in ("cost_gen_gb", "gstar_gb", "gstar_gb_linear", "gstar_gb_agree"):
            t[e + q + k] = tok[q + k]
        t[e + "headline." + q + "gstar_gb"] = tok["headline." + q + "gstar_gb"]
        for k in ("table_break_even", "table_price", "table_per_gb_by_gen_rung", "table_gen_step_by_rung", "table_budget"):
            t[e + q + k] = md_merge(tokens[p + q + k], tok[q + k])
        t[e + q + "table_price_sensitivity"] = sensitivity_merge(tokens[p + q + "table_price_sensitivity"], tok[q + "table_price_sensitivity"])
        # how much larger the per-GB bar is with the estimate: the ratio of the two generator steps' GB
        t[e + q + "per_gb_bar_factor"] = f(measured[name]["costs"]["generator_step"]["gb"]
                                           / est["datasets"][name]["costs"]["generator_step"]["gb"], 2)
        # the per-GB fit with the generator step priced in front of each rung: how many rungs cross, either way
        _MARK["on"] = True
        for tag, dd in (("", est["datasets"][name]), ("_measured", measured[name])):
            rung = dd["dial"]["roots"]["per_gb_by_generator_rung"]
            t[e + q + "rungs_with_crossing" + tag] = f"{sum(has_root((r or {}).get('isotonic')) for r in rung.values())} of {len(rung)}"
            t[e + q + "rungs_crossing_list" + tag] = "; ".join(
                f"{rid}, where the step costs {f(r['generator_cost'], 2)} GB: g* = {interval_text(r['isotonic'])}"
                for rid, r in rung.items() if has_root((r or {}).get("isotonic"))) or "none"
        assert tok[q + "cost_ret_gb"] == tokens[p + q + "cost_ret_gb"], "the retriever step's GB must not move"
    _MARK["on"] = False
    t[e + "agree_per_gb"] = tok["agree_per_gb"]

    # the budget tables, the peak range and what fits, on the estimated peaks
    by_dir = {(r["sweep"], r["config_id"]): r["estimate_gb"] for r in rows}
    est_tables = budget_tables(REPO, sweeps, "", peaks=by_dir)
    for name in sweeps:
        t[e + f"budget.table_{name}"] = md_merge(tokens[p + f"budget.table_{name}"], est_tables[f"budget.table_{name}"])
    n = len(rows)
    t[e + "budget.peak_rss_range_gb"] = f"{f(min(r['estimate_gb'] for r in rows), 1)} to {f(max(r['estimate_gb'] for r in rows), 1)}"
    grid = [Budget.from_dict(b) for b in rows[0]["budgets"]]
    fits = sum(any(v.fits for v in check_all(Measurement(**{**r["measurement"], "peak_rss_gb": r["estimate_gb"]}), grid)) for r in rows)
    t[e + "budget.configs_fitting"] = f"{fits} of {n}"
    for ram in sorted({b.ram_gb for b in grid}):
        t[e + f"budget.under_ram_{ram:g}gb"] = f"{sum(r['estimate_gb'] <= ram for r in rows)} of {n}"
        t[e + f"budget.under_ram_{ram:g}gb_measured"] = f"{sum(r['measured_peak_rss_gb'] <= ram for r in rows)} of {n}"
    t[e + "guard.rows_above_bound"] = f"{sum(r['above_bound'] for r in rows)} of {n}"
    t[e + "guard.min_margin_gb"] = f(min(r["measured_minus_surplus_gb"] - r["bound_peak_rss_gb"] for r in rows), 2)
    below = [f"{Path(r['sweep']).name} {r['config_id']}" for r in rows if not r["above_bound"]]
    t[e + "guard.rows_below_bound"] = ", ".join(below) if below else "none"

    # the three HotpotQA rows timed again: both of their peaks, less the surplus
    retimed = []
    batches = read_jsonl(REPO / hf["calibration"] / "batches.jsonl")
    current: dict = {}
    for b in batches:
        if b.get("superseded"):
            if b["stage"] in current:
                retimed.append(current.pop(b["stage"]))
        elif b.get("accepted"):
            current[b["stage"]] = b
    pairs = [(b, current[b["stage"]]) for b in retimed]
    t[e + "hygiene.retimed_peaks"] = "; ".join(
        f"{Path(b['config']).stem.replace('_ministral', '').replace('hotpotqa_k5', 'HotpotQA').replace('nq_k5', 'NQ')} "
        f"{b['config_id'].replace('ministral-3-', '')} {f(b['peak_rss_gb'] - probe[gen]['surplus_gb'], 2)} GB, then "
        f"{f(nb['peak_rss_gb'] - probe[gen]['surplus_gb'], 2)} GB"
        for b, nb in pairs for gen in [b["config_id"].split("+", 1)[1]])

    # the probe itself
    pp = e + "probe."
    loads = [x for x in lines if x.get("record") == "load"]
    mach = next(x for x in lines if x.get("record") == "machine")
    for gen, v in probe.items():
        size = gen.rsplit("-", 1)[1]
        t[pp + f"surplus.{size}"] = f(v["surplus_gb"], 2)
        lo, hi = f(min(v["differences_gb"]), 3), f(max(v["differences_gb"]), 3)
        t[pp + f"differences.{size}"] = f"all {lo}" if lo == hi else f"{lo} to {hi}"
        t[pp + f"peak_on.{size}"] = f(statistics.median(v["peak_on_gb"]), 2)
        t[pp + f"peak_off.{size}"] = f(statistics.median(v["peak_off_gb"]), 2)
        on = [x for x in loads if x["model"] == gen and x["use_mmap"] and x.get("vmmap_mapped_file")]
        t[pp + f"mapped_resident.{size}"] = on[0]["vmmap_mapped_file"]["resident"] if on else "n/a"
        t[pp + f"mapped_virtual.{size}"] = on[0]["vmmap_mapped_file"]["virtual"] if on else "n/a"
        for b, tag in ((True, "on"), (False, "off")):
            t[pp + f"answer_s_{tag}.{size}"] = f(statistics.median(x["answer_s"] for x in loads if x["model"] == gen and x["use_mmap"] is b), 1)
    t[pp + "repeats"] = str(min(v["repeats"] for v in probe.values()))
    t[pp + "loads"] = str(len(loads))
    t[pp + "answers_identical"] = ("every answer was the same string with mapping on and off"
                                   if all(v["answers_identical"] for v in probe.values()) else
                                   "some answers differed between mapping on and off")
    a0 = datetime.fromisoformat(min(x["utc_start"] for x in loads).replace("Z", "+00:00"))
    a1 = datetime.fromisoformat(max(x["utc_end"] for x in loads).replace("Z", "+00:00"))
    t[pp + "utc"] = f"{a0:%H:%M} to {a1:%H:%M} UTC on {a0.day} {a0:%b %Y}"
    t[pp + "macos"] = mach.get("macos") or mach["machine"].get("os_version", "n/a")
    t[pp + "llama_cpp_python"] = mach["llama_cpp_python"]
    t[pp + "available_min_gb"] = f(min(x["available_before_gb"] for x in loads), 1)

    doc = {"note": ("Peak RSS of the English Ministral 3 rows without the double-counted mapped pages: an estimate, "
                    "beside the measured values in results/crossover.json. Written by scripts/crossover.py from "
                    f"{ecfg['probe']} and the rows' results.json."),
           "labels": {"measured": MEASURED, "estimate": ESTIMATE},
           "rule": ("estimate = measured peak RSS minus the surplus of the row's generator, the median over the probe's "
                    "repeats of (peak RSS with the model file mapped) minus (without); a row whose estimate is not "
                    f"above the peak of its retriever's row with {ecfg['bound_generator']} on the same slice, which "
                    "bounds the index build from above, is given min(measured, max(that bound, the estimate))"),
           "settings": ecfg,
           "probe": {"record": ecfg["probe"], "surplus": probe},
           "rows": [{k: v for k, v in r.items() if k not in ("budgets", "measurement")} for r in rows],
           "datasets": {name: {"costs": d["costs"], "budget": d["budget"],
                               "per_gb": {k: v for k, v in d["dial"]["roots"]["per_unit"].items() if k.startswith("per_gb|")},
                               "per_gb_by_generator_rung": d["dial"]["roots"]["per_gb_by_generator_rung"],
                               "break_even_gold_gain": [{"p": x["p"], "g_any": x["g_any"], "per_gb": x.get("per_gb")}
                                                        for x in d["dial"]["roots"]["break_even_gold_gain"]],
                               "price_table": [{k: v for k, v in x.items() if k in ("p", "g_any", "ret_per_gb", "gen_per_gb")}
                                               for x in d["dial"]["price_table"]]}
                        for name, d in est["datasets"].items()},
           "agreement_per_gb": est["agreement"].get("per_gb")}
    return doc, t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crossover.yaml")
    ap.add_argument("--out", default="results/crossover.json")
    ap.add_argument("--stats", default="results/stats.json")
    ap.add_argument("--estimate", default="results/crossover/mmap_estimate.json")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    # the estimate's settings stay out of crossover.json, whose analysis does not read them (mmap_estimate.json does)
    result = {"config": {k: v for k, v in cfg.items() if k != "mmap_estimate"}, "datasets": {}}
    for name, dcfg in cfg["datasets"].items():
        if args.only and name != args.only:
            continue
        slice_dir = REPO / dcfg["slice"]
        if not dcfg.get("records") and not (slice_dir / "meta.json").exists():
            result["datasets"][name] = {"status": "no slice"}
            continue
        result["datasets"][name] = analyse_dataset(name, dcfg, cfg, args.allow_partial)
        # v0.4.2: another family's controlled dial, read the same way, beside the release's own. Only
        # the dial: its answers are untimed, so no harness, budget or price is read for it, and nothing of Qwen2.5's changes.
        for fam, (fb, fg) in (dcfg.get("dial_families") or {}).items():
            fcache = REPO / dcfg["cache"]
            if not dcfg.get("records") and not ((fcache / f"gen_{fb}.jsonl").exists() and (fcache / f"gen_{fg}.jsonl").exists()):
                continue
            sub = analyse_dataset(name, {**dcfg, "harness": []}, {**cfg, "base_generator": fb, "big_generator": fg}, args.allow_partial)
            result["datasets"][name].setdefault("dial_families", {})[fam] = {
                "generators": [fb, fg], "dial": sub.get("dial"), "dial_missing_answers": sub.get("dial_missing_answers"),
                "partial": sub.get("partial"), "generations": sub.get("generations")}

    result["agreement_all"], result["agreement"] = agreements(result["datasets"], cfg)

    # v0.4.2: a family timed on the laptop beside Qwen2.5, read by the same analysis, dial and natural sweeps
    # alike, and kept apart, so that nothing of Qwen2.5's changes; its tokens carry the family's prefix
    result["headline_families"] = {}
    for fam, hf in (cfg.get("headline_families") or {}).items():
        fb, fg = hf["generators"]
        fcfg = {**cfg, "base_generator": fb, "big_generator": fg, "dial_same_machine": hf.get("dial_same_machine", True)}
        fds = {}
        for name, harness in hf["harness"].items():
            if not any((REPO / h / "results.json").exists() for h in harness):
                continue
            fds[name] = analyse_dataset(name, {**cfg["datasets"][name], "harness": harness}, fcfg, True)
        fall, fagree = agreements(fds, fcfg)
        result["headline_families"][fam] = {"generators": [fb, fg], "datasets": fds, "agreement_all": fall, "agreement": fagree}

    text = json.dumps(clean(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    (REPO / args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    cleaned = json.loads(text)
    tokens = build_tokens(cleaned, cfg)
    from stats_tokens import extra_tokens, headline

    tokens.update(extra_tokens(cleaned, cfg, REPO, tokens))
    for fam, hf in (cfg.get("headline_families") or {}).items():
        fr = cleaned["headline_families"].get(fam) or {}
        if fr.get("datasets"):
            fcfg = {**cfg, "base_generator": fr["generators"][0], "big_generator": fr["generators"][1], "mark_censored": True}
            sub = {"config": fcfg, "datasets": fr["datasets"], "agreement_all": fr["agreement_all"], "agreement": fr["agreement"]}
            tokens.update({hf["prefix"] + k: v for k, v in build_tokens(sub, fcfg).items()})
            tokens.update({hf["prefix"] + k: v for k, v in headline(sub, tokens, mark_censored=True).items()})
            _MARK["on"] = False
    if not args.only:
        fam_est = (cfg.get("mmap_estimate") or {}).get("family")
        doc, est_tokens = mmap_estimate(cfg, tokens, (cleaned["headline_families"].get(fam_est) or {}).get("datasets", {}))
        if doc is not None:
            (REPO / args.estimate).write_text(json.dumps(clean(doc), indent=2, sort_keys=True, allow_nan=False) + "\n",
                                              encoding="utf-8")
            print(f"wrote {args.estimate}")
            tokens.update(est_tokens)
    stats = {"source": args.out, "tokens": dict(sorted(tokens.items()))}
    (REPO / args.stats).write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.stats} ({len(tokens)} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
