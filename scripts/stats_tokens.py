"""Tokens beyond the crossover itself, for every document that quotes a number.

scripts/crossover.py merges these into results/stats.json. Each family reads
committed results or cached data only, and nothing here depends on the clock,
so a second run writes the same bytes.

  design.*    the experiment's constants (dial grid, k, seeds, budgets, cap)
  run1.*    run 1's two operating points, from results/en_rag
  models.*    sizes and revisions, from results/crossover/models_manifest.json
  data.*      slice sizes and the NQ gold-mapping coverage
  runs.*      every run's dataset, size and licence note, from its manifest
  hygiene.*   the calibration probe and the timed batches (Rule 6)
  gens.*      generations run in run 2, against the cap
  budget.*    one table per natural sweep, each row with the OS it ran on
  remeasure.* the re-measured k = 5 ladder against the rows it replaced
  repro.*     the byte-for-byte reproduction check
  headline.*  the sentences README and the report lead with
  czech_checks.*  the SQAD 3.0 copy and the filter read from the LINDAT archive
  cz.*        the Czech tables, from results/czech/analysis.json (scripts/czech_tokens.py)
  czr.*       v0.4.0's retrievers on SQAD, from results/czech/v040/retrievers.json (scripts/czech_tokens.py)
"""

from __future__ import annotations

import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

import yaml

GEN_SHORT = {"qwen2.5-1.5b": "1.5B", "qwen2.5-3b": "3B"}


def f(v, nd=3, signed=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def thousands(n) -> str:
    return f"{int(n):,}"


def pct(v, nd=1) -> str:
    return "n/a" if v is None else f"{100 * v:.{nd}f}%"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def levels_text(levels) -> str:
    return ", ".join("0" if p == 0 else f"{p:.1f}" for p in levels)


# --------------------------------------------------------------------------
def design(cfg: dict, repo: Path) -> dict:
    t = {}
    ds = cfg["datasets"].get("hotpotqa") or next(iter(cfg["datasets"].values()))
    h = ds["dial"]
    t["design.levels"] = levels_text(h["levels"])
    t["design.dial_step"] = f"{h['levels'][1] - h['levels'][0]:.1f}"
    t["design.k_dial"] = str(h["k"])
    sweeps = [yaml.safe_load((repo / f"configs/crossover/hotpotqa_k{k}.yaml").read_text(encoding="utf-8"))
              for k in (3, 5, 10)]
    t["design.ks_natural"] = ", ".join(str(s["retrieval"]["context_k"]) for s in sweeps)
    b = sweeps[1]["budgets"]
    t["design.budget_ram_gb"] = ", ".join(str(x) for x in b["ram_gb"])
    t["design.budget_retrieval_ms"] = ", ".join(str(x) for x in b["retrieval_ms"])
    t["design.budget_e2e_s"] = ", ".join(str(x) for x in b["e2e_s"])
    # the same grids as prose, unit on every value ("3 s and 8 s"); the plain lists above stay as they are
    for key, unit in (("ram_gb", "GB"), ("retrieval_ms", "ms"), ("e2e_s", "s")):
        v = [f"{x} {unit}" for x in b[key]]
        t[f"design.budget_{key}_text"] = v[0] if len(v) == 1 else ", ".join(v[:-1]) + " and " + v[-1]
    t["design.seed"] = str(cfg["seed"])
    t["design.gen_cap"] = thousands(cfg.get("generation_cap", 30000))
    en = read_json(repo / "results/en_rag/results.json")
    t["design.max_tokens_run1"] = str(en["config"]["rag"]["max_tokens"]) if en else "n/a"
    t["design.n_ctx"] = str(sweeps[1]["rag"]["n_ctx"])
    t["design.top_k"] = str(sweeps[1]["retrieval"]["top_k"])
    try:
        import sys

        sys.path.insert(0, str(repo / "scripts"))
        import calibration_probe as cp
        import sweep_queue as q

        t["design.probe_n"] = str(cp.N)
        t["design.probe_tolerance"] = f"{round(100 * cp.TOLERANCE)}%"
        t["design.probe_wait_min"] = str(q.DRIFT_WAIT_S // 60)
        t["design.contended_max"] = f"{q.CONTENDED_MAX:.0f}%"
        t["design.contended_sum"] = f"{q.CONTENDED_SUM:.0f}%"
        t["design.sample_s"] = str(q.SAMPLE_S)
        t["design.stuck_min"] = str(q.STUCK_S // 60)
        t["design.max_attempts"] = str(q.MAX_ATTEMPTS)
        t["design.max_contended"] = str(q.MAX_CONTENDED)
        t["design.contended_frac"] = f"{round(100 * q.CONTENDED_FRAC)}%"
        t["design.contended_min"] = str(q.CONTENDED_MIN)
        t["design.max_drift_waits"] = str(q.MAX_DRIFT_WAITS)
        t["design.max_drift_hours"] = f"{q.MAX_DRIFT_WAITS * q.DRIFT_WAIT_S / 3600:.0f}"
        t["design.probe_k"] = str(cp.K)
    except Exception:  # pragma: no cover - the scripts are part of the repo
        pass
    return t


def run1(repo: Path) -> dict:
    """Run 1's two regimes (results/en_rag, k = 5, max_tokens 256)."""
    r = read_json(repo / "results/en_rag/results.json")
    if not r:
        return {}
    by = {(x["retriever_id"], x["generator"]["key"]): x for x in r["results"] if x.get("status") == "ok"}
    em = {k: v["answers"]["em"] for k, v in by.items()}
    g = {rid: by[(rid, "qwen2.5-1.5b")]["metrics"]["gold_in_context@5"] for rid, _ in by}
    t = {}
    t["run1.g_low"] = f(g["tfidf-svd"])
    t["run1.g_high"] = f(g["bm25"])
    t["run1.g_e5small"] = f(g["e5-small"])
    t["run1.gap_points"] = f"{round(100 * (g['bm25'] - g['tfidf-svd']))}"
    ret_low = em[("bm25", "qwen2.5-1.5b")] - em[("tfidf-svd", "qwen2.5-1.5b")]
    gen_low = em[("tfidf-svd", "qwen2.5-3b")] - em[("tfidf-svd", "qwen2.5-1.5b")]
    ret_high = em[("e5-small", "qwen2.5-1.5b")] - em[("bm25", "qwen2.5-1.5b")]
    gen_high = em[("bm25", "qwen2.5-3b")] - em[("bm25", "qwen2.5-1.5b")]
    t["run1.ret_gain_low"] = f(ret_low, signed=True)
    t["run1.gen_gain_low"] = f(gen_low, signed=True)
    t["run1.ret_gain_high"] = f(ret_high, signed=True)
    t["run1.gen_gain_high"] = f(gen_high, signed=True)
    t["run1.ratio_low"] = f"{ret_low / gen_low:.1f}" if gen_low else "n/a"
    t["run1.ratio_high"] = f"{gen_high / ret_high:.1f}" if ret_high else "n/a"
    t["run1.n"] = str(int(by[("bm25", "qwen2.5-1.5b")]["answers"]["n_answered"]))
    return t


def models(repo: Path) -> dict:
    m = read_json(repo / "results/crossover/models_manifest.json")
    if not m:
        return {}
    t = {}
    for key, v in m["models"].items():
        b = v.get("bytes") or v.get("shipped_bytes")
        t[f"models.{key}.gb"] = f(b / 1e9, 2)
        t[f"models.{key}.revision"] = v["revision"][:8]
        t[f"models.{key}.repo"] = v["repo"]
    t["models.all_verified"] = "yes" if all(v.get("verified") for v in m["models"].values()) else "no"
    return t


def data(cfg: dict, repo: Path) -> dict:
    t = {}
    for name, d in cfg["datasets"].items():
        meta = read_json(repo / d["slice"] / "meta.json")
        if not meta:
            continue
        fp = meta.get("fingerprint")
        if isinstance(fp, str):
            fp = None
        docs = repo / d["slice"] / "corpus.jsonl"
        n_docs = sum(1 for _ in docs.open(encoding="utf-8")) if docs.exists() else None
        t[f"data.{name}.docs"] = thousands(n_docs) if n_docs is not None else "n/a"
        notes = meta.get("notes") or []
        if isinstance(notes, str):
            notes = [notes]
        for note in notes:
            mm = dict(re.findall(r"(\w+)=([\d.]+)", note))
            if "joined_to_nq_open" in mm:
                q, j, g = int(mm["beir_test_queries_with_qrels"]), int(mm["joined_to_nq_open"]), int(mm["with_mapped_gold"])
                t["data.nq.beir_queries"] = thousands(q)
                t["data.nq.joined"] = thousands(j)
                t["data.nq.joined_pct"] = pct(j / q)
                t["data.nq.mapped"] = thousands(g)
                t["data.nq.mapped_pct"] = pct(g / j)
        qrels = repo / d["slice"] / "qrels.jsonl"
        if qrels.exists():
            rows = [json.loads(line) for line in qrels.open(encoding="utf-8") if line.strip()]
            t[f"data.{name}.gold_passages"] = thousands(sum(1 for r in rows if r.get("relevance", 0) > 0))
    return t


RUNS = ["en_small", "en_nfcorpus", "en_fiqa", "cz_small", "cz_full", "cz_rag", "cz_rag_32", "en_rag",
        "crossover/natural/hotpotqa_k5", "crossover/natural/nq_k5"]


def runs(repo: Path) -> dict:
    """Each run's dataset as its manifest recorded it: name, size, licence note."""
    t = {}
    for r in RUNS:
        m = read_json(repo / "results" / r / "manifest.json")
        key = r.split("/")[-1]
        if not m:
            for fld in ("dataset", "docs", "queries", "licence_note"):
                t[f"runs.{key}.{fld}"] = "not measured"
            continue
        d = m["datasets"][0]
        t[f"runs.{key}.dataset"] = d.get("name", "n/a")
        t[f"runs.{key}.docs"] = thousands(d["n_docs"]) if d.get("n_docs") is not None else "n/a"
        t[f"runs.{key}.queries"] = thousands(d["n_queries"]) if d.get("n_queries") is not None else "n/a"
        t[f"runs.{key}.licence_note"] = d.get("license", "n/a")
    return t


def machine(repo: Path) -> dict:
    """The machine every number was measured on, as the latest run recorded it."""
    m = (read_json(repo / "results/crossover/natural/hotpotqa_k3/manifest.json") or {}).get("machine") or {}
    return {"machine.chip": m.get("chip", "n/a"), "machine.cores": str(m.get("cpu_count", "n/a")),
            "machine.ram_gb": f"{m['ram_gb']:.0f}" if m.get("ram_gb") else "n/a"}


def retrieval_runs(repo: Path) -> dict:
    """Run 1's retrieval-only runs: nDCG@10 and peak RSS per retriever."""
    t = {}
    for r in ("en_small", "en_nfcorpus", "en_fiqa", "cz_small", "cz_full"):
        res = read_json(repo / "results" / r / "results.json")
        for x in (res or {}).get("results", []):
            if x.get("status") != "ok":
                continue
            rid = x["config_id"]
            t[f"runs.{r}.{rid}.ndcg10"] = f(x["metrics"]["ndcg@10"], 4)
            t[f"runs.{r}.{rid}.peak_gb"] = f(x["measurement"]["peak_rss_gb"], 2)
            t[f"runs.{r}.{rid}.p95_ms"] = f(x["measurement"]["retrieval_ms_p95"], 1)
    return t


def czech_rag(repo: Path) -> dict:
    """Run 1's Czech end-to-end RAG (results/cz_rag, SQAD 3.0): one row per configuration."""
    r = read_json(repo / "results/cz_rag/results.json")
    if not r:
        return {}
    k = int(r["config"]["retrieval"]["context_k"])
    t, rows = {}, []
    for x in r["results"]:
        if x.get("status") != "ok":
            continue
        cid, m, a = x["config_id"], x["metrics"], x["answers"]
        t[f"runs.cz_rag.{cid}.em"] = f(a["em"])
        t[f"runs.cz_rag.{cid}.f1"] = f(a["f1"])
        t[f"runs.cz_rag.{cid}.gic"] = f(m[f"gold_in_context@{k}"])
        t["runs.cz_rag.n"] = str(int(a["n_answered"]))
        # no peak RSS or latency here: they cannot be recomputed from the tree
        rows.append([cid, f(m["ndcg@10"]), f(m[f"gold_in_context@{k}"]), f(a["em"]), f(a["f1"])])
    t["runs.cz_rag.k"] = str(k)
    t["runs.cz_rag.max_tokens"] = str(r["config"]["rag"]["max_tokens"])
    t["runs.cz_rag.table"] = md_table(["configuration", "nDCG@10", f"g @{k}", "EM", "F1"], rows)
    return t


def czech_v030(repo: Path) -> dict:
    """Every Czech table the public text quotes, from results/czech/analysis.json."""
    a = read_json(repo / "results/czech/analysis.json")
    if not a:
        return {}
    import sys

    sys.path.insert(0, str(repo / "scripts"))
    from czech_tokens import tokens

    return tokens(a)


def czech_retrievers(repo: Path) -> dict:
    """v0.4.0's retrievers on SQAD's test splits, from results/czech/v040/retrievers.json."""
    r = read_json(repo / "results/czech/v040/retrievers.json")
    if not r:
        return {}
    import sys

    sys.path.insert(0, str(repo / "scripts"))
    from czech_tokens import retriever_tokens

    return retriever_tokens(r)


def czech_checks(repo: Path) -> dict:
    """The copy of SQAD 3.0 and its filter, from results/czech_checks.json; and BenCzechMark's filter, which the debias rule follows."""
    import sys

    sys.path.insert(0, str(repo / "src"))
    from measurerag.datasets.sqad32 import BENCZECHMARK_FILTER, STEFANIK_2023_FILTER

    c = read_json(repo / "results/czech_checks.json")
    if not c:
        return {}

    fl = c["filter_read_from_the_comparison"]
    opens = fl["answer_sentence_opens_the_article"]
    return {
        "czech_checks.copy_repo": c["copy"]["repo_id"], "czech_checks.copy_revision": c["copy"]["revision"],
        "czech_checks.archive_md5": c["archive"]["md5"], "czech_checks.threshold_stated": f(c["thresholds"]["archive_p90"], 3), "czech_checks.threshold_copy": f(c["thresholds"]["copy_p90"], 2),
        "czech_checks.copy_rows": thousands(c["copy"]["rows"]), "czech_checks.archive_records": thousands(c["archive"]["records"]),
        "czech_checks.archive_records_stated": thousands(c["archive"]["records_stated_on_lindat"]),
        "czech_checks.opens_all": thousands(opens["kept"] + opens["dropped"]), "czech_checks.opens_dropped": thousands(opens["dropped"]),
        "czech_checks.opens_dropped_pct": pct(opens["dropped"] / (opens["kept"] + opens["dropped"])),
        "czech_checks.later_kept": thousands(fl["answer_sentence_comes_later"]["kept"]),
        "czech_checks.not_located": str(fl["answer_sentence_not_located"]["dropped"]),
        # BenCzechMark's filter, which the debias rule follows: cited for the comparison, not measured
        "bcm.first_chars": str(BENCZECHMARK_FILTER["first_chars"]), "bcm.share_removed": pct(BENCZECHMARK_FILTER["share_removed"], 0),
        "bcm.tfidf_cosine": f(BENCZECHMARK_FILTER["tfidf_cosine"], 1),
        # Štefánik et al. (2023), who reported the position bias and whose filtered SQAD 3.0 is the copy: cited, not measured
        "stefanik.first_chars": str(STEFANIK_2023_FILTER["first_chars"]), "stefanik.share_removed": pct(STEFANIK_2023_FILTER["share_removed"], 0),
    }


def timing_totals(repo: Path, cal_rel: str = "results/crossover/calibration", prefix: str = "") -> dict:
    """Wall-clock of the accepted timed batches, for the README's expected run time."""
    rows = read_jsonl(repo / cal_rel / "batches.jsonl")
    current: dict = {}
    for b in rows:                                  # the current accepted batch of each stage (a superseded one is replaced)
        if b.get("superseded"):
            current.pop(b["stage"], None)
        elif b.get("accepted"):
            current[b["stage"]] = b
    acc = list(current.values())
    walls = [b["wall_s"] for b in acc if b.get("wall_s")]
    if not walls:
        return {}
    return {prefix + "hygiene.wall_hours_accepted": f(sum(walls) / 3600, 1), prefix + "hygiene.n_wall_recorded": str(len(walls))}


def max_tokens_check(repo: Path) -> dict:
    """max_tokens 64 against run 1's 256, per dataset checked."""
    t = {}
    for tag, fn in (("hotpotqa", "max_tokens_check.json"), ("czech", "max_tokens_check_czech.json")):
        d = read_json(repo / "results/crossover" / fn)
        if not d:
            continue
        g = d["generators"].values()
        same = sum(v["identical_strings"] for v in g)
        total = d["n_questions"] * len(d["generators"])
        t[f"mtcheck.{tag}.identical"] = f"{same} of {total}"
        t[f"mtcheck.{tag}.n_questions"] = str(d["n_questions"])
        t[f"mtcheck.{tag}.hit_cap"] = str(sum(v["hit_cap"] for v in g))
        t[f"mtcheck.{tag}.longest"] = str(max(v.get("max_completion_tokens_at_64", 0) for v in g))
        t[f"mtcheck.{tag}.tolerance"] = str(d["tolerance"])
        t[f"mtcheck.{tag}.verdict"] = d["verdict"]
    return t


# A batch the author stopped part-way. Its record says only that the harness exited, so the reason is named
# here, and it is counted apart from the batches the queue's rules rejected.
STOPPED = {("hotpotqa_k5_ministral:bm25+ministral-3-8b", 3):
           "stopped part-way by the author, who judged that another application was slowing it; timed again from the start"}


def hygiene(repo: Path, cal_rel: str = "results/crossover/calibration", prefix: str = "", off_label: str = "off") -> dict:
    cal = repo / cal_rel
    base = read_json(cal / "baseline.json")
    probes = read_jsonl(cal / "probes.jsonl")
    batches = read_jsonl(cal / "batches.jsonl")
    t = {}
    if base:
        t[prefix + "hygiene.baseline_s"] = f(base["median_s"], 3)
        t[prefix + "hygiene.macos"] = base["macos"]
    checked = [p for p in probes if p.get("label") != "baseline" and p.get("drift") is not None]
    t[prefix + "hygiene.n_probes"] = str(len(checked))
    pre = [p for p in checked if str(p.get("label", "")).startswith("pre:")]
    waits = [p for p in pre if not p.get("ok")]
    t[prefix + "hygiene.n_drift_waits"] = str(len(waits))
    t[prefix + "hygiene.max_drift_waited"] = pct(max((abs(p["drift"]) for p in waits), default=0.0))
    # a batch a later "superseded" row takes back was measured again (scripts/sweep_queue.py --retime): the current
    # accepted batches are read for every count and table, and the superseded ones are listed apart
    current: dict = {}
    superseded = []
    for b in batches:
        if b.get("superseded"):
            if b["stage"] in current:
                superseded.append(current.pop(b["stage"]))
        elif b.get("accepted"):
            current[b["stage"]] = b
    acc = [b for b in batches if b.get("accepted") and any(b is c for c in current.values())]
    rej = [b for b in batches if b.get("accepted") is False and (b.get("stage"), b.get("attempt")) not in STOPPED]
    stopped = [b for b in batches if b.get("accepted") is False and (b.get("stage"), b.get("attempt")) in STOPPED]
    if stopped:
        t[prefix + "hygiene.batches_stopped"] = str(len(stopped))
        t[prefix + "hygiene.stopped_contention"] = ", ".join(
            f"{b['contention']['n_contended']} of {b['contention']['n_samples']}" for b in stopped)
    if superseded:
        def span(bs: list) -> str:
            a0 = datetime.fromisoformat(min(b["utc_start"] for b in bs))
            a1 = datetime.fromisoformat(max(b["utc_end"] for b in bs))
            return (f"{a0:%H:%M} to {a1:%H:%M} UTC on {a0.day} {a0:%b %Y}" if a0.date() == a1.date() else
                    f"{a0:%H:%M} UTC on {a0.day} {a0:%b} to {a1:%H:%M} UTC on {a1.day} {a1:%b %Y}")

        def label(b: dict) -> str:
            sweep = Path(b["config"]).stem.replace("_ministral", "")
            return (f"{sweep.replace('hotpotqa_k5', 'HotpotQA').replace('nq_k5', 'NQ')} "
                    f"{b['config_id'].replace('ministral-3-', '').replace('qwen2.5-', '')}")

        again = [current[b["stage"]] for b in superseded]
        first = [b for b in acc if not any(b is a for a in again)]
        t[prefix + "hygiene.retimed"] = f"{len(superseded)} of {len(acc)}"
        t[prefix + "hygiene.retimed_list"] = "; ".join(
            f"{label(b)} (end-to-end p50 {f(b['e2e_s_p50'], 2)} s when first timed, {f(n['e2e_s_p50'], 2)} s when "
            f"timed again; peak RSS {f(b['peak_rss_gb'], 2)} GB, then {f(n['peak_rss_gb'], 2)} GB)"
            for b, n in zip(superseded, again))
        t[prefix + "hygiene.retimed_utc"] = span(again)
        t[prefix + "hygiene.first_sitting"] = str(len(first))
        t[prefix + "hygiene.first_sitting_utc"] = span(first)
        cont = [f"{n['contention']['n_contended']} of {n['contention']['n_samples']}" for n in again]
        t[prefix + "hygiene.retimed_contention"] = ", ".join(cont[:-1]) + " and " + cont[-1] if len(cont) > 1 else cont[0]
        t[prefix + "hygiene.rows_marked_accepted"] = str(sum(1 for b in batches if b.get("accepted")))
        for b, n in zip(superseded, again):
            key = f"{prefix}hygiene.retimed.{Path(b['config']).stem.replace('_ministral', '')}.{b['config_id']}"
            t[key + ".p50_shift_ms"] = str(round(abs(n["e2e_s_p50"] - b["e2e_s_p50"]) * 1000))
            t[key + ".peak_shift_gb"] = f(abs(n["peak_rss_gb"] - b["peak_rss_gb"]), 2)
    t[prefix + "hygiene.batches_accepted"] = str(len(acc))
    t[prefix + "hygiene.batches_gated"] = str(sum(1 for b in acc if b.get("hygiene", "on") == "on"))
    t[prefix + "hygiene.batches_ungated"] = str(sum(1 for b in acc if b.get("hygiene") == "off"))
    t[prefix + "hygiene.batches_rejected"] = str(len(rej))
    reasons: dict[str, int] = {}
    for b in rej:
        reasons[b.get("reason", "?")] = reasons.get(b.get("reason", "?"), 0) + 1
    t[prefix + "hygiene.rejected_reasons"] = ("none" if not reasons else
                                     "; ".join(f"{k}: {v}" for k, v in sorted(reasons.items())))
    posts = [b["post_probe"]["drift"] for b in acc if (b.get("post_probe") or {}).get("drift") is not None]
    t[prefix + "hygiene.post_drift_range"] = (f"{pct(min(posts))} to {pct(max(posts))}" if posts else "n/a")
    t[prefix + "hygiene.post_drift_median"] = pct(statistics.median(posts)) if posts else "n/a"
    rows = []
    for b in acc:
        pre_d = (b.get("pre_probe") or {}).get("drift")
        post_d = (b.get("post_probe") or {}).get("drift")
        c = b.get("contention") or {}
        rows.append([Path(b["config"]).stem.replace("_ministral", "").replace("hotpotqa_", "HotpotQA ").replace("nq_", "NQ "), b["config_id"],
                     "gated" if b.get("hygiene", "on") == "on" else off_label,
                     f(b["wall_s"] / 60, 0) if b.get("wall_s") else "n/a", pct(pre_d), pct(post_d),
                     f"{c['n_contended']} of {c['n_samples']}" if c else "not recorded", b.get("macos") or "n/a"])
    if prefix:      # v0.4.2: a family's table also says which batches the switched-off rule would have rejected, and the chip
        rows = [r + ["yes" if (b.get("contention") or {}).get("contaminated") else "no",
                     " → ".join(f"{x:.0f}" for x in b["chip_c"]) if b.get("chip_c") else "n/a"] for r, b in zip(rows, acc)]
        t[prefix + "hygiene.table_batches"] = md_table(
            ["sweep", "configuration", "hygiene", "wall min", "probe before", "probe after", "contended samples", "macOS",
             "over the gate's contention rule", "chip °C, start → end"], rows) if rows else "_no timed batch has run yet._"
        over = [b for b in acc if (b.get("contention") or {}).get("contaminated")]
        t[prefix + "hygiene.would_reject"] = f"{len(over)} of {len(acc)}"
        t[prefix + "hygiene.would_reject_list"] = "; ".join(
            f"{Path(b['config']).stem.replace('_ministral', '').replace('hotpotqa_k5', 'HotpotQA').replace('nq_k5', 'NQ')} "
            f"{b['config_id'].replace('ministral-3-', '').replace('qwen2.5-', '')} ({b['contention']['n_contended']} of {b['contention']['n_samples']})"
            for b in over) or "none"
    else:
        t[prefix + "hygiene.table_batches"] = md_table(
            ["sweep", "configuration", "hygiene", "wall min", "probe before", "probe after", "contended samples", "macOS"], rows) \
            if rows else "_no timed batch has run yet._"
    # ungated batches whose contention would have rejected them, and badly so
    heavy = [b for b in acc if (b.get("contention") or {}).get("n_samples")
             and b["contention"]["n_contended"] / b["contention"]["n_samples"] > 0.25]
    t[prefix + "hygiene.heavily_contended"] = ("none" if not heavy else "; ".join(
        f"{Path(b['config']).stem} {b['config_id'].replace('qwen2.5-', '').replace('ministral-3-', '')} "
        f"({round(100 * b['contention']['n_contended'] / b['contention']['n_samples'])}% of samples)" for b in heavy))
    rrows = [[Path(b["config"]).stem.replace("_ministral", ""), b["config_id"], b.get("attempt"),
              STOPPED.get((b.get("stage"), b.get("attempt")), b.get("reason"))]
             for b in sorted(rej + stopped, key=lambda b: b.get("utc_start", ""))]
    t[prefix + "hygiene.table_rejected"] = md_table(["sweep", "configuration", "attempt", "reason"], rrows) \
        if rrows else "_none._"
    return t


RUN2_HARNESS = ["results/crossover/natural/hotpotqa_k3", "results/crossover/natural/hotpotqa_k5",
                     "results/crossover/natural/hotpotqa_k10", "results/crossover/natural/nq_k5",
                     "results/crossover/superseded/hotpotqa_k5_2026-09-14"]


def harness_rows(repo: Path, d: str) -> list[dict]:
    r = read_json(repo / d / "results.json")
    return [x for x in (r or {}).get("results", []) if x.get("status") == "ok"]


def gens(cfg: dict, repo: Path) -> dict:
    t = {}
    dial = 0
    for name, d in cfg["datasets"].items():
        for g in (cfg["base_generator"], cfg["big_generator"]):
            p = repo / d["cache"] / f"gen_{g}.jsonl"
            if p.exists():
                dial += sum(1 for line in p.open(encoding="utf-8") if line.strip())
    harness = sum(int(x["answers"]["n_answered"]) for d in RUN2_HARNESS for x in harness_rows(repo, d))
    batches = read_jsonl(repo / "results/crossover/calibration/batches.jsonl")
    discarded = sum(int(b.get("n_answered") or 0) for b in batches if b.get("accepted") is False)
    session = sum(int(b.get("n_answered") or 0) for b in batches)
    total = dial + harness + discarded
    t["gens.dial_cache"] = thousands(dial)
    t["gens.harness"] = thousands(harness)
    t["gens.discarded"] = thousands(discarded)
    t["gens.total_run2"] = thousands(total)
    t["gens.this_run"] = thousands(session)
    cap = int(cfg.get("generation_cap", 30000))
    t["gens.cap"] = thousands(cap)
    t["gens.under_cap"] = "yes" if total <= cap else "no"
    rates: dict[str, list[float]] = {}
    for b in batches:
        if b.get("accepted") and b.get("n_answered") and b.get("e2e_s_p50"):
            g = b["config_id"].split("+", 1)[1]
            rates.setdefault(g, []).append(b["e2e_s_p50"])
    for g, v in rates.items():
        t[f"gens.p50_s.{GEN_SHORT.get(g, g)}"] = f"{statistics.median(v):.2f}"
    return t


def budget_tables(repo: Path, sweeps: dict | None = None, prefix: str = "", peaks: dict | None = None) -> dict:
    """`peaks`, {(sweep folder, config_id): GB}, gives each row another peak RSS (the estimate without the
    double-counted mapped pages, scripts/crossover.py) and checks its budgets again; the rows keep the measured order."""
    t = {}
    sweeps = sweeps or {"hotpotqa_k5": "results/crossover/natural/hotpotqa_k5",
                        "hotpotqa_k3": "results/crossover/natural/hotpotqa_k3",
                        "hotpotqa_k10": "results/crossover/natural/hotpotqa_k10",
                        "nq_k5": "results/crossover/natural/nq_k5",
                        "run1_k5": "results/en_rag"}
    for name, d in sweeps.items():
        res = read_json(repo / d / "results.json")
        if not res:
            t[f"{prefix}budget.table_{name}"] = "_not measured._"
            t[f"{prefix}budget.os_{name}"] = "n/a"
            continue
        k = res["config"]["retrieval"]["context_k"]
        rows = []
        for x in sorted(harness_rows(repo, d), key=lambda x: (x["measurement"]["peak_rss_gb"], x["config_id"])):
            m = x["measurement"]
            fits = [v["budget"] for v in x.get("verdicts", []) if v.get("fits")]
            if peaks is not None:
                from measurerag.budget import Budget, Measurement, check_all

                m = {**m, "peak_rss_gb": peaks[(d, x["config_id"])]}
                fits = [v.budget for v in check_all(Measurement(**m), [Budget.from_dict(b) for b in res["budgets"]]) if v.fits]
            ram = sorted({int(re.match(r"ram(\d+)g", b).group(1)) for b in fits if re.match(r"ram(\d+)g", b)})
            e2e3 = any(b.endswith("e2e3s") for b in fits)
            rows.append([
                x["config_id"].replace("qwen2.5-", "").replace("ministral-3-", ""), f(x["metrics"].get(f"gold_in_context@{k}")),
                f(x["answers"]["em"]), f(m["peak_rss_gb"], 2), f(m["index_bytes"] / 1e6, 1),
                f(m["model_bytes"] / 1e9, 2), f(m["retrieval_ms_p50"], 1), f(m["e2e_s_p50"], 2),
                f(m["e2e_s_p95"], 2), f"{ram[0]} GB" if ram else "none", "yes" if e2e3 else "no",
                (x.get("machine") or {}).get("os_version", "n/a"),
            ])
        t[f"{prefix}budget.table_{name}"] = md_table(
            ["configuration", f"g @{k}", "EM", "peak RSS GB", "index MB", "model GB", "retrieval p50 ms",
             "e2e p50 s", "e2e p95 s", "smallest RAM budget", "fits 3 s e2e", "macOS"], rows) \
            if rows else "_not measured._"
        oses = sorted({r[-1] for r in rows})
        t[f"{prefix}budget.os_{name}"] = ", ".join(oses) if oses else "n/a"
    return t


def remeasure(repo: Path) -> dict:
    """The re-measured k = 5 ladder against run 1's rows and the superseded e5-base pair."""
    new_dir = repo / "results/crossover/natural/hotpotqa_k5"
    old = {}
    for d in ("results/en_rag", "results/crossover/superseded/hotpotqa_k5_2026-09-14"):
        for x in harness_rows(repo, d):
            old[x["config_id"]] = (d, x)
    rows, same, n = [], 0, 0
    for x in sorted(harness_rows(repo, "results/crossover/natural/hotpotqa_k5"), key=lambda x: x["config_id"]):
        cid = x["config_id"]
        if cid not in old:
            continue
        d, o = old[cid]
        pn = read_json(new_dir / f"predictions_{cid}.json")
        po = read_json(repo / d / f"predictions_{cid}.json")
        if pn is None or po is None:
            ident = None        # prediction files are cached data, not committed
        else:
            ident = pn == po
            n += 1
            same += ident
        mo, mn = o["measurement"], x["measurement"]
        rows.append([cid.replace("qwen2.5-", ""), "not checked (no cached predictions)" if ident is None else
                     ("yes" if ident else "NO"),
                     f"{f(mo['peak_rss_gb'], 2)} → {f(mn['peak_rss_gb'], 2)}",
                     f"{f(mo['e2e_s_p50'], 2)} → {f(mn['e2e_s_p50'], 2)}",
                     pct((mn["e2e_s_p50"] - mo["e2e_s_p50"]) / mo["e2e_s_p50"]),
                     f"{(o.get('machine') or {}).get('os_version', '?')} → {(x.get('machine') or {}).get('os_version', '?')}"])
    t = {"remeasure.identical": f"{same} of {n}",
         "remeasure.table": md_table(["configuration", "answers identical", "peak RSS GB (old → new)",
                                      "e2e p50 s (old → new)", "p50 change", "macOS (old → new)"], rows)
         if rows else "_the re-measured ladder has no rows yet._"}
    return t


def repro(repo: Path) -> dict:
    import hashlib

    r = read_json(repo / "results/crossover/repro_check.json")
    if not r:
        return {"repro.crossover_identical": "not yet run", "repro.stats_identical": "not yet run"}
    # the record speaks for the analysis that ships only if it checked this very crossover.json
    shipped = repo / "results/crossover.json"
    if shipped.exists() and r.get("crossover_sha256") and hashlib.sha256(shipped.read_bytes()).hexdigest() != r["crossover_sha256"]:
        return {"repro.crossover_identical": "not run on this analysis", "repro.stats_identical": "not run on this analysis"}
    return {"repro.crossover_identical": "yes" if r.get("crossover_identical") else "NO",
            "repro.stats_identical": "yes" if r.get("stats_identical") else "NO"}


def headline(result: dict, tokens: dict, mark_censored: bool = False) -> dict:
    """The g* sentence for a dataset, stated honestly when there is no root."""
    t = {}
    for name, d in result.get("datasets", {}).items():
        roots = (d.get("dial") or {}).get("roots")
        if not roots:
            continue
        step = "->".join(result["config"]["priced_step"])
        pu = roots.get("per_unit") or {}
        for tag, r in (("raw", roots["raw"]["isotonic"]),
                       ("gb", pu.get(f"per_gb|{step}", {})),
                       ("ms_total", pu.get(f"per_ms_total|{step}", {}))):
            key = f"headline.{name}.gstar_{tag}"
            if r and r.get("undefined"):
                t[key] = "undefined: the priced retriever step does not add this cost"
                continue
            r = r.get("isotonic") if r and "isotonic" in r else r
            if not r:
                t[key] = "not measured"
                continue
            lv = [x for x in d["dial"]["levels"] if x.get("ret_minus_gen")]
            lo_g = min(x["g_any"]["point"] for x in lv)
            hi_g = max(x["g_any"]["point"] for x in lv)
            if r.get("point") is not None:
                cens = (f"; {r['n_no_root']} of {r['n_boot']} resamples have no root" if r.get("n_no_root") else "")
                # does the crossing rest on the no-gold level alone?
                lv = [x for x in d["dial"]["levels"] if x.get("ret_minus_gen")]
                ret = [x for x in lv if x["ret_minus_gen"]["point"] > 0]
                gen = [x for x in lv if x["ret_minus_gen"]["point"] < 0]
                minority = ret if len(ret) <= len(gen) else gen
                rests = ("; it rests on the no-gold level alone" if tag == "raw" and minority
                         and all(x["g_any"]["point"] == 0 for x in minority) else "")
                side = "upper" if r.get("no_root_above", 0) >= r.get("no_root_below", 0) else "lower"
                edge = f", the {side} end censored" if mark_censored and r.get("interval_censored") else ""
                t[key] = f"g* = {f(r['point'], 2)} [{f(r['lo'], 2)}, {f(r['hi'], 2)}]{edge}{cens}{rests}"
            else:
                side = "above" if r.get("no_root_above", 0) >= r.get("no_root_below", 0) else "below"
                edge = hi_g if side == "above" else lo_g
                t[key] = (f"no crossing inside g = {f(lo_g, 2)} to {f(hi_g, 2)} (censored {side} {f(edge, 2)}; "
                          f"{r['n_no_root']} of {r['n_boot']} resamples have no root)")
    return t


def family_timing(repo: Path, cfg: dict) -> dict:
    """A family timed on the laptop beside Qwen2.5 (configs/crossover.yaml, headline_families): its budget tables,
    its own measurement record, and what its sweeps fit, under the family's prefix."""
    t: dict = {}
    for fam, hf in (cfg.get("headline_families") or {}).items():
        p = hf["prefix"]
        sweeps = {f"{name}_k5": dirs[0] for name, dirs in hf["harness"].items()}
        t.update(budget_tables(repo, sweeps, p))
        if hf.get("calibration"):
            t.update(hygiene(repo, hf["calibration"], p, "off"))
            t.update(timing_totals(repo, hf["calibration"], p))
        rows = [x for d in sweeps.values() for x in harness_rows(repo, d)]
        if rows:
            fit = [x for x in rows if any(v.get("fits") for v in x.get("verdicts", []))]
            t[p + "budget.configs_fitting"] = f"{len(fit)} of {len(rows)}"
            t[p + "budget.e2e_p95_min_s"] = f(min(x["measurement"]["e2e_s_p95"] for x in rows), 1)
            t[p + "budget.e2e_p95_max_s"] = f(max(x["measurement"]["e2e_s_p95"] for x in rows), 1)
            t[p + "budget.e2e_p50_min_s"] = f(min(x["measurement"]["e2e_s_p50"] for x in rows), 1)
            t[p + "budget.peak_rss_range_gb"] = (f"{f(min(x['measurement']['peak_rss_gb'] for x in rows), 1)} to "
                                                 f"{f(max(x['measurement']['peak_rss_gb'] for x in rows), 1)}")
            t[p + "budget.macos"] = ", ".join(sorted({(x.get("machine") or {}).get("os_version", "n/a") for x in rows}))
            # what a reader needs to scale the tables: index build, retrieval p95, and which ceiling binds
            t[p + "budget.build_s_range"] = (f"{thousands(round(min(x['build']['build_seconds'] for x in rows)))} to "
                                             f"{thousands(round(max(x['build']['build_seconds'] for x in rows)))}")
            t[p + "budget.retrieval_p95_range_ms"] = (f"{f(min(x['measurement']['retrieval_ms_p95'] for x in rows), 1)} to "
                                                      f"{f(max(x['measurement']['retrieval_ms_p95'] for x in rows), 1)}")
            grid = read_json(repo / next(iter(sweeps.values())) / "results.json")["budgets"]
            ram, ret, e2e = (max(g["ram_gb"] for g in grid), min(g["retrieval_ms"] for g in grid),
                             max(g["e2e_s"] for g in grid if g.get("e2e_s")))
            t[p + "budget.under_largest_ram"] = f"{sum(x['measurement']['peak_rss_gb'] <= ram for x in rows)} of {len(rows)}"
            t[p + "budget.under_tightest_retrieval"] = f"{sum(x['measurement']['retrieval_ms_p95'] <= ret for x in rows)} of {len(rows)}"
            t[p + "budget.under_loosest_e2e"] = f"{sum(x['measurement']['e2e_s_p95'] <= e2e for x in rows)} of {len(rows)}"
        night = NIGHT.get(fam)
        if night and hf.get("calibration"):
            acc = [b for b in read_jsonl(repo / hf["calibration"] / "batches.jsonl") if b.get("accepted")]
            # the build's own batch of each: the earliest accepted one, so that a later re-timing appended to the log
            # (docs/REPRODUCE.md, step 3) does not move the span
            nb = [min(bs, key=lambda b: b["utc_start"]) for k in sorted(night)
                  if (bs := [b for b in acc if (Path(b["config"]).stem, b["config_id"]) == k])]
            if len(nb) == len(night):
                start, end = min(b["utc_start"] for b in nb), max(b["utc_end"] for b in nb)
                nxt = min((b["utc_start"] for b in acc if b["utc_start"] > end), default=None)
                day = datetime.fromisoformat(start)
                t[p + "hygiene.night_utc"] = (f"{start[11:16]} to {end[11:16]} UTC on {day.day} {day.strftime('%b %Y')}")
                if nxt:
                    t[p + "hygiene.night_next_gap_s"] = str(int((datetime.fromisoformat(nxt) - datetime.fromisoformat(end)).total_seconds()))
                prev = max((b["utc_end"] for b in acc if b["utc_end"] <= start), default=None)
                if prev:
                    t[p + "hygiene.night_prev_gap_s"] = str(int((datetime.fromisoformat(start) - datetime.fromisoformat(prev)).total_seconds()))
    return t


# The three HotpotQA batches of Ministral 3's sweeps that ran in one night and answered more slowly. The author picked
# them out after the run, by their latencies; no recorded condition marks them. Their bounds are read from batches.jsonl.
NIGHT = {"ministral-3": {("hotpotqa_k5_ministral", "e5-base+ministral-3-8b"), ("hotpotqa_k5_ministral", "bm25+ministral-3-3b"),
                         ("hotpotqa_k5_ministral", "bm25+ministral-3-8b")}}


def extra_tokens(result: dict, cfg: dict, repo: Path, tokens: dict) -> dict:
    t: dict = {}
    for fam in (design(cfg, repo), run1(repo), models(repo), data(cfg, repo), runs(repo), machine(repo), retrieval_runs(repo), czech_rag(repo), czech_checks(repo), czech_v030(repo), czech_retrievers(repo), timing_totals(repo), max_tokens_check(repo), hygiene(repo),
                gens(cfg, repo), budget_tables(repo), remeasure(repo), repro(repo), headline(result, tokens), family_timing(repo, cfg)):
        t.update(fam)
    return t
