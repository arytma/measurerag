"""Result tables and the two curves.

The table is the deliverable: one row per configuration, quality next to the
cost that bought it, and a FITS/NO column per budget. A quality number whose
configuration misses the budget is printed in brackets - it is real, but it is
not available to you on that box.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

GB = 1024**3
MB = 1024**2


def flatten(result: dict) -> dict:
    """One result -> one flat CSV row."""
    m = result.get("measurement", {})
    metrics = result.get("metrics", {})
    row: dict[str, Any] = {
        "config_id": result.get("config_id"),
        "retriever_id": result.get("retriever_id"),
        "retriever_name": result.get("retriever_name", result.get("retriever_id")),
        "dataset": result.get("dataset"),
        "language": result.get("language"),
        "device": result.get("device"),
        "status": result.get("status"),
        "n_docs": result.get("n_docs"),
        "n_queries": result.get("n_queries"),
        "ndcg@10": metrics.get("ndcg@10"),
        "recall@10": metrics.get("recall@10"),
        "recall@100": metrics.get("recall@100"),
        "peak_rss_gb": m.get("peak_rss_gb"),
        "index_mb": round(m.get("index_bytes", 0) / MB, 2),
        "model_mb": round(m.get("model_bytes", 0) / MB, 2),
        "retrieval_ms_p50": round(m.get("retrieval_ms_p50", 0.0), 2),
        "retrieval_ms_p95": round(m.get("retrieval_ms_p95", 0.0), 2),
        "build_seconds": round(result.get("build", {}).get("build_seconds", 0.0), 2),
    }
    # Provenance for the dense rows. A mean-pooled model that should be CLS
    # produces a plausible number, not an error, so which
    # pooling bought a score belongs next to the score.
    build = result.get("build", {})
    if build.get("pooling"):
        row["pooling"] = build["pooling"]
        row["pooling_source"] = build.get("pooling_source")
    for key, value in metrics.items():
        if key.startswith("gold_in_context@"):
            row[key] = value
    if result.get("rag_status") == "ok":
        gen = result.get("generator", {})
        ans = result.get("answers", {})
        row["em"] = ans.get("em")
        row["f1"] = ans.get("f1")
        row["e2e_s_p50"] = round(m.get("e2e_s_p50") or 0.0, 3)
        row["e2e_s_p95"] = round(m.get("e2e_s_p95") or 0.0, 3)
        # NOT a decode rate. The denominator is the whole call, and prefilling
        # five passages dominates a four-token extractive answer - SQAD reports
        # 0.72 here, which reads as broken hardware and is nothing of the kind.
        # Named for what it measures.
        row["gen_tok_per_s_incl_prefill"] = round(m.get("gen_tokens_per_s") or 0.0, 2)
        answered = ans.get("n_answered") or 0
        tokens = gen.get("completion_tokens") or 0
        row["completion_tokens"] = tokens
        row["tokens_per_answer"] = round(tokens / answered, 2) if answered else None
        row["generator"] = gen.get("key")
    for v in result.get("verdicts", []):
        row[f"fits:{v['budget']}"] = "FITS" if v["fits"] else "NO"
    if result.get("status") != "ok":
        row["error"] = result.get("error", "")
    return row


def write_csv(rows: Sequence[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------
# terminal table
# --------------------------------------------------------------------------
def budget_table(results: Sequence[dict], budgets: Sequence[dict], quality_key: str = "ndcg@10") -> str:
    """Plain-text budget table. No colour, no rich - it has to paste into a
    thesis, an email and a CI log unchanged."""
    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]

    headers = ["configuration", quality_key, "peak RSS", "index", "ret p95"]
    has_rag = any(r.get("rag_status") == "ok" for r in ok)
    if has_rag:
        headers += ["EM", "F1", "e2e p95"]
    budget_names = [b["name"] for b in budgets]
    headers += budget_names

    rows: list[list[str]] = []
    for r in sorted(ok, key=lambda x: -(x.get("metrics", {}).get(quality_key) or 0)):
        m = r.get("measurement", {})
        verdicts = {v["budget"]: v["fits"] for v in r.get("verdicts", [])}
        fits_any = any(verdicts.values())
        q = r.get("metrics", {}).get(quality_key)
        q_str = "-" if q is None else (f"{q:.4f}" if fits_any else f"[{q:.4f}]")
        row = [
            str(r.get("config_id") or r.get("retriever_id")),
            q_str,
            f"{m.get('peak_rss_gb', 0):.2f} GB",
            f"{m.get('index_bytes', 0) / MB:.1f} MB",
            f"{m.get('retrieval_ms_p95', 0):.1f} ms",
        ]
        if has_rag:
            a = r.get("answers", {})
            row += [
                f"{a.get('em', 0):.3f}" if r.get("rag_status") == "ok" else "-",
                f"{a.get('f1', 0):.3f}" if r.get("rag_status") == "ok" else "-",
                f"{(m.get('e2e_s_p95') or 0):.2f} s" if m.get("e2e_s_p95") else "-",
            ]
        row += ["FITS" if verdicts.get(n) else "no" for n in budget_names]
        rows.append(row)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out += [line(r) for r in rows]

    if not rows:
        out.append("(no configuration completed)")

    out.append("")
    out.append("Budgets:")
    for b in budgets:
        e2e = f", e2e p95 <= {b['e2e_s']:g} s" if b.get("e2e_s") is not None else ""
        out.append(
            f"  {b['name']}: RAM <= {b['ram_gb']:g} GB, retrieval p95 <= "
            f"{b['retrieval_ms']:g} ms{e2e}"
        )
    out.append("")
    out.append("[value] = measured, but the configuration fits no budget in this grid.")

    if failed:
        out.append("")
        out.append("Failed configurations:")
        for r in failed:
            out.append(f"  {r.get('retriever_id')}: {r.get('error', 'unknown error')[:160]}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# curves
# --------------------------------------------------------------------------
def _plot_curve(results, x_getter, x_label, title, out_path, budget_lines, quality_key):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    for r in ok:
        x = x_getter(r)
        y = r.get("metrics", {}).get(quality_key)
        if x is None or y is None:
            continue
        fits = any(v["fits"] for v in r.get("verdicts", []))
        ax.scatter(
            x,
            y,
            s=90,
            marker="o" if fits else "X",
            color="#1f77b4" if fits else "#999999",
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        ax.annotate(
            r.get("config_id") or r.get("retriever_id"),
            (x, y),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=8,
        )

    for value, label in budget_lines:
        ax.axvline(value, linestyle="--", linewidth=1, color="#c44e52", alpha=0.7)
        ax.annotate(
            label,
            (value, ax.get_ylim()[0]),
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=7,
            color="#c44e52",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(quality_key)
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle=":")
    fig.text(
        0.01,
        0.01,
        "circle = fits at least one budget;  X = fits none",
        fontsize=7,
        color="#444444",
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)


def curve_quality_vs_memory(results, budgets, out_path, quality_key="ndcg@10"):
    lines = sorted({(b["ram_gb"], f"{b['ram_gb']:g} GB") for b in budgets})
    return _plot_curve(
        results,
        lambda r: r.get("measurement", {}).get("peak_rss_gb"),
        "peak resident memory (GB)",
        "Quality vs memory, with the budget walls",
        out_path,
        lines,
        quality_key,
    )


def curve_quality_vs_latency(results, budgets, out_path, quality_key="ndcg@10"):
    lines = sorted({(b["retrieval_ms"], f"{b['retrieval_ms']:g} ms") for b in budgets})
    return _plot_curve(
        results,
        lambda r: r.get("measurement", {}).get("retrieval_ms_p95"),
        "retrieval latency p95 (ms, single query)",
        "Quality vs latency, with the budget walls",
        out_path,
        lines,
        quality_key,
    )
