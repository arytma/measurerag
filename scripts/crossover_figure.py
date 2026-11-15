"""Draw the crossover figure from results/crossover.json (never from caches).

    uv run python scripts/crossover_figure.py

One figure per dataset, two panels sharing the g axis: raw EM gain, and EM gain
per GB of resident memory. Blue is the retriever move, orange the generator
move; circles with bands are the controlled dial, open diamonds the 1.5B -> 3B
upgrade measured at real retrievers (the natural sweep at the dial's k). The
shaded vertical strip is g* and its 95% interval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RET, GEN = "#2a78d6", "#eb6834"   # categorical slots 1 and 2, validated
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def panel(ax, d: dict, cfg: dict, unit: str) -> None:
    levels = d["dial"]["levels"]
    base, big = cfg["base_generator"], cfg["big_generator"]
    costs = d["costs"]
    if unit == "gb":
        key = f"per_gb|{'->'.join(cfg['priced_step'])}"
        rs = d["dial"]["roots"]["per_unit"].get(key, {})
        if rs.get("undefined") or not rs:
            ax.axis("off")
            ax.text(0.5, 0.5, "per-GB price undefined or not yet measured", ha="center", transform=ax.transAxes, color=MUTED)
            return
        cr, cg = rs["retriever_cost"], rs["generator_cost"]
        iso = rs["isotonic"]
    else:
        cr = cg = 1.0
        iso = d["dial"]["roots"]["raw"]["isotonic"]

    def series(field, cost):
        xs, ys, lo, hi = [], [], [], []
        for lv in levels:
            v = lv.get(field)
            if not v:
                continue
            xs.append(lv["g_any"]["point"])
            ys.append(v["point"] / cost)
            lo.append(v["lo"] / cost)
            hi.append(v["hi"] / cost)
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        return [[a[i] for i in order] for a in (xs, ys, lo, hi)]

    if iso.get("point") is not None:
        ax.axvspan(iso["lo"], iso["hi"], color=GRID, alpha=0.7, lw=0, zorder=0)
        ax.axvline(iso["point"], color=MUTED, lw=1, ls="--", zorder=1)
        ax.annotate(f"g* {iso['point']:.2f} [{iso['lo']:.2f}, {iso['hi']:.2f}]", (iso["point"], 1.0),
                    xycoords=("data", "axes fraction"), xytext=(4, -12), textcoords="offset points",
                    fontsize=8, color=INK)
    elif iso:
        # no crossing inside the dial's range: say so where the mark would be
        gs = [lv["g_any"]["point"] for lv in levels if lv.get("ret_minus_gen")]
        side = "above" if iso.get("no_root_above", 0) >= iso.get("no_root_below", 0) else "below"
        ax.annotate(f"no crossing in g {min(gs):.2f} to {max(gs):.2f}: g* censored {side}",
                    (0.02, 0.97), xycoords="axes fraction", va="top", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    for field, cost, color, label in (("delta_ret", cr, RET, "retriever step"), ("delta_gen", cg, GEN, "generator 1.5B → 3B")):
        x, y, lo, hi = series(field, cost)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)
        ax.plot(x, y, color=color, lw=2, zorder=3)
        ax.plot(x, y, "o", color=color, ms=6, mec="white", mew=1.5, zorder=4, label=f"{label} (dial)")
        if x:
            ax.annotate(label, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8,
                        color=INK, va="center")

    # Real retrievers' generator step at their own g is comparable with the dial's
    # Δ_gen at that g. Real retriever steps are not: each moves g by its own amount,
    # so they are checked against the dial's prediction in tables, not drawn here.
    nat = d.get("natural", {}).get(str(d["dial"]["k"]))
    if nat:
        checks = [c for c in nat["generator_checks"] if not c.get("by_construction")]
        gx = [c["g_any"] for c in checks]
        gy = [c["delta_gen"]["point"] / cg for c in checks]
        ax.plot(gx, gy, "D", mfc="white", mec=GEN, mew=1.8, ms=7, zorder=5,
                label="generator step at a real retriever (natural sweep)")
    ax.set_xlim(-0.02, 1.32)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("gold-in-context g (any gold passage in the k passages)", fontsize=9, color=INK)
    ax.set_ylabel("EM gain" if unit == "raw" else "EM gain per GB resident memory", fontsize=9, color=INK)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/crossover.json")
    ap.add_argument("--out-dir", default="results/crossover")
    args = ap.parse_args()
    data = json.loads((REPO / args.inp).read_text(encoding="utf-8"))
    cfg = data["config"]
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, d in data["datasets"].items():
        if "dial" not in d:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
        panel(axes[0], d, cfg, "raw")
        axes[0].set_title(f"{name}: raw EM gain of each move (n = {d['n_questions']})", fontsize=10, color=INK, loc="left")
        panel(axes[1], d, cfg, "gb")
        axes[1].set_title("per GB of peak RSS", fontsize=10, color=INK, loc="left")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        path = out_dir / f"crossover_{name}.png"
        fig.savefig(path, metadata={"Software": None})
        plt.close(fig)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
