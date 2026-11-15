"""The release's figures, drawn from shipped results files only.

    .venv/bin/python scripts/figures.py              # write every figure to results/figures/
    .venv/bin/python scripts/figures.py --check      # draw into a temporary folder; exit 1 unless every file is byte-identical

The data figures come in a light and a dark version, for GitHub's two themes, and the documents embed both
with <picture>. Every value a figure draws is read through `Source.get`, which records the file and the JSON
path it came from; results/figures/figures.json lists them, and scripts/number_audit.py reads each one again.
No number is typed into this file except settings of the drawing itself.

The files are deterministic: SVG with its text as paths in the DejaVu Sans that matplotlib bundles, a fixed
hash salt and no date, and one PNG, the social preview card, with no metadata. The clean-clone test draws them
again and compares them byte for byte.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from czech_tokens import COMPLETE_SETS, FAMILY_LABEL, FAMILY_MODELS, FULL_LABEL, PREFIX, SIZE_NAME  # noqa: E402,F401

OUT = Path("results/figures")
ANALYSIS = "results/czech/analysis.json"
CROSSOVER = "results/crossover.json"

# The family palette. Ink, grid and surfaces are GitHub's own, so a figure sits on the page in either theme.
THEMES = {
    "light": {"surface": "#ffffff", "ink": "#1f2328", "ink2": "#59636e", "grid": "#e8ebef", "axis": "#c8d1da", "box": "#f6f8fa",
              "family": {"gemma-3n": "#4ba739", "gemma-4": "#287951", "ministral-3": "#4d94e4", "qwen2.5": "#d02716", "qwen3.5": "#7e353a"}},
    "dark": {"surface": "#0d1117", "ink": "#f0f6fc", "ink2": "#9198a1", "grid": "#21262d", "axis": "#3d444d", "box": "#151b23",
             "family": {"gemma-3n": "#0c7407", "gemma-4": "#539d71", "ministral-3": "#5a83cd", "qwen2.5": "#a05457", "qwen3.5": "#f84d54"}},
}
MARKER = {"gemma-3n": "v", "gemma-4": "s", "ministral-3": "D", "qwen2.5": "o", "qwen3.5": "^"}     # a second encoding, so identity is never colour alone
# Each family takes its colour by its maker's country, as the author asked: green for the US (Google), blue for
# France (Mistral AI), red and dark red for China (Alibaba's Qwen). One maker's two generations share the hue; the newer takes the step
# with more contrast against the surface. Every slot passes the dataviz validator over all pairs in both themes, and every slot
# is at least 3:1 against its surface. Gemma 3n has the least contrast in both themes, and each maker's newer generation more than
# its older. In the light theme Qwen3.5's red and Gemma 4's green are 7.9 apart for deuteranopia, and in the dark theme Qwen3.5's
# red and Gemma 3n's green 7.2 apart for protanopia; both are legal only with a second encoding, which the markers, the labels and
# the flags give. Bringing Gemma 3n's light green to 3:1 took Gemma 4's and Qwen2.5's light colours a step darker, so that the
# red and green slots stay apart by lightness under deuteranopia. Every figure draws the families
# in this order, the US models first, then the French, then the Chinese, each maker's generations older first; the families figure
# alone orders each panel's models by their score on the released 200 (below).
ORDER = ("gemma-3n", "gemma-4", "ministral-3", "qwen2.5", "qwen3.5")
COUNTRY = {"gemma-3n": "US", "gemma-4": "US", "ministral-3": "FR", "qwen2.5": "CN", "qwen3.5": "CN"}
RC = {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path", "svg.hashsalt": "measurerag",
      "axes.unicode_minus": False, "path.simplify": False}
SVG_META = {"Date": None, "Creator": None}


class Source:
    """A shipped results file. `get` returns one value and records where it came from."""

    def __init__(self, root: Path, rel: str, record: list):
        self.rel, self.record = rel, record
        self.data = json.loads((root / rel).read_text(encoding="utf-8"))

    def get(self, *path):
        v = self.data
        for p in path:
            v = v[p]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise TypeError(f"{self.rel} {list(path)} is not a number")
        entry = {"source": self.rel, "path": list(path), "value": v}
        if entry not in self.record:
            self.record.append(entry)
        return v

    def raw(self, *path):
        """Structure only (which keys exist, flags); never a value that is drawn."""
        v = self.data
        for p in path:
            v = v[p]
        return v


def interval(src: Source, *path) -> dict:
    return {k: src.get(*path, k) for k in ("point", "lo", "hi")}


def family_view(*path_to_set, fam):
    return list(path_to_set) if fam == "qwen2.5" else [*path_to_set, "families", fam]


def style_axes(ax, t: dict, grid_axis: str) -> None:
    ax.set_facecolor(t["surface"])
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    if grid_axis == "x":
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(t["axis"])
    else:
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(t["axis"])
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], labelsize=8.5, length=0)


def heading(fig, t: dict, title: str, subtitle: str, top: float = 0.975) -> None:
    fig.text(0.012, top, title, ha="left", va="top", fontsize=12.5, color=t["ink"], weight="bold")
    fig.text(0.012, top - 0.068, subtitle, ha="left", va="top", fontsize=8.8, color=t["ink2"], linespacing=1.45)


# ---- flags: drawn as vector shapes (an emoji font cannot be drawn by matplotlib or kept as outlines), simplified to label size ----
FLAG_RATIO = {"US": 1.9, "FR": 1.5, "CN": 1.5, "CZ": 1.5}


def _star(cx: float, cy: float, r: float, sx: float, rot: float = 0.0) -> list[tuple[float, float]]:
    import math

    pts = []
    for i in range(10):
        a = math.pi / 2 + rot + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.382
        pts.append((cx + rr * math.cos(a) * sx, cy + rr * math.sin(a)))
    return pts


def draw_flag(fig, x: float, y: float, h: float, country: str, t: dict) -> float:
    """A flag of height h (figure fraction), its left edge at x and its middle at y; returns its width."""
    from matplotlib.patches import Polygon, Rectangle

    W, H = fig.get_size_inches()
    sx = H / W                                                   # one unit of height, in figure-width fraction
    w = h * FLAG_RATIO[country] * sx
    y0 = y - h / 2
    add = lambda p: fig.add_artist(p)                            # noqa: E731
    # Each flag is one solid field with its other colours laid on top, never shapes edge to edge: where two shapes meet,
    # antialiasing lets a hairline of the surface show through the seam, which reads as banding (dark lines in the dark theme)
    if country == "FR":
        add(Rectangle((x, y0), w, h, transform=fig.transFigure, facecolor="#FFFFFF", edgecolor="none", zorder=10))
        for i, c in ((0, "#0055A4"), (2, "#EF4135")):
            add(Rectangle((x + i * w / 3, y0), w / 3, h, transform=fig.transFigure, facecolor=c, edgecolor="none", zorder=10))
    elif country == "CN":
        add(Rectangle((x, y0), w, h, transform=fig.transFigure, facecolor="#EE1C25", edgecolor="none", zorder=10))
        u = h / 20
        add(Polygon(_star(x + 5 * u * sx, y0 + 15 * u, 3 * u, sx), closed=True, transform=fig.transFigure, facecolor="#FFFF00", edgecolor="none", zorder=11))
        for cx, cy in ((10, 18), (12, 16), (12, 13), (10, 11)):
            add(Polygon(_star(x + cx * u * sx, y0 + cy * u, u, sx), closed=True, transform=fig.transFigure, facecolor="#FFFF00", edgecolor="none", zorder=11))
    elif country == "US":
        # seven stripes, not thirteen: at label size a thirteenth of the flag is under two pixels, and the screen rounds
        # such thin stripes unevenly, which shows as dithering; seven keep each stripe about three pixels tall
        sh = h / 7
        add(Rectangle((x, y0), w, h, transform=fig.transFigure, facecolor="#FFFFFF", edgecolor="none", zorder=10))
        for i in range(0, 7, 2):                                # four red stripes on the white field
            add(Rectangle((x, y0 + i * sh), w, sh, transform=fig.transFigure, facecolor="#B22234", edgecolor="none", zorder=10))
        add(Rectangle((x, y0 + 3 * sh), w * 0.4, 4 * sh, transform=fig.transFigure, facecolor="#3C3B6E", edgecolor="none", zorder=11))
    elif country == "CZ":
        add(Rectangle((x, y0), w, h, transform=fig.transFigure, facecolor="#FFFFFF", edgecolor="none", zorder=10))
        add(Polygon([(x, y0), (x + w, y0), (x + w, y0 + h / 2), (x, y0 + h / 2)], closed=True, transform=fig.transFigure, facecolor="#D7141A", edgecolor="none", zorder=10))
        add(Polygon([(x, y0), (x + w / 2, y0 + h / 2), (x, y0 + h)], closed=True, transform=fig.transFigure, facecolor="#11457E", edgecolor="none", zorder=11))
    add(Rectangle((x, y0), w, h, transform=fig.transFigure, facecolor="none", edgecolor=t["axis"], linewidth=0.4, zorder=12))
    return w


def flag_height(fig, txt, scale: float) -> float:
    """A flag's height, in figure fraction, from its label's font size: the same for every label of that size, whatever
    its letters (a label's measured box is taller with ascenders and descenders)."""
    return txt.get_fontsize() / 72 / fig.get_figheight() * 1.2 * scale


CAP_HEIGHT = 0.729     # DejaVu Sans's capital height, in em (1493 of 2048 font units)


def cap_middle(fig, txt, r) -> float:
    """The figure-fraction height of the middle of a label's capital letters: a flag centred there sits level with the
    text before it. The label's measured box runs from below the baseline (room for descenders) to above the capitals,
    so its middle sits lower. The baseline is the one the SVG was written with, where the measuring pass recorded it:
    the SVG renderer draws a label about 0.03 em above the baseline its layout reports."""
    _, y0 = fig.transFigure.inverted().transform(txt.get_window_extent(r))[0]
    prop = txt.get_fontproperties()
    # matplotlib lays a line out with at least the descent of "lp", whatever its letters, so the baseline sits that high
    descent = max(r.get_text_width_height_descent(txt.get_text(), prop, ismath=False)[2],
                  r.get_text_width_height_descent("lp", prop, ismath=False)[2])
    em = txt.get_fontsize() * fig.dpi / 72
    base = y0 + descent / fig.bbox.height
    drawn = list(getattr(r, "baselines", {}).get(txt.get_text(), []))
    if getattr(r, "ink", None) is not None:
        drawn += [b for b in [ink_baseline(fig, txt, r)] if b is not None]
    if drawn:
        nearest = min(drawn, key=lambda b: abs(b - base))
        if abs(nearest - base) < 0.25 * em / fig.bbox.height:
            base = nearest
    return base + CAP_HEIGHT * em / 2 / fig.bbox.height


class saved_layout:
    """A renderer that lays text out as the saved file does, so that a flag placed from it lands where the text is
    drawn: the SVG renderer at 72 dpi for the SVG figures (unhinted metrics), the figure's own canvas for the PNG card. The
    screen renderer's hinted metrics place tick and legend labels a little differently, which left flags low."""

    def __init__(self, fig, svg: bool = True):
        self.fig, self.svg, self.dpi = fig, svg, fig.dpi

    def __enter__(self):
        if not self.svg:
            import numpy as np

            self.fig.canvas.draw()
            r = self.fig.canvas.get_renderer()
            r.ink = np.asarray(self.fig.canvas.buffer_rgba())[..., :3].astype(int)   # what the PNG will hold
            return r
        import io

        from matplotlib.backends.backend_svg import RendererSVG

        import html
        import re

        self.fig.set_dpi(72)                                   # as savefig does for SVG
        w, h = self.fig.get_size_inches()
        buf = io.StringIO()
        r = RendererSVG(w * 72, h * 72, buf, image_dpi=72)
        self.fig.draw(r)
        # each label's baseline as written: the SVG puts a comment with the text before the group that places its glyphs
        r.baselines = {}
        for m in re.finditer(r'<!-- (.*?) -->(?:(?!<!--).)*?<g [^>]*?transform="translate\(([-0-9.e]+) ([-0-9.e]+)\)', buf.getvalue(), re.S):
            r.baselines.setdefault(html.unescape(m.group(1)), []).append(1 - float(m.group(3)) / (h * 72))
        return r

    def __exit__(self, *exc):
        self.fig.set_dpi(self.dpi)
        return False


def ink_baseline(fig, txt, r) -> float | None:
    """A label's baseline as the PNG canvas drew it: in its box, the row most of its letters end on."""
    import numpy as np
    from matplotlib.colors import to_rgb

    img = r.ink
    H = img.shape[0]
    bb = txt.get_window_extent(r)
    top, bottom = max(int(H - bb.y1), 0), min(int(np.ceil(H - bb.y0)), H)
    crop = img[top:bottom, max(int(bb.x0), 0):int(np.ceil(bb.x1))]
    ink = np.abs(crop - np.array(to_rgb(fig.get_facecolor())) * 255).sum(axis=2) > 200
    ends = [np.nonzero(ink[:, c])[0][-1] for c in range(ink.shape[1]) if ink[:, c].any()]
    return 1 - (top + np.bincount(ends).argmax() + 1) / H if ends else None


def flags_after(fig, items: list, t: dict, gap: float = 0.009, scale: float = 0.78, svg: bool = True) -> None:
    """After the layout is final: a flag right after each text artist, sized to its line. items: [(text_artist, country), ...]."""
    with saved_layout(fig, svg) as r:
        inv = fig.transFigure.inverted()
        for txt, country in items:
            if not country or not txt.get_text():
                continue
            (x0, y0), (x1, y1) = inv.transform(txt.get_window_extent(r))
            draw_flag(fig, x1 + gap, cap_middle(fig, txt, r), flag_height(fig, txt, scale), country, t)


def name_run(fig, ax, items: list, t: dict, y_axes: float = 1.02, fontsize: float = 8.3, per_line: int = 3, sep: float = 0.014) -> None:
    """After the layout is final: names written left to right from the axes' top-left corner, each followed by its flag,
    wrapped every `per_line` names onto the line above. items: [(name, country), ...], or [(name, country, family), ...] to
    put the family's marker, in its colour, before each name, so that the run is the panel's legend."""
    with saved_layout(fig) as r:
        _name_run(fig, ax, items, t, r, y_axes, fontsize, per_line, sep)


def _name_run(fig, ax, items, t, r, y_axes, fontsize, per_line, sep) -> None:
    inv = fig.transFigure.inverted()
    left, base = inv.transform(ax.transAxes.transform((0, y_axes)))
    lines = [items[i:i + per_line] for i in range(0, len(items), per_line)]
    line_h = None
    placed = []                                                 # (text, marker x, flag x, country, family)
    W, H = fig.get_size_inches()
    for row, line in enumerate(reversed(lines)):               # the last line sits lowest, the first on top
        x = left
        y = base + row * (line_h or 0)
        for name, country, *fam in line:
            if fam:                                             # the marker, then the name a little to its right
                x += 0.006 + (0.008 if x > left else 0)          # and a wider gap after the previous flag
            txt = fig.text(x + (0.008 if fam else 0), y, name, fontsize=fontsize, color=t["ink"] if fam else t["ink2"],
                           ha="left", va="bottom")
            (x0, y0), (x1, y1) = inv.transform(txt.get_window_extent(r))
            line_h = line_h or (y1 - y0) * 1.35
            placed.append((txt, x, x1 + 0.004, country, fam[0] if fam else None))
            x = x1 + 0.004 + flag_height(fig, txt, 0.78) * FLAG_RATIO[country] * H / W + sep
    # the names now exist: measure them as the file will draw them, then put each marker and flag level with its name
    with saved_layout(fig) as r2:
        for txt, xm, xf, country, fam in placed:
            mid = cap_middle(fig, txt, r2)
            if fam:
                fig.add_artist(plt.Line2D([xm], [mid], marker=MARKER[fam], markersize=7, color=t["family"][fam],
                                          markeredgecolor=t["surface"], markeredgewidth=1.5, linestyle="none",
                                          transform=fig.transFigure))
            draw_flag(fig, xf, mid, flag_height(fig, txt, 0.78), country, t)


def country_of_model(model: str) -> str:
    fam = next(f for f, pair in FAMILY_MODELS.items() if model in pair)
    return COUNTRY[fam]


def save_svg(fig, path: Path, t: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", facecolor=t["surface"], metadata=SVG_META)
    plt.close(fig)


# ---- F1: the families, model for model -----------------------------------------------------------------

def families_values(a: Source) -> dict:
    out = {}
    for k, _ in COMPLETE_SETS:
        for fam, pair in FAMILY_MODELS.items():
            for i, g in enumerate(pair):
                out[(k, fam, i)] = interval(a, *family_view("sets", k, fam=fam), "answers", f"e5-small+{g}", "secondary_em_yes_no_mapped")
    return out


def set_tick(label: str) -> str:
    """A set's name on two lines, as the README's prose names it: 'SQAD 3.2, debiased at 0.138' -> 'SQAD 3.2\\ndebiased'."""
    head, _, tail = label.partition(", ")
    return f"{head}\n{tail.replace(' at 0.138', '').replace('the released 200', 'released 200')}"


# The families figure's sets, left to right: the SQAD 3.0 copy's released 200 first, then the copy debiased, then SQAD 3.2's two
# sets. The author's choice, made after the results were read; every table keeps the plan's order.
FAMILIES_SETS = ("released", "copy_debiased", "sqad32_debiased", "sqad32_full")
FAMILIES_ORDER_BY = "released"


def families_model_order(pts: dict, i: int) -> list[str]:
    """One panel's families, best first by the point estimate on the released 200; a tie keeps the country order."""
    score = lambda f: pts[(FAMILIES_ORDER_BY, f, i)]["point"]  # noqa: E731
    return sorted(ORDER, key=lambda f: (score(f) != score(f), -score(f) if score(f) == score(f) else 0, ORDER.index(f)))


def families(a: Source, t: dict, path: Path) -> None:
    # The score goes up, so a mark higher on the page is a better score; the sets run along the bottom and, inside
    # each set, the models stand side by side, in each panel best first on the released 200, the order the names above
    # the panel give. (Rows of families, one above the other, read as a ranking they were not.)
    pts = families_values(a)
    lo = min(v["lo"] for v in pts.values())
    hi = max(v["hi"] for v in pts.values())
    fig, axes = plt.subplots(1, 2, figsize=(9, 6.2), sharey=True, facecolor=t["surface"])
    label = dict(COMPLETE_SETS)
    sets = [(k, label[k]) for k in FAMILIES_SETS]
    orders = [families_model_order(pts, i) for i in (0, 1)]
    for i, (ax, title) in enumerate(zip(axes, ("Each family's smaller model", "Each family's larger model"))):
        style_axes(ax, t, "y")
        ax.spines["left"].set_visible(False)
        offset = dict(zip(orders[i], (-0.32, -0.16, 0.0, 0.16, 0.32)))   # left to right, as the names above the panel
        for col, (k, _) in enumerate(sets):
            for fam in orders[i]:
                v, c, x = pts[(k, fam, i)], t["family"][fam], col + offset[fam]
                ax.plot([x, x], [v["lo"], v["hi"]], color=c, linewidth=2, solid_capstyle="round", zorder=2)
                ax.plot([x], [v["point"]], marker=MARKER[fam], markersize=7.5, color=c, markeredgecolor=t["surface"], markeredgewidth=2, linestyle="none", zorder=3)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([set_tick(lab) for _, lab in sets], color=t["ink"], fontsize=8.3, linespacing=1.3)
        ax.set_xlim(-0.55, len(sets) - 0.45)
        ax.set_ylim(lo - 0.02, hi + 0.02)
        ax.set_title(title, fontsize=10, color=t["ink"], loc="left", pad=34)
    axes[0].set_ylabel("Exact match, yes and no mapped (higher is better)", fontsize=8.8, color=t["ink2"])
    axes[1].tick_params(axis="y", labelleft=False)
    # no figure-wide legend: each panel names its models once, each with its marker and colour, in the order of its marks
    heading(fig, t, "Czech RAG: five model families on four SQAD sets",
            "Higher is better: a mark is the share of questions answered right, the line through it its 95% interval.\n"
            "Each model answers with e5-small's top 3 passages; models are paired by place in their family, not by size.\n"
            "Compare the marks within a set: the sets keep different questions. Each panel lists its models best first on the released 200.")
    fig.tight_layout(rect=(0, 0, 1, 0.80), w_pad=2.0)
    for i, ax in enumerate(axes):
        name_run(fig, ax, [(FULL_LABEL[FAMILY_MODELS[f][i]], COUNTRY[f], f) for f in orders[i]], t)
    save_svg(fig, path, t)


# ---- F2: what the budget buys (the English dial, as levels) --------------------------------------------

DIAL_FAMILY = "ministral-3"        # v0.4.2: the dial figure shows Ministral 3's two models, the author's choice


def dial_values(c: Source, ds: str) -> dict:
    out = {"n": c.get("datasets", ds, "n_questions"), "levels": []}
    small, big = c.raw("datasets", ds, "dial_families", DIAL_FAMILY, "generators")
    for i, _ in enumerate(c.raw("datasets", ds, "dial_families", DIAL_FAMILY, "dial", "levels")):
        base = ("datasets", ds, "dial_families", DIAL_FAMILY, "dial", "levels", i)
        out["levels"].append({"p": c.get(*base, "p"), "g": c.get(*base, "g_any", "point"),
                              "small": interval(c, *base, "em", small), "big": interval(c, *base, "em", big)})
    out["levels"].sort(key=lambda lv: lv["g"])
    return out


def dial(c: Source, t: dict, path: Path) -> None:
    col = t["family"][DIAL_FAMILY]
    small_name, big_name = (FULL_LABEL[g] for g in FAMILY_MODELS[DIAL_FAMILY])
    flagged: list = []
    data = {ds: dial_values(c, ds) for ds in ("hotpotqa", "nq")}
    top = max(lv["big"]["hi"] for d in data.values() for lv in d["levels"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True, facecolor=t["surface"])
    for ax, (ds, name) in zip(axes, (("hotpotqa", "HotpotQA"), ("nq", "Natural Questions"))):
        d = data[ds]
        style_axes(ax, t, "y")
        xs = [lv["g"] for lv in d["levels"]]
        ends = {key: d["levels"][-1][key]["point"] for key in ("small", "big")}
        nudge = max(0.0, 0.035 * top - abs(ends["big"] - ends["small"])) / 2          # keep the two end labels apart
        hi_key = "big" if ends["big"] >= ends["small"] else "small"
        for key, ls, face, label in (("small", (0, (4, 2.5)), t["surface"], small_name), ("big", "-", col, big_name)):
            ys = [lv[key]["point"] for lv in d["levels"]]
            for lv in d["levels"]:
                ax.plot([lv["g"], lv["g"]], [lv[key]["lo"], lv[key]["hi"]], color=col, linewidth=1, alpha=0.55, zorder=2)
            ax.plot(xs, ys, color=col, linewidth=2, linestyle=ls, solid_joinstyle="round", zorder=3)
            ax.plot(xs, ys, linestyle="none", marker="o", markersize=8, markerfacecolor=face, markeredgecolor=col if key == "small" else t["surface"],
                    markeredgewidth=1.8 if key == "small" else 2, zorder=4)
            flagged.append((ax.text(xs[-1] + 0.03, ys[-1] + (nudge if key == hi_key else -nudge), label, va="center", ha="left", fontsize=8.5,
                                    color=t["ink"]), COUNTRY[DIAL_FAMILY]))
        ax.set_xlim(-0.03, 1.36)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_ylim(0, top * 1.08)
        ax.set_title(f"{name}, {d['n']} questions", fontsize=10, color=t["ink"], loc="left", pad=8)
        ax.set_xlabel("gold-in-context g: a gold passage among the 5", fontsize=8.8, color=t["ink2"])
    axes[0].set_ylabel("exact match (EM)", fontsize=8.8, color=t["ink2"])
    # how to read it, once, on the left panel: the generator step is the gap between the lines at one level;
    # the retriever step is one dial level to the right along a line
    lv = data["hotpotqa"]["levels"]
    a, b = lv[-3], lv[-2]
    ax = axes[0]
    ax.add_patch(FancyArrowPatch((a["g"], a["small"]["point"]), (b["g"], b["small"]["point"]), arrowstyle="-|>", mutation_scale=11,
                                 color=t["ink2"], linewidth=1.1, shrinkA=7, shrinkB=7, connectionstyle="arc3,rad=0.28", zorder=5))
    ax.text((a["g"] + b["g"]) / 2 + 0.02, (a["small"]["point"] + b["small"]["point"]) / 2 - 0.035, "retriever step:\none dial level more gold",
            fontsize=8, color=t["ink2"], ha="left", va="top", linespacing=1.3)
    g = lv[-4]
    ax.add_patch(FancyArrowPatch((g["g"], g["small"]["point"]), (g["g"], g["big"]["point"]), arrowstyle="-|>", mutation_scale=11,
                                 color=t["ink2"], linewidth=1.1, shrinkA=6, shrinkB=6, zorder=5))
    ax.text(g["g"] - 0.02, g["big"]["point"] + 0.022, f"generator step: {SIZE_NAME[FAMILY_MODELS[DIAL_FAMILY][0]]} to {SIZE_NAME[FAMILY_MODELS[DIAL_FAMILY][1]]}",
            fontsize=8, color=t["ink2"], ha="right", va="bottom")
    handles = [plt.Line2D([], [], color=col, linewidth=2, marker="o", markersize=8, markerfacecolor=col, markeredgecolor=t["surface"], markeredgewidth=2, label=big_name),
               plt.Line2D([], [], color=col, linewidth=2, linestyle=(0, (4, 2.5)), marker="o", markersize=8, markerfacecolor=t["surface"], markeredgecolor=col,
                          markeredgewidth=1.8, label=small_name)]
    leg = axes[0].legend(handles=handles, loc="upper left", frameon=False, fontsize=8.8, handlelength=4.4)   # long enough to show the dash
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
        flagged.append((txt, COUNTRY[DIAL_FAMILY]))
    # the subtitle's last sentence is read from the levels drawn, not written ahead of them
    gold = [lv for d in data.values() for lv in d["levels"] if lv["g"] > 0]
    better = all(lv["big"]["point"] > lv["small"]["point"] for lv in gold)
    tail = (f" Wherever gold is present, {SIZE_NAME[FAMILY_MODELS[DIAL_FAMILY][1]]} scores above {SIZE_NAME[FAMILY_MODELS[DIAL_FAMILY][0]]}"
            " (point estimates)." if better else "")
    heading(fig, t, f"What the budget buys: more gold in the context, or a bigger generator ({FAMILY_LABEL[DIAL_FAMILY]})",
            "The controlled dial: e5-base's top 5 passages, each gold passage swapped out for the next non-gold one with probability p.\n"
            f"Each point is one dial level, with its 95% interval.{tail}")
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=2.5)
    flags_after(fig, flagged, t)
    save_svg(fig, path, t)


# ---- F3: answer in Czech, please -----------------------------------------------------------------------

LANGUAGE_SET = "sqad32_full"


def answer_language_values(a: Source) -> list[dict]:
    rows = []
    for fam in ORDER:
        for g in FAMILY_MODELS[fam]:
            base = (*family_view("sets", LANGUAGE_SET, fam=fam), "answers", f"e5-small+{g}")
            rows.append({"fam": fam, "model": g, "english": a.get(*base, "yes_no_answer_classes", "yes or no in English"),
                         "questions": a.get(*base, "compliance", "yes_no_questions")})
    return rows


def answer_language(a: Source, t: dict, path: Path) -> None:
    rows = answer_language_values(a)
    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=t["surface"])
    style_axes(ax, t, "x")
    ys, y = [], 0.0
    for i, r in enumerate(rows):
        if i and r["fam"] != rows[i - 1]["fam"]:
            y += 0.45                               # a little air between the families
        ys.append(y)
        y += 1
    ys = [max(ys) - v for v in ys]                  # first row on top
    for r, y in zip(rows, ys):
        share = r["english"] / r["questions"]
        ax.barh(y, share, height=0.6, color=t["family"][r["fam"]], linewidth=0, zorder=2)
        ax.text(share + 0.008, y, f"{r['english']} of {r['questions']}", va="center", ha="left", fontsize=8.8, color=t["ink"])
    ax.set_yticks(ys)
    ax.set_yticklabels([FULL_LABEL[r["model"]] for r in rows], color=t["ink"], fontsize=9.2)
    ax.tick_params(axis="y", pad=30)                # room for the flag after each name, before the axis
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_xlabel("share of the yes/no questions answered yes or no, in English", fontsize=8.8, color=t["ink2"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=t["family"][f], label=FAMILY_LABEL[f]) for f in ORDER]
    leg = fig.legend(handles=handles, loc="upper left", ncol=5, frameon=False, bbox_to_anchor=(0.004, 0.80), fontsize=9, handlelength=1.1, columnspacing=3.6)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    n = rows[0]["questions"]
    heading(fig, t, "Asked for ano or ne, who answers yes or no in English?",
            f"The {n} yes/no questions of SQAD 3.2 full, answered with e5-small's top 3 under the prompt that asks for ano or ne.\n"
            "The rest answer ano or ne, decline, or say something else. Every set and configuration is in docs/CROSSOVER.md.")
    fig.tight_layout(rect=(0, 0, 1, 0.74))
    flags_after(fig, [(txt, COUNTRY[f]) for txt, f in zip(leg.get_texts(), ORDER)]
                + [(lab, country_of_model(r["model"])) for lab, r in zip(ax.get_yticklabels(), rows)], t)
    save_svg(fig, path, t)


# ---- F4: the gap between the SQAD versions -------------------------------------------------------------

GAP_METRIC, GAP = "em_yes_no_mapped", "G1"


def version_gap_values(a: Source) -> list[dict]:
    rows = []
    for fam in ORDER:
        small, big = FAMILY_MODELS[fam]
        for cfg, model, retr in ((f"bm25+{small}", small, "BM25"), (f"e5-small+{small}", small, "e5-small"), (f"e5-small+{big}", big, "e5-small")):
            base = ("planned_gap", GAP_METRIC, cfg, GAP)
            rows.append({"fam": fam, "label": f"{retr}, {FULL_LABEL[model]}", "model": model,
                         "all": interval(a, *base, "gap", "gap"), "passes": interval(a, *base, "on_questions_the_rule_passes", "gap"),
                         "n_all": (a.get(*base, "gap", "n_target"), a.get(*base, "gap", "n_copy")),
                         "n_passes": (a.get(*base, "on_questions_the_rule_passes", "n_target"), a.get(*base, "on_questions_the_rule_passes", "n_copy"))})
    return rows


def excludes_zero(v: dict) -> bool:
    return v["lo"] > 0 or v["hi"] < 0


def version_gap(a: Source, t: dict, path: Path) -> None:
    rows = version_gap_values(a)
    fig, ax = plt.subplots(figsize=(9, 8.0), facecolor=t["surface"])
    style_axes(ax, t, "x")
    ys, y = [], 0.0
    for i, r in enumerate(rows):
        if i and r["fam"] != rows[i - 1]["fam"]:
            y += 0.5
        ys.append(y)
        y += 1
    ys = [max(ys) - v for v in ys]
    ax.axvline(0, color=t["axis"], linewidth=1, zorder=1)
    for r, y in zip(rows, ys):
        c = t["family"][r["fam"]]
        for key, dy, face in (("all", 0.17, c), ("passes", -0.17, t["surface"])):
            v = r[key]
            ax.plot([v["lo"], v["hi"]], [y + dy, y + dy], color=c, linewidth=1.6, solid_capstyle="round", zorder=2)
            ax.plot([v["point"]], [y + dy], marker="o", markersize=8, markerfacecolor=face, markeredgecolor=c if key == "passes" else t["surface"],
                    markeredgewidth=1.8 if key == "passes" else 2, linestyle="none", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], color=t["ink"], fontsize=9)
    ax.tick_params(axis="y", pad=22)                # room for the flag after each name, before the axis
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    span = max(max(abs(r[k]["lo"]), abs(r[k]["hi"])) for r in rows for k in ("all", "passes"))
    ax.set_xlim(-span * 1.15, span * 1.15)
    ax.set_xlabel("EM with yes and no mapped, SQAD 3.2 full minus the SQAD 3.0 copy's released 200 [95%]",
                  fontsize=8.8, color=t["ink2"])
    ax.text(0.004, 1.0, "SQAD 3.2 higher →", transform=ax.get_xaxis_transform(), fontsize=8, color=t["ink2"], ha="left", va="bottom")
    ax.text(-0.004, 1.0, "← SQAD 3.0 copy higher", transform=ax.get_xaxis_transform(), fontsize=8, color=t["ink2"], ha="right", va="bottom")
    ink = t["ink2"]
    handles = [plt.Line2D([], [], color=ink, marker="o", markersize=8, markerfacecolor=ink, markeredgecolor=t["surface"], markeredgewidth=2, linewidth=1.6,
                          label=f"all questions ({rows[0]['n_all'][0]:,} and {rows[0]['n_all'][1]})"),
               plt.Line2D([], [], color=ink, marker="o", markersize=8, markerfacecolor=t["surface"], markeredgecolor=ink, markeredgewidth=1.8, linewidth=1.6,
                          label=f"the questions the debias rule passes ({rows[0]['n_passes'][0]:,} and {rows[0]['n_passes'][1]})")]
    leg = fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.005, 0.845), fontsize=8.8)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    est_all = sum(excludes_zero(r["all"]) for r in rows)
    est_passes = sum(excludes_zero(r["passes"]) for r in rows)
    heading(fig, t, "The gap between the SQAD versions falls without the questions the debias rule flags",
            f"On all questions the gap's interval excludes zero in {est_all} of {len(rows)} configurations; "
            f"on the questions the rule passes, in {est_passes} of {len(rows)}.\nColour is the family; filled and open marks are the two sets of questions.")
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    flags_after(fig, [(lab, country_of_model(r["model"])) for lab, r in zip(ax.get_yticklabels(), rows)], t)
    save_svg(fig, path, t)


# ---- F5: the pipeline (no data) ------------------------------------------------------------------------

def pipeline(t: dict, path: Path) -> None:
    fig = plt.figure(figsize=(9, 4.0), facecolor=t["surface"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")
    ink, ink2 = t["ink"], t["ink2"]

    def box(x, y, w, h, title, sub="", dashed=False, fill=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", linewidth=1.1,
                                    edgecolor=t["axis"] if not dashed else ink2, facecolor=fill or t["box"], linestyle=(0, (3, 2)) if dashed else "-", zorder=2))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=10, color=ink, weight="bold", zorder=3)
        if sub:
            ax.text(x + w / 2, y + h - 5.4, sub, ha="center", va="top", fontsize=8.2, color=ink2, linespacing=1.35, zorder=3)

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12, color=ink2, linewidth=1.2, zorder=3))

    def bracket(x0, x1, y, label):
        ax.plot([x0, x0, x1, x1], [y + 1, y, y, y + 1], color=ink2, linewidth=1, zorder=3)
        ax.text((x0 + x1) / 2, y - 1, label, ha="center", va="top", fontsize=8, color=ink2, zorder=3)

    def knob(x, y, label):
        ax.text(x, y, label, ha="center", va="center", fontsize=7.8, color=ink, zorder=4,
                bbox={"boxstyle": "round,pad=0.35,rounding_size=0.8", "facecolor": t["surface"], "edgecolor": ink2, "linewidth": 0.9})

    # the machine
    ax.add_patch(FancyBboxPatch((1.2, 2.4), 97.6, 38.6, boxstyle="round,pad=0,rounding_size=1.6", linewidth=1, edgecolor=t["axis"], facecolor=t["surface"], zorder=1))
    ax.text(2.6, 39.6, "One machine, no network: no API keys, no cloud models", ha="left", va="top", fontsize=9, color=ink, weight="bold")
    ax.text(97.4, 39.6, "budget: peak resident memory of the whole process, GB", ha="right", va="top", fontsize=8, color=ink2)

    y, h = 14.5, 14.5
    box(3, y, 11, h, "Question")
    box(18, y, 18, h, "Retriever", "BM25, or a dense\nencoder: e5-small,\ne5-base, bge-m3, ...")
    box(40, y, 13, h, "Top k", "the first k\npassages of\nthe ranking")
    box(57, y, 18, h, "Generator", "a local GGUF model,\nllama.cpp, greedy")
    box(79, y, 8.2, h, "Answer")
    box(88.8, y, 9.0, h, "Score", "EM and F1\nagainst gold")
    for x0, x1 in ((14, 18), (36, 40), (53, 57), (75, 79), (87.2, 88.8)):
        arrow(x0, x1, y + h / 2)
    # the families, as a key beside their colour: the US's and France's in the first column, China's in the second
    for i, fam in enumerate(ORDER):
        col, row = (0, i) if i < 3 else (1, i - 3)
        yy = y + 3.4 - row * 1.15
        xx = 57.8 + col * 9.2
        ax.plot([xx], [yy + 1.2], marker=MARKER[fam], markersize=5.5, color=t["family"][fam], markeredgecolor=t["box"], markeredgewidth=1, zorder=4)
        ax.text(xx + 1.3, yy + 1.2, FAMILY_LABEL[fam], ha="left", va="center", fontsize=8, color=ink2, zorder=4)
    # the moves that are compared, and the dial
    knob(27, y - 0.2, "retriever step")
    knob(66, y - 0.2, "generator step")
    ax.add_patch(FancyBboxPatch((28.5, 30.8), 36, 5.8, boxstyle="round,pad=0,rounding_size=1.2", linewidth=1.1, edgecolor=ink2, facecolor=t["surface"],
                                linestyle=(0, (3, 2)), zorder=2))
    ax.text(46.5, 34.9, "The dial, in the experiment only", ha="center", va="center", fontsize=9, color=ink, weight="bold", zorder=3)
    ax.text(46.5, 32.4, "each gold passage swapped out with probability p", ha="center", va="center", fontsize=7.6, color=ink2, zorder=3)
    ax.add_patch(FancyArrowPatch((46.5, 30.8), (46.5, y + h), arrowstyle="-|>", mutation_scale=10, color=ink2, linewidth=1, linestyle=(0, (3, 2)), zorder=3))
    # the latency budgets
    bracket(18, 36, 10.4, "budget: retrieval latency, ms")
    bracket(3, 87.2, 6.0, "budget: end-to-end latency, s")
    save_svg(fig, path, t)


# ---- F6: the social preview card -----------------------------------------------------------------------

def social_card(a: Source, about: str, path: Path) -> None:
    t = THEMES["light"]
    pts = families_values(a)
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=t["surface"])
    for i, fam in enumerate(ORDER):                                   # bands of the family colours along the top
        fig.add_artist(plt.Rectangle((i / len(ORDER), 0.975), 1 / len(ORDER), 0.025, transform=fig.transFigure, color=t["family"][fam], linewidth=0))
    fig.text(0.06, 0.80, "measureRAG", fontsize=56, weight="bold", color=t["ink"], ha="left", va="top")
    fig.text(0.06, 0.60, about, fontsize=20, color=t["ink2"], ha="left", va="top", linespacing=1.4, wrap=True)
    names = []
    order = families_model_order(pts, 0)                             # as in F1's smaller panel: best first on the released 200
    for i, fam in enumerate(order):
        y = 0.32 - i * 0.058
        fig.add_artist(plt.Line2D([0.065], [y], marker=MARKER[fam], markersize=13, color=t["family"][fam], markeredgecolor=t["surface"], markeredgewidth=2, transform=fig.transFigure))
        names.append((fig.text(0.085, y, FAMILY_LABEL[fam], fontsize=17, color=t["ink"], va="center", ha="left"), COUNTRY[fam]))
    ax = fig.add_axes((0.62, 0.17, 0.35, 0.56))
    style_axes(ax, t, "y")
    ax.spines["left"].set_visible(False)
    label = dict(COMPLETE_SETS)
    sets = [(k, label[k]) for k in FAMILIES_SETS]                   # as in F1: the released 200 first
    offset = dict(zip(order, (-0.32, -0.16, 0.0, 0.16, 0.32)))     # as in F1: score up, models side by side
    for col, (k, _) in enumerate(sets):
        for fam in order:
            v, c, x = pts[(k, fam, 0)], t["family"][fam], col + offset[fam]
            ax.plot([x, x], [v["lo"], v["hi"]], color=c, linewidth=3, solid_capstyle="round")
            ax.plot([x], [v["point"]], marker=MARKER[fam], markersize=10, color=c, markeredgecolor=t["surface"], markeredgewidth=2, linestyle="none")
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([set_tick(lab).replace("SQAD ", "").replace("released ", "") for _, lab in sets],
                       fontsize=12, color=t["ink"], linespacing=1.25)
    ax.set_xlim(-0.55, len(sets) - 0.45)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))                  # a scale, so a mark's height reads as a score
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelsize=11, colors=t["ink2"], length=0)
    ax.set_title("Czech RAG, the smaller models:\nEM with yes and no mapped, higher is better", fontsize=13, color=t["ink2"], loc="left", pad=10)
    flags_after(fig, names, t, gap=0.008, svg=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", dpi=100, facecolor=t["surface"], metadata={"Software": None})
    plt.close(fig)


# ---- all of them ---------------------------------------------------------------------------------------

ABOUT = "Budget-aware retrieval and RAG\nbenchmark for Czech and English,\non one offline machine."

FIGURES = (
    ("families", "Czech RAG: each model's EM with yes and no mapped, with its 95% interval, on the four sets", (ANALYSIS,)),
    ("dial", "the English dial as levels: EM of Ministral 3 3B and 8B against gold-in-context, HotpotQA and NQ", (CROSSOVER,)),
    ("answer_language", "the share of SQAD 3.2 full's yes/no questions each model answers with yes or no in English, although the prompt asks for ano or ne", (ANALYSIS,)),
    ("version_gap", "the gap between the SQAD versions, on all questions and on the questions the debias rule passes", (ANALYSIS,)),
    ("pipeline", "one configuration on one machine, with the budgets and the two moves marked; no data", ()),
    ("social_card", "the social preview card for the repository's settings, 1280 x 640, with the smaller models of families", (ANALYSIS,)),
)


def draw_all(root: Path, out: Path) -> dict:
    plt.rcParams.update(RC)
    manifest: dict = {"note": "Every value each figure draws, with the file and the JSON path it was read from. Written by scripts/figures.py; "
                              "scripts/number_audit.py reads each value again.", "figures": {}}
    for name, what, sources in FIGURES:
        record: list = []
        src = {rel: Source(root, rel, record) for rel in sources}
        files = []
        if name == "social_card":
            social_card(src[ANALYSIS], ABOUT, out / f"{name}.png")
            files.append(f"{name}.png")
        else:
            for mode, t in THEMES.items():
                p = out / f"{name}_{mode}.svg"
                if name == "families":
                    families(src[ANALYSIS], t, p)
                elif name == "dial":
                    dial(src[CROSSOVER], t, p)
                elif name == "answer_language":
                    answer_language(src[ANALYSIS], t, p)
                elif name == "version_gap":
                    version_gap(src[ANALYSIS], t, p)
                elif name == "pipeline":
                    pipeline(t, p)
                label_svg(p, TITLE[name], what)
                files.append(p.name)
        manifest["figures"][name] = {"what": what, "files": files, "sources": list(sources), "values": record}
    (out / "figures.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


# Each SVG's <title> and <desc>, which a screen reader reads when the figure is opened on its own; on a page, the <img>'s alt
# text does that. The title is the figure's own heading.
TITLE = {"families": "Czech RAG: five model families on four SQAD sets",
         "dial": "What the budget buys: more gold in the context, or a bigger generator (Ministral 3)",
         "answer_language": "Asked for ano or ne, who answers yes or no in English?",
         "version_gap": "The gap between the SQAD versions falls without the questions the debias rule flags",
         "pipeline": "One measured configuration on one machine with no network"}


def label_svg(path: Path, title: str, what: str) -> None:
    from xml.sax.saxutils import escape

    text = path.read_text(encoding="utf-8")
    head, sep, rest = text.partition("<svg ")
    tag, close, body = rest.partition(">")
    desc = f"{what[0].upper()}{what[1:]}. Every value it draws is listed in results/figures/figures.json."
    path.write_text(f'{head}{sep}{tag} role="img" aria-labelledby="fig-title fig-desc"{close}\n'
                    f' <title id="fig-title">{escape(title)}</title>\n <desc id="fig-desc">{escape(desc)}</desc>{body}',
                    encoding="utf-8")


def expected_files(manifest: dict) -> list[str]:
    return sorted([f for m in manifest["figures"].values() for f in m["files"]] + ["figures.json"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--out-dir", default=None, help="default: <root>/results/figures")
    ap.add_argument("--check", action="store_true", help="draw into a temporary folder and compare byte for byte")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out = Path(args.out_dir).resolve() if args.out_dir else root / OUT
    if not args.check:
        m = draw_all(root, out)
        print(f"figures: wrote {len(expected_files(m))} files to {out}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        m = draw_all(root, Path(tmp))
        bad = [f for f in expected_files(m) if not (out / f).exists() or not filecmp.cmp(Path(tmp) / f, out / f, shallow=False)]
        stray = sorted(p.name for p in out.glob("*") if p.is_file() and p.name not in expected_files(m)) if out.exists() else []
    for f in bad:
        print(f"figures: {OUT / f} differs from what the script draws (stale, or edited by hand)")
    for f in stray:
        print(f"figures: {OUT / f} is not drawn by this script")
    print("figures:", "FAILED" if bad or stray else f"OK, {len(expected_files(m))} files byte-identical")
    return 1 if bad or stray else 0


if __name__ == "__main__":
    sys.exit(main())
