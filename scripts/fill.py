"""Fill {{token}} placeholders from results/stats.json.

    uv run python scripts/fill.py --all            # every template under templates/
    uv run python scripts/fill.py --check          # fail if a generated document is stale
    uv run python scripts/fill.py --template T --stats S --out O   # one file

Every document whose prose carries results is generated from a template of the
same path under templates/ (configs/docs.yaml). No number in the prose is typed
by hand. {{token|neg}} reads a difference the other way round (A minus B as B minus A); {{table|family_columns}}
and {{table|western_first}} put a table's models in the documents' order (below). A Czech document writes {{token|cs}}: the same value with Czech number
marks (a decimal comma, a no-break space between thousands, "až" for "to"), and
a token that holds any other English word cannot be written that way. A token a template uses and the stats file does not define fails the
build, and nothing is written - a document with a hole in it is worse than no
document, because it looks finished.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

# a token name is anything without spaces or braces: it may name a step "a->b"
# (hotpotqa.step.tfidf-svd->bm25.k5.ret) or a configuration "a+b"
# (runs.cz_rag.e5-small+qwen2.5-1.5b.em)
TOKEN = re.compile(r"\{\{\s*([^\s{}]+)\s*\}\}")
ROOT = Path(__file__).resolve().parents[1]

# ---- Czech number marks: {{token|cs}} ---------------------------------------------------------------
NBSP = "\u00a0"
_EN_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_EN_LIST = re.compile(r"(\d), ([+\-\u2212]?\d)")        # "[+0.016, +0.061]": the comma between two numbers
_UNITS = {"GB", "MB", "KB", "ms", "s", "k", "x"}


def czech_number(value: str) -> str:
    """A token's value with Czech number marks: 0.8399 -> 0,8399, 4,771 -> 4 771 (a no-break space),
    [+0.016, +0.061] -> [+0,016; +0,061], "+0.126 to +0.170" -> "+0,126 až +0,170". Anything else in English is an
    error, so a Czech sentence never carries an English phrase from a token."""
    v = _EN_LIST.sub(r"\1; \2", value.replace(" to ", " až "))

    def one(m: re.Match) -> str:
        whole, _, frac = m.group(0).partition(".")
        return whole.replace(",", NBSP) + ("," + frac if frac else "")

    v = _EN_NUMBER.sub(one, v)
    words = [w for w in re.findall(r"[^\W\d_]+", v) if w not in _UNITS and w != "až"]
    if words:
        raise ValueError(f"not a number in Czech: {value!r} holds {' '.join(words)!r}")
    return v


_SIGNED = r"[+\-\u2212]?\d[\d,]*(?:\.\d+)?"


def _flip(number: str) -> str:
    sign, digits = (number[0], number[1:]) if number[0] in "+-\u2212" else ("+", number)
    if float(digits.replace(",", "")) == 0:
        return "+" + digits
    return ("+" if sign in "-\u2212" else "-") + digits


def negated(value: str) -> str:
    """A difference read the other way round, so that a sentence can name its models in the order it lists them:
    "-0.038 to -0.013" -> "+0.013 to +0.038", "-0.025 [-0.044, -0.007]" -> "+0.025 [+0.007, +0.044]". A zero stays +0.000."""
    m = re.fullmatch(rf"({_SIGNED}) to ({_SIGNED})", value)
    if m:
        return f"{_flip(m.group(2))} to {_flip(m.group(1))}"
    m = re.fullmatch(rf"({_SIGNED}) \[({_SIGNED}), ({_SIGNED})\]((?: \(not established\))?)", value)
    if m:
        return f"{_flip(m.group(1))} [{_flip(m.group(3))}, {_flip(m.group(2))}]{m.group(4)}"
    if re.fullmatch(_SIGNED, value):
        return _flip(value)
    raise ValueError(f"not a difference to read the other way round: {value!r}")


# ---- the order models are named in: the United States first, then France, then China -------------------------------
# The documents' prose names models in this order, and a table a document shows can be put in it at render time; the
# stats file, and the tables of docs/CROSSOVER.md, keep the order and the direction they were computed in.
FAMILY_ORDER = ("Gemma 3n", "Gemma 4", "Ministral 3", "Qwen2.5", "Qwen3.5")
COUNTRY = {"Gemma 3n": 0, "Gemma 4": 0, "Ministral 3": 1, "Qwen2.5": 2, "Qwen3.5": 2}
_FAMILY = re.compile(r"Gemma 3n|Gemma 4|Ministral 3|Qwen2\.5|Qwen3\.5")


def _family(cell: str) -> str | None:
    m = _FAMILY.search(cell)
    return m.group(0) if m else None


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _line(cells: list[str]) -> str:
    if all(re.fullmatch(r"-+", c) for c in cells):
        return "|" + "|".join(cells) + "|"
    return "| " + " | ".join(cells) + " |"


def family_columns(table: str) -> str:
    """A table whose columns name models, with those columns grouped by family in FAMILY_ORDER (each family's models
    keep their order); the columns that name no model stay first. Every row is permuted alike."""
    lines = table.split("\n")
    head = _cells(lines[0])
    if not any(_family(c) for c in head):
        raise ValueError("no column names a model")
    order = sorted(range(len(head)), key=lambda i: (-1, i) if _family(head[i]) is None else (FAMILY_ORDER.index(_family(head[i])), i))
    return "\n".join(_line([_cells(x)[i] for i in order]) for x in lines)


def _cell_other_way(cell: str) -> str:
    m = re.fullmatch(r"(\S+) minus (\S+): (.*)", cell)
    if not m:
        raise ValueError(f"not a difference cell: {cell!r}")
    return f"{m.group(2)} minus {m.group(1)}: {negated(m.group(3))}"


def western_first(table: str) -> str:
    """A table of differences between families ("A minus B" in its "families" column): a row whose first family comes
    from a later country than its second is written the other way round, B minus A, every value negated and its
    interval bounds swapped; within each set the rows then follow FAMILY_ORDER, by the first family, then the second."""
    lines = table.split("\n")
    head = _cells(lines[0])
    col = head.index("families")
    rows, sets = [], []
    for line in lines[2:]:
        c = _cells(line)
        a, b = re.fullmatch(r"(.+) minus (.+)", c[col]).groups()
        fa, fb = _family(a), _family(b)
        if COUNTRY[fa] > COUNTRY[fb]:
            c = c[:col] + [f"{b} minus {a}"] + [_cell_other_way(x) for x in c[col + 1:]]
            fa, fb = fb, fa
        if c[0] not in sets:
            sets.append(c[0])
        rows.append(((sets.index(c[0]), FAMILY_ORDER.index(fa), FAMILY_ORDER.index(fb)), c))
    rows.sort(key=lambda r: r[0])
    return "\n".join([lines[0], lines[1]] + [_line(c) for _, c in rows])


FORMATS = {"cs": czech_number, "neg": negated, "family_columns": family_columns, "western_first": western_first}


def token_name(name: str) -> str:
    """The stats key a placeholder reads: "a.b|cs" reads a.b."""
    return name.split("|", 1)[0]


def token_value(name: str, stats: dict) -> str:
    """A placeholder's text: the token's value, then each format in turn ("a.b|neg|cs")."""
    base, *fmts = name.split("|")
    value = str(stats[base])
    for fmt in fmts:
        value = FORMATS[fmt](value)
    return value


def fill(template: str, stats: dict) -> tuple[str, list[str]]:
    missing = set()
    for m in set(TOKEN.findall(template)):
        base, *fmts = m.split("|")
        if base not in stats or any(f not in FORMATS for f in fmts):
            missing.add(m)
            continue
        try:
            token_value(m, stats)
        except ValueError as e:
            missing.add(f"{m} ({e})")
    if missing:
        return "", sorted(missing)
    return TOKEN.sub(lambda m: token_value(m.group(1), stats), template), []


# Every table with a Qwen2.5 3B row carries that model's licence next to it (a release decision): the generator
# adds the line, scripts/check_docs.py checks it. A bare "3B" is Qwen2.5's only beside Qwen2.5's 1.5B or its flag;
# Ministral 3 3B (Apache 2.0) carries no such line.
QWEN_3B_NAMED = re.compile(r"(?i)qwen2\.5[- ]3b\b|(?<![\w.])3b \U0001F1E8\U0001F1F3")
BARE_3B = re.compile(r"(?i)(?<![\w.])3b\b")
QWEN_SMALL = re.compile(r"(?i)(?<![\w.])(?:qwen2\.5-)?1\.5b\b")
LABEL_3B_MARK = "Qwen Research License"


def has_qwen_3b(text: str) -> bool:
    return bool(QWEN_3B_NAMED.search(text) or (BARE_3B.search(text) and QWEN_SMALL.search(text)))


def label_3b(prefix: str) -> str:
    return (f"*Qwen2.5 3B rows: Qwen2.5-3B-Instruct, under the {LABEL_3B_MARK} (non-commercial research and "
            f"evaluation use only; see [THIRD_PARTY.md]({prefix}THIRD_PARTY.md)).*")


def tables_with_3b(lines: list[str]) -> list[tuple[int, int]]:
    """(first, last) line index of each Markdown table with a Qwen2.5 3B row, outside code fences."""
    out, fence, i = [], False, 0
    while i < len(lines):
        if lines[i].lstrip().startswith(("```", "~~~")):
            fence = not fence
        if not fence and lines[i].lstrip().startswith("|"):
            j = i
            while j + 1 < len(lines) and lines[j + 1].lstrip().startswith("|"):
                j += 1
            if j > i and has_qwen_3b("\n".join(lines[i:j + 1])):
                out.append((i, j))
            i = j
        i += 1
    return out


def next_text(lines: list[str], j: int) -> str:
    for line in lines[j + 1:]:
        if line.strip():
            return line
    return ""


def add_3b_labels(text: str, rel: str) -> str:
    prefix = "../" * (len(Path(rel).parts) - 1)
    lines = text.split("\n")
    for first, last in reversed(tables_with_3b(lines)):
        if LABEL_3B_MARK in next_text(lines, last):
            continue
        gap = [""] if last + 1 < len(lines) and lines[last + 1].strip() else []
        lines[last + 1:last + 1] = ["", label_3b(prefix), *gap]
    return "\n".join(lines)


# v0.4.2: a flag after each model's name in the documents a reader reads for the results, in prose
# and tables; never in a heading, code, HTML, a link's target, a licence line or the licence and reproduction documents
FLAGGED_DOCS = {"README.md", "README.cs.md", "CHANGELOG.md", "docs/CROSSOVER.md", "docs/LIMITATIONS.md", "docs/METHOD.md", "docs/BUDGETS.md"}
UNFLAGGED_SPANS = re.compile(r"(`[^`]*`|\]\([^)]*\))")


def flag_prose(text: str) -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    from czech_tokens import flagged

    out, fence = [], False
    for line in text.split("\n"):
        s = line.lstrip()
        if s.startswith(("```", "~~~")):
            fence = not fence
        if fence or s.startswith(("```", "~~~", "#", "<")) or LABEL_3B_MARK in line:
            out.append(line)
            continue
        parts = UNFLAGGED_SPANS.split(line)
        out.append("".join(x if i % 2 else flagged(x) for i, x in enumerate(parts)))
    return "\n".join(out)


def header(rel: str) -> str:
    return (f"<!-- Generated from templates/{rel} and results/stats.json by scripts/fill.py. "
            "Edit the template, not this file. -->\n")


def load_tokens(path: Path) -> dict:
    stats = json.loads(path.read_text(encoding="utf-8"))
    return stats.get("tokens", stats)


def docs_config(root: Path) -> dict:
    """configs/docs.yaml, with the documents and exemptions of any configs/docs_*.yaml beside it added."""
    cfg = yaml.safe_load((root / "configs" / "docs.yaml").read_text(encoding="utf-8"))
    for extra in sorted((root / "configs").glob("docs_*.yaml")):
        more = yaml.safe_load(extra.read_text(encoding="utf-8")) or {}
        cfg["documents"] = [*cfg.get("documents", []), *more.get("documents", [])]
        for key in ("number_rule_exempt", "number_rule_from_heading"):
            cfg[key] = {**(cfg.get(key) or {}), **(more.get(key) or {})}
    return cfg


def render_all(root: Path) -> tuple[dict[Path, str], dict[str, list[str]]]:
    """{output path: text} for every template, and {template: missing tokens}."""
    cfg = docs_config(root)
    tokens = load_tokens(root / cfg["stats"])
    tdir = root / cfg["templates"]
    out: dict[Path, str] = {}
    missing: dict[str, list[str]] = {}
    for t in sorted(p for p in tdir.rglob("*") if p.is_file() and p.suffix == ".md"):
        rel = t.relative_to(tdir).as_posix()
        text, miss = fill(t.read_text(encoding="utf-8"), tokens)
        if miss:
            missing[rel] = miss
        else:
            out[root / rel] = header(rel) + add_3b_labels(flag_prose(text) if rel in FLAGGED_DOCS else text, rel)
    return out, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="fill every template under templates/")
    ap.add_argument("--check", action="store_true", help="fail if any generated document is stale")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--template")
    ap.add_argument("--stats")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.all or args.check:
        root = Path(args.root)
        out, missing = render_all(root)
        for rel, miss in missing.items():
            print(f"fill: templates/{rel}: {len(miss)} token(s) missing from stats: {', '.join(miss)}", file=sys.stderr)
        if missing:
            return 1
        stale = [p for p, text in out.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
        if args.check:
            for p in stale:
                print(f"fill: {p.relative_to(root)} is stale or edited by hand; edit templates/ and run --all",
                      file=sys.stderr)
            return 1 if stale else 0
        for p in stale:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(out[p], encoding="utf-8")
            print(f"fill: wrote {p.relative_to(root)}")
        return 0

    if not (args.template and args.stats and args.out):
        ap.error("give --all, --check, or --template/--stats/--out")
    tokens = load_tokens(Path(args.stats))
    text, missing = fill(Path(args.template).read_text(encoding="utf-8"), tokens)
    if missing:
        print(f"fill: {len(missing)} token(s) missing from {args.stats}: {', '.join(missing)}", file=sys.stderr)
        return 1
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"fill: wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
