"""Check the documentation set (configs/docs.yaml). Exit 1 on any failure.

    uv run python scripts/check_docs.py              # everything, including `# test` blocks
    uv run python scripts/check_docs.py --skip-exec  # without running shell blocks

It fails on:
* a document of the set that is missing;
* a generated document that differs from its template filled from
  results/stats.json (stale, or edited by hand), or a template token stats.json
  does not define;
* an unfilled {{token}} anywhere in the set;
* a number in prose that does not trace to results/stats.json - that is, does
  not appear among the numbers rendered in its tokens - unless it is a small
  counting integer, a year, a date, a time, a version, or matches an exempt
  pattern (configs/docs.yaml says which documents are exempt, and why);
* a broken internal link (a relative path that does not exist, or a #anchor
  with no matching heading), in Markdown or in an embedded HTML image's `src` or
  `srcset`;
* a table with a 3B row that is not followed by the 3B model's licence line;
* an en dash or an em dash anywhere in a document of the set;
* a fenced shell block whose first line is `# test` that does not exit 0 with
  the network cut off (proxies pointed at a closed port, Hugging Face and uv
  offline). The cut is by environment only: a tool that ignores proxy
  variables and opens its own socket would still reach the network.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill import LABEL_3B_MARK, TOKEN, docs_config, load_tokens, next_text, render_all, tables_with_3b  # noqa: E402

FENCE = re.compile(r"^(```+|~~~+)([^\n]*)\n(.*?)^\1[ \t]*$", re.M | re.S)
INLINE_CODE = re.compile(r"`[^`\n]*`")
COMMENT = re.compile(r"<!--.*?-->", re.S)
LINK = re.compile(r"!?\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
# the image paths of embedded HTML: <img src="..."> and <source srcset="...">, as <picture> uses them
HTML_SRC = re.compile(r"\b(?:src|srcset)\s*=\s*\"([^\"]+)\"")
URL = re.compile(r"https?://\S+")
# a number, optionally glued to a unit ("1.81GB", "250ms", "2.5x", "100k"), not part
# of an identifier ("e5-small", "Qwen2.5-1.5B", "p95", "N12")
NUMBER = re.compile(r"(?<![\w.#@§^])([+\-−]?\d[\d,]*(?:\.\d+)?)(?:GB|MB|KB|ms|s|x|×|k|M)?(?![\w@])")
RANGE = re.compile(r"(\d)[-\u2013](\d)")
ANY_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
# not results: dates, times, version strings, licence and software versions
NOT_RESULTS = [
    re.compile(rf"\b\d{{1,2}}(?:(?:[-\u2013]|\s+to\s+)\d{{1,2}})?\s+{MONTHS}(?:\s+\d{{4}})?"),
    re.compile(rf"\b{MONTHS}\s+\d{{1,2}}(?:,\s*\d{{4}})?"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ][\d:]+Z?)?"),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
    re.compile(r"\bv?\d+\.\d+\.\d+(?:[-.\w]*)"),
    re.compile(r"\b(?:Python|macOS|CC BY(?:-[A-Z]{2})*|Apache(?:\s+License,?)?(?:\s+Version)?|Version|GPL|LGPL|MPL)(?:\s+|-)v?\d+(?:\.\d+)*(?:(?:[\u2013-]|\s+(?:to|a\u017e)\s+)\d+(?:\.\d+)*)?"),
    re.compile(r"\bSHA-?256\b", re.I),        # a hash function's name
    re.compile(r"\b(?:Granite|Gemma|Ministral|Mistral|Qwen|Llama)\s+\d+(?:\.\d+)?\b"),   # a model's name
    re.compile(r"\bInstruct\s+\d{4}\b"),                                                 # a model's release tag (Ministral 3 Instruct 2512)
    re.compile(r"\bsections?\s+\d+(?:(?:,\s*|\s+and\s+|\s+to\s+)\d+)*\b"),          # a reference to a section of a document
]
SLUG_DROP = re.compile(r"[^\w\- ]", re.U)


def _blank(m: re.Match) -> str:
    """A space for what is left out, and its line breaks, so that a line number still points into the template."""
    return " " + "\n" * m.group(0).count("\n")


def prose_of(text: str) -> str:
    """The text a reader reads as sentences: no code, comments, link targets or URLs."""
    text = COMMENT.sub(_blank, text)
    text = FENCE.sub(_blank, text)
    text = INLINE_CODE.sub(_blank, text)
    text = LINK.sub(lambda m: m.group(1) + "\n" * m.group(0).count("\n"), text)
    text = URL.sub(_blank, text)
    for pat in NOT_RESULTS:
        text = pat.sub(_blank, text)
    return text


def norm(num: str) -> str:
    return num.lstrip("+-−").replace(",", "").rstrip(".")


def traced_numbers(tokens: dict) -> set[str]:
    out: set[str] = set()
    for v in tokens.values():
        for m in ANY_NUMBER.findall(str(v)):
            out.add(norm(m))
    return out


def exempt_number(num: str, small_max: int) -> bool:
    n = norm(num)
    if re.fullmatch(r"\d+", n):
        v = int(n)
        return v <= small_max or 1900 <= v <= 2100
    return False


CZECH_THOUSANDS = re.compile(r"(?<=\d)[\u00a0\u202f](?=\d{3}(?!\d))")
CZECH_DECIMAL = re.compile(r"(?<=\d),(?=\d)")


def from_czech_numbers(text: str) -> str:
    """A Czech document's numbers as the stats file writes them: 4 771 (a no-break space) -> 4771, 0,8399 -> 0.8399.
    In Czech prose a comma is followed by a space unless it is a decimal comma."""
    return CZECH_DECIMAL.sub(".", CZECH_THOUSANDS.sub("", text))


def czech_documents(cfg: dict) -> set[str]:
    return set(cfg.get("czech_number_marks") or [])


def untraced(text: str, traced: set[str], small_max: int, czech: bool = False) -> list[str]:
    bad = []
    if czech:
        text = from_czech_numbers(text)
    for line_no, line in enumerate(prose_of(text).splitlines(), 1):
        line = RANGE.sub(r"\1 to \2", line)          # "0.30-0.99" is two numbers
        for m in NUMBER.finditer(line):
            raw = m.group(1).rstrip(",")
            if not raw or exempt_number(raw, small_max) or norm(raw) in traced:
                continue
            bad.append(f"{raw!r} in: {line.strip()[:120]}")
    return bad


def slug(heading: str) -> str:
    h = INLINE_CODE.sub(lambda m: m.group(0)[1:-1], heading.strip().lower())
    h = re.sub(r"<[^>]+>", "", h)
    return SLUG_DROP.sub("", h).replace(" ", "-")


def anchors(text: str) -> set[str]:
    out, seen = set(), {}
    for m in re.finditer(r"^#{1,6}\s+(.+?)\s*#*\s*$", FENCE.sub("", text), re.M):
        s = slug(m.group(1))
        n = seen.get(s, 0)
        out.add(s if n == 0 else f"{s}-{n}")
        seen[s] = n + 1
    return out


def link_targets(text: str) -> list[str]:
    """Every relative or absolute target a reader can follow: Markdown links and images, and HTML image paths."""
    body = COMMENT.sub(" ", FENCE.sub(" ", text))
    out = [m.group(2) for m in LINK.finditer(body)]
    for m in HTML_SRC.finditer(INLINE_CODE.sub(" ", body)):
        out += [c.split()[0] for c in m.group(1).split(",") if c.strip()]   # a srcset may list several candidates
    return out


def broken_links(doc: Path, text: str, root: Path) -> list[str]:
    bad = []
    for target in link_targets(text):
        if re.match(r"^[a-z][a-z0-9+.\-]*:", target, re.I):   # http:, mailto:, ...
            continue
        path, _, frag = target.partition("#")
        dest = doc if not path else (doc.parent / path).resolve()
        if not dest.exists():
            bad.append(f"{target} (no such file)")
            continue
        if frag and dest.is_file() and dest.suffix == ".md":
            if frag.lower() not in anchors(dest.read_text(encoding="utf-8")):
                bad.append(f"{target} (no heading #{frag})")
        try:
            dest.relative_to(root.resolve())
        except ValueError:
            bad.append(f"{target} (outside the repository)")
    return bad


def test_blocks(text: str) -> list[str]:
    blocks = []
    for m in FENCE.finditer(text):
        lang, body = m.group(2).strip().lower(), m.group(3)
        if lang in ("bash", "sh", "shell", "zsh") and body.lstrip().startswith("# test"):
            blocks.append(body)
    return blocks


def run_offline(block: str, root: Path, timeout: int) -> tuple[bool, str]:
    dead = "http://127.0.0.1:9"
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               UV_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", http_proxy=dead, https_proxy=dead,
               HTTP_PROXY=dead, HTTPS_PROXY=dead, ALL_PROXY=dead, all_proxy=dead, no_proxy="", NO_PROXY="")
    try:
        p = subprocess.run(["bash", "-euo", "pipefail", "-c", block], cwd=root, env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout} s"
    return p.returncode == 0, (p.stdout + p.stderr)[-600:]


def check(root: Path, skip_exec: bool = False, timeout: int = 900) -> list[str]:
    cfg = docs_config(root)
    problems: list[str] = []

    rendered, missing = render_all(root)
    for rel, miss in missing.items():
        problems.append(f"templates/{rel}: tokens missing from {cfg['stats']}: {', '.join(miss)}")
    for path, text in rendered.items():
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            problems.append(f"{path.relative_to(root)}: differs from its template filled from {cfg['stats']} "
                            "(stale, or edited by hand)")

    traced = traced_numbers(load_tokens(root / cfg["stats"]))
    exempt = cfg.get("number_rule_exempt") or {}
    from_heading = cfg.get("number_rule_from_heading") or {}
    small_max = int(cfg.get("small_integer_max", 10))
    for rel in cfg["documents"]:
        doc = root / rel
        if not doc.exists():
            problems.append(f"{rel}: missing")
            continue
        text = doc.read_text(encoding="utf-8")
        # token syntax quoted as code is documentation, not a hole
        prose = INLINE_CODE.sub(" ", FENCE.sub(" ", COMMENT.sub(" ", text)))
        for m in TOKEN.finditer(prose):
            problems.append(f"{rel}: unfilled token {m.group(0)}")
        # a backstop that does not depend on TOKEN: any "{{" left in prose is a hole
        if not TOKEN.search(prose) and "{{" in prose:
            for n, line in enumerate(prose.splitlines(), 1):
                if "{{" in line:
                    problems.append(f"{rel}: unfilled placeholder in: {line.strip()[:100]}")
        if rel not in exempt:
            body = text
            if rel in from_heading:
                hm = re.search(r"^" + re.escape(from_heading[rel]) + r"\s*$", text, re.M)
                if hm is None:
                    problems.append(f"{rel}: heading {from_heading[rel]!r} not found, so the number rule "
                                    "cannot tell the frozen part from the checked part")
                    body = text
                else:
                    body = text[hm.start():]
            for b in untraced(body, traced, small_max, rel in czech_documents(cfg)):
                problems.append(f"{rel}: number not in {cfg['stats']}: {b}")
        for b in broken_links(doc, text, root):
            problems.append(f"{rel}: broken link {b}")
        lines = text.split("\n")
        for first, last in tables_with_3b(lines):
            if LABEL_3B_MARK not in next_text(lines, last):
                problems.append(f"{rel}: the table at line {first + 1} has 3B rows but no {LABEL_3B_MARK} line after it")
        for n, line in enumerate(lines, 1):
            if "\u2013" in line or "\u2014" in line:
                problems.append(f"{rel}: line {n} has an en or em dash: {line.strip()[:80]}")
        if not skip_exec:
            for i, block in enumerate(test_blocks(text), 1):
                ok, out = run_offline(block, root, timeout)
                if not ok:
                    problems.append(f"{rel}: `# test` block {i} failed offline: {out.strip()[-300:]}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--skip-exec", action="store_true", help="do not run `# test` shell blocks")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per `# test` block")
    args = ap.parse_args()
    problems = check(Path(args.root), args.skip_exec, args.timeout)
    for p in problems:
        print(f"check_docs: {p}")
    print(f"check_docs: {'FAILED, ' + str(len(problems)) + ' problem(s)' if problems else 'OK'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
