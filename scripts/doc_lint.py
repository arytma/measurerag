"""Fill and check ONE template, without touching the generated documents.

    uv run python scripts/doc_lint.py templates/docs/METHOD.md [--stats results/stats.json]

Fills the template into logs/doc_lint/<same path>, then applies check_docs.py's
rules to that one file: missing tokens, unfilled tokens, numbers that do not
trace to the stats file, broken internal links (resolved as if the file were
at its real path). `# test` shell blocks are listed, not run. For writing a
document while the full set is still in progress; check_docs.py is the real gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_docs import TOKEN, broken_links, czech_documents, test_blocks, traced_numbers, untraced  # noqa: E402
from fill import docs_config, fill, header, load_tokens  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("template")
    ap.add_argument("--stats", default=None, help="default: the stats file in configs/docs.yaml")
    args = ap.parse_args()
    cfg = docs_config(ROOT)
    tpl = Path(args.template).resolve()
    rel = tpl.relative_to(ROOT / cfg["templates"]).as_posix()
    tokens = load_tokens(Path(args.stats) if args.stats else ROOT / cfg["stats"])
    text, missing = fill(tpl.read_text(encoding="utf-8"), tokens)
    problems = [f"token not in the stats file: {m}" for m in missing]
    if not missing:
        text = header(rel) + text
        out = ROOT / "logs" / "doc_lint" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"doc_lint: filled -> {out.relative_to(ROOT)}")
        problems += [f"unfilled token {m.group(0)}" for m in TOKEN.finditer(text)]
        if rel not in (cfg.get("number_rule_exempt") or {}):
            body = text
            heading = (cfg.get("number_rule_from_heading") or {}).get(rel)
            if heading:
                import re
                hm = re.search(r"^" + re.escape(heading) + r"\s*$", text, re.M)
                body = text[hm.start():] if hm else text
            problems += [f"number not in the stats file: {b}" for b in
                         untraced(body, traced_numbers(tokens), int(cfg.get("small_integer_max", 10)), rel in czech_documents(cfg))]
        problems += [f"broken link {b}" for b in broken_links(ROOT / rel, text, ROOT)]
        for i, b in enumerate(test_blocks(text), 1):
            print(f"doc_lint: `# test` block {i} (check_docs.py runs it offline):\n{b.rstrip()}")
    for p in problems:
        print(f"doc_lint: {p}")
    print(f"doc_lint: {'FAILED' if problems else 'OK'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
