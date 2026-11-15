"""Write tests/frozen_v020.json: what v0.3.0 may not change, read from v0.2.0's commit.

    .venv/bin/python scripts/freeze_english.py

English results and CsFEVER are not re-run, changed or rounded in this release. The file lists the
SHA-256 of every results file they are in, a hash of each English block of results/crossover.json, and
the value of every token that is not about SQAD. tests/test_english_frozen.py holds the tree to it.
It reads v0.2.0's commit, which only the build history holds: the public repository does not have it, and
the file it wrote ships, so a reader runs the test, not this script.

v0.2.0's records carried two reference labels, in the note of a rejected batch and in the hygiene column of the batch
table; they were removed before publication (no number changed), and the freeze holds the records without them. They
also named other processes on the machine, beside the contention counts; scripts/redact_records.py writes each as
"another process" (no number changed), and the freeze holds the records, and the token that quoted a name, so.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from redact_records import redact_text, redact_token  # noqa: E402
V020 = "ea121b5d1f29c93792b3a99034814eb1f6659e19"
FROZEN_DIRS = ("results/en_small/", "results/en_nfcorpus/", "results/en_fiqa/", "results/en_rag/", "results/cz_small/", "results/cz_full/", "results/crossover/natural/", "results/crossover/superseded/", "results/crossover/calibration/")
FROZEN_FILES = ("results/crossover/crossover_hotpotqa.png", "results/crossover/crossover_nq.png", "results/crossover/max_tokens_check.json")
BLOCKS = ("hotpotqa", "nq")
# Tokens about SQAD change on purpose; a token counting what ran depends on the machine's cache.
SQAD_TOKENS = ("cz.", "czech.", "czech_checks.", "headline.czech.", "data.czech.", "mtcheck.czech.", "runs.cz_rag.", "runs.cz_rag_32.", "gens")


LABEL = re.compile(r" \((?:[A-Z_]+\.md )?N\d+\)")


def unlabelled(text: str) -> str:
    return LABEL.sub("", text)


def show(path: str) -> bytes:
    raw = subprocess.run(["git", "show", f"{V020}:{path}"], cwd=REPO, check=True, capture_output=True).stdout
    if path.endswith((".json", ".jsonl", ".csv", ".txt")):
        text = raw.decode("utf-8")
        if LABEL.search(text):
            text = unlabelled(text)
        if path.endswith(".jsonl"):
            text = redact_text(text)
        return text.encode("utf-8")
    return raw


def block_hash(block: dict) -> str:
    return hashlib.sha256(json.dumps(block, sort_keys=True).encode("utf-8")).hexdigest()


def main() -> int:
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", V020, "results"], cwd=REPO, check=True, capture_output=True, text=True).stdout.split("\n")
    files = {n: hashlib.sha256(show(n)).hexdigest() for n in names if n.startswith(FROZEN_DIRS) or n in FROZEN_FILES}
    crossover = json.loads(show("results/crossover.json"))
    stats = json.loads(show("results/stats.json"))["tokens"]
    out = {"v020_commit": V020, "files": files,
           "crossover_blocks": {k: block_hash(crossover["datasets"][k]) for k in BLOCKS},
           "agreement": block_hash(crossover["agreement"]) if "agreement" in crossover else None,
           "tokens": {k: redact_token(unlabelled(v)) if isinstance(v, str) else v for k, v in sorted(stats.items()) if not k.startswith(SQAD_TOKENS)}}
    (REPO / "tests/frozen_v020.json").write_text(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(files)} files, {len(out['tokens'])} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
