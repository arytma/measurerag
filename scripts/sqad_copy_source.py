"""Where the Hugging Face copy of SQAD 3.0 comes from: compare it, row by row, with the filtered SQAD 3.0 that
Štefánik et al. (2023) published, both at pinned revisions.

    .venv/bin/python scripts/sqad_copy_source.py           # download both (into data/hf), compare with the tree
    .venv/bin/python scripts/sqad_copy_source.py --write   # rewrite results/czech/copy_source.json

The copy's card names no source. This compares, in each split, the ids, and for each shared id the question, the
context and the set of answers. Nothing of SQAD is written: counts only. Exit 1 if a count differs from the tree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# The dataset of Štefánik et al. (2023): their organization publishes the paper's templates and models, and this
# paper's section 3 states the filter as 90% of the questions whose answer starts in the first 50 characters; their card describes removing the bias of answers in the article's first sentence.
SOURCE = "fewshot-goes-multilingual/cs_squad-3.0"
SOURCE_REVISION = "e2a26bd1bc6d1ff15a33ab9d2ed118054151a784"
SPLITS = ("train", "validation", "test")
OUT = REPO / "results/czech/copy_source.json"


def copy_split(split: str, cache_dir: str) -> dict[str, dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from measurerag.datasets.sqad import MIRROR, REVISION, SPLIT_FILES

    path = hf_hub_download(MIRROR, SPLIT_FILES[split], repo_type="dataset", revision=REVISION, cache_dir=cache_dir)
    return {str(r.id): {"question": str(r.question), "context": str(r.context), "answers": sorted(str(a) for a in r.answers["text"])}
            for r in pd.read_parquet(path).itertuples(index=False)}


def source_split(split: str, cache_dir: str) -> dict[str, dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SOURCE, f"{split}.json", repo_type="dataset", revision=SOURCE_REVISION, cache_dir=cache_dir)
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[str(r["item_id"])] = {"question": str(r["question"]), "context": str(r["context"]), "answers": sorted(str(a) for a in r["answers"])}
    return out


def compare(copy: dict[str, dict], source: dict[str, dict]) -> dict:
    common = set(copy) & set(source)
    return {"copy_rows": len(copy), "source_rows": len(source), "ids_in_common": len(common),
            "only_in_copy": len(set(copy) - common), "only_in_source": len(set(source) - common),
            "same_question": sum(copy[i]["question"] == source[i]["question"] for i in common),
            "same_context": sum(copy[i]["context"] == source[i]["context"] for i in common),
            "same_answers": sum(copy[i]["answers"] == source[i]["answers"] for i in common)}


def identical(c: dict) -> bool:
    n = c["copy_rows"]
    return n == c["source_rows"] == c["ids_in_common"] == c["same_question"] == c["same_context"] == c["same_answers"]


def main(argv: list[str] | None = None) -> int:
    from measurerag.datasets.sqad import MIRROR, REVISION

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(REPO / "data" / "hf"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    splits = {s: compare(copy_split(s, args.cache_dir), source_split(s, args.cache_dir)) for s in SPLITS}
    out = {"copy": {"repo_id": MIRROR, "revision": REVISION}, "source": {"repo_id": SOURCE, "revision": SOURCE_REVISION},
           "splits": splits, "identical_row_for_row": all(identical(c) for c in splits.values()),
           "note": "Counts only, no SQAD text. scripts/sqad_copy_source.py recomputes them from the two pinned Hugging Face revisions."}
    text = json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    print(text, end="")
    path = Path(args.out)
    if args.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print("sqad_copy_source: wrote", path)
        return 0
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        print("sqad_copy_source: DIFFERS from", path)
        return 1
    print("sqad_copy_source: OK, identical to", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
