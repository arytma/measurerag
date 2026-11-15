"""Materialise a dataset slice without running the harness.

    uv run python scripts/make_slice.py --dataset nq-rag --max-docs 100000 \
        --max-queries 300 --out results/crossover/slices/nq
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main() -> int:
    from measurerag.datasets import load_dataset
    from measurerag.datasets.base import cache_root

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ds = load_dataset(args.dataset, cache_dir=cache_root(), max_docs=args.max_docs, max_queries=args.max_queries)
    ds.save(args.out)
    print(json.dumps(ds.fingerprint(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
