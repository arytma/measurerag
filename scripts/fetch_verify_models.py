"""Fetch the run-2 models and verify them against what run 1 recorded.

Run 1's manifests record, for each generator, the Hugging Face snapshot
revision it loaded (in `generator.model_path`) and the GGUF's byte count; for
each dense encoder, the byte count of what would be shipped (`build.model_bytes`).
They record no content hash. So the check is:

* GGUF: download at the pinned revision; the file's sha256 must equal the LFS
  object id that revision points to, and its size must equal run 1's bytes.
  A git revision content-addresses the LFS pointer, so this chains the file to
  the revision run 1 recorded.
* Dense encoders: the repos are unchanged since before run 1 (their last
  commit predates 2026-09-11); download that revision and require the shipped
  byte count to equal run 1's. The functional check - identical nDCG@10 from
  a fresh embedding - comes from the harness re-runs.

Writes results/crossover/models_manifest.json with a sha256 per file, so the next
run has real hashes to compare against. Exits non-zero on any mismatch.

    HF_HUB_CACHE=data/hf-hub uv run python scripts/fetch_verify_models.py [--bge-m3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "crossover" / "models_manifest.json"

# (key, repo, revision, filename, run-1 bytes)
GGUF = [
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
     "91cad51170dc346986eccefdc2dd33a9da36ead9", "qwen2.5-1.5b-instruct-q4_k_m.gguf", 1117320736),
    ("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct-GGUF",
     "7dabda4d13d513e3e842b20f0d435c732f172cbe", "qwen2.5-3b-instruct-q4_k_m.gguf", 2104932768),
]
# (key, repo, revision, run-1 shipped bytes; bge-m3's are from run 1's SciFact run)
DENSE = [
    ("e5-small", "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3", 493292828),
    ("e5-base", "intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a", 1134534380),
]
BGE_M3 = ("bge-m3", "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", 2293331623)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bge-m3", action="store_true", help="also fetch bge-m3 (step 5)")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    sys.path.insert(0, str(ROOT / "src"))
    from measurerag.retrievers.dense import _model_disk_bytes

    api = HfApi()
    record: dict = {"schema": "measurerag/models/1", "models": {}}
    ok = True

    for key, repo, rev, fn, nbytes in GGUF:
        info = api.model_info(repo, revision=rev, files_metadata=True)
        lfs = next(s for s in info.siblings if s.rfilename == fn).lfs
        # "main", as the kit loads it (this writes refs/main for offline runs),
        # and it must resolve to the revision run 1 recorded
        path = Path(hf_hub_download(repo, fn))
        got_sha, got_bytes = sha256(path), path.stat().st_size
        good = (path.parent.name == rev and got_sha == lfs.sha256
                and got_bytes == nbytes == lfs.size)
        ok &= good
        record["models"][key] = {
            "repo": repo, "revision": rev, "file": fn, "bytes": got_bytes,
            "sha256": got_sha, "night_one_bytes": nbytes, "lfs_sha256": lfs.sha256,
            "verified": good,
        }
        print(f"{'OK ' if good else 'BAD'} {key}: {got_bytes} bytes, sha256 {got_sha[:16]}...")

    dense = DENSE + ([BGE_M3] if args.bge_m3 else [])
    for key, repo, rev, nbytes in dense:
        # Fetch through sentence-transformers, exactly as `measurerag fetch
        # models` and run 1 did: it downloads only the files a dense load
        # needs, and the shipped byte count is defined over that set. A full
        # snapshot would add README images, ColBERT/sparse heads and repo
        # metadata, and change model_bytes without changing the model.
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(repo, device="cpu")
        snap = Path(snapshot_download(repo, local_files_only=True))
        if snap.name != rev:
            print(f"BAD {key}: main resolves to {snap.name}, pinned {rev}")
            ok = False
            continue
        files = {}
        for f in sorted(snap.rglob("*")):
            if f.is_file():
                files[str(f.relative_to(snap))] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
        shipped = _model_disk_bytes(repo)
        good = nbytes is None or shipped == nbytes
        ok &= good
        record["models"][key] = {
            "repo": repo, "revision": rev, "shipped_bytes": shipped,
            "night_one_shipped_bytes": nbytes, "files": files, "verified": good,
        }
        tag = "OK " if good else "BAD"
        print(f"{tag} {key}: shipped {shipped} bytes (run 1 {nbytes}), {len(files)} files")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
