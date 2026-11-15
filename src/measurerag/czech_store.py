"""The shipped Czech records: reading and writing them, and what the dial analysis needs from them.

results/czech/records/ holds ids, labels and rankings, and answers/ the generated answers, under their
sources' licences. No SQAD question, passage or gold answer is stored, so EM and F1 are stored per row. `scripts/czech_records.py`
writes the answers and rankings, `scripts/czech_diagnostics.py` the labels, and
`scripts/rescore_czech.py` recomputes every Czech number from them.
"""

from __future__ import annotations

import gzip
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from .hashing import sha256_json
from .metrics import ndcg_at_k

REPO = Path(__file__).resolve().parents[2]
RECORDS = "results/czech/records"
FIRST_CUT = "results/czech/first_cut"
ANSWERS = "answers"
SOURCES = ("sqad30_copy", "sqad32")


def write_jsonl_gz(path: Path, rows: list[dict]) -> int:
    """Same input, same bytes: no timestamp and no file name in the gzip header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=buf, mtime=0) as gz:
        for r in rows:
            gz.write((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    data = buf.getvalue()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return len(data)


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_sets(root: Path) -> dict:
    return json.loads((Path(root) / "sets.json").read_text(encoding="utf-8"))


def is_first_cut(root: Path) -> bool:
    """Does this records root describe the first v0.3.0 cut? Its answers stay in a folder of their own."""
    path = Path(root) / "sets.json"
    return path.exists() and bool(json.loads(path.read_text(encoding="utf-8")).get("first_cut"))


def answers_path(root: Path, name: str) -> Path:
    """Where an answers file is. Answers are data under their source's licence: answers/<source>/, and
    answers/<source>/first_cut/ for the first cut. A file beside the root's sets.json comes first."""
    root = Path(root)
    if (root / name).exists():
        return root / name
    folders = [REPO / ANSWERS / source for source in SOURCES]
    if is_first_cut(root):
        folders = [*(f / "first_cut" for f in folders), *folders]
    for folder in folders:
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"{name}: not beside {root} and not in {ANSWERS}/")


def load_answers(root: Path, key: str) -> list[dict]:
    return read_jsonl_gz(answers_path(root, f"answers_{key}.jsonl.gz"))


def load_first_cut_same_questions() -> list[dict]:
    """The first cut's one question on both corpora: a file per source. Empty if they are absent."""
    rows: list[dict] = []
    for source in SOURCES:
        path = REPO / ANSWERS / source / "first_cut" / "same_questions.jsonl.gz"
        if path.exists():
            rows += read_jsonl_gz(path)
    return rows


def load_cross_version() -> list[dict]:
    """One question on each version's passages, a file per source. Empty before that run has finished."""
    rows: list[dict] = []
    for source in SOURCES:
        path = REPO / ANSWERS / source / "answers_cross_version.jsonl.gz"
        if path.exists():
            rows += read_jsonl_gz(path)
    return rows


def records_path(root: Path, name: str) -> Path:
    """Rankings and labels describe the test splits, so a root without its own reads the main records."""
    path = Path(root) / name
    return path if path.exists() else REPO / RECORDS / name


def load_rankings(root: Path, population: str) -> dict[str, dict]:
    return {r["q"]: r for r in read_jsonl_gz(records_path(root, f"rankings_{population}.jsonl.gz"))}


def load_labels(root: Path, population: str) -> dict[str, dict]:
    return {r["q"]: r for r in read_jsonl_gz(records_path(root, f"labels_{population}.jsonl.gz"))}


@dataclass
class DialInputs:
    """What scripts/crossover.py's analysis reads, taken from the records instead of the caches."""

    name: str
    corpus_sha256: str
    queries_sha256: str
    qids: list[str]
    gold: dict[str, set[str]]
    retrieval: dict[str, dict]
    em: dict[tuple, int] = field(default_factory=dict)
    f1: dict[tuple, float] = field(default_factory=dict)
    context: dict[tuple, list[str]] = field(default_factory=dict)
    leak: dict[tuple, int] = field(default_factory=dict)


def dial_inputs(root: Path, key: str, retrievers: list[str], levels: list[float], leak_field: str = "leak", qids: list[str] | None = None, em_field: str = "em") -> DialInputs:
    """`qids` restricts the set, for a subset measured with the same outputs. `em_field` reads a
    labelled secondary metric in EM's place."""
    root = Path(root)
    meta = load_sets(root)["sets"][key]
    rank = load_rankings(root, meta["ranking"])
    use = meta["questions"] if qids is None else qids
    keep = set(use)
    inp = DialInputs(
        name=meta["label"], corpus_sha256=meta["corpus_sha256"], queries_sha256=meta["queries_sha256"], qids=list(use),
        gold={q: set(rank[q]["gold"]) for q in use}, retrieval={},
    )
    for rid in retrievers:
        rows = {q: rank[q][rid] for q in use if rid in rank[q]}
        if len(rows) == len(use):
            scores = [ndcg_at_k(rows[q], {g: 1 for g in rank[q]["gold"]}, 10) for q in use]
            inp.retrieval[rid] = {"ndcg@10": sum(scores) / len(scores), "run_sha256": sha256_json(rows)}
    for r in load_answers(root, key):
        if r["kind"] != "dial" or r["q"] not in keep or r["p"] not in levels:
            continue
        g = r["config"].split("+", 1)[1]
        k = (float(r["p"]), r["q"], g)
        inp.em[k] = int(r[em_field])
        inp.f1[k] = float(r["f1"])
        inp.context[k] = list(r["context"])
        inp.leak[(float(r["p"]), r["q"])] = int(r[leak_field])
    return inp
