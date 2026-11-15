"""SQAD 3.2 - Czech question answering, read from the LINDAT archive.

SQAD 3.2 is at http://hdl.handle.net/11234/1-5019, licence CC BY-SA 3.0. The
archive stays under the git-ignored `.local/data/sqad_3.2/`; no file of it enters
this repository. It is read in place with `tarfile`, never extracted: it holds
over 120,000 symbolic links, many with absolute targets on its authors' machine.

Three things differ from the Hugging Face copy of SQAD 3.0 (`datasets/sqad.py`).

1. Format. Each record is a directory of vertical files (word, lemma, tag), and
   its text is a whole Wikipedia article. The copy of 3.0 has short contexts.
2. Yes/no answers. 3.2 stores `yes` or `no`; the copy of 3.0 stores `ano` or `ne`.
   They are mapped to `ano` and `ne` here, so exact match sees the same strings.
3. Contexts. The copy of 3.0 cut each context from its article around the answer
   sentence, by a rule it does not state. `cut_window` is a reconstruction of it,
   compared with the real contexts in `scripts/sqad32_bias.py`.

Everything else is `sqad.py`'s rule: every distinct context across all splits is
a document, a question's gold document is its own context, and the questions are
a seeded sample of the test split. 3.2 ships an official 60/30/10 split by
question type, which is used as it stands.

Not read: `08question_context.vert` (present for some records only, and the
copy of 3.0 has no such field), the archive's `test/` directory of four records,
and `trash/`.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import unicodedata
from pathlib import Path
from typing import Iterable

from .base import Dataset, DatasetUnavailable, Doc
from ..seeds import rng

HANDLE = "http://hdl.handle.net/11234/1-5019"
LICENCE = f"CC BY-SA 3.0 (SQAD 3.2 on LINDAT, {HANDLE})"
ARCHIVE = Path(".local/data/sqad_3.2/sqad_v3.2.tar.xz")
ARCHIVE_MD5 = "39c29a416d2542ecc924542ad79b4cfd"
ARCHIVE_30 = Path(".local/data/sqad_3.0/sqad_v3.tar.xz")
ARCHIVE_30_MD5 = "e5935598d999102c53237f44e69fd6f9"
WINDOW_CHARS = 1000
YES_NO = {"yes": "ano", "no": "ne"}
_FILES = {
    "01question.vert", "02answer.vert", "02yes_no_answer.vert", "03text.vert", "04url.txt", "05metadata.txt",
    "06answer.selection.vert", "09answer_extraction.vert", "10title.vert",
}
_WORD = re.compile(r"\w+", re.UNICODE)


# --------------------------------------------------------------------------
# vertical format
# --------------------------------------------------------------------------
def parse_vert(text: str) -> dict:
    """Plain text, words and lemmas of a vertical file.

    `<g/>` means no space before the next token and `</s>` ends a line. Other
    structure tags are skipped.
    """
    out: list[str] = []
    words: list[str] = []
    lemmas: list[str] = []
    glue = True
    for line in text.split("\n"):
        line = line.rstrip("\r")
        if not line:
            continue
        if line.startswith("<g/>"):
            glue = True
            continue
        if line.startswith("<") and line.endswith(">") and "\t" not in line:
            if line.startswith("</s") or line.startswith("</p"):
                out.append("\n")
                glue = True
            continue
        cols = line.split("\t")
        words.append(cols[0])
        lemmas.append(cols[1] if len(cols) > 1 else cols[0])
        if not glue:
            out.append(" ")
        out.append(cols[0])
        glue = False
    return {"text": re.sub(r"\n{2,}", "\n", "".join(out)).strip(), "words": words, "lemmas": lemmas}


def read_archive(tar_path: str | Path, out_dir: str | Path) -> dict:
    """Parse the archive into records.jsonl, articles.jsonl and forms.jsonl.

    Reads SQAD 3.2 (`sqad_v3.2/data/<id>/`) and SQAD 3.0 (`data/<id>/`, no split, and
    `02answer.vert` where 3.2 has `02yes_no_answer.vert`). forms.jsonl maps each lowercased word form the archive contains to the lemmas
    its vertical files give it. The case-only check in `czech_sensitivity` uses it.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs: dict[str, dict] = {}
    articles: dict[str, str] = {}
    splits: dict[str, str] = {}
    forms: dict[str, dict[str, int]] = {}

    def add_forms(p: dict) -> None:
        for w, l in zip(p["words"], p["lemmas"]):
            d = forms.setdefault(w.lower(), {})
            d[l.lower()] = d.get(l.lower(), 0) + 1

    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf:
            parts = m.name.split("/")
            if len(parts) == 6 and parts[1] == "split_q_type_even_distribution" and parts[5] == "01question.vert":
                splits[parts[4]] = parts[3]
            if len(parts) == 3 and parts[0] == "data":
                rid, fn = parts[1], parts[2]
            elif len(parts) == 4 and parts[1] == "data":
                rid, fn = parts[2], parts[3]
            else:
                continue
            if fn not in _FILES:
                continue
            r = recs.setdefault(rid, {"id": rid})
            if m.issym():
                if fn == "03text.vert":
                    t = re.search(r"(\d{6})/03text\.vert$", m.linkname)
                    r["article"] = t.group(1) if t else None
                continue
            if not m.isreg():
                continue
            raw = tf.extractfile(m).read().decode("utf-8", "replace")
            if fn == "03text.vert":
                r["article"] = rid
                p = parse_vert(raw)
                add_forms(p)
                articles[rid] = p["text"]
            elif fn == "01question.vert":
                r["question"] = parse_vert(raw)["text"].replace("\n", " ")
            elif fn == "02yes_no_answer.vert":
                r["yes_no"] = raw.strip().lower()
            elif fn == "02answer.vert":
                r["answer02"] = parse_vert(raw)["text"].replace("\n", " ")
            elif fn == "04url.txt":
                r["url"] = raw.strip()
            elif fn == "05metadata.txt":
                for k in ("q_type", "a_type"):
                    mm = re.search(rf"<{k}>(.*?)</{k}>", raw)
                    r[k] = mm.group(1) if mm else None
            elif fn == "06answer.selection.vert":
                p = parse_vert(raw)
                add_forms(p)
                r["selection"] = p["text"].replace("\n", " ")
            elif fn == "09answer_extraction.vert":
                p = parse_vert(raw)
                add_forms(p)
                r["answer"] = p["text"].replace("\n", " ")
                r["answer_lemmas"] = p["lemmas"]
                r["answer_words"] = p["words"]
            elif fn == "10title.vert":
                r["title"] = parse_vert(raw)["text"].replace("\n", " ")
    for rid, r in recs.items():
        a, seen = r.get("article"), set()
        while a is not None and a not in articles and a in recs and a not in seen:
            seen.add(a)
            a = recs[a].get("article")
        r["article"] = a if a in articles else None
        r["split"] = splits.get(rid)
    with open(out / "records.jsonl", "w", encoding="utf-8") as fh:
        for rid in sorted(recs):
            fh.write(json.dumps(recs[rid], ensure_ascii=False, sort_keys=True) + "\n")
    with open(out / "articles.jsonl", "w", encoding="utf-8") as fh:
        for a in sorted(articles):
            fh.write(json.dumps({"article": a, "text": articles[a]}, ensure_ascii=False) + "\n")
    with open(out / "forms.jsonl", "w", encoding="utf-8") as fh:
        for f in sorted(forms):
            fh.write(json.dumps({"form": f, "lemmas": forms[f]}, ensure_ascii=False, sort_keys=True) + "\n")
    return {"records": len(recs), "articles": len(articles), "with_split": len(splits), "forms": len(forms)}


def load_parsed(parsed_dir: str | Path) -> tuple[list[dict], dict[str, str]]:
    p = Path(parsed_dir)
    recs = [json.loads(l) for l in open(p / "records.jsonl", encoding="utf-8")]
    arts = {}
    for l in open(p / "articles.jsonl", encoding="utf-8"):
        d = json.loads(l)
        arts[d["article"]] = d["text"]
    return recs, arts


# --------------------------------------------------------------------------
# contexts
# --------------------------------------------------------------------------
def squeeze(s: str) -> str:
    """No whitespace, NFC, lowercase: for matching across different spacing."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s)).lower()


def find_ignoring_space(hay: str, needle: str, start: int = 0) -> int:
    """Offset in `hay` of the first match of `needle` at or after `start`, ignoring
    whitespace and case; -1 if none. NFC differences are not bridged, because an
    offset into `hay` has to stay valid."""
    target = re.sub(r"\s+", "", needle).lower()
    if not target:
        return -1
    index = [i for i, c in enumerate(hay) if not c.isspace()]
    squeezed = "".join(hay[i] for i in index).lower()
    if len(squeezed) != len(index):      # a character whose lowercase is longer
        return hay.lower().find(needle.lower(), start)
    from_pos = next((k for k, i in enumerate(index) if i >= start), len(index))
    k = squeezed.find(target, from_pos)
    return index[k] if k >= 0 else -1


def sentences(article: str) -> list[str]:
    return [s.strip() for s in article.split("\n") if s.strip()]


def locate(sents: list[str], selection: str) -> tuple[int, int] | None:
    """First and last sentence index that hold the answer selection."""
    target = squeeze(selection)
    if not target:
        return None
    joined, starts = "", []
    for s in sents:
        starts.append(len(joined))
        joined += squeeze(s)
    i = joined.find(target)
    if i < 0:
        return None
    j = i + len(target)
    return (max(k for k, x in enumerate(starts) if x <= i), max(k for k, x in enumerate(starts) if x < j))


def cut_window(article: str, selection: str, tag: str, limit: int = WINDOW_CHARS) -> str:
    """A context of whole sentences around the answer sentence.

    `limit` characters less the selection's own are split left and right by one
    seeded share, drawn from a normal distribution with mean 0.5 and sd 0.25 and
    clipped to [0, 1]. Sentences are added while they fit each side's part. If the
    answer sentence has a sentence before it that fits at all, that one is kept:
    in the copy of SQAD 3.0 a window cut on the left almost never opens on the
    answer sentence. A cut side is marked with `...`, as in that copy.
    """
    sents = sentences(article)
    loc = locate(sents, selection)
    if loc is None:
        return selection
    a, b = loc
    room = max(0, limit - sum(len(s) + 1 for s in sents[a : b + 1]))
    share = float(rng("sqad32", "window", tag).normal(0.5, 0.25))
    left = room * min(1.0, max(0.0, share))
    right = room - left
    lo, hi, used = a, b, 0
    if lo > 0 and len(sents[lo - 1]) + 1 <= room:
        lo -= 1
        used = len(sents[lo]) + 1
        left = max(left, used)
        right = room - left
    while lo > 0 and used + len(sents[lo - 1]) + 1 <= left:
        lo -= 1
        used += len(sents[lo]) + 1
    used = 0
    while hi < len(sents) - 1 and used + len(sents[hi + 1]) + 1 <= right:
        hi += 1
        used += len(sents[hi]) + 1
    text = " ".join(sents[lo : hi + 1])
    return ("... " if lo > 0 else "") + text + ("..." if hi < len(sents) - 1 else "")


def window_tag(rec: dict) -> str:
    """One window per question. The copy of SQAD 3.0 does the same: of its 725 answer
    sentences with two or more questions, 688 have different contexts."""
    return str(rec["id"])


def gold_answers(rec: dict) -> list[str]:
    if rec.get("yes_no") in YES_NO:
        return [YES_NO[rec["yes_no"]]]
    return [rec["answer"]] if rec.get("answer") else []


def opens_article(article: str, selection: str) -> bool | None:
    """Does the answer sentence open the article? None if it cannot be found.

    This is the record the copy of SQAD 3.0 filtered: it kept 10.0% of them and
    every other record it could locate.
    """
    s = squeeze(selection)
    if not s:
        return None
    i = squeeze(article).find(s)
    return None if i < 0 else i == 0


# --------------------------------------------------------------------------
# the debiasing flags
# --------------------------------------------------------------------------
def tokens(text: str, fold: bool = False) -> set[str]:
    text = text.lower()
    if fold:
        text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))
    return set(_WORD.findall(text))


def jaccard(a: str, b: str, fold: bool = True) -> float:
    x, y = tokens(a, fold), tokens(b, fold)
    return len(x & y) / len(x | y) if x | y else 0.0


def answer_position(context: str, answers: Iterable[str], after: int = 0) -> int | None:
    """Where the first locatable answer starts in `context`, or None.

    `ano` and `ne` are not spans, so a yes/no answer has no position. `after` is
    where the answer sentence starts, if known: an answer word can occur earlier.
    """
    best = None
    for ans in answers:
        if ans.strip().lower() in ("ano", "ne", "yes", "no"):
            continue
        pos = find_ignoring_space(context, ans, after)
        if pos < 0 and after:
            pos = find_ignoring_space(context, ans, 0)
        if pos >= 0 and (best is None or pos < best):
            best = pos
    return best


def pair_position(context: str, answers: Iterable[str], selection: str = "") -> int | None:
    """The answer's position, searched from where its annotated sentence starts."""
    start = find_ignoring_space(context, selection) if selection else -1
    return answer_position(context, answers, max(start, 0))


# The filter this rule follows: BenCzechMark's for SQAD 3.2 (Fajcik, Docekal, Dolezal et al., arXiv:2412.17933, v2 of 22 May 2025,
# Appendix B.7). It removes 90% of the pairs whose answer is in the first 50 characters of the context, and 90% of the pairs whose
# question and context have a TF-IDF cosine similarity above 0.9. Here the overlap is the Jaccard similarity of word sets, above a
# threshold read from the SQAD 3.0 archive, and every flagged question is left out. Kept for the documents' comparison only.
BENCZECHMARK_FILTER = {"first_chars": 50, "share_removed": 0.9, "tfidf_cosine": 0.9}
# Its first rule is Štefánik, Kadlčík, Gramacki and Sojka's (2023, SlavicNLP, section 3), who reported that about half of SQAD's
# questions have the answer at the beginning of the article, and removed 90% of those whose answer starts in the first 50 characters.
# Their filtered SQAD 3.0 is the Hugging Face copy this release reads (scripts/sqad_copy_source.py).
STEFANIK_2023_FILTER = {"first_chars": 50, "share_removed": 0.9}


def debias_flags(question: str, context: str, position: int | None, threshold: float, first: int = 50) -> dict:
    """The two rules: the answer starts in the first 50 characters, or the question overlaps the context heavily.
    They follow BenCzechMark's filter (BENCZECHMARK_FILTER), with a different overlap measure, and nothing kept at random."""
    overlap = jaccard(question, context, fold=True)
    early = position is not None and position < first
    heavy = overlap > threshold
    return {"overlap": overlap, "early": early, "heavy": heavy, "flagged": early or heavy}


# --------------------------------------------------------------------------
# the loader
# --------------------------------------------------------------------------
def contexts_and_rows(recs: list[dict], arts: dict[str, str]) -> list[dict]:
    rows = []
    for r in recs:
        ans = gold_answers(r)
        if not ans or not r.get("article"):
            continue
        ctx = cut_window(arts[r["article"]], r.get("selection", ""), window_tag(r))
        rows.append({"id": r["id"], "split": r.get("split"), "question": r["question"], "answers": ans,
                     "context": ctx, "selection": r.get("selection", ""), "a_type": r.get("a_type"),
                     "article": r["article"], "opens_article": opens_article(arts[r["article"]], r.get("selection", ""))})
    return rows


def doc_id(text: str) -> str:
    return "ctx-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load(
    cache_dir: str | Path | None = None,
    split: str = "test",
    max_docs: int | None = 100_000,
    max_queries: int | None = 2_000,
    archive: str | Path = ARCHIVE,
    questions_file: str | Path | None = None,
    questions_key: str | None = None,
    **_,
) -> Dataset:
    """`questions_file` and `questions_key` name a list of ids in a JSON file: exactly those questions.
    A seeded draw of a larger size does not hold a smaller one, so a set that grows names its questions."""
    archive = Path(archive)
    parsed = archive.parent / "parsed"
    if not (parsed / "records.jsonl").exists():
        if not archive.exists():
            raise DatasetUnavailable(f"{archive} not found; SQAD 3.2 comes from {HANDLE}")
        read_archive(archive, parsed)
    recs, arts = load_parsed(parsed)
    rows = contexts_and_rows(recs, arts)
    docs: dict[str, Doc] = {}
    for x in rows:
        did = doc_id(x["context"])
        if did not in docs:
            docs[did] = Doc(doc_id=did, text=x["context"], title="")
    queries, qrels, answers = {}, {}, {}
    for x in rows:
        if x["split"] != split:
            continue
        queries[x["id"]] = x["question"]
        qrels[x["id"]] = {doc_id(x["context"]): 1}
        answers[x["id"]] = x["answers"]
    if questions_file is not None:
        wanted = json.loads(Path(questions_file).read_text(encoding="utf-8"))[questions_key]
        missing = [q for q in wanted if q not in queries]
        if missing:
            raise DatasetUnavailable(f"{questions_file}: {len(missing)} questions are not in the {split} split")
        queries = {q: queries[q] for q in wanted}
        qrels = {q: qrels[q] for q in wanted}
        answers = {q: answers[q] for q in wanted}
    elif max_queries is not None and len(queries) > max_queries:
        keep = sorted(queries)
        idx = rng("sqad", split, "queries").choice(len(keep), size=max_queries, replace=False)
        chosen = {keep[i] for i in sorted(idx.tolist())}
        queries = {q: queries[q] for q in chosen}
        qrels = {q: qrels[q] for q in chosen}
        answers = {q: answers[q] for q in chosen}
    return Dataset(
        name=f"sqad-3.2-{split}",
        language="cs",
        docs=sorted(docs.values(), key=lambda d: d.doc_id),
        queries=queries,
        qrels=qrels,
        answers=answers,
        license=LICENCE,
        source=HANDLE,
        notes=[
            "Czech QA. Corpus is every distinct context across all three splits; "
            "gold is the context cut around the question's answer sentence.",
            "Contexts are reconstructed windows of whole sentences (cut_window), since "
            "3.2 ships whole articles. Yes/no golds are mapped to ano and ne.",
            "Answers are short spans, so EM is a meaningful metric; a generator "
            "that writes a sentence around the span will lose EM and keep F1.",
        ],
    )
