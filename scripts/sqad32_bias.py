"""What changed between SQAD 3.0 and 3.2, and the bias statistics. No model runs.

    .venv/bin/python scripts/sqad32_bias.py --out results/sqad32/bias.json

Reads the Hugging Face copy of SQAD 3.0 from the cache, and the 3.2 archive from
.local/data/sqad_3.2 (parsed once, in place). Writes aggregate numbers only.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from measurerag.datasets.sqad32 import (  # noqa: E402
    ARCHIVE,
    ARCHIVE_30,
    YES_NO,
    answer_position,
    contexts_and_rows,
    debias_flags,
    gold_answers,
    jaccard,
    load_parsed,
    opens_article,
    pair_position,
    read_archive,
    squeeze,
)
from measurerag.metrics import normalize_answer  # noqa: E402

MIRROR = "data/hf/datasets--Pehy--cs_sqad-3.0"


def deciles(v):
    return [round(x, 4) for x in st.quantiles(v, n=10)] if len(v) >= 2 else []


def load_mirror() -> pd.DataFrame:
    frames = []
    for name in ("train", "validation", "test"):
        p = glob.glob(str(REPO / MIRROR / f"snapshots/*/data/{name}-*.parquet"))[0]
        frames.append(pd.read_parquet(p).assign(split=name))
    return pd.concat(frames)


def mirror_rows(mir: pd.DataFrame) -> list[dict]:
    rows = []
    for r in mir.itertuples(index=False):
        texts = [str(t) for t in r.answers["text"]]
        starts = [int(s) for s in r.answers["answer_start"]]
        rows.append({"id": str(r.id), "split": r.split, "question": str(r.question), "context": str(r.context),
                     "answers": texts, "sentence_start": starts[0] if starts else 0})
    return rows


def describe(rows: list[dict], threshold: float, label: str) -> dict:
    """The bias statistics over a list of pairs carrying context, question, position."""
    span = [x for x in rows if x["position"] is not None]
    pos = [x["position"] for x in span]
    lens = [len(x["context"]) for x in rows]
    bins = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 700), (700, 10**9)]
    n = len(rows)
    flags = [debias_flags(x["question"], x["context"], x["position"], threshold) for x in rows]
    return {
        "label": label,
        "n_pairs": n,
        "yes_no_or_no_position": n - len(span),
        "context_chars_deciles": deciles(lens),
        "overlap_jaccard_diacritics_kept_deciles": deciles([jaccard(x["question"], x["context"], fold=False) for x in rows]),
        "overlap_jaccard_diacritics_stripped_deciles": deciles([f["overlap"] for f in flags]),
        "overlap_question_answer_sentence_deciles": deciles([x["overlap_sentence"] for x in rows if x.get("overlap_sentence") is not None]),
        "answer_first_50_chars_share_of_span_answers": round(sum(p < 50 for p in pos) / max(len(pos), 1), 4),
        "answer_first_50_chars_share_of_all_pairs": round(sum(p < 50 for p in pos) / max(n, 1), 4),
        "answer_position_deciles": deciles(pos),
        "answer_position_bins": {f"{a}-{b if b < 10**9 else 'end'}": sum(a <= p < b for p in pos) for a, b in bins},
        "flag_early": sum(f["early"] for f in flags),
        "flag_heavy": sum(f["heavy"] for f in flags),
        "flag_both": sum(f["early"] and f["heavy"] for f in flags),
        "flag_any": sum(f["flagged"] for f in flags),
        "flag_any_share": round(sum(f["flagged"] for f in flags) / max(n, 1), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sqad32/bias.json")
    args = ap.parse_args()
    os.chdir(REPO)
    parsed = ARCHIVE.parent / "parsed"
    if not (parsed / "forms.jsonl").exists():
        print(read_archive(ARCHIVE, parsed), flush=True)
    recs, arts = load_parsed(parsed)
    r32 = contexts_and_rows(recs, arts)
    by_id = {r["id"]: r for r in recs}
    mir = load_mirror()
    m30 = mirror_rows(mir)

    # ---- positions and overlap, on the context each version really uses ------
    by_q: dict[str, list[dict]] = collections.defaultdict(list)
    for r in recs:
        by_q[squeeze(r.get("question", ""))].append(r)
    for x in m30:
        x["position"] = answer_position(x["context"], x["answers"], x["sentence_start"])
        hit = by_q.get(squeeze(x["question"]))
        x["record"] = hit[0] if hit else None
        x["overlap_sentence"] = jaccard(x["question"], hit[0]["selection"]) if hit and hit[0].get("selection") else None
    for x in r32:
        x["position"] = pair_position(x["context"], x["answers"], x["selection"])
        x["overlap_sentence"] = jaccard(x["question"], x["selection"]) if x["selection"] else None

    threshold = st.quantiles([jaccard(x["question"], x["context"], fold=True) for x in m30], n=10, method="inclusive")[-1]
    out: dict = {"threshold_overlap_p90_of_sqad30_copy": round(threshold, 6), "threshold_full": threshold}

    # ---- what changed ---------------------------------------------------------
    ids32 = {r["id"] for r in recs}
    joined = [x for x in m30 if x["record"]]
    same_id = sum(any(c["id"] == x["id"] for c in by_q[squeeze(x["question"])]) for x in joined)
    gold_same = gold_diff = 0
    for x in joined:
        g32 = {normalize_answer(a) for c in by_q[squeeze(x["question"])] for a in gold_answers(c)}
        g30 = {normalize_answer(a) for a in x["answers"]}
        if g30 & g32:
            gold_same += 1
        else:
            gold_diff += 1
    matched = {x["record"]["id"] for x in joined}
    out["inventory"] = {
        "sqad30_lindat_records_stated": 13477,
        "sqad30_copy_rows": len(m30),
        "sqad30_copy_splits": collections.Counter(x["split"] for x in m30),
        "sqad30_copy_distinct_contexts": len({x["context"] for x in m30}),
        "sqad32_records": len(recs),
        "sqad32_splits": collections.Counter(r["split"] for r in recs),
        "sqad32_articles": len(arts),
        "sqad32_distinct_windows": len({x["context"] for x in r32}),
        "sqad32_yes_no": sum(1 for r in recs if r.get("yes_no")),
        "sqad32_span_answers": sum(1 for r in recs if r.get("answer")),
        "sqad32_answer_types": collections.Counter(r.get("a_type") for r in recs).most_common(),
        "copy_questions_found_in_32": len(joined),
        "copy_questions_not_found_in_32": len(m30) - len(joined),
        "copy_questions_same_id_in_32": same_id,
        "copy_records_matched_distinct_32_ids": len(matched),
        "copy_gold_same_as_32_after_normalising": gold_same,
        "copy_gold_differs_from_32": gold_diff,
        "copy_alternatives_per_question": collections.Counter(len(x["answers"]) for x in m30),
        "copy_answer_start_is_answer_text": sum(
            x["context"][x["sentence_start"] : x["sentence_start"] + len(x["answers"][0])] == x["answers"][0] for x in m30 if x["answers"]
        ),
    }

    # ---- the real SQAD 3.0 archive --------------------------------------------------
    arch = ARCHIVE_30.parent / "parsed"
    if not (arch / "forms.jsonl").exists():
        print(read_archive(ARCHIVE_30, arch), flush=True)
    r30, a30 = load_parsed(arch)
    q30: dict[str, list[dict]] = collections.defaultdict(list)
    for r in r30:
        q30[squeeze(r.get("question", ""))].append(r)
    d30 = {r["id"]: r for r in r30}
    both = set(q30) & set(by_q)
    same_a = same_s = same_t = 0
    for k in both:
        x, y = q30[k][0], by_q[k][0]
        same_a += normalize_answer(x.get("answer") or "") == normalize_answer(YES_NO.get(y.get("yes_no"), "") or y.get("answer") or "")
        same_s += squeeze(x.get("selection", "")) == squeeze(y.get("selection", ""))
        same_t += squeeze(a30.get(x["article"], "")) == squeeze(arts.get(y["article"], ""))
    ids_common = set(d30) & {r["id"] for r in recs}
    d32 = {r["id"]: r for r in recs}
    out["sqad30_vs_32"] = {
        "sqad30_lindat_records_stated": 13477,
        "sqad30_records_in_archive": len(r30),
        "sqad30_distinct_questions": len(q30),
        "sqad30_articles": len(a30),
        "sqad30_has_official_split": False,
        "sqad30_answer_types": collections.Counter(r.get("a_type") for r in r30).most_common(),
        "sqad32_distinct_questions": len(by_q),
        "questions_in_both": len(both),
        "questions_only_30": len(set(q30) - set(by_q)),
        "questions_only_32": len(set(by_q) - set(q30)),
        "in_both_same_answer": same_a,
        "in_both_same_answer_sentence": same_s,
        "in_both_same_article_text": same_t,
        "ids_in_common": len(ids_common),
        "ids_in_common_with_same_question": sum(squeeze(d30[i].get("question", "")) == squeeze(d32[i].get("question", "")) for i in ids_common),
        "files_30": ["01question", "02answer", "03text", "04url", "05metadata", "06answer.selection", "07answer.prediction", "08question_context", "09answer_extraction"],
        "files_32": ["01question", "02yes_no_answer", "03text", "04url", "05metadata", "06answer.selection", "08question_context", "09answer_extraction", "10title", "11source.html"],
    }

    # ---- what the Hugging Face copy kept of the real 3.0 ----------------------------
    same_id_q = {x["id"] for x in m30 if x["id"] in d30 and squeeze(d30[x["id"]].get("question", "")) == squeeze(x["question"])}
    kept_tab = collections.Counter()
    jac_tab = collections.Counter()
    for r in r30:
        k = r["id"] in same_id_q
        art = squeeze(a30.get(r["article"], ""))
        sel = squeeze(r.get("selection", ""))
        i = art.find(sel) if sel else -1
        if i < 0:
            kept_tab[("unlocated", k)] += 1
            continue
        kept_tab[("opens_article" if i == 0 else "later", k)] += 1
        yn = r.get("a_type") == "YES_NO"
        j = sel.find(squeeze(r.get("answer") or "")) if (r.get("answer") and not yn) else -1
        early = (not yn) and j >= 0 and i + j < 50
        kept_tab[("answer_first_50", k) if early else ("answer_later_or_yes_no", k)] += 1
        if i > 0:
            jac_tab[(min(int(jaccard(r["question"], r["selection"]) * 10), 9), k)] += 1
    out["copy_is_filtered"] = {
        "copy_rows_with_same_id_and_question_in_30": len(same_id_q),
        "answer_sentence_opens_article": {"kept": kept_tab[("opens_article", True)], "dropped": kept_tab[("opens_article", False)]},
        "answer_sentence_later_in_article": {"kept": kept_tab[("later", True)], "dropped": kept_tab[("later", False)]},
        "answer_sentence_not_located": {"kept": kept_tab[("unlocated", True)], "dropped": kept_tab[("unlocated", False)]},
        "answer_in_first_50_chars_of_article": {"kept": kept_tab[("answer_first_50", True)], "dropped": kept_tab[("answer_first_50", False)]},
        "answer_later_or_yes_no": {"kept": kept_tab[("answer_later_or_yes_no", True)], "dropped": kept_tab[("answer_later_or_yes_no", False)]},
        "later_sentences_by_question_sentence_jaccard": {f"{b_ / 10:.1f}": {"kept": jac_tab[(b_, True)], "dropped": jac_tab[(b_, False)]} for b_ in range(10)},
    }
    r30w = contexts_and_rows(r30, a30)
    for x in r30w:
        x["position"] = pair_position(x["context"], x["answers"], x["selection"])
        x["overlap_sentence"] = jaccard(x["question"], x["selection"]) if x["selection"] else None

    # ---- window fit against the real contexts -----------------------------------
    pairs_b, pairs_a, pairs_l = [], [], []
    for x in joined:
        c = x["context"]
        c = c[3:] if c.startswith("...") else c
        c = c.rstrip()
        c = c[:-3].rstrip() if c.endswith("...") else c
        pairs_l.append(len(c))
        n, s = squeeze(c), squeeze(x["record"].get("selection", ""))
        i = n.find(s) if s else -1
        if i >= 0:
            pairs_b.append(i)
            pairs_a.append(len(n) - i - len(s))
    rec_ctx = {x["id"]: x["context"] for x in r32}
    ours = [rec_ctx[x["record"]["id"]] for x in joined if x["record"]["id"] in rec_ctx]
    ol, ob, oa = [], [], []
    for x in joined:
        c0 = rec_ctx.get(x["record"]["id"])
        if c0 is None:
            continue
        c = c0[3:] if c0.startswith("...") else c0
        c = c.rstrip()
        c = c[:-3].rstrip() if c.endswith("...") else c
        ol.append(len(c))
        n, s = squeeze(c), squeeze(x["record"].get("selection", ""))
        i = n.find(s) if s else -1
        if i >= 0:
            ob.append(i)
            oa.append(len(n) - i - len(s))
    out["window_fit"] = {
        "questions": len(joined),
        "real_length_deciles": deciles(pairs_l),
        "ours_length_deciles": deciles(ol),
        "real_chars_before_answer_sentence_deciles": deciles(pairs_b),
        "ours_chars_before_answer_sentence_deciles": deciles(ob),
        "real_chars_after_answer_sentence_deciles": deciles(pairs_a),
        "ours_chars_after_answer_sentence_deciles": deciles(oa),
        "real_starts_with_dots": round(sum(x["context"].startswith("...") for x in joined) / len(joined), 4),
        "ours_starts_with_dots": round(sum(c.startswith("...") for c in ours) / len(ours), 4),
        "real_ends_with_dots": round(sum(x["context"].rstrip().endswith("...") for x in joined) / len(joined), 4),
        "ours_ends_with_dots": round(sum(c.rstrip().endswith("...") for c in ours) / len(ours), 4),
    }

    # ---- bias statistics --------------------------------------------------------
    slice_ids = {json.loads(l)["query_id"] for l in open(REPO / "results/cz_rag/slice/queries.jsonl", encoding="utf-8")}
    out["bias"] = {
        "sqad30_copy_all": describe(m30, threshold, "SQAD 3.0 copy, all rows, its own contexts"),
        "sqad30_copy_test": describe([x for x in m30 if x["split"] == "test"], threshold, "SQAD 3.0 copy, test split"),
        "sqad30_as_released_200": describe([x for x in m30 if x["split"] == "test" and x["id"] in slice_ids], threshold,
                                           "SQAD 3.0 copy, the 200 questions of the released results"),
        "sqad32_all": describe(r32, threshold, "SQAD 3.2, all records, reconstructed windows"),
        "sqad32_test": describe([x for x in r32 if x["split"] == "test"], threshold, "SQAD 3.2, test split"),
        "sqad30_archive_all": describe(r30w, threshold, "SQAD 3.0 archive, all records, reconstructed windows"),
    }

    # the rule against the record the copy filtered: the answer sentence opens the article
    for name, rows in (("sqad32_test", [x for x in r32 if x["split"] == "test"]), ("sqad30_archive", r30w)):
        opens = [x for x in rows if x["opens_article"]]
        early = [debias_flags(x["question"], x["context"], x["position"], threshold)["early"] for x in opens]
        anyf = [debias_flags(x["question"], x["context"], x["position"], threshold)["flagged"] for x in opens]
        out[f"opens_article_{name}"] = {
            "pairs": len(rows), "answer_sentence_opens_article": len(opens),
            "share": round(len(opens) / len(rows), 4),
            "caught_by_rule_1": sum(early), "caught_by_either_rule": sum(anyf),
            "share_caught_by_either_rule": round(sum(anyf) / max(len(opens), 1), 4),
        }
    mirror_opens = sum(1 for x in m30 if x["id"] in d30 and opens_article(a30.get(d30[x["id"]]["article"], ""), d30[x["id"]].get("selection", "")))
    out["opens_article_sqad30_copy"] = {"pairs": len(m30), "answer_sentence_opens_article": mirror_opens, "share": round(mirror_opens / len(m30), 4)}

    # how well does the windows' rule see the article-level bias?
    def article_pos(r):
        a_ = squeeze(arts.get(r["article"], ""))
        s_ = squeeze(r.get("selection", ""))
        i_ = a_.find(s_) if s_ else -1
        if i_ < 0:
            return None
        j_ = s_.find(squeeze(r["answer"])) if r.get("answer") else -1
        return i_ + j_ if j_ >= 0 else None

    art_early = {r["id"]: (lambda p: p is not None and p < 50)(article_pos(r)) for r in recs}
    sub = [x for x in r32 if x["split"] == "test" and x["position"] is not None]
    fl = {x["id"]: debias_flags(x["question"], x["context"], x["position"], threshold) for x in sub}
    out["rule1_against_article_position_32_test"] = {
        "span_answers": len(sub),
        "first_50_of_article": sum(art_early[x["id"]] for x in sub),
        "first_50_of_window": sum(fl[x["id"]]["early"] for x in sub),
        "both": sum(art_early[x["id"]] and fl[x["id"]]["early"] for x in sub),
    }
    # what drives the Jaccard rule: context length or real copying?
    def corr(xs, ys):
        mx, my = st.mean(xs), st.mean(ys)
        sx = sum((a - mx) ** 2 for a in xs) ** 0.5
        sy = sum((b - my) ** 2 for b in ys) ** 0.5
        return round(sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy), 4)

    for name, rows in (("sqad30_copy", m30), ("sqad32", r32)):
        ov_ = [jaccard(x["question"], x["context"]) for x in rows]
        ln = [len(x["context"]) for x in rows]
        os_ = [x["overlap_sentence"] if x["overlap_sentence"] is not None else 0.0 for x in rows]
        out[f"jaccard_rule_drivers_{name}"] = {
            "corr_with_context_chars": corr(ov_, ln),
            "corr_with_question_answer_sentence_overlap": corr(ov_, os_),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
