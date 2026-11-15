# Metrics: exact definitions

Every number this kit prints is defined here. Where a definition has a choice in
it, the choice and its reason are stated. If you change a definition in
`src/measurerag/metrics.py`, change it here in the same commit.

Notation: a query `q` has a set of judgements `rel(q)` mapping document id to an
integer relevance grade; a retriever returns a ranked list `R(q)`.

---

## Retrieval

### nDCG@k

```
DCG@k  = sum over i in 1..k of  rel(R_i) / log2(i + 1)
IDCG@k = DCG@k of the judgements sorted by grade, descending
nDCG@k = DCG@k / IDCG@k          (0 if IDCG@k == 0)
```

Linear gains, not `2^rel - 1`. BEIR qrels are mostly binary, where the two agree;
for the graded sets (NFCorpus, DaReCzech) linear gains are the more conservative
choice and are what the BEIR leaderboard reports.

**Reported at k = 10.** This is the headline quality number and the y-axis of
both curves.

### Recall@k

```
P(q)      = { d : rel(q)[d] > 0 }
Recall@k  = |P(q) ∩ top-k of R(q)| / |P(q)|      (0 if P(q) is empty)
```

**Reported at k = 10 and k = 100.** Recall@100 is the ceiling a reranker would
have to work inside; Recall@10 is what a user scrolling one screen sees.

### gold-in-context@k

```
1 if P(q) ∩ top-k of R(q) is non-empty, else 0
```

The bridge metric, and the most useful single diagnostic in the kit. `k` here is
the number of passages actually handed to the generator (`retrieval.context_k`,
default 5), **not** the retrieval k. It separates two failures that otherwise
look identical from the end:

* low gold-in-context, low EM → the retriever failed; a bigger generator is
  wasted money.
* high gold-in-context, low EM → the evidence was on the page and the generator
  did not use it; a better retriever is wasted money.

On multi-hop HotpotQA a question needs two paragraphs, so this metric means "at
least one arrived". It is a floor, not full evidence coverage.

### Aggregation

All retrieval metrics are **macro-averaged over queries**: each query counts
once, regardless of how many judgements it has.

Two conventions worth stating:

* **Unjudged documents count as non-relevant.** The standard BEIR assumption. It
  does not move every system by the same amount, nor every metric the same way. A
  relevant but unjudged document inside the top k counts as a miss, which lowers
  nDCG@k, Recall@k and gold-in-context. A relevant but unjudged document that a
  retriever does not return is missing from IDCG@k and from |P(q)|, which raises
  that retriever's nDCG@k and Recall@k; gold-in-context can only be lowered. The
  judgements here come from how each set was built, not from pooling the results
  of the systems compared: HotpotQA's supporting paragraphs, each SQAD question's
  own context, CsFEVER's evidence pages, and for NQ a mapping with the same
  property (a passage that holds the answer but is not judged is not gold). A
  retriever that ranks another passage holding the answer high loses against one
  that ranks the source passage, and a question that shares words with its source
  passage can favour lexical matching, so comparisons between lexical and dense
  retrievers carry that bias.
* **Queries with no judgements at all are skipped, not scored zero.** Scoring
  them zero would punish a retriever for a gap in the annotation, and the size
  of that gap differs between datasets. `n_queries` in the results reports how
  many queries were actually scored.

---

## End-to-end answer quality

### Normalisation

Applied to both prediction and gold before comparing:

1. Unicode NFC.
2. Lowercase.
3. Remove punctuation.
4. Drop English articles (`a`, `an`, `the`).
5. Collapse whitespace.

**Diacritics are kept.** SQuAD's original normaliser is English-only; stripping
diacritics would merge distinct Czech words (`byt`/`být`, `pas`/`pás`) and score
a wrong answer as right. Diacritic-insensitivity is a legitimate *retrieval*
trade-off and is exposed there as the `strip_diacritics` retriever knob, where it
is measured rather than assumed.

**On Czech, two steps can over-credit.** Dropping English articles also drops the
Czech conjunction `a` ("and"), so `Petr a Pavel` and `Petr Pavel` match. Removing
punctuation joins the two parts of a decimal comma, so `3,5` and `35` match. Czech
exact match mostly under-counts, because an inflected form of the right answer
scores 0, but these two steps mean it is not strictly a lower bound.

### Exact match (EM)

`1.0` if the normalised prediction equals any normalised gold answer, else `0.0`.

EM punishes a correct answer wrapped in a sentence. That is intentional and it is
why F1 is reported beside it: the gap between them tells you whether a
configuration is wrong or merely verbose, and verbosity is fixable with a prompt.

### Czech: EM with yes and no mapped, and compliance

**EM with yes and no mapped** is EM after one change to the prediction: a whole
answer that normalises to `yes` becomes `ano`, and one that normalises to `no`
becomes `ne` (`yes_no_in_czech` in `src/measurerag/czech_sensitivity.py`). Nothing
else is mapped. It is the primary Czech metric, because SQAD's gold for a yes/no
question is `ano` or `ne` and some models answer in English.

**Compliance** is read on the yes/no questions only. Each answer is classed as
`ano or ne`, `yes or no in English`, `unanswerable` (the model declined) or
`something else` (`yes_no_class`). Compliance is the share of `ano or ne` among
the answers that did not decline: a Czech span or sentence counts against it, as
an English yes or no does (`compliance`).

### Token F1

Bag-of-tokens F1 between normalised prediction and gold, maximised over the
acceptable gold answers:

```
common    = multiset intersection of prediction and gold tokens
precision = |common| / |prediction tokens|
recall    = |common| / |gold tokens|
F1        = 2PR / (P + R)              (0 if |common| == 0)
```

### No LLM-as-judge

Deliberately absent. A judge model is not deterministic across runs, versions or
hardware, and this kit promises byte-identical reruns. It would also need to be
either a cloud API (forbidden here) or another local model inside the same RAM
budget it is supposed to be measuring.

---

## Cost

Measured, never estimated, and each configuration is measured **in its own
process** so nothing inherits another's high-water mark.

| quantity | how |
|---|---|
| peak RSS | `psutil` sampling at 50 ms, combined with `getrusage(RUSAGE_SELF).ru_maxrss`, over build **and** search |
| index size on disk | the index is written out and the bytes counted |
| model size on disk | files in the HF snapshot that a load actually reads; duplicate weight formats (`.bin` beside `.safetensors`) and ONNX/OpenVINO exports counted once or not at all |
| retrieval latency | one query at a time, encoding included, `time.perf_counter` |
| end-to-end latency | prompt build + generation, per question |
| generator throughput | completion tokens / generation seconds |

Three things to know about these:

**Peak RSS is the whole process.** Weights, index, tokenizer, runtime, allocator
fragmentation. That is what makes a customer's 8 GB box swap, so that is what the
RAM budget is compared against. Reporting index size alone would understate a
dense retriever by roughly the size of its encoder.

**Latency is single-query service latency, not throughput.** A dense retriever's
query encoding is part of what the user waits for, so `search_one` encodes and
searches one query and the clock covers both. Batch-encoding 300 queries would
produce a far prettier and completely dishonest number.

**Percentiles are nearest-rank**, not interpolated: `p95` of *n* values is the
value at rank `ceil(0.95n)`. It returns an observation that actually happened,
and it is exactly reproducible.

Query-time thread count is pinned (`query_threads`, default 1) so latency does
not drift with machine load. Index building gets every core (`build_threads`),
because indexing is a one-off offline job and is reported separately as
`build_seconds`.

---

## Budget verdicts

A budget is a triple: `ram_gb`, `retrieval_ms` (p95), and optionally `e2e_s`
(p95). A configuration **FITS** only if every applicable ceiling holds. Ceilings
are inclusive: a configuration exactly at a limit fits it.

```
peak_rss_gb        <= ram_gb
retrieval_ms_p95   <= retrieval_ms
e2e_s_p95          <= e2e_s          (when the budget names one)
```

A retrieval-only configuration has no end-to-end latency, so it **cannot** fit a
budget that names `e2e_s`; the verdict records that reason rather than excusing
it. The `e2e_s` axis is therefore only added to the grid when the RAG stage is
enabled.

The default grid is `ram_gb ∈ {4, 8, 16} × retrieval_ms ∈ {200, 500}`, six
budgets, plus `e2e_s ∈ {3, 8}` when generating.

In the printed table, a quality value in `[brackets]` was really measured but its
configuration fits **no** budget in the grid. The number is true; it is just not
available to you on that box.

---

## Reproducibility

`results/<run>/manifest.json` records the SHA-256 of every dataset slice
(corpus, queries, qrels, hashed separately), of every result file, of the
resolved config, and of the metric payload with timings removed
(`results_stable_sha256`).

```
measurerag repro --config configs/en_small.yaml
```

runs the same config twice and diffs the manifests. Timings, peak RSS and the
verdicts derived from them are excluded, they are physics, not logic. Metrics
must match exactly; if they do not, the run is not reproducible and the diff
says where.
