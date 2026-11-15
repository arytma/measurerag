<!-- Generated from templates/docs/BUDGETS.md and results/stats.json by scripts/fill.py. Edit the template, not this file. -->
# Budgets: what each configuration costs, and how far to trust the timings

This document is for two readers: an engineer sizing a deployment, who wants to
know what a retriever and generator pair costs in memory and in waiting time,
and a reviewer checking the cost side of the crossover in
[CROSSOVER.md](CROSSOVER.md), whose per-GB and per-ms prices are differences
between rows of the tables below.

Every number here comes from one machine: an Apple M4 MacBook Air with
32 GB of memory and no fan. Every configuration was measured
one at a time, in its own process. Exact metric definitions are in [METRICS.md](METRICS.md). What to
discount for is in [LIMITATIONS.md](LIMITATIONS.md).

---

## 1. What is measured

A **configuration** is one retriever and one generator at one context size k:
`bm25+3b` is BM25 handing its top k passages to Ministral 3 3B 🇫🇷. Each
one runs in a fresh subprocess (`src/measurerag/worker.py`). The subprocess
builds the retriever's index, retrieves for every question, loads the generator
and answers every question. Nothing else is resident in it.

**Peak RSS** is the high-water mark of the resident set of that whole process:
retriever, index, tokenizer, generator weights, runtime and allocator
fragmentation together. It is not the index size. `src/measurerag/memory.py`
samples the process's RSS in a background thread and combines it with the
kernel's own maximum (`ru_maxrss`), taking the larger. The measurement covers
the whole run, **including the index build**, so a deployment that loads
a prebuilt index may peak differently.

**Peak RSS and the mapped model file.** The generator loads its GGUF memory-mapped, llama-cpp-python's default, and on
the CPU llama.cpp also copies most weights into its own buffers; the file's pages stay resident beside that copy, so
every row's peak RSS counts part of the generator's weights twice. The tables give each row's peak RSS as measured, model file memory-mapped (library default), and beside
it an estimate without the double-counted mapped pages (probe; see [LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings)): the measured peak less the surplus a probe measured for its generator, 1.36 GB
for 3B and 3.70 GB for 8B. Every budget verdict is the same either way (§2).

**What "GB" means.** The unit is not the same in every column:

* `peak RSS GB` is `bytes / 1024**3` (binary gigabytes). The RAM budgets are
  compared against this same number, so a RAM ceiling is in binary gigabytes too.
* `model GB` is `bytes / 1e9` and `index MB` is `bytes / 1e6` (decimal).

So peak RSS and model size are not directly subtractable to the last digit.

**Index MB** is what the retriever writes to disk: the FAISS flat index for a
dense retriever, the bm25s index directory for BM25 (an estimate from the token
lists if that directory cannot be saved), and the projected document matrix
for tfidf-svd. tfidf-svd's vocabulary and SVD components are held in memory and
counted nowhere on disk; its peak RSS includes them.

**Model GB** is the model files as you would ship them: for a dense
encoder, only the files a sentence-transformers load reads, with duplicate
weight formats counted once; for the generator, the GGUF file. BM25 and
tfidf-svd have no model file of their own, so for them the column is the
generator alone. Every model file is pinned to a revision
([THIRD_PARTY.md](../THIRD_PARTY.md)).

**Retrieval latency** is single-query service latency: one question at a time,
query encoding included, with the query side pinned to one thread.
Each search returns the top 100 passages, the ranking the analysis
uses, not only the k passages handed to the generator. The tables show the p50.
The budget verdicts use the p95, which is in each sweep's `results.json`.
Percentiles are nearest-rank.

**End-to-end latency** is, per question, building the prompt from the
already-retrieved passages plus generating the answer: prefill of the k
passages and decoding, with greedy decoding, `max_tokens` = 64, a
context window of 4096 tokens, on the CPU with no layers offloaded
to the GPU (`device: cpu`), and llama.cpp's default thread count. **Despite the name, it does not include retrieval.** The worker
retrieves for every question first, on its own clock, and starts the
end-to-end clock afterwards. A user waits for both, so the wait a user sees is
retrieval latency plus end-to-end latency. Both are reported at p50 and p95,
one question at a time. Neither is throughput, and no configuration was
measured under concurrent load.

### The budget grid

Each configuration is checked against every budget in a grid of RAM ceilings
(4 GB, 8 GB and 16 GB), retrieval p95 ceilings
(200 ms and 500 ms) and end-to-end p95 ceilings
(3 s and 8 s). A configuration **fits** a budget only if all three
of its ceilings hold. Ceilings are inclusive ([METRICS.md](METRICS.md), "Budget
verdicts").

Two columns of the tables summarise those verdicts:

* `smallest RAM budget` is the smallest RAM ceiling among the budgets the
  configuration fits. Every budget also carries latency ceilings, so this is
  not a memory verdict alone. A configuration whose memory fits but whose
  end-to-end p95 exceeds every end-to-end ceiling reads `none`. Read it as "the
  smallest box on which this configuration meets at least the loosest latency
  budget of the grid".
* `fits 3 s e2e` is `yes` when the configuration fits at least one budget with
  the tighter of the end-to-end ceilings. It is decided on the **p95**, not the
  p50 shown beside it.

---

## 2. The tables

One table per natural sweep, Ministral 3 3B 🇫🇷 and 8B with each of the five retrievers. Rows are sorted by peak RSS.
`g @k` is gold-in-context at that k (the share of questions whose k passages include a gold passage), and EM is
exact match over the sweep's questions. The `macOS` column is the operating system the worker recorded when it
measured that row: every row was measured on macOS 27.2.1, so any two rows of a table, and of the
two tables, are comparable as far as the operating system goes. Three HotpotQA rows first timed in one night were
timed again in a later sitting (§4).

**No configuration fits a budget of the grid.** Configurations that fit one: 0 of 20 with peak RSS as measured, model file memory-mapped (library default),
and 0 of 20 with the estimate without the double-counted mapped pages (probe; see [LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings)). The end-to-end p95 runs from 9.8 s to
45.9 s, against end-to-end ceilings of 3 s and 8 s, so `smallest RAM budget` reads `none`
on every row either way. One ceiling decides: rows under the largest RAM ceiling, 20 of 20 as measured, model file memory-mapped (library default) and
20 of 20 in the estimate; under the tightest retrieval ceiling,
20 of 20; under the loosest end-to-end ceiling, 0 of 20.
This describes Ministral 3 🇫🇷 at 3B and 8B on this CPU, not every generator.

**The setting behind every row, for scaling it.** Each corpus is a slice: 100,000 passages for HotpotQA
and 100,000 for NQ, with 500 and 300 questions; index size, build time and
retrieval latency grow with the corpus, at different rates for different retrievers. The index build ran inside
the measured process and took 2 to 11,389 s by configuration (`build_seconds` in each sweep's
`results.json`), and retrieval p95 runs from 1.2 to 87.0 ms.

Each table gives `peak RSS GB` twice, as measured and as the estimate; rows stay in the order of the measured peak.

### HotpotQA, k = 5: the prices come from here

| configuration | g @5 | EM | peak RSS GB, as measured, model file memory-mapped (library default) | peak RSS GB, estimate without the double-counted mapped pages (probe; see LIMITATIONS) | index MB | model GB | retrieval p50 ms | e2e p50 s | e2e p95 s | smallest RAM budget | fits 3 s e2e | macOS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25+3b | 0.984 | 0.360 | 5.66 | 4.29 | 33.1 | 2.15 | 1.1 | 11.07 | 15.74 | none | no | 27.2.1 |
| e5-small+3b | 0.986 | 0.380 | 6.60 | 5.24 | 153.6 | 2.64 | 10.7 | 8.60 | 13.54 | none | no | 27.2.1 |
| tfidf-svd+3b | 0.184 | 0.050 | 7.29 | 5.93 | 102.4 | 2.15 | 20.9 | 5.98 | 11.11 | none | no | 27.2.1 |
| e5-base+3b | 0.992 | 0.364 | 7.50 | 6.14 | 307.2 | 3.28 | 36.9 | 8.98 | 14.03 | none | no | 27.2.1 |
| bge-m3+3b | 0.992 | 0.348 | 8.25 | 6.88 | 409.6 | 4.44 | 69.2 | 9.71 | 13.86 | none | no | 27.2.1 |
| bm25+8b | 0.984 | 0.428 | 10.76 | 7.06 | 33.1 | 5.20 | 1.3 | 23.75 | 33.39 | none | no | 27.2.1 |
| e5-small+8b | 0.986 | 0.462 | 11.70 | 8.00 | 153.6 | 5.69 | 10.8 | 16.58 | 28.17 | none | no | 27.2.1 |
| e5-base+8b | 0.992 | 0.466 | 12.57 | 8.87 | 307.2 | 6.33 | 28.0 | 19.58 | 31.24 | none | no | 27.2.1 |
| tfidf-svd+8b | 0.184 | 0.080 | 12.83 | 9.13 | 102.4 | 5.20 | 13.1 | 13.94 | 26.70 | none | no | 27.2.1 |
| bge-m3+8b | 0.992 | 0.438 | 13.34 | 9.64 | 409.6 | 7.49 | 70.8 | 24.27 | 34.87 | none | no | 27.2.1 |

### Natural Questions, k = 5

| configuration | g @5 | EM | peak RSS GB, as measured, model file memory-mapped (library default) | peak RSS GB, estimate without the double-counted mapped pages (probe; see LIMITATIONS) | index MB | model GB | retrieval p50 ms | e2e p50 s | e2e p95 s | smallest RAM budget | fits 3 s e2e | macOS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25+3b | 0.733 | 0.343 | 5.85 | 4.49 | 50.0 | 2.15 | 1.0 | 13.38 | 19.24 | none | no | 27.2.1 |
| e5-small+3b | 0.943 | 0.377 | 6.69 | 5.33 | 153.6 | 2.64 | 10.8 | 13.50 | 21.33 | none | no | 27.2.1 |
| e5-base+3b | 0.960 | 0.407 | 7.53 | 6.17 | 307.2 | 3.28 | 27.2 | 12.40 | 20.25 | none | no | 27.2.1 |
| tfidf-svd+3b | 0.060 | 0.040 | 7.91 | 6.55 | 102.4 | 2.15 | 18.2 | 4.65 | 9.79 | none | no | 27.2.1 |
| bge-m3+3b | 0.950 | 0.390 | 8.73 | 7.37 | 409.6 | 4.44 | 70.6 | 12.21 | 19.68 | none | no | 27.2.1 |
| bm25+8b | 0.733 | 0.417 | 10.96 | 7.26 | 50.0 | 5.20 | 1.0 | 30.46 | 44.78 | none | no | 27.2.1 |
| bge-m3+8b | 0.950 | 0.497 | 11.79 | 8.09 | 409.6 | 7.49 | 71.0 | 26.31 | 40.06 | none | no | 27.2.1 |
| e5-small+8b | 0.943 | 0.487 | 11.80 | 8.10 | 153.6 | 5.69 | 10.6 | 28.49 | 45.89 | none | no | 27.2.1 |
| e5-base+8b | 0.960 | 0.507 | 12.64 | 8.94 | 307.2 | 6.33 | 27.0 | 27.68 | 45.44 | none | no | 27.2.1 |
| tfidf-svd+8b | 0.060 | 0.047 | 12.95 | 9.25 | 102.4 | 5.20 | 11.2 | 11.64 | 24.83 | none | no | 27.2.1 |

---

## 3. The price of each move

The crossover compares two moves, and each is priced from the bench rows at
the dial's k = 5:

* **The retriever step** is priced at the natural step bm25 → e5-small, both
  with the 3B generator. It is the lexical-to-dense upgrade a deployment
  actually faces.
* **The generator step** is 3B → 8B at the dial's own retriever. If that
  pair is not yet measured, it is priced at the retriever step's upper rung
  instead and `crossover.json` records the fallback.

Each cost is the difference between two rows: peak RSS in GB, and the added
latency in ms, which is the change in retrieval p50 plus the change in
end-to-end p50 (the whole wait a user sees, §1). The two
parts are shown as well.

| dataset | retriever step: GB | retriever step: ms added | of which retrieval | of which end-to-end | generator step: GB, as measured, model file memory-mapped (library default) | generator step: GB, estimate without the double-counted mapped pages (probe; see LIMITATIONS) | generator step: ms added |
|---|---|---|---|---|---|---|---|
| HotpotQA (dial on e5-base) | 0.95 | -2453 | 9.6 | -2462 | 5.07 | 2.73 | 10590 |
| Natural Questions (dial on e5-base) | 0.83 | 125 | 9.8 | 115 | 5.11 | 2.77 | 15274 |

The retriever step's GB is the same either way: its two rows load the same generator, so the surplus cancels.

The added latency is a sum of two medians, not the median of a sum, because
retrieval and generation are timed in separate passes over the questions. On
HotpotQA, both priced steps use a row that was timed again (§4), in GB and in ms:
the retriever step's lower rung, bm25 + 3B, and the generator step's upper row, e5-base + 8B.
With their first timings, from the night, the generator step's added latency in front of e5-base was the largest of
the five rungs' and the retriever step's change in generation-stage latency was more negative; the prices here use
the second timings ([CROSSOVER.md](CROSSOVER.md), §3), and the second timing of e5-base + 8B also peaked
0.77 GB higher, which sets the generator step's GB. Each step sets the row timed again against one timed once, in
the first sitting, so if the later sitting ran faster in general, the generator step's added latency is lower, and the
retriever step's change less negative, than a comparison within one sitting would give.

**How the prices are used.** The per-GB crossover is the g at which
Δ_ret / (retriever step's GB) equals Δ_gen / (generator step's GB). The per-ms
crossover is the same with the added latency. Two partial forms are kept as
sensitivities: per ms of end-to-end latency alone (which leaves out the
retriever's own cost), and the retriever priced at the retrieval latency it
adds against the generator at the end-to-end latency it adds (the "component"
form). The results are in [CROSSOVER.md](CROSSOVER.md), under each dataset's
"the price", together with a table of each move per unit of cost at every dial
level.

**The prices are point values.** The bootstrap that puts intervals on the
per-GB and per-ms g\* resamples questions. Each cost is a fixed difference
between two single runs and is not resampled. Run-to-run variation in peak RSS
or latency is therefore in no interval. The three rows timed again give its size here: the two runs of e5-base + 8B
on HotpotQA differ by 0.77 GB in peak RSS, and bm25 + 3B's by 3790 ms in end-to-end p50 (§4).

### Why a move that lowers latency has no per-ms price

A retriever step can make latency go **down**. The end-to-end clock does not
include retrieval (§1), so for a retriever step it measures only the change in
prompt building and generation. If the stronger retriever's passages are
shorter, the generator's prefill is cheaper, and that saving can outweigh the
retrieval time the step adds. On HotpotQA the priced step's lower rung was timed again (§4) and the change stayed
negative, but bm25 + 3B alone moved by 3790 ms between its two timings, more than the change; the prompts' token
counts are not recorded, so the records cannot show that shorter passages are the cause rather than variation between
runs. A step can also lower peak RSS. EM gained per
negative millisecond is not a price, so whenever a move's cost is zero or
negative, its per-unit crossover is recorded as **undefined**, with both
measured costs, and is not computed. The same rule applies to the
generator step.

### Price sensitivity

Every consecutive step of the retriever ladder, priced each way, with the g\* it
would give (`undefined` where a cost is not positive):

**HotpotQA**

| pricing | retriever cost | generator cost | g* isotonic | g* linear |
|---|---|---|---|---|
| per GB of peak RSS, as measured, model file memory-mapped (library default): bm25 → e5-small | 0.95 | 5.07 | no crossing in range | no crossing in range |
| per GB of peak RSS, as measured, model file memory-mapped (library default): e5-base → bge-m3 | 0.75 | 5.07 | no crossing in range | no crossing in range |
| per GB of peak RSS, as measured, model file memory-mapped (library default): e5-small → e5-base | 0.90 | 5.07 | no crossing in range | no crossing in range |
| per GB of peak RSS, as measured, model file memory-mapped (library default): tfidf-svd → bm25 | -1.64 | 5.07 | undefined | n/a |
| per GB of peak RSS, estimate without the double-counted mapped pages (probe; see LIMITATIONS): bm25 → e5-small | 0.95 | 2.73 | no crossing in range | no crossing in range |
| per GB of peak RSS, estimate without the double-counted mapped pages (probe; see LIMITATIONS): e5-base → bge-m3 | 0.75 | 2.73 | no crossing in range | no crossing in range |
| per GB of peak RSS, estimate without the double-counted mapped pages (probe; see LIMITATIONS): e5-small → e5-base | 0.90 | 2.73 | no crossing in range | no crossing in range |
| per GB of peak RSS, estimate without the double-counted mapped pages (probe; see LIMITATIONS): tfidf-svd → bm25 | -1.64 | 2.73 | undefined | n/a |
| retriever per ms of retrieval latency, generator per ms of generation: bm25 → e5-small | 9.58 | 10598.76 | no crossing in range | no crossing in range |
| retriever per ms of retrieval latency, generator per ms of generation: e5-base → bge-m3 | 32.38 | 10598.76 | no crossing in range | no crossing in range |
| retriever per ms of retrieval latency, generator per ms of generation: e5-small → e5-base | 26.15 | 10598.76 | no crossing in range | no crossing in range |
| retriever per ms of retrieval latency, generator per ms of generation: tfidf-svd → bm25 | -19.74 | 10598.76 | undefined | n/a |
| per ms of added latency (retrieval + generation): bm25 → e5-small | -2452.88 | 10589.96 | undefined | n/a |
| per ms of added latency (retrieval + generation): e5-base → bge-m3 | 766.00 | 10589.96 | no crossing in range | no crossing in range |
| per ms of added latency (retrieval + generation): e5-small → e5-base | 400.20 | 10589.96 | no crossing in range | no crossing in range |
| per ms of added latency (retrieval + generation): tfidf-svd → bm25 | 5064.19 | 10589.96 | no crossing in range | no crossing in range |
| per ms of generation-stage latency only: bm25 → e5-small | -2462.45 | 10598.76 | undefined | n/a |
| per ms of generation-stage latency only: e5-base → bge-m3 | 733.63 | 10598.76 | no crossing in range | no crossing in range |
| per ms of generation-stage latency only: e5-small → e5-base | 374.05 | 10598.76 | no crossing in range | no crossing in range |
| per ms of generation-stage latency only: tfidf-svd → bm25 | 5083.93 | 10598.76 | no crossing in range | no crossing in range |

**Natural Questions**

| pricing | retriever cost | generator cost | g* isotonic | g* linear |
|---|---|---|---|---|
| per GB of peak RSS, as measured, model file memory-mapped (library default): bm25 → e5-small | 0.83 | 5.11 | no crossing in range | no crossing in range |
| per GB of peak RSS, as measured, model file memory-mapped (library default): e5-base → bge-m3 | 1.21 | 5.11 | no crossing in range | no crossing in range |
| per GB of peak RSS, as measured, model file memory-mapped (library default): e5-small → e5-base | 0.84 | 5.11 | no crossing in range | no crossing in range |
| per GB of peak RSS, as measured, model file memory-mapped (library default): tfidf-svd → bm25 | -2.06 | 5.11 | undefined | n/a |
| per GB of peak RSS, estimate without the double-counted mapped pages (probe; see LIMITATIONS): bm25 → e5-small | 0.83 | 2.77 | no crossing in range | no crossing in range |
| per GB of peak RSS, estimate without the double-counted mapped pages (probe; see LIMITATIONS): e5-base → bge-m3 | 1.21 | 2.77 | no crossing in range | no crossing in range |
| per GB of peak RSS, estimate without the double-counted mapped pages (probe; see LIMITATIONS): e5-small → e5-base | 0.84 | 2.77 | no crossing in range | no crossing in range |
| per GB of peak RSS, estimate without the double-counted mapped pages (probe; see LIMITATIONS): tfidf-svd → bm25 | -2.06 | 2.77 | undefined | n/a |
| retriever per ms of retrieval latency, generator per ms of generation: bm25 → e5-small | 9.81 | 15274.65 | no crossing in range | no crossing in range |
| retriever per ms of retrieval latency, generator per ms of generation: e5-base → bge-m3 | 43.35 | 15274.65 | no crossing in range | no crossing in range |
| retriever per ms of retrieval latency, generator per ms of generation: e5-small → e5-base | 16.44 | 15274.65 | no crossing in range | no crossing in range |
| retriever per ms of retrieval latency, generator per ms of generation: tfidf-svd → bm25 | -17.22 | 15274.65 | undefined | n/a |
| per ms of added latency (retrieval + generation): bm25 → e5-small | 125.13 | 15274.47 | no crossing in range | no crossing in range |
| per ms of added latency (retrieval + generation): e5-base → bge-m3 | -149.76 | 15274.47 | undefined | n/a |
| per ms of added latency (retrieval + generation): e5-small → e5-base | -1080.64 | 15274.47 | undefined | n/a |
| per ms of added latency (retrieval + generation): tfidf-svd → bm25 | 8722.24 | 15274.47 | no crossing in range | no crossing in range |
| per ms of generation-stage latency only: bm25 → e5-small | 115.31 | 15274.65 | no crossing in range | no crossing in range |
| per ms of generation-stage latency only: e5-base → bge-m3 | -193.11 | 15274.65 | undefined | n/a |
| per ms of generation-stage latency only: e5-small → e5-base | -1097.09 | 15274.65 | undefined | n/a |
| per ms of generation-stage latency only: tfidf-svd → bm25 | 8739.45 | 15274.65 | no crossing in range | no crossing in range |

The `pricing` column names the unit and the step. `retriever cost` and
`generator cost` are in GB or ms accordingly. The rows show how much the
per-unit crossover depends on which real step stands in for the dial's synthetic
one.

---

## 4. Measurement hygiene, as a record

**Switched off after the first configuration.** Two attempts at the first configuration ran under the probe gate
and the contention rule, and both were set aside. The first was measured against a baseline taken while the
machine was warm and in use; that baseline and its record are kept in
`results/crossover/calibration_ministral/superseded_warm_baseline/`, and the baseline was taken again on a quiet,
cooled machine. The second was rejected for contention (1 rejected:
contention: 1). The author then switched both off. All 20
accepted batches ran one at a time, with no probe before or after them and their contention recorded but not
enforced: 17 of them back to back, from 03:03 UTC on 12 Nov to 06:39 UTC on 14 Nov 2026, and the
three timed again in a later sitting (below). A batch with more than a quarter of its samples contended would have
been run again (such batches: none). Read every latency, and every per-ms price, with that in view;
the answers do not depend on it. A **timed batch** is one configuration, and nothing else from the
build ran during one (`scripts/sweep_queue.py`).

**The contention record.** Under the switched-off rule, 4 of 20 accepted batches would have
been rejected for contention (the `over the gate's contention rule` column): HotpotQA e5-base+3b (37 of 228); HotpotQA tfidf-svd+8b (18 of 252); NQ e5-small+3b (11 of 178); NQ bge-m3+8b (52 of 648).
Their rows stay as measured. Among them are the lower row of HotpotQA's generator step (e5-base with 3B) and the
upper rung of NQ's priced retriever step (e5-small with 3B).

**One night, timed again.** Three HotpotQA batches ran in the same night, from 04:58 to 15:06 UTC on 12 Nov 2026:
e5-base + 8B, bm25 + 3B and bm25 + 8B. The author picked them out after the run, by their latencies; no recorded
condition marks that span. In their first timings, bm25 + 3B answered more slowly per question than any other 3B
configuration on HotpotQA, while on NQ bm25 + 3B is in line with the other retrievers, and the two 8B rows sat at
the slow end of HotpotQA's 8B rows and over the contention rule. After short probes suggested that the night had run
slower, the author chose to time the three again before the first publication, by a rule fixed before the new
batches ran: exactly these three rows, each by the same bench, configuration and protocol as every other row, one
batch at a time, on the same Mac and macOS, with no other application open. The new batches ran from
21:28 UTC on 14 Nov to 07:41 UTC on 15 Nov 2026, each with the same EM as when first measured: HotpotQA e5-base+8b (end-to-end p50 24.45 s when first timed, 19.58 s when timed again; peak RSS 11.80 GB, then 12.57 GB); HotpotQA bm25+3b (end-to-end p50 14.86 s when first timed, 11.07 s when timed again; peak RSS 5.65 GB, then 5.66 GB); HotpotQA bm25+8b (end-to-end p50 28.23 s when first timed, 23.75 s when timed again; peak RSS 10.77 GB, then 10.76 GB) (peak RSS as measured, model file memory-mapped (library default);
estimate without the double-counted mapped pages (probe; see [LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings)): HotpotQA e5-base+8b 8.10 GB, then 8.87 GB; HotpotQA bm25+3b 4.29 GB, then 4.29 GB; HotpotQA bm25+8b 7.07 GB, then 7.06 GB). All
three answered faster the second time, but that does not show that the night ran slower: no row from outside the
night was timed again, rows picked out because they were slow tend to come out faster when timed again, and two of
the three second timings started on a cooler chip than most batches of the first sitting. With its second timing,
bm25 + 3B is still the slowest 3B configuration on HotpotQA. The tables use the second timings.

The first timings stay in `results/crossover/natural/hotpotqa_k5_ministral/superseded/`, and in
`results/crossover/calibration_ministral/batches.jsonl` as the earlier rows of the same configurations, still marked
`"accepted": true`; a later row marked `"superseded": true`, with no timings of its own, takes each back and names the
kept result (`kept_as`). Its reason, one line that points to this section, replaced the note written when the re-time
began. Another application, open during the second timings at the author's request, departed from the rule; the
contention record counts 8 of 411, 1 of 185 and 1 of 395 samples contended in those batches, none over the gate's
rule. The shipped records keep every count and write each other process's name as "another process". The first attempt
to time bm25 + 8B again (attempt 3 in the record) was stopped part-way by the author, who judged that another
application was slowing it, and it was timed again from the start (attempt 4; 1 stopped batch,
listed below with the rejected one). Its record shows 5 of 186 samples contended, under the
gate's rule. A restart chosen because a batch looked slow can lower the timing that is kept. The same configuration on
the same Mac timed differently in two sittings, as these three rows show, so every latency and memory price carries
this caveat; the answers, and so every EM number, do not depend on it.

### The calibration probe

`scripts/calibration_probe.py` times 20 fixed generations: the
first 20 HotpotQA questions by id, with BM25's cached top 5,
answered by one small, fixed generator (Qwen2.5-1.5B-Instruct 🇨🇳, used here only as a
measure of the machine's speed) under the bench's generator settings, after one
untimed warm-up. It reports the median and hashes the answers. The baseline,
taken before the first batch on macOS 27.2, is 2.197 s.

Under the gate, the queue probes before each batch. If the median is more than
10% away from the baseline **in either direction**, it
waits 10 minutes and probes again. A probe whose answers
differ from the baseline's is a failure, not a delay: generation is meant to be
deterministic, and a different answer means something other than the
machine's speed has changed.

### The contention rule

Every 30 s during a batch, the queue reads `ps` for every
process outside the bench's own process tree. A sample is **contended** if any
one of those processes uses at least 80% of a core, or all
of them together at least 150%. Under the gate, a batch is rejected when
its contended samples reach `max(3, ceil(0.05 * n_samples))`, the `contaminated`
test in the queue's sampler. A rejected batch's result is moved to
`<out>/discarded/`, its row stays in
`results/crossover/calibration_ministral/batches.jsonl`, and the configuration is
measured again.

What the sampler cannot see: anything that starts and ends between two samples,
except as far as `ps`'s CPU figure reflects it, and load spread over many processes that each stay below
both thresholds. This is a laptop that is also used for other things.

### The record in numbers

| the record | |
|---|---|
| probes checked against the baseline | 2 |
| batches accepted | 20 |
| of which ungated | 20 |
| batches rejected, by reason | 1 (contention: 1) |
| batches stopped part-way by the author | 1 |
| accepted batches timed again | 3 of 20 |
| wall-clock of the accepted batches | 49.7 hours |

Every calibration probe the queue ran is a row in `results/crossover/calibration_ministral/probes.jsonl`; the short
probes read before the three rows were timed again ran outside the queue and are not recorded. Every attempt at a
batch, accepted or not, is a row in `results/crossover/calibration_ministral/batches.jsonl`. The file has
23 rows marked accepted: each configuration timed again keeps its first accepted
row, which a later row marked `superseded` takes back, so the 20 counted here are those
not taken back.

### Accepted batches

| sweep | configuration | hygiene | wall min | probe before | probe after | contended samples | macOS | over the gate's contention rule | chip °C, start → end |
|---|---|---|---|---|---|---|---|---|---|
| HotpotQA k5 | e5-base+ministral-3-3b | off | 115 | n/a | n/a | 37 of 228 | 27.2.1 | yes | 48 → 57 |
| HotpotQA k5 | e5-small+ministral-3-3b | off | 87 | n/a | n/a | 5 of 174 | 27.2.1 | no | 60 → 53 |
| HotpotQA k5 | e5-small+ministral-3-8b | off | 156 | n/a | n/a | 0 of 310 | 27.2.1 | no | 59 → 59 |
| HotpotQA k5 | bge-m3+ministral-3-3b | off | 171 | n/a | n/a | 6 of 341 | 27.2.1 | no | 62 → 54 |
| HotpotQA k5 | bge-m3+ministral-3-8b | off | 312 | n/a | n/a | 5 of 622 | 27.2.1 | no | 60 → 56 |
| HotpotQA k5 | tfidf-svd+ministral-3-3b | off | 55 | n/a | n/a | 0 of 109 | 27.2.1 | no | 61 → 54 |
| HotpotQA k5 | tfidf-svd+ministral-3-8b | off | 127 | n/a | n/a | 18 of 252 | 27.2.1 | yes | 59 → 55 |
| NQ k5 | e5-base+ministral-3-3b | off | 123 | n/a | n/a | 4 of 244 | 27.2.1 | no | 60 → 55 |
| NQ k5 | e5-base+ministral-3-8b | off | 202 | n/a | n/a | 10 of 402 | 27.2.1 | no | 59 → 54 |
| NQ k5 | bm25+ministral-3-3b | off | 68 | n/a | n/a | 0 of 134 | 27.2.1 | no | 60 → 54 |
| NQ k5 | bm25+ministral-3-8b | off | 155 | n/a | n/a | 5 of 307 | 27.2.1 | no | 60 → 54 |
| NQ k5 | e5-small+ministral-3-3b | off | 89 | n/a | n/a | 11 of 178 | 27.2.1 | yes | 58 → 54 |
| NQ k5 | e5-small+ministral-3-8b | off | 167 | n/a | n/a | 4 of 333 | 27.2.1 | no | 58 → 54 |
| NQ k5 | bge-m3+ministral-3-3b | off | 241 | n/a | n/a | 7 of 480 | 27.2.1 | no | 58 → 54 |
| NQ k5 | bge-m3+ministral-3-8b | off | 325 | n/a | n/a | 52 of 648 | 27.2.1 | yes | 59 → 56 |
| NQ k5 | tfidf-svd+ministral-3-3b | off | 26 | n/a | n/a | 0 of 52 | 27.2.1 | no | 59 → 56 |
| NQ k5 | tfidf-svd+ministral-3-8b | off | 66 | n/a | n/a | 4 of 131 | 27.2.1 | no | 59 → 55 |
| HotpotQA k5 | e5-base+ministral-3-8b | off | 206 | n/a | n/a | 8 of 411 | 27.2.1 | no | 49 → 54 |
| HotpotQA k5 | bm25+ministral-3-3b | off | 93 | n/a | n/a | 1 of 185 | 27.2.1 | no | 47 → 54 |
| HotpotQA k5 | bm25+ministral-3-8b | off | 198 | n/a | n/a | 1 of 395 | 27.2.1 | no | 59 → 55 |

`wall min` runs from the start of the batch to its end, index build included.
`probe before` and `probe after` read `n/a` because no probe ran with the gate off.
`contended samples` is contended samples out of all samples taken, and `over the gate's contention rule` says
whether they reach `max(3, ceil(0.05 * n_samples))`, the rule the gate would have applied. `chip °C` is the hottest
CPU die sensor at the batch's start and end. The three rows timed again appear with their second timings, after
the others; their first timings are the earlier rows of the same configurations in `batches.jsonl`, each taken back by
a later row marked `superseded`.

### Batches not accepted

| sweep | configuration | attempt | reason |
|---|---|---|---|
| hotpotqa_k5 | e5-base+ministral-3-3b | 1 | contention |
| hotpotqa_k5 | bm25+ministral-3-8b | 3 | stopped part-way by the author, who judged that another application was slowing it; timed again from the start |

A rejected batch's numbers go into no price. Its row in `batches.jsonl` keeps
them (EM, peak RSS, end-to-end p50 and p95, the probe, the contention
summary), so the rejection can be audited. The stopped batch wrote no result: its row keeps its start and end, its
wall-clock, the chip's temperatures and its contention summary, and its own reason reads `harness exit 1, status
None`; the reason in the table is the author's, named in `scripts/stats_tokens.py` (`STOPPED`). `attempt` is one more
than the number of rows already on record for that configuration since the baseline was last taken, a row marked
`superseded` included. It started again with the second baseline, so the rejected attempt reads 1 although it was the
configuration's second, and the configurations timed again have no attempt 2.

---

*Generated from `results/stats.json`. Measurement code: `src/measurerag/worker.py`,
`src/measurerag/memory.py`, `src/measurerag/budget.py`. Hygiene:
`scripts/sweep_queue.py`, `scripts/calibration_probe.py`.*
