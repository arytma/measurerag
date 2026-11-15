<!-- Generated from templates/docs/METHOD.md and results/stats.json by scripts/fill.py. Edit the template, not this file. -->
# Method: how every number is produced

This document is for judging whether the measurements are sound. It follows each
number from the code that produces it to the file that holds it, and it reports
no findings: those are in [CROSSOVER.md](CROSSOVER.md) and
[BUDGETS.md](BUDGETS.md), and what the design cannot show is in
[LIMITATIONS.md](LIMITATIONS.md).

Two things are defined elsewhere and only summarised here. The crossover's
terms (g, Δ_gen, Δ_ret, g\*, the dial, the per-GB and per-ms crossovers) are
defined in [CROSSOVER.md](CROSSOVER.md) §1 to 2 and nowhere else. The metrics
(nDCG, recall, gold-in-context, EM, F1) are defined in [METRICS.md](METRICS.md).
How to rerun all of it is in
[REPRODUCE.md](REPRODUCE.md).

Every setting quoted below is filled from `results/stats.json` (§6), so this
document changes when the configuration does.

---

## 1. The budget bench

### One configuration, one process

`measurerag run --config <yaml>` (`src/measurerag/runner.py`) resolves the
dataset slice and then launches `python -m measurerag.worker` once per
configuration: one retriever, or one retriever × one generator when the RAG
stage is on. The parent imports nothing heavy and stays small, nothing is
inherited from one configuration's high-water mark by the next, and a crash
loses only the configuration that crashed. Inside the child (`src/measurerag/worker.py`), in this order:

1. build the retriever's index over the whole slice, with every core
   (`build_threads`; indexing is a one-off offline job);
2. pin the query path to `query_threads` (one) and retrieve every question, one
   at a time, to depth 100;
3. score the ranking (nDCG, recall and gold-in-context at the context k);
4. if the RAG stage is on, answer every question from its top-k passages (§2);
5. compare the measurements with every budget in the grid.

A configuration that already finished under a byte-identical spec (slice path,
caps, seed, thread counts, retriever declaration) is not measured again;
`--no-resume` forces a new measurement. Before any of this, the OpenMP
preflight refuses an environment in which two OpenMP runtimes would load into
one process.

### Peak RSS, and what "GB" means

`PeakRSS` (`src/measurerag/memory.py`) wraps steps 1 to 4. A background thread
samples the child's own resident set size with `psutil`, and on exit the peak
is combined by `max()` with `getrusage`'s `ru_maxrss` (bytes on macOS,
kibibytes on Linux, converted); the sampler catches what a coarse `ru_maxrss`
misses, `ru_maxrss` catches a spike between samples. The window covers the
index build, retrieval, loading the generator and generating, so the number is
the whole process at its worst moment: encoder weights, index, generator
weights, tokenizer, runtimes and allocator fragmentation together. It is
not the index size.

The generator loads its GGUF with the model file memory-mapped (`use_mmap`, llama-cpp-python's default, a setting of
`LlamaGenerator` that each run records in its result), and every timed row was measured so. On the CPU, llama.cpp
also copies most weights into its own buffers, and the mapped file's pages stay resident beside that copy, so peak
RSS counts that part of the weights twice. The documents give every figure that rests on the English Ministral 3 🇫🇷
rows' peak RSS as measured and, beside it, as an estimate without the double count: each row's peak less the surplus
that `scripts/mmap_probe.py` measured for its generator, loaded alone with mapping on and off
([LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings)).

Units differ by column, and the tables keep them apart:

* **peak RSS** is in binary gigabytes, `peak_bytes / 1024**3`, and the RAM
  budgets are compared against that number;
* **model GB** and **index MB** in the budget tables are decimal,
  `model_bytes / 1e9` and `index_bytes / 1e6`; so are the model sizes taken
  from `results/crossover/models_manifest.json` (as in
  [REPRODUCE.md](REPRODUCE.md) §1).

### Index and model bytes

* **Index bytes** are the size of what the retriever writes into the
  configuration's work directory: the FAISS flat index, the tfidf-svd matrix,
  the bm25s index directory.
* **Model bytes** are the retriever's plus the generator's. For a dense encoder,
  `_model_disk_bytes` in `src/measurerag/retrievers/dense.py` counts what you
  would copy onto the target machine: every file of the local Hugging Face
  snapshot except ONNX, OpenVINO, Core ML and TFLite exports and `.git`, with
  duplicate weight formats counted once and safetensors preferred. It
  counts the local snapshot, so it depends on what was downloaded; the models
  are fetched through sentence-transformers, so the count means the same thing
  in every row. For a generator it is the
  GGUF file's size. bm25 and tfidf-svd count zero: the index is the model.

### Latency

* **Retrieval latency** is `time.perf_counter` around one call of `search_one`
  for one question. For a dense retriever that call encodes the question (a
  batch of one) and searches the index, so encoding is included. These are
  single-query service latencies, not throughput.
* **End-to-end latency** is per question: from fetching the passage texts,
  through building the prompt, to the generated answer. The question's
  retrieval is not inside it; it was timed in step 2.
* **p50 and p95** are nearest-rank percentiles: the value at rank
  `ceil(p/100 * n)` of the sorted observations. Each is an observation that
  happened, and the computation is exactly reproducible.
* **Threads.** Query encoding runs on one torch thread and bm25s searches with
  one thread. The generator runs on the CPU with nothing offloaded and with
  llama.cpp's default thread count; the sweep configurations do not set
  `n_threads`.

### Budget verdicts and the grid

A budget (`src/measurerag/budget.py`) is a RAM ceiling on peak RSS, a ceiling
on retrieval p95 and, optionally, a ceiling on end-to-end p95. A configuration
**fits** only if every ceiling it is subject to holds, and ceilings are
inclusive. A retrieval-only configuration has no end-to-end latency, so it
cannot fit a budget that names one, and its verdict says why; that axis
enters the grid only when the RAG stage is on. The grid in every crossover sweep
is the cross product of RAM ceilings of 4 GB, 8 GB and 16 GB, retrieval
p95 ceilings of 200 ms and 500 ms and end-to-end p95 ceilings
of 3 s and 8 s.
Details: [METRICS.md](METRICS.md#budget-verdicts).

### What a run writes

`results.json`, `results.csv`, `results_table.txt`, two curves, and
`manifest.json`: the SHA-256 of the slice's corpus, queries and qrels (each
hashed separately, over records rather than file bytes), of the resolved config
and of every result file, the package versions, the machine, and
`results_stable_sha256` over the metrics with every timing removed. Each
configuration's answers go to `predictions_<config>.json` beside them; those,
the slice, the specs and the indexes are not committed.

---

## 2. The RAG stage

`src/measurerag/rag/generator.py`, called by the worker.

* **One prompt.** A single template, `src/measurerag/prompt_template.txt`,
  serves every configuration, dataset and generator. Its first revision, the
  yes/no line, was made before any end-to-end number existed. One
  instruction depends on the language: Czech asks for `ano` or `ne`
  where the template asks for `yes` or `no`, because SQAD's gold is `ano` or `ne`
  and exact match compares strings. The top-k
  passages are numbered `[1]`, `[2]`, … and each is cut to a fixed number of
  characters (`max_chars_per_passage` in `build_prompt`), so a retriever cannot
  win by handing the generator more text than its rival. The prompt is sent as
  one user message through llama.cpp's chat completion, which applies the
  model's own chat template.
* **Generators.** All Q4_K_M GGUF, through llama-cpp-python. The English sweeps, and so the English
  headline, budgets and prices, use Ministral 3 3B 🇫🇷 and 8B Instruct 2512, on the Mac's CPU. The English dial
  ran with Gemma 3n E2B 🇺🇸 and E4B, Gemma 4 E2B 🇺🇸 and E4B, Ministral 3 🇫🇷, and Qwen2.5-1.5B-Instruct 🇨🇳 and 3B-Instruct.
  On Czech, Gemma 4 E2B 🇺🇸 and E4B, instruction-tuned, answer beside Qwen2.5 🇨🇳: a short benchmark on the dev
  set chose the family. Gemma 3n 🇺🇸, Ministral 3 🇫🇷 and Qwen3.5 2B 🇨🇳 and 4B were added at the author's request
  ([LIMITATIONS.md](LIMITATIONS.md#the-czech-and-english-families)). The files of every family but Qwen2.5 🇨🇳 are
  Unsloth's, pinned to a commit. Every Czech answer behind the reported sets, and the English dials of Gemma 3n 🇺🇸,
  Gemma 4 🇺🇸 and Ministral 3 🇫🇷, were generated with every layer on an NVIDIA GB10's GPU (`device: cuda`), by the same version
  of llama-cpp-python built with CUDA; Qwen2.5's 🇨🇳 English dial and every timing come from the Mac's CPU.
* **Decoding.** Greedy (`temperature = 0`, `top_k = 1`, `top_p = 1`), seed
  20260101, `max_tokens` = 64, context window `n_ctx` =
  4096.
* **The token cap.** Every answer is cut at `max_tokens` = 64. Before the English sweeps, the
  first questions of the HotpotQA slice (sorted ids) were answered at 64 and at
  256, from e5-small's top five, through the dial's generation path, by Qwen2.5's 🇨🇳 two
  models, and compared string for string (`scripts/max_tokens_check.py`); the rule was to keep the larger cap
  if EM or F1 moved by more than the script's tolerance. The record is `results/crossover/max_tokens_check.json`
  (and `max_tokens_check_czech.json` for the Czech slice). On the same machine, the same comparison showed that
  the dial's path (cached ranking, prompt bytes, decoding) reproduces the bench's answers when both walk the
  questions in the same order. Ministral 3's 🇫🇷 dial ran on the GX10 and its sweeps on the Mac, so its dial and
  bench answers come from two machines ([LIMITATIONS.md](LIMITATIONS.md#the-czech-and-english-families)).
* **Order.** One process answers many questions, and the answer to a near tie can
  turn on the prompt before it. So a rerun reproduces an answer exactly only in
  the same order, and the cache holds one answer per prompt, from whichever order
  reached that prompt first. The bench and a dial's first pass walk sorted question
  ids; a later question set reuses the answers already cached for its prompts. Clearing
  the state between prompts would remove the effect and would change the protocol
  behind every result, so it was not done.
* **Context k.** The generator sees the first k passages of the ranking, and
  gold-in-context is measured at the same k. The dial uses k =
  5, and the natural sweeps use the dial's k.
* **Scoring.** Exact match and token F1 after SQuAD-style normalisation, with
  diacritics kept; no model judges another
  ([METRICS.md](METRICS.md#end-to-end-answer-quality)).

---

## 3. The crossover measurements

What follows is how [CROSSOVER.md](CROSSOVER.md) §1 to 2 is implemented. Code:
`src/measurerag/crossover/`; analysis: `scripts/crossover.py`; inputs:
`configs/crossover.yaml`.

### The controlled dial

`dial.py`. Each gold passage of a question gets one uniform draw u from a
generator seeded by the kit's seed, the dataset's name and the question id. The
draw runs over all of the question's gold passages in sorted-id order, so it
does not depend on which retriever found them. A gold passage in the top k is
removed at every level p with u < p and replaced by the next-ranked non-gold
passage from below rank k of the cached top-100 ranking.
Substitutes are assigned in drop order (ascending u, then rank), so a removed
passage keeps its substitute at every higher level.

The levels are therefore nested: what is gone at one level is gone at every
higher one, which makes adjacent levels a paired comparison. At p = 0 the
context is the retriever's own top k; at p = 1 it holds no gold. The levels are
p = 0, 0.2, 0.4, 0.6, 0.8, 1.0, and one retriever step on the dial is one level, a
change of 0.2 in p, with the smaller generator. The dial
runs on e5-base for HotpotQA and e5-base for
NQ; for HotpotQA because it is the strongest cached retriever by nDCG@10, the kit's
quality key. Questions with no gold passage in the qrels are left out. Most
(level, question) pairs share their prompt with a lower level, so the dial
needs far fewer generations than levels × questions.

### Stated assumption: the NQ gold mapping

The dial needs to know which passages are gold. For HotpotQA these are BEIR's
relevance judgements, and the short answers come from the original dataset,
joined on the question id. NQ has no such pairing: BEIR's nq ships passages and
judgements without short answers, and nq-open ships short answers without
passages. measureRAG therefore makes an assumption on which every NQ number
rests:

> **A passage is gold for a question only if BEIR judges it relevant and its
> body text contains an nq-open answer as a whole-token span** after SQuAD
> normalisation. Questions are joined on normalised question text.

Containment alone would make every passage that mentions the answer string
gold; relevance alone would count passages about the right page that do not
carry the answer. Of BEIR's 3,452 NQ test questions,
2,255 (65.3%) join to nq-open, and
2,083 of those (92.4%) have at least one passage
that passes the rule; the 300 questions measured are a seeded sample of
them. Judged passages that fail the rule stay in the corpus as ordinary
non-gold passages, so the corpus a retriever searches does not depend on it.

Two consequences travel with every NQ result. A judged passage that gives the
answer in other words is not gold, so g can understate what reached the
generator. And a passage that is not gold can still hold the answer: at NQ's
no-gold level (p = 1), 27% of the contexts still contain an
nq-open answer in a non-gold passage (HotpotQA: 5%).
On NQ, the difference is established at 1 of 5 levels, and only at the no-gold level (g = 0), so nowhere that gold is present, so NQ's g\* (g* = 0.56 [0.29, 0.83], the upper end censored; 1847 of 10000 resamples have no root)
should be read with this assumption in view. The rule is an assumption, not an
annotation, and it is stated as one in the README and in
[LIMITATIONS.md](LIMITATIONS.md).

### The natural sweep

Real retrievers through the unchanged bench (`configs/crossover/*.yaml`),
weakest first: tfidf-svd (a deliberately weak instrument rung),
bm25, multilingual-e5-small, multilingual-e5-base and bge-m3, each with
both generators, at the dial's k on both datasets. The analysis computes each
configuration's g from the cached ranking at the sweep's k and records whether
it equals the bench's own gold-in-context. Each bench configuration builds its
own index, so its peak RSS includes the build.

### What is cached

* **Rankings**, one per retriever × dataset:
  `results/crossover/cache/<dataset>/retrieval_<retriever>.json` (`retrieval.py`,
  written by `scripts/retrieval_cache.py`). The top-100 ids and
  scores for every question, produced exactly as the bench produces them
  (`search_one`, one question at a time, one query thread), with the slice's
  corpus and query hashes and a SHA-256 of the ranking. A dial context at p = 0
  is therefore byte for byte the context the bench handed the generator.
* **Answers**: `results/crossover/cache/<dataset>/gen_<generator>.jsonl`
  (`generate.py`, written by `scripts/dial_generate.py`). Append-only, one row
  per distinct prompt, keyed on the SHA-256 of (generator, `max_tokens`, seed,
  prompt). Thread count and seconds are recorded beside each answer but are not
  in the key, which is safe only because this path was checked to reproduce
  the bench's strings on one machine (the token cap, §2).
* **Natural-sweep answers** are the bench's `predictions_<config>.json` files,
  beside each sweep's `results.json`.

`scripts/crossover.py` reads only these, the slices and the bench's
`results.json` files; it loads no model. It refuses a ranking or a bench run
whose corpus hash differs from the slice it is analysing, and with missing dial
answers it records the dataset as incomplete (`--allow-partial` gives a progress
look that is never committed).

### Statistics

`stats.py`, applied by `scripts/crossover.py`:

* **Paired bootstrap**, 10000 resamples. A resample is an integer count
  vector over the questions, drawn (multinomial) from a generator seeded by the
  kit's seed and the dataset. Every condition (each dial level, each generator,
  each natural configuration) is re-scored on the same counts, so every
  difference is paired. EM is 0 or 1, so every resampled sum is an exact
  integer and every resampled mean an exact ratio.
* **Intervals** are 95% percentile intervals (inverted-CDF quantiles over the
  resamples). An effect is *established* when its interval excludes zero.
* **Roots.** The curve Δ_ret − Δ_gen over the dial's g is fitted two ways. The
  isotonic fit is a monotone non-increasing least-squares fit by pool adjacent
  violators, written out rather than taken from a library so that the bootstrap
  is a fixed sequence of floating-point operations; its root is the first zero
  crossing of the linear interpolation through the fitted points. The linear
  fit is ordinary least squares in closed form; its root counts only inside the
  observed g range. The headline uses the isotonic root, and the fits
  *agree* when each root lies inside the other's interval. The same roots are
  computed on the all-gold axis and for a retriever step worth two dial levels.
* **Censoring.** A resample whose fitted curve never crosses zero is
  given the edge of the observed g range it fell off (the top when that
  resample's Δ_ret − Δ_gen is positive on average, the bottom otherwise), not
  dropped. The count is reported, and an interval is flagged as censored
  when more resamples were censored than one tail of the 95% interval holds.
* **Break-even gold gain.** At each operating point, the smallest
  increase in g along the dial's smaller-generator EM curve (linearly interpolated) whose EM
  gain reaches Δ_gen at that point. Per GB, the target is Δ_gen scaled by the
  ratio of the priced retriever step's GB to the generator step's GB. Where the
  headroom above the operating point is not enough, the resample is censored at
  the headroom and counted.
* **Dial against reality.** The dial's curves, interpolated in g within each
  resample, predict every natural point; the two checks reported ("inside
  band", "consistent") are described in [CROSSOVER.md](CROSSOVER.md) §2.

### Prices

A price is a difference between two bench rows at the dial's k, never across
k: peak RSS in GB, end-to-end p50 in ms and, for one variant, retrieval p50 in
ms. Because peak RSS covers the index build (§1), a GB price is the difference
of two whole-process high-water marks, wherever in each run they occurred.

* **The retriever step.** The dial's step is synthetic and has no cost of its
  own, so it is priced at the natural step bm25 → e5-small with the smaller
  generator. Every other ladder step with a positive cost is reported as a
  sensitivity.
* **The generator step** is the smaller to the larger generator (3B → 8B) in
  front of the dial's retriever. Until
  that pair of rows exists it is priced at the priced step's upper rung, and
  `crossover.json` marks it as a fallback.
* **Per-unit crossovers** are the roots of Δ_ret / (retriever cost) − Δ_gen /
  (generator cost), with the same fits, bootstrap and censoring. The
  "component" variant prices the retriever step at the retrieval p50 it adds and
  the generator step at the end-to-end p50 it adds.
* **A move that does not add cost has no per-unit price.** When either
  cost is zero or negative, the per-unit crossover is recorded as undefined,
  with both costs, rather than computed.

---

## 4. Measurement hygiene

Every timing that sets a price comes from one machine, a fanless MacBook Air
(Apple M4, 32 GB), measured one configuration at a time
by `scripts/sweep_queue.py`.

**The rules below were switched off after the first configuration.** Two attempts at it ran under them and
were set aside ([BUDGETS.md](BUDGETS.md), §4); the author then had the queue restarted with `--no-hygiene`: no
probe, no gate, no rejection for contention, batches back to back, the contention sampler still recording, and
a batch with more than a quarter of its samples contended run again. 0 accepted
batches ran under the rules and 20 without them; `batches.jsonl` marks each row
`"hygiene": "on"` or `"off"`. Everything below describes the rules as they apply under the gate.

### The calibration probe

`scripts/calibration_probe.py` times 20 fixed generations: the
first 20 questions of the HotpotQA slice (sorted ids), with
bm25's cached top five passages, by Qwen2.5-1.5B-Instruct 🇨🇳, a small fixed generator used only to measure
the machine's speed, at the bench's settings (seed 20260101, 64 tokens, `n_ctx`
4096, llama.cpp's default threads), after one untimed warm-up. It records the median
latency, the SHA-256 of the answers, the one-minute load average and the macOS
version, and appends the row to `results/crossover/calibration_ministral/probes.jsonl`.
The baseline (`--baseline`, written to `baseline.json`) is 2.197
s, taken on macOS 27.2. A probe passes when its median is within
10% of the baseline, in either direction, **and** its
answers hash to the baseline's; the second condition checks, before every
batch, that generation is still deterministic.

### One timed batch is one configuration

For each configuration (one retriever × one generator at one k), the queue:

1. waits while anyone holds the queue through `scripts/quiet.sh`, then takes the
   phase's `RUNNING` marker (`logs/sweep_ministral/RUNNING` for Ministral 3 🇫🇷);
2. probes; on drift it waits 10 minutes and probes again,
   up to a fixed number of waits. A probe that errors or whose answers differ
   from the baseline is not waited out: the attempt fails;
3. runs `measurerag run --config <sweep> --only <retriever>+<generator>` alone,
   while the contention sampler runs;
4. probes again (recorded, not enforced; see below);
5. accepts or rejects the batch.

### The contention sampler

Every 30 s the queue lists every process with `ps` and leaves
out the bench's process tree (the `measurerag run` process and its
descendants) and itself. A sample is *contended* when one foreign process uses
at least 80% of a core, or all foreign processes together
use at least 150%. A batch is rejected for contention when
its contended samples reach a fixed share of all its samples, with a minimum
count (`Sampler.summary` in the script). The sampler also watches the bench:
no CPU progress for 45 minutes and the run is killed and
recorded as stuck.

### Rejection and rerun

A batch is accepted only if the bench exited cleanly with status ok, the
batch was not contended, and it did not stall. Every attempt, accepted or not,
is a row in `results/crossover/calibration_ministral/batches.jsonl` (both probes, the
contention summary, EM, peak RSS, p50 and p95, the macOS version, the reason).
A rejected attempt's result moves to `<out_dir>/discarded/`, and the sweep's
`results.json` is reassembled from accepted configurations only. The queue
commits the sweep directory and the calibration log after every attempt.

A bench failure counts against a limit of 3 attempts. A
rejection
for contention is the hygiene rule doing its job, not a failure: it is
counted separately and retried after a 10-minute wait
outside the RUNNING marker. A stage that ends failed, stuck, contended or never within tolerance is
retried once at the end. On a restart, a configuration with an accepted batch
on record is skipped, and a result on disk without one is discarded and
measured again.

### Why the probe after a batch is recorded, not enforced

The machine has no fan. A long batch warms it by itself, which is the
sustained-load condition every configuration meets alike, not outside
interference. The probe before a batch is what keeps outside heat and load out;
the probe after it is kept as information.

### Heavy work outside the queue

Tests, the bootstrap analysis and any other heavy command run through
`scripts/quiet.sh <command>`, which takes a hold the queue honours before its
next batch, waits for the batch in flight to finish, runs the command and
releases the hold. Generation is never run in parallel.

### Where the timed rows are

Ministral 3's 🇫🇷 sweeps are in `results/crossover/natural/hotpotqa_k5_ministral` and
`results/crossover/natural/nq_k5_ministral`, and their record in `results/crossover/calibration_ministral/`.
Qwen2.5's 🇨🇳 English sweeps, measured the same way on the same Mac in earlier sittings, are in
`results/crossover/natural/` and their record in `results/crossover/calibration/`; no document tabulates them
([LIMITATIONS.md](LIMITATIONS.md#the-czech-and-english-families)).

---

## 5. Determinism and reproducibility

**Byte-identical by construction, given the same inputs:**

* **Slices.** Seeded sampling (`rng` in `src/measurerag/seeds.py` derives an
  independent stream from the seed and a string tag), caps that keep every gold
  passage, and corpus, query and qrels hashes recorded in each slice's
  `meta.json`, in every run's manifest and in `results/crossover.json`.
* **Rankings**, with a SHA-256 of each recorded in `results/crossover.json`.
  The cached bm25, e5-small and tfidf-svd rankings were checked to reproduce
  the bench's own retrieval metrics. Identity has been checked on one
  machine only.
* **Answers.** Greedy decoding, a fixed seed and the same prompt bytes give the
  same strings on one machine. Checked by the token-cap comparison (§2) and by
  the calibration probe's answer hash before every gated batch (§4).
* **Everything computed from answers and rankings:** EM, F1, g, every Δ, every
  raw g\* and break-even gain.
* **`results/crossover.json`.** Integer bootstrap counts, exact integer sums, a
  hand-written isotonic fit, floats rounded to six places and keys sorted.
  `scripts/repro_check.py` runs `scripts/crossover.py` twice into
  `logs/sweep/repro/` and requires both outputs to equal each other and the
  committed `results/crossover.json` and `results/stats.json` (the latter
  compared without its `repro.*` tokens and its `source` field), and writes
  `results/crossover/repro_check.json`. Its last result, as `stats.json`
  carries it: crossover.json identical: yes;
  stats.json identical: yes.
  `tests/test_crossover_determinism.py` runs the same
  check offline on a synthetic slice with synthetic caches.
* **`results/stats.json`** and the documents filled from it: nothing in either
  depends on the clock.

**Not reproducible, and not claimed to be:** retrieval and end-to-end
latencies, build seconds, wall-clock, generator throughput, peak RSS (equal
only within noise), the budget verdicts derived from them, and therefore every
price and every per-GB or per-ms g\*. `crossover.json` is a deterministic
function of the bench files it reads, but re-measuring a sweep changes those
files. `measurerag repro --config <yaml>` runs a configuration twice and compares
the manifests with these fields removed
([METRICS.md](METRICS.md#reproducibility)).

**Models are verified, not trusted.** `scripts/fetch_verify_models.py`
requires Qwen2.5's 🇨🇳 two GGUFs' `main` to resolve to the pinned revision, their
SHA-256 to equal that revision's LFS object id and their size to equal the
recorded size, and each encoder's `main` to resolve to the pinned commit and its
shipped byte count to equal the recorded count exactly. It writes
`results/crossover/models_manifest.json` with a SHA-256 for every model file it
checks. All verified: yes. The other families' GGUFs are pinned
to a commit in `src/measurerag/rag/generator.py`, and every result records the
generator file's size.

---

## 6. Documents from data

* **`results/stats.json`** holds every number the documents quote, already
  formatted as text. `scripts/crossover.py` writes it together with
  `results/crossover.json`: the crossover tokens from its own analysis, and the
  other families (`design.*`, `run1.*`, `models.*`, `data.*`, `runs.*`,
  `hygiene.*`, `gens.*`, `budget.*`, `remeasure.*`, `repro.*`, `headline.*`) from
  `scripts/stats_tokens.py`, which reads committed results, the configurations,
  the calibration logs and cached data only. Ministral 3's 🇫🇷 English tokens carry the
  prefix `en_m3.`; the documents quote those for every English sweep, budget and price.
* **Templates.** Each document whose prose carries a result is generated from a
  file of the same path under `templates/`, in which every number is a
  double-brace placeholder naming a token. `scripts/fill.py --all` fills every
  template; a token the stats file does not define fails the build and nothing
  is written. `--check` fails if any generated document differs from its
  template filled from the current stats file. Each generated file starts with
  a comment naming its template.
* **`scripts/check_docs.py`** checks the whole set listed in
  `configs/docs.yaml`, and fails on: a missing document; a generated document
  that is stale or was edited by hand; a template token the stats file lacks; an
  unfilled placeholder; a number in prose that does not appear among the numbers
  rendered in the stats file, unless it is a small counting integer, a year, a
  date, a time or a version (the licence texts and the third-party attributions are
  exempt, with the reasons in `configs/docs.yaml`); a
  broken relative link or `#anchor`; and a shell block tagged `# test` that does
  not exit 0 with the network cut off by environment variables.
  `scripts/doc_lint.py` applies the same rules to one template while a
  document is being written.
