# The crossover: where does the next gigabyte go?

A RAG system on one machine can spend its next gigabyte on a stronger retriever or on a bigger generator.
Which buys more exact match (EM) depends on how often the retriever already puts a gold passage in front of the
generator, the gold-in-context rate g. On HotpotQA at k = 5, the step from the weak tfidf-svd to bm25 buys
{{en_m3.hotpotqa.step.tfidf-svd->bm25.k5.ret}} EM, and the generator step at tfidf-svd {{en_m3.hotpotqa.step.tfidf-svd->bm25.k5.gen}}:
{{en_m3.hotpotqa.step.tfidf-svd->bm25.k5.larger}}. One step up, bm25 to e5-small buys {{en_m3.hotpotqa.step.bm25->e5-small.k5.ret}}
and the generator step at bm25 {{en_m3.hotpotqa.step.bm25->e5-small.k5.gen}}: {{en_m3.hotpotqa.step.bm25->e5-small.k5.larger}}
(generator minus retriever {{en_m3.hotpotqa.step.bm25->e5-small.k5.gen_minus_ret}}; every step, with its g, is in §3).
This document looks for the crossing between such points with a controlled dial, puts an interval on it, prices
both moves in gigabytes and milliseconds, checks the dial against real retrievers, and repeats it on a second
dataset.

The generator step is Ministral 3 3B to Ministral 3 8B (Instruct 2512, Q4_K_M). Its dial answered on a second
machine's GPU and its natural sweeps were timed on the laptop's CPU
([LIMITATIONS.md](LIMITATIONS.md#the-czech-and-english-families)). The other families' dials are in
[Four families' English dials](#four-families-english-dials).

**The answer in one paragraph.**

* **HotpotQA, raw:** {{en_m3.headline.hotpotqa.gstar_raw}}; {{en_m3.hotpotqa.established_text}}.
* **HotpotQA, per GB:** as measured, model file memory-mapped (library default), {{en_m3.headline.hotpotqa.gstar_gb}}; estimate without the double-counted mapped pages (probe; see [LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings)),
  {{en_m3.est.headline.hotpotqa.gstar_gb}}.
* **Natural Questions, raw:** {{en_m3.headline.nq.gstar_raw}}; {{en_m3.nq.established_text}}.
* **Do they agree?** {{en_m3.headline.nq_agrees}}. In full: {{en_m3.agreement_verdict}}.
* **What a crossing point cannot say.** A retriever step and a generator step are
  not fixed quantities, so the crossover has to be read with its step size. On
  the dial, a retriever step is one level of gold removed.
* **The usable form is a break-even gold gain.** For each operating point,
  §3 and §4 state how much gold-in-context a retriever upgrade must add before it
  matches the 3B → 8B upgrade. Measure what your upgrade adds, and compare.

---

## 1. Definitions

These are the only definitions of these terms in the kit. The code in
`src/measurerag/crossover/` implements them, and `scripts/crossover.py` is the
only thing that computes the numbers.

**Gold-in-context, g.** The fraction of questions for which the k passages
handed to the generator contain a gold passage. HotpotQA questions have two gold
passages, so two versions are recorded:

* *g any*: at least one gold passage is in the context. **This is the primary g.**
* *g all*: every gold passage is in the context.

A question with no gold passage in the qrels counts as neither.

**The two moves.** Both are measured at the same k, the same prompt template
(`src/measurerag/prompt_template.txt`) and the same
seed, with greedy decoding at `max_tokens` = {{max_tokens}}.

* **Δ_gen(g)**: the EM gain from replacing Ministral 3 3B with Ministral 3 8B
  (both Instruct 2512, Q4_K_M), with the retrieval held fixed, at operating
  point g.
* **Δ_ret(g)**: the EM gain, with the 3B generator, from moving one step up
  the retriever ladder, starting from operating point g.

**The crossover, g\*.** The g at which Δ_ret(g) = Δ_gen(g). Below g\*, the next
unit of budget buys more on the retriever; above it, on the generator.

**Per GB and per ms.** Each move is also divided by what it costs, as measured
by the budget bench: the increase in peak resident set size of the whole
process (retriever, index and generator together), and the increase in the
latency a user waits. The bench times retrieval and the generation stage
(prompt build and generation; it calls this "end-to-end") separately, so the
added latency is the change in retrieval p50 plus the change in generation
p50: a sum of medians, not the median of a sum. The per-GB
crossover is where Δ_ret / GB = Δ_gen / GB.
A move that does not *add* cost (a retriever change that lowers p50 because
its passages are shorter) has no per-unit price, and its crossover is reported
as undefined rather than computed from a negative denominator.

### Two ways to move g, never mixed

**The controlled dial.** Take the strongest retriever's top k. Replace each gold
passage, with probability p ∈ {{{design.levels}}}, by the next-ranked
non-gold passage from below rank k. The corpus, k, the prompt and the amount of
text stay fixed; only whether the gold is there changes. At p = 0 the context
is the retriever's own top k; at p = 1 it contains no gold.

The draws are nested: each gold passage gets one uniform u, seeded by the
question id, and is removed at every level with u < p. Whatever is gone at one
level is also gone at every higher one. That makes adjacent levels a paired
comparison.

On the dial, **one step up the retriever ladder is one dial level**: from p to
p − {{design.dial_step}}. This is the clean estimate of g\*.

**The natural sweep.** Real retrievers, through the unchanged budget bench:
tfidf-svd (a deliberately weak instrument), bm25,
multilingual-e5-small, multilingual-e5-base and bge-m3, at k = {{en_m3.hotpotqa.natural_ks}}.
This is the check that real retrievers behave the way the dial says they should.

### A property of the definition that travels with every g\*

**g\* depends on how much gold one retriever step recovers.** A retriever step
that recovers more gold buys more EM and moves the crossing. A dial level
removes a sizeable share of the gold; a real upgrade from bm25 to e5-small, where
bm25 already puts gold in nearly every context of this {{data.hotpotqa.docs}}-passage slice (§3), recovers almost
none. So the dial's g\* answers "if a
retriever step is worth one dial level of gold, where does it stop paying?" It
does not say that every real upgrade above g\* is wasted. For the same reason
the natural sweep is compared with the dial not on raw Δ_ret, but against the
gain the dial predicts for the same move in g. g\* for a step worth two dial
levels is reported as a sensitivity.

---

## 2. Statistics and measurement

* **Paired bootstrap over questions**, {{n_boot}} resamples, seeded. One resample
  draws one multiset of questions and re-scores every condition on it, so every
  difference is paired. Resamples are integer count vectors and EM is 0/1, so
  every resampled mean is an exact ratio and the analysis is byte-identical
  across runs (reproduced: {{repro.crossover_identical}}).
* **95% percentile intervals** on every Δ at every operating point.
  "Established" means the interval excludes zero. No p-values.
* **g\*** is the root of Δ_ret(g) − Δ_gen(g) under two fits over the dial levels:
  a monotone non-increasing (isotonic, pool-adjacent-violators) fit, linearly
  interpolated, and an ordinary least-squares line. The headline uses the
  isotonic root. The two fits **agree** when each root lies inside the other's
  interval. A resample whose curve never crosses zero is censored at the edge of
  the observed g range it fell off, not dropped. The count is reported.
* **Dial versus reality.** For each natural retriever step, the dial's EM curve,
  interpolated in g, predicts the gain of moving from the lower rung's g to the
  upper rung's. For each natural configuration, it predicts Δ_gen at that g. Two
  checks are reported. *Inside band*: the measured point lies inside the dial's
  95% band. *Consistent*: the paired interval on measured minus predicted
  includes zero. The first ignores the natural point's own noise, which the
  second accounts for.
* **Timing hygiene, switched off.** Every configuration was measured alone, with every other process
  sampled for CPU contention every {{design.sample_s}} s. Two attempts at the first configuration ran under the
  calibration-probe gate ({{design.probe_n}} fixed generations, median within {{design.probe_tolerance}} of a
  baseline) and were set aside: one was measured against a baseline taken on a warm machine in use, and one, against
  the baseline taken again ({{en_m3.hygiene.baseline_s}} s), was rejected for contention. The author then switched the
  gate and the rejections off: all {{en_m3.hygiene.batches_ungated}} accepted batches ran one at a time,
  {{en_m3.hygiene.first_sitting}} of them back to back and the three timed again in a later sitting
  ([BUDGETS.md](BUDGETS.md), §4), their contention recorded but not enforced, except that a batch more than a quarter contended would have been run again (batches
  that were: {{en_m3.hygiene.heavily_contended}}). Under the switched-off rule {{en_m3.hygiene.would_reject}} accepted
  batches would have been rejected. Their timings, and any price that uses them, carry that caveat; the answers do
  not depend on it. Three HotpotQA configurations first timed in one night ({{en_m3.hygiene.night_utc}}),
  picked out after the run by their latencies, were timed again: {{en_m3.hygiene.retimed_list}} (peak RSS
  as measured, model file memory-mapped (library default); estimate without the double-counted mapped pages (probe; see [LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings)): {{en_m3.est.hygiene.retimed_peaks}}).
  The tables use the second timings. Rows vary from batch to batch too, so every latency price carries the
  caveat ([LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings)).
  Details: [BUDGETS.md](BUDGETS.md).

---

## 3. HotpotQA

### HotpotQA: the controlled dial

{{en_m3.hotpotqa.n}} questions; dial on {{en_m3.hotpotqa.dial_retriever}}'s top {{en_m3.hotpotqa.k}}; the dial's top operating point is
g = {{en_m3.hotpotqa.g_top}}.

{{en_m3.hotpotqa.table_dial}}

**Crossover:** {{en_m3.hotpotqa.verdict}}.

* The retriever step is the larger point estimate at {{en_m3.hotpotqa.levels_ret_larger}} levels.
* Its advantage is established (the paired interval excludes zero) at {{en_m3.hotpotqa.levels_ret_established}}.
* The generator's advantage is established at {{en_m3.hotpotqa.levels_gen_established}}.

| fit | g* raw [95%] | g* on the all-gold axis | g* with a two-level retriever step |
|---|---|---|---|
| isotonic (headline) | {{en_m3.hotpotqa.gstar_raw}} | {{en_m3.hotpotqa.gstar_gall}} | {{en_m3.hotpotqa.gstar_step2}} |
| linear | {{en_m3.hotpotqa.gstar_raw_linear}} | {{en_m3.hotpotqa.gstar_gall_linear}} | {{en_m3.hotpotqa.gstar_step2_linear}} |
| fits agree | {{en_m3.hotpotqa.gstar_raw_agree}} | {{en_m3.hotpotqa.gstar_gall_agree}} | {{en_m3.hotpotqa.gstar_step2_agree}} |

### HotpotQA: the break-even gold gain

How much gold-in-context a retriever upgrade must add, starting from each operating point, to buy as much EM as the
3B → 8B upgrade does there. "Censored" counts resamples in which the headroom above the operating point was not
enough. The last two columns are each a bar for one upgrade that costs as much peak memory as the priced retriever step,
{{priced_step}} ({{en_m3.hotpotqa.cost_ret_gb}} GB), against the generator step's own cost ({{en_m3.hotpotqa.cost_gen_gb}} GB as measured, model file memory-mapped (library default);
{{en_m3.est.hotpotqa.cost_gen_gb}} GB, estimate without the double-counted mapped pages (probe; see [LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings))): the gold gain whose EM gain equals the generator step's times the ratio
of the two costs, so the table gives it twice. It is not a rate. For an upgrade that costs C GB, the bar is the gold gain
that buys the generator step's EM gain times C / (the generator step's GB); where the dial's EM curve is close to
straight, that is about the raw bar times C / (the generator step's GB). The cost C here is the memory of {{priced_step}}, a step that on HotpotQA moves
g any by only {{en_m3.hotpotqa.step.bm25->e5-small.k5.g_change}} ({{en_m3.hotpotqa.step.bm25->e5-small.k5.g}}): the columns
say what an upgrade of that cost would have to add, not what that step adds.

{{en_m3.est.hotpotqa.table_break_even}}

### HotpotQA: the price

**Two values for every price in GB.** The generator loads its model file memory-mapped, llama-cpp-python's default,
and on the CPU llama.cpp also copies most weights into its own buffers, so each row's peak RSS counts part of the
generator's weights twice. Every figure below that rests on peak RSS is given as measured, model file memory-mapped
(library default), and beside it as an estimate without the double-counted mapped pages (probe; see
[LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings)): each row's measured peak less the surplus a probe
measured for its generator. The retriever step's GB is the same either way, since its two rows load the same
generator; no answer and no timing depends on it.

Each move priced by the bench at k = {{en_m3.hotpotqa.k}}:
* **Retriever step ({{priced_step}}):** {{en_m3.hotpotqa.cost_ret_gb}} GB of peak RSS; {{en_m3.hotpotqa.cost_ret_ms_total}} ms of added latency, of
  which {{en_m3.hotpotqa.cost_ret_retrieval_ms}} ms is the change in retrieval p50 and {{en_m3.hotpotqa.cost_ret_ms}} ms the change in
  generation-stage p50, as measured. bm25 with Ministral 3 3B was timed again after the night it was first timed in,
  and the change stayed negative. Two explanations fit, and the records cannot separate them: e5-small's passages may
  make shorter prompts, so less prefill, or the change is variation between runs, since bm25 + 3B alone moved by
  {{en_m3.hygiene.retimed.hotpotqa_k5.bm25+ministral-3-3b.p50_shift_ms}} ms between its two timings. The prompts' token counts are not recorded.
* **Generator step:** {{en_m3.hotpotqa.cost_gen_gb}} GB as measured, model file memory-mapped (library default), {{en_m3.est.hotpotqa.cost_gen_gb}} GB estimate without the double-counted mapped pages (probe; see [LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings));
  {{en_m3.hotpotqa.cost_gen_ms_total}} ms of added latency.

The generator step measured in front of every rung. Peak RSS is one maximum over the whole process, index build
included, so the difference between two configurations' peaks moves with the retriever in front
of them; the dial prices the step at one rung, and this table is its sensitivity:

{{en_m3.est.hotpotqa.table_gen_step_by_rung}}

bm25's two rows are both from the later sitting ([BUDGETS.md](BUDGETS.md), §4), e5-base's 8B row is from the later
sitting and its 3B row from the first, and the other rungs' rows are both from the first.

The per-GB crossover with the generator step priced in front of each rung
instead. A large spread means the per-GB answer depends on a price the bench
measures only as a difference of two process-wide peaks:

{{en_m3.est.hotpotqa.table_per_gb_by_gen_rung}}

Rungs at which the per-GB fit crosses: as measured, model file memory-mapped (library default), {{en_m3.est.hotpotqa.rungs_crossing_list_measured}}; estimate without the double-counted mapped pages, {{en_m3.est.hotpotqa.rungs_crossing_list}}.

| crossover | isotonic [95%] | linear [95%] | agree |
|---|---|---|---|
| per GB, as measured, model file memory-mapped (library default) | {{en_m3.hotpotqa.gstar_gb}} | {{en_m3.hotpotqa.gstar_gb_linear}} | {{en_m3.hotpotqa.gstar_gb_agree}} |
| per GB, estimate without the double-counted mapped pages (probe; see LIMITATIONS) | {{en_m3.est.hotpotqa.gstar_gb}} | {{en_m3.est.hotpotqa.gstar_gb_linear}} | {{en_m3.est.hotpotqa.gstar_gb_agree}} |
| per ms of added latency (retrieval + generation) | {{en_m3.hotpotqa.gstar_ms_total}} | {{en_m3.hotpotqa.gstar_ms_total_linear}} | {{en_m3.hotpotqa.gstar_ms_total_agree}} |
| per ms of generation-stage latency only | {{en_m3.hotpotqa.gstar_ms}} | {{en_m3.hotpotqa.gstar_ms_linear}} | {{en_m3.hotpotqa.gstar_ms_agree}} |
| retriever per ms of retrieval latency only | {{en_m3.hotpotqa.gstar_ms_component}} | {{en_m3.hotpotqa.gstar_ms_component_linear}} | {{en_m3.hotpotqa.gstar_ms_component_agree}} |

On the per-GB rows, the retriever step is one dial level of gold priced at the {{en_m3.hotpotqa.cost_ret_gb}} GB of
{{priced_step}}, a step that moves g any on HotpotQA by only {{en_m3.hotpotqa.step.bm25->e5-small.k5.g_change}}, so the
per-GB comparison prices one move and gains from another.

Each move on each side of the crossing, per unit of cost:

{{en_m3.est.hotpotqa.table_price}}

Every other priced retriever step, as a sensitivity:

{{en_m3.est.hotpotqa.table_price_sensitivity}}

### HotpotQA: the dial against real retrievers

Natural sweep measured at k = {{en_m3.hotpotqa.natural_ks}}. At the dial's k, {{en_m3.hotpotqa.natural_k5_inside}} checks fall inside the
dial's band and {{en_m3.hotpotqa.natural_k5_consistent}} are consistent with it (paired difference interval includes zero).
The dial retriever's own generator check counts: its prompts are the dial's p = 0 prompts, but the dial answered on
the GX10 and the sweep on the laptop, so it measures the two machines, {{en_m3.hotpotqa.natural_k5_dial_own}}.
Inconsistent: {{en_m3.hotpotqa.natural_k5_inconsistent}}. Part of the other disagreements can be the machines; a
retriever change that leaves g where it was and still moves EM shows that EM depends on more than whether a gold
passage is present.

At every real step: the retriever upgrade against the generator upgrade made at the lower rung, paired over
questions.

{{en_m3.hotpotqa.table_steps_k5}}

{{en_m3.hotpotqa.table_natural_k5}}

Retriever steps: measured gain against the gain the dial predicts for the same move in g.

{{en_m3.hotpotqa.table_check_ret_k5}}

Generator step at each real retriever against the dial's Δ_gen at the same g:

{{en_m3.hotpotqa.table_check_gen_k5}}

**Budget rows used for the prices** (peak RSS is the whole process; every row's
operating system is in [BUDGETS.md](BUDGETS.md)). bm25 + 3B, bm25 + 8B and e5-base + 8B were first timed in one
night ({{en_m3.hygiene.night_utc}}) and timed again from {{en_m3.hygiene.retimed_utc}}; their rows below are the
second timings, and the priced retriever step (bm25 → e5-small, with 3B) and the generator step where it is priced
(in front of e5-base) use them, in peak memory and in latency. Rows vary from batch to batch, so every latency price carries
that caveat ([LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings)):

{{en_m3.est.hotpotqa.table_budget}}


## 4. Natural Questions

### Natural Questions: the controlled dial

{{en_m3.nq.n}} questions; dial on {{en_m3.nq.dial_retriever}}'s top {{en_m3.nq.k}}; the dial's top operating point is
g = {{en_m3.nq.g_top}}.

**The dial leaks on NQ.** The last column of the table below is the share of
contexts in which a *non-gold* passage still contains an nq-open answer:
{{en_m3.nq.leak_no_gold}} even with every gold passage removed (HotpotQA:
{{en_m3.hotpotqa.leak_no_gold}}). Removing gold does not remove the answer, so the
dial understates what gold is worth on NQ, and the no-gold level, the only level
where NQ's difference between the two moves is established, is the leakiest. Read
NQ's g\* with that in view.

Gold passages are mapped, not annotated: a passage is gold if
BEIR judges it relevant **and** it contains an nq-open answer. Of
{{data.nq.beir_queries}} BEIR test questions with judgements, {{data.nq.joined}}
({{data.nq.joined_pct}}) join to nq-open, and {{data.nq.mapped}} of those
({{data.nq.mapped_pct}}) have a mapped gold passage; {{en_m3.nq.n}} are sampled.

{{en_m3.nq.table_dial}}

**Crossover:** {{en_m3.nq.verdict}}.

* The retriever step is the larger point estimate at {{en_m3.nq.levels_ret_larger}} levels.
* Its advantage is established (the paired interval excludes zero) at {{en_m3.nq.levels_ret_established}}.
* The generator's advantage is established at {{en_m3.nq.levels_gen_established}}.

| fit | g* raw [95%] | g* on the all-gold axis | g* with a two-level retriever step |
|---|---|---|---|
| isotonic (headline) | {{en_m3.nq.gstar_raw}} | {{en_m3.nq.gstar_gall}} | {{en_m3.nq.gstar_step2}} |
| linear | {{en_m3.nq.gstar_raw_linear}} | {{en_m3.nq.gstar_gall_linear}} | {{en_m3.nq.gstar_step2_linear}} |
| fits agree | {{en_m3.nq.gstar_raw_agree}} | {{en_m3.nq.gstar_gall_agree}} | {{en_m3.nq.gstar_step2_agree}} |

### Natural Questions: the break-even gold gain

As for HotpotQA; the last columns are the bar for one upgrade of {{en_m3.nq.cost_ret_gb}} GB, the priced step's memory,
against a generator step of {{en_m3.nq.cost_gen_gb}} GB as measured, model file memory-mapped (library default), and of {{en_m3.est.nq.cost_gen_gb}} GB, estimate without the double-counted mapped pages (probe; see [LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings)).

{{en_m3.est.nq.table_break_even}}

### Natural Questions: the price

Each move priced by the bench at k = {{en_m3.nq.k}}:
* **Retriever step ({{priced_step}}):** {{en_m3.nq.cost_ret_gb}} GB of peak RSS; {{en_m3.nq.cost_ret_ms_total}} ms of added latency, of
  which {{en_m3.nq.cost_ret_retrieval_ms}} ms is the change in retrieval p50 and {{en_m3.nq.cost_ret_ms}} ms the change in generation-stage p50 (other passages
  change the prompt the generator reads).
* **Generator step:** {{en_m3.nq.cost_gen_gb}} GB as measured, model file memory-mapped (library default), {{en_m3.est.nq.cost_gen_gb}} GB estimate without the double-counted mapped pages (probe; see [LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings));
  {{en_m3.nq.cost_gen_ms_total}} ms of added latency.

The generator step measured in front of every rung. Peak RSS is one maximum over the whole process, index build
included, so the difference between two configurations' peaks moves with the retriever in front
of them; the dial prices the step at one rung, and this table is its sensitivity:

{{en_m3.est.nq.table_gen_step_by_rung}}

The per-GB crossover with the generator step priced in front of each rung
instead. A large spread means the per-GB answer depends on a price the bench
measures only as a difference of two process-wide peaks:

{{en_m3.est.nq.table_per_gb_by_gen_rung}}

Rungs at which the per-GB fit crosses: as measured, model file memory-mapped (library default), {{en_m3.est.nq.rungs_crossing_list_measured}}; estimate without the double-counted mapped pages, {{en_m3.est.nq.rungs_crossing_list}}.

| crossover | isotonic [95%] | linear [95%] | agree |
|---|---|---|---|
| per GB, as measured, model file memory-mapped (library default) | {{en_m3.nq.gstar_gb}} | {{en_m3.nq.gstar_gb_linear}} | {{en_m3.nq.gstar_gb_agree}} |
| per GB, estimate without the double-counted mapped pages (probe; see LIMITATIONS) | {{en_m3.est.nq.gstar_gb}} | {{en_m3.est.nq.gstar_gb_linear}} | {{en_m3.est.nq.gstar_gb_agree}} |
| per ms of added latency (retrieval + generation) | {{en_m3.nq.gstar_ms_total}} | {{en_m3.nq.gstar_ms_total_linear}} | {{en_m3.nq.gstar_ms_total_agree}} |
| per ms of generation-stage latency only | {{en_m3.nq.gstar_ms}} | {{en_m3.nq.gstar_ms_linear}} | {{en_m3.nq.gstar_ms_agree}} |
| retriever per ms of retrieval latency only | {{en_m3.nq.gstar_ms_component}} | {{en_m3.nq.gstar_ms_component_linear}} | {{en_m3.nq.gstar_ms_component_agree}} |

{{en_m3.est.nq.table_price}}

### Natural Questions: the dial against real retrievers

At k = 5, {{en_m3.nq.natural_k5_inside}} checks fall inside the dial's band and
{{en_m3.nq.natural_k5_consistent}} are consistent with it (the dial retriever's own check counted, as for HotpotQA:
the two machines differ by {{en_m3.nq.natural_k5_dial_own}}). Inconsistent: {{en_m3.nq.natural_k5_inconsistent}}.

{{en_m3.nq.table_steps_k5}}

{{en_m3.nq.table_natural_k5}}

{{en_m3.nq.table_check_ret_k5}}

{{en_m3.nq.table_check_gen_k5}}

{{en_m3.est.nq.table_budget}}


## Four families' English dials

The dial above is Ministral 3's 🇫🇷, the family whose English sweeps were timed on this laptop. The same controlled
dial ran with three more families of the Czech comparison, each family's smaller model in 3B's place and its larger in
8B's: Gemma 4 E2B and E4B 🇺🇸, chosen first, and, after Gemma 4's dial was read, Gemma 3n E2B and E4B 🇺🇸 and
Ministral 3, all three answered on the GX10's GPU; Qwen2.5 1.5B and 3B 🇨🇳 answered it on the laptop's CPU. Qwen3.5 🇨🇳
did not answer the English dial
([LIMITATIONS.md](LIMITATIONS.md#the-czech-and-english-families)). Same questions, rankings (e5-base, k = 5), levels,
prompt, token cap, seed and resamples. The README's dial figure shows Ministral 3's dial, the author's choice after
these dials were read, and Ministral 3's English sweeps were timed after the same reading. The documents read no
budget or price for the other families.

**Gemma 3n 🇺🇸**, E2B to E4B: {{hotpotqa.dial_family.gemma_3n.gen_step_range}} on HotpotQA, established above zero at
{{hotpotqa.dial_family.gemma_3n.gen_step_established_above}} levels, and {{nq.dial_family.gemma_3n.gen_step_range}} on NQ, at
{{nq.dial_family.gemma_3n.gen_step_established_above}}.

**Gemma 4 🇺🇸**, E2B to E4B: {{hotpotqa.dial_family.gemma_4.gen_step_range}} on HotpotQA, where E4B scores below E2B, established at
{{hotpotqa.dial_family.gemma_4.gen_step_established_below}} levels, and {{nq.dial_family.gemma_4.gen_step_range}} on NQ. In English, at this
prompt and cap, Gemma 4's larger model buys nothing on the dial.

**Ministral 3 🇫🇷**, 3B to 8B: the generator step is {{hotpotqa.dial_family.ministral_3.gen_step_range}} on HotpotQA, above zero and
established at {{hotpotqa.dial_family.ministral_3.gen_step_established_above}} levels, and {{nq.dial_family.ministral_3.gen_step_range}} on
NQ, established at {{nq.dial_family.ministral_3.gen_step_established_above}}. Its full tables are in §3 and §4.

Qwen2.5 🇨🇳, 1.5B to 3B: {{hotpotqa.dial_family.qwen2_5.gen_step_range}} on HotpotQA, established above zero at
{{hotpotqa.dial_family.qwen2_5.gen_step_established_above}} levels, and {{nq.dial_family.qwen2_5.gen_step_range}} on NQ, at
{{nq.dial_family.qwen2_5.gen_step_established_above}}.

HotpotQA, Gemma 3n 🇺🇸:

{{hotpotqa.dial_family.gemma_3n.table}}

NQ, Gemma 3n 🇺🇸:

{{nq.dial_family.gemma_3n.table}}

HotpotQA, Gemma 4 🇺🇸:

{{hotpotqa.dial_family.gemma_4.table}}

NQ, Gemma 4 🇺🇸:

{{nq.dial_family.gemma_4.table}}

HotpotQA, Qwen2.5 🇨🇳:

{{hotpotqa.dial_family.qwen2_5.table}}

NQ, Qwen2.5 🇨🇳:

{{nq.dial_family.qwen2_5.table}}

## 5. Do the two datasets agree?

{{en_m3.table_agreement}}

{{en_m3.agreement_verdict}}.

Overlap of the HotpotQA and NQ g\* intervals (the specification's test), raw: {{en_m3.agree_raw}}; per GB, as measured, model file memory-mapped (library default):
{{en_m3.agree_per_gb}}; per GB, estimate without the double-counted mapped pages (probe; see [LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings)): {{en_m3.est.agree_per_gb}}.
Both raw intervals are wide, so their overlap says little about where the two datasets cross.

## 6. What this does not show

* **g\* depends on step size.** Every g\* here is conditional on a retriever step being one dial level. The break-even
  gold gain removes that dependence; the crossing point does not.
* **Two generators, one family.** Ministral 3 at 3B and 8B, Q4_K_M, greedy. Nothing here says where a smaller or a
  larger model, or another family, would sit; the other families' dials are above, untimed.
* **Exact match only.** EM is strict. F1 moves with it in the tables but is not bootstrapped.
* **HotpotQA is multi-hop.** "Any gold" is a floor on the evidence. EM tracks the all-gold rate closely, which is why
  g\* on the all-gold axis is reported beside it.
* **The dial is a model of retrieval quality, not a retriever.** A real upgrade changes *which* non-gold passages
  appear too. The natural-sweep checks are the evidence about how much that matters, and they are only as good as
  the range of g the real retrievers reach.
* **The dial and the natural sweeps come from two machines.** The dial's answers are the GX10's, the natural sweeps'
  the laptop's. On the same prompts the two machines' generator step differs by {{en_m3.hotpotqa.natural_k5_dial_own}} on
  HotpotQA and {{en_m3.nq.natural_k5_dial_own}} on NQ, so part of any disagreement between the dial and the sweeps can be
  the machines ([LIMITATIONS.md](LIMITATIONS.md#the-czech-and-english-families)). Comparing Qwen2.5's dial, answered on
  the laptop, with the others' carries the same machine effect.
* **NQ's gold is mapped, and its dial leaks.** A judged passage that does not contain the answer string is not gold,
  and an unjudged passage that does is not gold either; {{en_m3.nq.leak_no_gold}} of NQ's no-gold contexts still contain
  an answer in a non-gold passage.
* **EM depends on more than g.** The natural sweeps disagree with the dial at some points: on HotpotQA
  {{en_m3.hotpotqa.natural_k5_inconsistent}}; on NQ {{en_m3.nq.natural_k5_inconsistent}}. The break-even gold gain assumes EM
  follows g; it is a guide, not a law.
* **The per-GB result is priced at one rung**, and the generator step's memory price is a difference of two
  process-wide peaks that moves with the rung; the per-rung tables in §3 and §4 price it in front of each.
* **Where the difference is established.** HotpotQA: {{en_m3.hotpotqa.established_text}}. NQ:
  {{en_m3.nq.established_text}}.
* **Samples.** {{en_m3.hotpotqa.n}} HotpotQA and {{en_m3.nq.n}} NQ questions; intervals widen accordingly.
* **One machine for the timings.** Every English timing is from one fanless laptop on one macOS version
  ({{en_m3.budget.macos}}), without the full measurement hygiene; three HotpotQA configurations first timed in one night
  ({{en_m3.hygiene.night_utc}}) were timed again, and the tables use the second timings. Every latency price carries
  that caveat
  ([BUDGETS.md](BUDGETS.md), [LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings)).

---

## Appendix: Czech (SQAD 3.2 and the SQAD 3.0 copy)

This section is **separate from the English result above**, and it is in no headline, verdict or
agreement statement. The prompt asks Czech questions for ano or ne, the form of SQAD's gold ([The
prompt](#the-prompt), below). Every Czech generation ran under an analysis fixed before any of it: sets sized
for power, and every generation on one second machine ([LIMITATIONS.md](LIMITATIONS.md)). Qwen2.5 and
Gemma 4 answered first; Ministral 3, Qwen3.5 and Gemma 3n were added later, each run alone on the same
machine with everything else unchanged ([LIMITATIONS.md](LIMITATIONS.md#the-czech-and-english-families)).
Every number here recomputes from `answers/` and `results/czech/records/` with
`scripts/rescore_czech.py`, which needs no model and no data
([REPRODUCE.md](REPRODUCE.md#10-checking-every-czech-number)). SQAD 3.0 is under LGPL-3.0 on LINDAT
(http://hdl.handle.net/11234/1-3069) and SQAD 3.2 under CC BY-SA 3.0
(http://hdl.handle.net/11234/1-5019; [THIRD_PARTY.md](../THIRD_PARTY.md)).

**How to read it.** EM with yes and no mapped to ano and ne is the primary Czech metric, because Qwen2.5
still answers many yes/no questions in English, and raw EM would then score the language of the answer
more than the answer. Raw EM and compliance stand beside it: compliance is, of the yes/no questions a model did not
decline, the share it answered with exactly ano or ne ([METRICS.md](METRICS.md)). Read differences within a set, not levels across sets. Every
table lists SQAD 3.2 debiased, SQAD 3.2 full, the released 200 and the SQAD 3.0 copy debiased, in that
order, and none of them is the headline. E2B and E4B are Gemma 4; Gemma 3n E2B and E4B are Gemma 3n, Gemma 4's
previous generation; Ministral 3B and 8B are Ministral 3; 1.5B and 3B are Qwen2.5; Qwen3.5 2B and 4B are Qwen3.5.
The generator
step is one model size up with e5-small's top 3, and the retriever step is e5-small over BM25 with the
smaller model.

### What holds on all four sets

**Qwen2.5.** In raw EM the generator step is established on all four sets,
{{cz.range.generator_step.em}}, and so is its lead over the retriever step,
{{cz.range.generator_step_minus_retriever_step.em}}. With yes and no mapped, the generator step
({{cz.range3.generator_step.mapped}}) and the retriever step ({{cz.range3.retriever_step.mapped}}) are each
established on the three larger sets, and the generator step's lead over the retriever step on the two
SQAD 3.2 sets only. On the released 200, neither step is established with yes and no mapped.

**Gemma 4.** The two metrics nearly agree, because Gemma 4 answers yes/no questions in Czech. The
generator step ({{cz.g4.range3.generator_step.mapped}}) and the retriever step
({{cz.g4.range3.retriever_step.mapped}}) are each established on the three larger sets, and the generator
step's lead over the retriever step on the two SQAD 3.2 sets only. On the copy debiased the retriever step
is the larger point estimate. The released 200 establishes no step for Gemma 4.

**Gemma 3n.** It gave {{cz.g3n.yes_no_in_english}} yes/no answers in English, so its two metrics are
{{cz.g3n.metrics_agree}}. The generator step, E4B over E2B, is established on all four sets,
{{cz.g3n.range.generator_step.mapped}}. The retriever step ({{cz.g3n.range3.retriever_step.mapped}} on the three
larger sets) is established on {{cz.holds.g3n.retriever_step.mapped_on}} only, and the generator step's lead over it
on {{cz.holds.g3n.generator_step_minus_retriever_step.mapped_on}}.

**Ministral 3.** It gave {{cz.m3.yes_no_in_english}} yes/no answers in English, so its two metrics are
{{cz.m3.metrics_agree}}. The generator step, 8B over 3B, is established on all four sets,
{{cz.m3.range.generator_step.mapped}}. The retriever step ({{cz.m3.range3.retriever_step.mapped}} on the three
larger sets) is established on {{cz.holds.m3.retriever_step.mapped_on}} only, and the generator step's lead
over it on {{cz.holds.m3.generator_step_minus_retriever_step.mapped_on}} only.

**Qwen3.5.** It gave {{cz.q35.yes_no_in_english}} yes/no answers in English, so its two metrics are
{{cz.q35.metrics_agree}}. The generator step, 4B over 2B, is established on all four sets,
{{cz.q35.range.generator_step.mapped}}. The retriever step ({{cz.q35.range3.retriever_step.mapped}} on the three
larger sets) is established on {{cz.holds.q35.retriever_step.mapped_on}}, and the generator step's lead over it
on {{cz.holds.q35.generator_step_minus_retriever_step.mapped_on}} only.

**The five families.** Gemma 4 E2B, Gemma 3n E2B, Ministral 3 3B and Qwen3.5 2B each score above Qwen2.5 1.5B,
and Gemma 4 E4B, Gemma 3n E4B, Ministral 3 8B and Qwen3.5 4B above 3B, on every set and in both metrics, each
difference established: with yes and no mapped, {{cz.family.smaller.mapped_range}} and
{{cz.family.larger.mapped_range}} for Gemma 4, {{cz.g3n.family.smaller.mapped_range}} and
{{cz.g3n.family.larger.mapped_range}} for Gemma 3n, {{cz.m3.family.smaller.mapped_range}} and
{{cz.m3.family.larger.mapped_range}} for Ministral 3, and {{cz.q35.family.smaller.mapped_range}} and
{{cz.q35.family.larger.mapped_range}} for Qwen3.5. Gemma 4 minus Gemma 3n, one maker's two generations, is
{{cz.g4_minus_g3n.smaller.mapped_sign}} for the smaller models, established on
{{cz.holds.g4_minus_g3n.smaller.mapped_on}}, and {{cz.g4_minus_g3n.larger.mapped_sign}} for the larger,
established on {{cz.holds.g4_minus_g3n.larger.mapped_on}}. Ministral 3 minus
Gemma 4 is {{cz.m3_minus_g4.smaller.mapped_sign}} for the smaller models and
{{cz.m3_minus_g4.larger.mapped_sign}} for the larger, and established on {{cz.holds.m3_minus_g4.smaller.mapped_on}}
(3B against E2B) and on {{cz.holds.m3_minus_g4.larger.mapped_on}} (8B against E4B): it does not hold on all four.
Qwen3.5 minus Gemma 4 is {{cz.q35_minus_g4.smaller.mapped_sign}} for the smaller models, established on
{{cz.holds.q35_minus_g4.smaller.mapped_on}}, and {{cz.q35_minus_g4.larger.mapped_sign}} for the larger,
established on {{cz.holds.q35_minus_g4.larger.mapped_on}}. Qwen3.5 minus Ministral 3 is
{{cz.q35_minus_m3.smaller.mapped_sign}} for the smaller models, established on
{{cz.holds.q35_minus_m3.smaller.mapped_on}}, and {{cz.q35_minus_m3.larger.mapped_sign}} for the larger
({{cz.q35_minus_m3.larger.mapped_range}}), {{cz.holds.q35_minus_m3.larger.mapped}}.

{{cz.table_holds}}

### The sets and the answers

{{cz.table_sets_g4}}

{{cz.table_answers}}

**Compliance.** Gemma 4's compliance is {{cz.g4.compliance_range}}, Gemma 3n's {{cz.g3n.compliance_range}},
Ministral 3's {{cz.m3.compliance_range}}, Qwen2.5's {{cz.compliance_range}} and Qwen3.5's
{{cz.q35.compliance_range}}, across models and sets.
Qwen2.5 3B answers in Czech more often than 1.5B, by {{cz.range.compliance_step}},
established on the three larger sets, so part of Qwen2.5's generator step in raw EM is the language of the
answer. Between Gemma 4's two models compliance moves by {{cz.g4.range.compliance_step}}, between Gemma 3n's by
{{cz.g3n.range.compliance_step}} (E2B answers some yes/no questions with something other than ano or ne), between
Ministral 3's by {{cz.m3.range.compliance_step}}, and between Qwen3.5's by {{cz.q35.range.compliance_step}}. A declined answer
is the other common miss on a yes/no question, and it counts in neither the numerator nor the denominator of
compliance.

{{cz.table_compliance_generators}}

{{cz.table_compliance_classes}}

### The steps, and what each set could detect

For Gemma 4 and Qwen2.5, both steps are established on the three larger sets, in both metrics. For Gemma 3n
the generator step is established on all four sets and the retriever step on
{{cz.holds.g3n.retriever_step.mapped_on}} only; for Ministral 3 the generator step on all four sets and the
retriever step on {{cz.holds.m3.retriever_step.mapped_on}} only; for Qwen3.5 the generator step on all four sets
and the retriever step on {{cz.holds.q35.retriever_step.mapped_on}}. On the released 200 the only steps
established are the generator steps of Gemma 3n, Ministral 3 and Qwen3.5, and Qwen2.5's in raw EM. Where
the retriever step is not established, the last column of the third table gives the sizes its interval rules
out.

{{cz.table_steps_both}}

The sizes were planned for Qwen2.5, from a first round of its answers, made before this run:

{{cz.table_power}}

What each set can detect, read from each family's own answers in this run:

{{cz.table_power_observed}}

On the questions outside that first round, which had no part in forming the hypotheses, every step keeps its sign.
For Gemma 4 and Qwen2.5 all but one are established; for Gemma 3n the generator step is established on each
set and the retriever step on the two SQAD 3.2 sets; for Ministral 3 the generator step on each set and the
retriever step on SQAD 3.2 debiased only; for Qwen3.5 both steps on each set:

{{cz.table_added}}

### The five families, model for model

Gemma 4 joined Qwen2.5 by a benchmark on the dev set, whose rule was fixed before it ran. Ministral 3 joined
because the author asked for Mistral AI's models beside the other two, and Qwen3.5 because the author asked for
Qwen's newer generation beside them; no rule chose either, and the analysis of each was fixed before it answered
any question outside the dev set. Qwen3.5 had answered the dev set in the benchmark, as a comparison only, and its
analysis was fixed knowing that result. Gemma 3n, Gemma 4's previous generation, was added at the author's request
after the other families' results were read and after an exploratory screen of models on the dev set, run outside
the release; no rule chose it, and its analysis was fixed before it answered any test question
([LIMITATIONS.md](LIMITATIONS.md#the-czech-and-english-families)). Gemma 4, Gemma 3n, Ministral 3 and Qwen3.5 each score above Qwen2.5, established on every set, at
both sizes and in both metrics, and on the cross-version questions with either version's passages
({{cz.cross.family.mapped_range}} for Gemma 4, {{cz.cross.g3n.family.mapped_range}} for Gemma 3n,
{{cz.cross.m3.family.mapped_range}} for Ministral 3 and {{cz.cross.q35.family.mapped_range}} for Qwen3.5, with yes
and no mapped). Gemma 4 minus Gemma 3n is established on {{cz.holds.g4_minus_g3n.smaller.mapped_on}} at the
smaller place and on {{cz.holds.g4_minus_g3n.larger.mapped_on}} at the larger, Gemma 4 higher on each, and on the
cross-version questions it is {{cz.cross.g4_minus_g3n.mapped_range}}. Ministral 3 minus Gemma 4 is established on two of the four sets at each size, Gemma 4 higher
on each, and on the cross-version questions it is {{cz.cross.m3_minus_g4.mapped_range}}. Qwen3.5 2B is below
Gemma 4 E2B on {{cz.holds.q35_minus_g4.smaller.mapped_on}}, and its difference from Ministral 3 3B is established
on {{cz.holds.q35_minus_m3.smaller.mapped_on}}. Qwen3.5 4B's difference from Gemma 4 E4B is established on
{{cz.holds.q35_minus_g4.larger.mapped_on}}, and it is above Ministral 3 8B on
{{cz.holds.q35_minus_m3.larger.mapped_on}}.
On the cross-version questions Qwen3.5 minus Gemma 4 is {{cz.cross.q35_minus_g4.mapped_range}} and Qwen3.5 minus
Ministral 3 {{cz.cross.q35_minus_m3.mapped_range}}. The pairs are by place in the family, not by memory: at
Q4_K_M, the smaller models' files are {{cz.bench.qwen2.5-1.5b.gb}} GB (1.5B), {{cz.bench.qwen3.5-2b.gb}} GB
(Qwen3.5 2B), {{cz.bench.gemma-4-e2b.gb}} GB (E2B) and {{cz.bench.ministral-3-3b-unsloth.gb}} GB (Ministral 3B),
and the larger models' {{cz.bench.qwen2.5-3b.gb}}, {{cz.bench.qwen3.5-4b.gb}}, {{cz.bench.gemma-4-e4b.gb}} and
{{cz.bench.ministral-3-8b-unsloth.gb}} GB.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../results/figures/families_dark.svg">
  <img alt="Dot plot in two panels, each family's smaller model and each family's larger model, with the four Czech SQAD sets along the bottom, the released 200 of the SQAD 3.0 copy first, and exact match with yes and no mapped going up, each mark with its 95% interval; in each panel the models stand best first by their score on the released 200, and each name carries its maker's flag. On every set and at both sizes Gemma 4, Gemma 3n, Ministral 3 and Qwen3.5 score higher than Qwen2.5. Among the smaller models Gemma 4 E2B is highest on every set; among the larger, Gemma 4 E4B and Qwen3.5 4B are level, and both are above Ministral 3 8B and Gemma 3n E4B on the two SQAD 3.2 sets." src="../results/figures/families_light.svg">
</picture>

*Qwen2.5-3B-Instruct is under the Qwen Research License ([THIRD_PARTY.md](../THIRD_PARTY.md)).*

{{cz.table_families_levels_mapped}}

{{cz.table_families_differences_mapped}}

{{cz.table_family}}

{{cz.table_cross_family}}

**The benchmark.** {{cz.bench.questions}} dev questions, {{cz.bench.yes_no}} of them yes/no, prompt V1, each
model once. A family's score is EM with yes and no mapped, averaged over its two models. A Western family
joined if its difference from Qwen2.5 was established above zero; {{cz.bench.joins}} did. Qwen3.5 is the
same age as the Western candidates and is a comparison only. Mistral AI's own Ministral 3 files did not load
in the release's llama.cpp build; Unsloth's build of the same weights did, and is reported beside the
benchmark, after its result was read, so it decides nothing. Those Unsloth files, at the same revisions, are
the Ministral 3 that answers every set.

{{cz.table_model_bench_families}}

{{cz.table_model_bench}}

{{cz.table_model_bench_ministral}}

Both seconds-per-answer columns are the GX10's: its GPU over every question, and its CPU, with the CUDA build, over
the first questions only. They compare speeds within the benchmark; no budget, latency or price comes from them, and
the laptop's CPU is far slower (the second machine, below).

### The dial and the rest of the protocol

Gemma 3n's, Gemma 4's and Ministral 3's dials lean towards the retriever: on every set the retriever step is the
larger at most levels. Qwen2.5's leans the other way; with yes and no mapped it crosses on every set, and in raw EM on
none. Qwen3.5's leans towards the generator on the two SQAD 3.2 sets and towards the retriever on the copy's two.
e5-small answers better than retromae-cs on every set, in every family.

{{cz.table_dial}}

{{cz.table_protocol}}

**The dial on the released {{czech.n}}**, Gemma 4, in raw EM. Dial on
{{czech.dial_retriever}}'s top {{czech.k}}; the dial's top operating point is g = {{czech.g_top}}. With
every gold passage removed (p = 1, the first row), EM stays well above zero: {{cz.released.g4.leak_no_gold}} of those
contexts still carry the answer in a non-gold passage, so the dial moves gold more than it moves the answer.
The generator step, E2B to E4B, is {{cz.released.g4.dial_gen_step_range}}, established above zero at
{{cz.released.g4.dial_gen_step_established_above}} levels.

{{cz.released.g4.table_dial_levels}}

The same dial with Qwen2.5 1.5B and 3B, with its fits, in raw EM. With every gold passage removed, its EM is
{{czech.em_no_gold_base}} (1.5B) and {{czech.em_no_gold_big}} (3B).

{{czech.table_dial}}

**Crossover:** {{czech.verdict}}.

| fit | g* raw [95%] | g* on the all-gold axis | g* with a two-level retriever step |
|---|---|---|---|
| isotonic | {{czech.gstar_raw}} | {{czech.gstar_gall}} | {{czech.gstar_step2}} |
| linear | {{czech.gstar_raw_linear}} | {{czech.gstar_gall_linear}} | {{czech.gstar_step2_linear}} |
| fits agree | {{czech.gstar_raw_agree}} | {{czech.gstar_gall_agree}} | {{czech.gstar_step2_agree}} |

{{czech.table_break_even}}

### The prompt

**The yes-or-no line against the ano-or-ne line, read on one machine.** Czech questions are asked with the English
template whose yes/no line asks for ano or ne; its other instructions, labels and decline word stay English. It
replaced the same template asking for yes or no, under which every yes/no question scored 0 against SQAD's gold. Both
were answered on the laptop, on the SQAD 3.0 copy's released {{cz.defect_mac.questions}}
({{cz.defect_mac.yes_no}} of them yes/no).

{{cz.table_prompt_same_machine}}

**The attempt to fix it, on the dev set only.** {{cz.dev.questions}} questions, {{cz.dev.yes_no}} yes/no,
in none of the reported sets. V1 is the prompt in use; V2 and V3 were written in Czech, V3 with a worked example,
and V4 with two. V3 passed the first rule only through its fallback and cost span answers; V4 answered in Czech and
lost span answers too. So the prompt stays: {{cz.dev.chosen}}.

{{cz.table_dev}}

### The second machine

Every answer behind this appendix's sets was generated on the GX10's GPU; the comparison of the two yes/no lines
(above) and the dev set's prompt selection were answered on the laptop. Before the move, the dev set's eight
configurations ran there from an empty cache, under three conditions fixed beforehand. The
repeat gave the same string on {{cz.pilot.repeat_same}} answers, and the GX10 took {{cz.pilot.share.1_5b}}
of the laptop's seconds per answer. The third condition, that its rankings match the shipped ones, failed:
they differ in the top 10 on {{cz.pilot.top10_differs}} rankings, each time at rank 10 between passages
whose scores agree to six decimals. The top 3 is the same on every question, so no prompt changes, and the
author moved. Against the laptop, {{cz.pilot.same_range}} of the dev answers are the same string.

{{cz.table_pilot_agreement}}

{{cz.table_pilot_rankings}}

### Labelled secondary metrics

These show how much of each score is the scoring: EM, EM with yes and no mapped (the primary metric), a
match on the first five characters to tolerate declension, and token F1 after lowercasing and stripping
diacritics. The five-character match is a proxy: it misses an inflection that changes a word within its first five
characters, as short words do (Prahy against Praha), and credits two different words that share them
(Československá against Československo). EM itself can credit a wrong answer in two ways ([METRICS.md](METRICS.md#normalisation)).

{{cz.table_secondary}}

### Retrieval at large samples

Retrieval only: no answer is generated, so the prompt plays no part in these.

{{cz.table_large}}

### Ten retrievers on SQAD

Six retrievers joined BM25 (with and without diacritics), e5-small and retromae-cs on both test splits, ranked on the
Mac under an analysis fixed before any of their results was read: e5-base, IBM's
granite-embedding-107m-multilingual and granite-embedding-278m-multilingual, Google's EmbeddingGemma 300m, Seznam's
simcse-cs-en, and BM25 over Czech lemmas from simplemma. BM25 over lemmas lowercases each word and keeps its
diacritics before simplemma lemmatises it, so a capitalised name can lose its sense ("Prahy" becomes "práh", threshold,
not "Praha"); its difference from BM25 without diacritics changes two things, lemmas and diacritics, and its
difference from plain BM25, which keeps diacritics too, changes one. That difference from plain BM25 was added at the author's
request after the rankings had been read, to change one factor at a time; it is outside the analysis fixed before them,
and the rule for an end-to-end run does not read it. Every dense encoder runs at e5-small's sequence length, with
the pooling and prompts its repository declares. e5-small was encoded again on the Mac first, and its top 10 equals
the shipped rankings on every question of both splits. Paired over questions, 95% intervals, with the resamples of
the large-sample rows above, so e5-small's and BM25's rows here are theirs.

SQAD 3.2, all {{czr.sqad32_test.n}} test questions:

{{czr.table_sqad32_test}}

{{czr.table_sqad32_test_differences}}

The SQAD 3.0 copy, all {{czr.copy_test.n}} test questions:

{{czr.table_copy_test}}

{{czr.table_copy_test_differences}}

The analysis fixed before these rankings were read sends a retriever end to end only if its gain in the gold in the top 3 over e5-small on SQAD 3.2
is established and at least e5-small's own lead over BM25 there, {{czr.rule.anchor}}. The largest gain is
{{czr.rule.retriever}}'s, {{czr.rule.gain}}, so the end-to-end comparison {{czr.rule.verdict}}.

**What each costs.** One timed row per retriever with the budget bench on SQAD 3.2's corpus and the
{{czr.cost.n_queries}} questions of SQAD 3.2 full, on the {{czr.cost.chip}} MacBook Air, with the hygiene on: peak
resident memory of the whole process, the model and index files, and retrieval latency one query at a time,
encoding included. They are one machine's, read as a price beside each
difference, not as a test ([LIMITATIONS.md](LIMITATIONS.md) has the conditions).

{{czr.cost.table}}

e5-base's nDCG@10 over e5-small on SQAD 3.2, {{czr.sqad32_test.e5_base_minus_e5_small.ndcg10}}, costs
{{czr.cost.e5_base_minus_e5_small.gb}} GB of peak memory and {{czr.cost.e5_base_minus_e5_small.p95_ms}} ms of p95
latency. BM25 over lemmas' gain over BM25 without diacritics, {{czr.sqad32_test.bm25_lemma_minus_bm25_nodiacritics.ndcg10}},
costs {{czr.cost.bm25_lemma_minus_bm25_nodiacritics.gb}} GB and {{czr.cost.bm25_lemma_minus_bm25_nodiacritics.p95_ms}} ms,
and e5-small's own lead over BM25 without diacritics costs {{czr.cost.e5_small_minus_bm25_nodiacritics.gb}} GB and
{{czr.cost.e5_small_minus_bm25_nodiacritics.p95_ms}} ms.

### The copy's own threshold, as a sensitivity row

The stated threshold is {{czech_checks.threshold_stated}}, the 90th percentile on the whole SQAD 3.0
archive. On the copy alone it would be {{czech_checks.threshold_copy}}. Each set debiased at
{{czech_checks.threshold_copy}} has its own run, sized as its 0.138 set or to every eligible question; the
rows marked "no top-up" take only the questions of the 0.138 set that also pass
{{czech_checks.threshold_copy}}. Every step keeps its sign at {{czech_checks.threshold_copy}}, and every step
established at 0.138 stays established, except Gemma 4's generator step on the copy.

{{cz.table_sensitivity_both}}

**The order of the queue.** The copy's {{czech_checks.threshold_copy}} set holds questions of its 0.138
set, with the same prompts, answered in another run and so after other prompts. An answer near a tie can
turn on the prompt before it in the same process; this measures how
often that happens at this size:

{{cz.table_order}}

### Why is EM higher on SQAD 3.2?

**The planned decomposition**, fixed before the answers were generated. G1 is the gap
between the unfiltered sets, SQAD 3.2 full against the released 200; G2 the gap between the two debiased
sets. With yes and no mapped, G1 is {{cz.planned.mapped.g1_range}}, established in
{{cz.planned.mapped.g1_established}} configurations ({{cz.planned.mapped.g1_established_sign}}). Matching answer type and
length does not close it; with yes and no mapped it widens it. Where G1 is established
({{cz.planned.mapped.rule_share_g1_established_n}} configurations), the questions the debias rule flags account
for {{cz.planned.mapped.rule_share_where_g1_established}} of it. Across every configuration the planned share is
{{cz.planned.mapped.rule_share}}: above 100% where G1 on the questions the rule passes falls below zero, and
below zero where G1 itself does, as for BM25 + Qwen3.5 2B, whose G1 is not established. A share of a gap
that is near zero can take any value, so only the first range is read.
On the questions the rule passes G1 is established in {{cz.planned.mapped.g1_rule_established}}. G2 is
{{cz.planned.mapped.g2_range}}, established in {{cz.planned.mapped.g2_established}}
({{cz.planned.mapped.g2_established_sign}}). Answering the same question on each version's passages makes an
established difference for {{cz.cross.mapped_established}} models ({{cz.cross.mapped_range}}), and what is left of G2
after that is established in {{cz.planned.mapped.left_established}}. The gap between the unfiltered sets is smaller on the
questions the rule passes in every configuration (point estimates), but the data cannot say how much of it the rule
accounts for: where G1 is established the point share is {{cz.planned.mapped.rule_share_where_g1_established}}, and each
interval on the passing questions contains both zero and the full gap. Between the debiased sets a gap is
established in {{cz.planned.mapped.g2_established}} configurations ({{cz.planned.mapped.g2_established_sign}}).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../results/figures/version_gap_dark.svg">
  <img alt="Dot plot with 95% intervals for fifteen configurations, each family's smaller model with BM25 and with e5-small and its larger model with e5-small: the gap in exact match with yes and no mapped between SQAD 3.2 full and the SQAD 3.0 copy's released 200, on all questions and on the questions the debias rule passes. In every configuration the gap is lower on the questions the rule passes, and no interval there excludes zero." src="../results/figures/version_gap_light.svg">
</picture>

*G1 with yes and no mapped, from the table below: filled marks on all questions, open marks on the questions the
rule passes; colour is the family. Qwen2.5-3B-Instruct is under the Qwen Research License ([THIRD_PARTY.md](../THIRD_PARTY.md)).*

With yes and no mapped (the primary metric):

{{cz.table_planned_gap_mapped}}

The same decomposition in raw EM:

{{cz.table_planned_gap_em}}

**The same question on each version's passages**, {{cz.cross.n}} questions, {{cz.cross.same_gold}} with the
same gold in both. The two versions' passages make an established difference for {{cz.cross.mapped_established}}
models, about what chance alone gives with this many unadjusted intervals, though e5-small puts the
gold in its top 3 less often on SQAD 3.2's, and the generator step is about the same size on either:

{{cz.table_cross}}

{{cz.table_cross_steps}}

**Exploratory cuts**, Qwen2.5 and raw EM, chosen after seeing an earlier round of answers on small cells.
On the unfiltered sets the gap is {{cz.gap.bm25}} with BM25 + 1.5B,
{{cz.gap.e5}} with e5-small + 1.5B and {{cz.gap.e5_3b}} with e5-small + 3B. The sets hold different
questions, so each interval resamples them independently.

**Answer type and length explain little.** Matching them closes {{cz.gap_closed.type_length}} of the gap in
raw EM.

{{cz.table_gap}}

{{cz.table_em_by_type}}

{{cz.table_em_by_length}}

**The questions the debias rule flags account for more.** The rule flags a question when its answer starts in the
first 50 characters of the context or its words overlap the context heavily. Štefánik et al. (2023) reported the first
bias, about half of SQAD's questions having their answer at the beginning of the article, and removed
{{stefanik.share_removed}} of the questions whose answer starts in the first {{stefanik.first_chars}} characters. The rule follows BenCzechMark's filter for
SQAD 3.2 (Fajcik, Docekal, Dolezal et al., https://arxiv.org/abs/2412.17933, Appendix B.7), which removes
{{bcm.share_removed}} of the pairs whose answer is in the first {{bcm.first_chars}} characters of the context and
{{bcm.share_removed}} of those whose question and context have a TF-IDF cosine similarity above
{{bcm.tfidf_cosine}}. Here the overlap is the Jaccard similarity of the question's and the context's sets of words,
lowercased and without diacritics, above {{czech_checks.threshold_stated}}, and every flagged question is left out of
a debiased set rather than {{bcm.share_removed}} of them. A yes/no question's answer, ano or ne, is not a span of the
context, so the rule can flag it only by overlap, never by position, and the debiased sets keep a larger share of
yes/no questions than the unfiltered ones (the sets table above). It flags {{cz.rule.share_32}} of
SQAD 3.2's whole test split and {{cz.rule.share_copy}} of the copy's, because the copy leaves out most records
whose answer sentence opens the article. In SQAD 3.2's unfiltered set, the flagged questions score
{{cz.rule.em_flagged_32}} and the rest {{cz.rule.em_passing_32}} (e5-small + 1.5B). Keeping only the questions
the rule passes closes {{cz.rule.closed}} of the gap, and the remaining gap's interval then includes both zero and
the original gap. The same goes for
the two debiased sets, which share many of their questions with the unfiltered ones.

{{cz.table_gap_rule}}

{{cz.table_em_by_rule}}

Matching the copy to SQAD 3.2's flagged share, the last column of the first table, does not work: the released
set has only {{cz.released.flagged}} flagged questions, and they score unlike SQAD 3.2's. That is why this is
read from the questions the rule passes.

**What does not explain it.**

* **Retrieval.** On the {{cz.same_retrieval.n}} questions in both test splits, g @3 with e5-small is
  {{cz.same_retrieval.e5_copy}} on the copy and {{cz.same_retrieval.e5_32}} on SQAD 3.2, a difference of
  {{cz.same_retrieval.e5_diff}}. It goes the other way.
* **How a passage is cut.** On those questions the median gold passage has {{cz.same_retrieval.chars_copy}} and
  {{cz.same_retrieval.chars_32}} characters, and the answer starts at character {{cz.same_retrieval.start_copy}} and
  {{cz.same_retrieval.start_32}}.
* **Whether the answer is in front of the model.** With the answer's string in the top 3, EM on SQAD 3.2 is
  still the higher.
* **Gold at all.** The gap is there at every dial level, down to no gold at all: {{cz.gap_no_gold.e5}} with 1.5B
  and {{cz.gap_no_gold.e5_3b}} with 3B.
* **The gold answers.** Of {{cz.join.joined}} questions joined across the two versions, {{cz.join.same}}
  ({{cz.join.same_pct}}) have the same gold after normalising.

{{cz.table_em_by_answer_in_top3}}

{{cz.table_gap_levels}}

**Two more cuts settle nothing.** Matching the opening sentence closes from {{cz.gap_closed.opening}} of the gap,
depending on the configuration, and it rests on the {{cz.released.opens}} such questions in the released set. A
negative share means the match widens the gap. Matching the gold passage's length closes
{{cz.gap_closed.passage}}.

{{cz.table_em_by_opening}}

{{cz.table_em_by_passage}}

**What is left.** On SQAD 3.2's questions that the copy also holds, the unfiltered gap stays positive and is
established only with retromae-cs + 1.5B; between the debiased sets it is near zero, either way:

{{cz.table_gap_origin}}

---

*Generated from `results/stats.json`. Inputs: `configs/crossover.yaml`.*
