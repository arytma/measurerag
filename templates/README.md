English | [Čeština](README.cs.md)

# measureRAG

**A budget-aware retrieval and RAG benchmark in Czech and English, for a
single machine with no network access.**

measureRAG measures retrieval and retrieval-augmented generation (RAG) the way
a deployment on one CPU-only machine with a few gigabytes of RAM and no network
meets them. The budget is imposed first: a configuration either fits a
deployment envelope (memory, retrieval latency, end-to-end latency) or it does
not, and answer quality is reported inside the envelope it fits. Everything
runs locally: no API keys, no cloud models, no telemetry.

Czech is measured beside English, end to end: Czech retrieval on CsFEVER v2 and
Czech RAG on SQAD 3.2 and on a filtered copy of SQAD 3.0, answered by five model families (Gemma 3n,
Gemma 4, Ministral 3, Qwen2.5 and Qwen3.5), with the same quality measures and statistics as the English sets; the Czech
answers were generated on a GPU, so no Czech generator has a CPU latency or a budget verdict. Every Czech quality number
recomputes from this repository, or from it and a file you download and check by a published MD5 or SHA-256 hash
([docs/REPRODUCE.md](docs/REPRODUCE.md#10-checking-every-czech-number)); the SQAD retrievers' timings are one machine's record.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/families_dark.svg">
  <img alt="Dot plot in two panels, each family's smaller model and each family's larger model, with the four Czech SQAD sets along the bottom, the released 200 of the SQAD 3.0 copy first, and exact match with yes and no mapped going up, each mark with its 95% interval; in each panel the models stand best first by their score on the released 200, and each name carries its maker's flag. On every set and at both sizes Gemma 3n, Gemma 4, Ministral 3 and Qwen3.5 score higher than Qwen2.5. Among the smaller models Gemma 4 E2B is highest on every set; among the larger, Gemma 4 E4B is highest on the released 200." src="results/figures/families_light.svg">
</picture>

*Czech RAG, five model families. Each model's EM with yes and no mapped, e5-small's top 3, with its 95% bootstrap
interval; left, each family's smaller model, right, its larger, each panel's models in order of their score on the
released 200 (a tie keeps the makers' country order), and the sets with the released 200 first. Compare the marks within a set; the numbers are in
[Five families, model for model](#five-families-model-for-model). Qwen2.5-3B-Instruct is under the Qwen Research License ([THIRD_PARTY.md](THIRD_PARTY.md)).*

## The question it answers

A RAG system on a fixed machine has two obvious places to spend its next
gigabyte: a stronger retriever or a bigger generator. Which one buys more
answer quality depends on how often the retriever already puts a gold passage
in front of the generator, the *gold-in-context* rate g.

measureRAG answers this with a controlled dial. It takes the strongest
retriever's top {{design.k_dial}} passages and replaces each gold passage, with
probability p, by the next-ranked non-gold passage. Corpus, k, prompt and amount
of text stay fixed; only whether the gold is there changes. At each dial level,
a retriever step (one dial level more gold in the context) is set against a
generator step (the generator one size up), with paired bootstrap intervals over
questions ({{n_boot}} resamples). Exact match (EM) is the quality measure. The
dial is then checked against real retrievers.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/pipeline_dark.svg">
  <img alt="Diagram of one measured configuration on one machine with no network. A question goes to a retriever, BM25 or a dense encoder; the first k passages of its ranking go to a local generator; the answer is scored against gold by exact match and F1. Marked on it: the budgets (retrieval latency, end-to-end latency, the peak resident memory of the whole process), the two moves the experiment compares (the retriever step and the generator step), and the dial, which swaps gold passages out with probability p." src="results/figures/pipeline_light.svg">
</picture>

*One configuration, as the bench runs it. The budgets bind the whole process; the dial exists only in the
experiment, to move gold-in-context while everything else stays fixed.*

## Result: the break-even gold gain

The usable result is not a crossing point but a bar. For each operating point
g, the tables give how much gold-in-context a retriever upgrade must add before
it buys as much EM as the generator step does there, raw, and for an upgrade that
costs as much peak resident memory as the priced retriever step. To use it: measure how much gold-in-context your retriever
upgrade adds on your data at your k, and compare it with the bar at your g.

**HotpotQA** ({{en_m3.hotpotqa.n}} questions):

{{en_m3.est.hotpotqa.table_break_even}}

**Natural Questions** (NQ, {{en_m3.nq.n}} questions):

{{en_m3.est.nq.table_break_even}}

*The generator step is Ministral 3 3B to Ministral 3 8B (Instruct 2512, Q4_K_M), the family whose English
sweeps were timed on this laptop's CPU. Its dial's answers were generated on a second machine's GPU
([docs/LIMITATIONS.md](docs/LIMITATIONS.md#the-czech-and-english-families)).*

How to read the tables: p is the dial level; "headroom to dial top" is how much
gold-in-context there is left to add; "> headroom" means no reachable gain
would match the generator step; "censored" counts the bootstrap resamples in
which the headroom above the operating point ran out, and an upper end written
"> headroom" is such an edge, not an estimate. Every table and the prices in
gigabytes and milliseconds are in [docs/CROSSOVER.md](docs/CROSSOVER.md).

The last two columns are not rates. Each is the bar for one upgrade that costs as much peak memory as the priced
retriever step, {{priced_step}} ({{en_m3.hotpotqa.cost_ret_gb}} GB on HotpotQA, {{en_m3.nq.cost_ret_gb}} GB on NQ),
set against the generator step's cost ({{en_m3.hotpotqa.cost_gen_gb}} GB on HotpotQA and {{en_m3.nq.cost_gen_gb}} GB on NQ as measured, model file memory-mapped (library default);
{{en_m3.est.hotpotqa.cost_gen_gb}} GB and {{en_m3.est.nq.cost_gen_gb}} GB, estimate without the double-counted mapped pages (probe; see [docs/LIMITATIONS.md](docs/LIMITATIONS.md#the-machine-and-the-timings))): the gold gain whose EM gain equals the generator
step's EM gain times the ratio of the two costs. Peak memory as measured counts part of each Ministral 3 model file
twice, because the file is memory-mapped while the CPU backend also copies the weights, so the tables give that bar
twice: without the double count the generator step costs less, and the bar is about
{{en_m3.est.hotpotqa.per_gb_bar_factor}} times as large on HotpotQA and {{en_m3.est.nq.per_gb_bar_factor}} times on NQ. For an upgrade
that costs C GB, the bar is the gold gain that buys the generator step's EM gain times C / (the generator step's GB);
where the dial's EM curve is close to straight, that is about the raw bar times C / (the generator step's GB). The
cost C here is the memory of {{priced_step}}, a step that on HotpotQA moves g any (the share of questions with at
least one gold passage in the context) by only {{en_m3.hotpotqa.step.bm25->e5-small.k5.g_change}}
({{en_m3.hotpotqa.step.bm25->e5-small.k5.g}}): the columns say what an upgrade of that cost would have to add, not
what that step adds.

## What is and is not established

* **HotpotQA:** {{en_m3.headline.hotpotqa.gstar_raw}}, raw EM. Per GB, as measured, model file memory-mapped (library default):
  {{en_m3.headline.hotpotqa.gstar_gb}}; per GB, estimate without the double-counted mapped pages (probe; see [docs/LIMITATIONS.md](docs/LIMITATIONS.md#the-machine-and-the-timings)): {{en_m3.est.headline.hotpotqa.gstar_gb}}.
* **NQ:** {{en_m3.headline.nq.gstar_raw}}, raw EM. With every gold passage removed,
  {{en_m3.nq.leak_no_gold}} of the contexts still contain the answer in a non-gold
  passage. Per GB, as measured, model file memory-mapped (library default): {{en_m3.headline.nq.gstar_gb}}; per GB, estimate without the double-counted mapped
  pages (probe; see LIMITATIONS): {{en_m3.est.headline.nq.gstar_gb}}.
* **No located crossing.** The fitted curves cross on both datasets, but the
  generator step's advantage is established at {{en_m3.hotpotqa.levels_gen_established}}
  levels on HotpotQA and {{en_m3.nq.levels_gen_established}} on NQ, and wherever gold is
  present the gap between the two moves is established at
  {{en_m3.hotpotqa.established_with_gold}} levels on HotpotQA and
  {{en_m3.nq.established_with_gold}} on NQ; some resamples never cross (above). The data
  do not establish that the two moves cross, nor where. Do the two datasets agree?
  {{en_m3.headline.nq_agrees}}.
* **On HotpotQA, among strong retrievers, the generator step is the larger.** At
  k = {{design.k_dial}}, on this {{data.hotpotqa.docs}}-passage slice, from bm25 upwards every real retriever
  step leaves g any almost where it was (bm25 to e5-small: {{en_m3.hotpotqa.step.bm25->e5-small.k5.g}}), and the
  generator step at the lower rung is larger, the difference established at each: bm25 to e5-small buys
  {{en_m3.hotpotqa.step.bm25->e5-small.k5.ret}} EM and 3B to 8B in front of bm25
  {{en_m3.hotpotqa.step.bm25->e5-small.k5.gen}} (generator minus retriever
  {{en_m3.hotpotqa.step.bm25->e5-small.k5.gen_minus_ret}}). e5-base to bge-m3, a change of
  retriever rather than an upgrade, leaves g any where it was and changes EM by
  {{en_m3.hotpotqa.step.e5-base->bge-m3.k5.ret}}. From the weak tfidf-svd to bm25, the
  retriever step is the larger: {{en_m3.hotpotqa.step.tfidf-svd->bm25.k5.ret}} against
  {{en_m3.hotpotqa.step.tfidf-svd->bm25.k5.gen}}. On NQ, bm25 to e5-small does move
  g any ({{en_m3.nq.step.bm25->e5-small.k5.g}}), and there generator minus retriever is
  {{en_m3.nq.step.bm25->e5-small.k5.gen_minus_ret}}: {{en_m3.nq.step.bm25->e5-small.k5.larger}}. That bm25 already
  puts a gold passage in nearly every HotpotQA context is a property of the slice, which holds every judged
  passage of the sampled questions and a sample of the rest; a full corpus has far more distractors.
* **Per millisecond, HotpotQA has no price.** The priced retriever step
  ({{priced_step}}) changes the generation-stage latency by
  {{en_m3.hotpotqa.cost_ret_ms}} ms on HotpotQA, as measured (its retrieval p50 adds
  {{en_m3.hotpotqa.cost_ret_retrieval_ms}} ms), so there a cost ratio is undefined. Its lower rung, bm25 with
  Ministral 3 3B, was timed again after the night it was first timed in (below), and the change stayed negative.
  Two explanations fit, and the records cannot separate them: e5-small's passages may make shorter prompts for the
  generator to read, or the change is variation between runs, since bm25 with Ministral 3 3B alone moved by
  {{en_m3.hygiene.retimed.hotpotqa_k5.bm25+ministral-3-3b.p50_shift_ms}} ms between its two timings. The prompts' token counts are not recorded
  ([docs/LIMITATIONS.md](docs/LIMITATIONS.md#the-machine-and-the-timings)). On NQ it adds
  {{en_m3.nq.cost_ret_ms}} ms, and the per-ms fit finds
  {{en_m3.headline.nq.gstar_ms_total}}.
* **Per GB, no crossing.** The generator step costs {{en_m3.hotpotqa.cost_gen_gb}} GB of
  peak memory on HotpotQA and {{en_m3.nq.cost_gen_gb}} GB on NQ as measured, model file memory-mapped (library default), and
  {{en_m3.est.hotpotqa.cost_gen_gb}} GB and {{en_m3.est.nq.cost_gen_gb}} GB, estimate without the double-counted mapped pages (probe; see [docs/LIMITATIONS.md](docs/LIMITATIONS.md#the-machine-and-the-timings)). As measured, the
  per-GB fit finds no crossing, with the generator step priced in front of e5-base or of each of the five rungs in
  turn. Measured peak memory counts part of each model file twice; with that taken out, the fit still finds no crossing
  in front of e5-base on either dataset, but priced in front of each rung it crosses at
  {{en_m3.est.hotpotqa.rungs_with_crossing}} rungs on HotpotQA and {{en_m3.est.nq.rungs_with_crossing}} on NQ
  ({{en_m3.est.nq.rungs_crossing_list}}; [docs/CROSSOVER.md](docs/CROSSOVER.md), §3 and §4). In every per-GB fit the retriever step is one dial level of gold priced at the
  {{en_m3.hotpotqa.cost_ret_gb}} GB of {{priced_step}}, a step that moves g any on HotpotQA by only
  {{en_m3.hotpotqa.step.bm25->e5-small.k5.g_change}}, so the per-GB comparison prices one move and gains from another.
* **No Ministral 3 configuration fits a budget in the grid.** Configurations that fit
  one: {{en_m3.budget.configs_fitting}} with peak memory as measured, and {{en_m3.est.budget.configs_fitting}} with the estimate. Their end-to-end p95 runs from
  {{en_m3.budget.e2e_p95_min_s}} s to {{en_m3.budget.e2e_p95_max_s}} s, against the grid's
  end-to-end limits ({{design.budget_e2e_s_text}}), and their peak memory from
  {{en_m3.budget.peak_rss_range_gb}} GB as measured, model file memory-mapped (library default), and from {{en_m3.est.budget.peak_rss_range_gb}} GB, estimate without the double-counted mapped pages (probe; see [docs/LIMITATIONS.md](docs/LIMITATIONS.md#the-machine-and-the-timings))
  ([docs/BUDGETS.md](docs/BUDGETS.md)). This describes
  Ministral 3 at 3B and 8B on this CPU, not every generator
  ([docs/LIMITATIONS.md](docs/LIMITATIONS.md#the-czech-and-english-families)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/dial_dark.svg">
  <img alt="Line chart in two panels, HotpotQA and Natural Questions: exact match of Ministral 3 3B and Ministral 3 8B against gold-in-context on the controlled dial, with 95% intervals. Both lines rise as more gold reaches the generator, and wherever gold is present 8B scores above 3B. A small French flag follows each model name." src="results/figures/dial_light.svg">
</picture>

*The dial as levels, on both English sets, for Ministral 3 3B and 8B 🇫🇷. A retriever step is one dial level
to the right along a line; the generator step is the gap between the two lines at one level. The headline above, the
budgets and the prices are Ministral 3's: its dial answered on a second machine's GPU, its sweeps were timed on this
laptop's CPU. The figure shows one of the three Western families' dials, Ministral 3's, the one in which the larger
model gains most, chosen by the author after all three were read, and Ministral 3 became the family whose English
sweeps were timed after the same reading ([docs/LIMITATIONS.md](docs/LIMITATIONS.md#the-czech-and-english-families)).
The other dials are in [docs/CROSSOVER.md](docs/CROSSOVER.md#four-families-english-dials): Gemma 3n's 🇺🇸 and Gemma 4's 🇺🇸,
where the larger model buys little or nothing, and Gemma 4 E4B's score below E2B is established at
{{hotpotqa.dial_family.gemma_4.gen_step_established_below}} HotpotQA levels, and Qwen2.5's 🇨🇳. Qwen3.5 🇨🇳 did not answer the English dial.*

## Czech

**Retrieval** (CsFEVER v2, {{runs.cz_small.docs}} Wikipedia pages,
{{runs.cz_small.queries}} claims): BM25 with diacritics stripped scores nDCG@10
{{runs.cz_small.bm25-nodiacritics.ndcg10}}, plain BM25
{{runs.cz_small.bm25.ndcg10}}, multilingual-e5-small
{{runs.cz_small.e5-small.ndcg10}}. The order holds at {{runs.cz_full.docs}}
documents ({{runs.cz_full.bm25-nodiacritics.ndcg10}}, {{runs.cz_full.bm25.ndcg10}}
and {{runs.cz_full.e5-small.ndcg10}}). These are means over the claims with no interval: no per-claim score
ships, so the order is not tested. CsFEVER is entity-heavy, which favours
lexical matching, and every configuration fits every budget at these sizes.
That lead is CsFEVER's. On SQAD, e5-small leads BM25 without diacritics at large
samples: {{cz.large.copy_test.lead_nd}} in nDCG@10 on all {{cz.large.copy_test.n}} test questions of
the SQAD 3.0 copy, and {{cz.large.sqad32_test.lead_nd}} on all {{cz.large.sqad32_test.n}} test
questions of SQAD 3.2 (95% intervals).

**Ten retrievers on SQAD.** Six more retrievers ranked
both test splits: e5-base, IBM's granite-embedding 107m and 278m multilingual, Google's EmbeddingGemma 300m,
Seznam's simcse-cs-en, and BM25 over Czech lemmas. None leads e5-small by as much as e5-small leads BM25. e5-base, more
than twice its size, gains {{czr.sqad32_test.e5_base_minus_e5_small.ndcg10}} in nDCG@10 on SQAD 3.2's
{{czr.sqad32_test.n}} test questions and nothing established on the copy's {{czr.copy_test.n}}
({{czr.copy_test.e5_base_minus_e5_small.ndcg10}}). Lemmas help BM25: {{czr.sqad32_test.bm25_lemma_minus_bm25_nodiacritics.ndcg10}}
over BM25 without diacritics on SQAD 3.2, about half of e5-small's lead over it on that set, and
{{czr.copy_test.bm25_lemma_minus_bm25_nodiacritics.ndcg10}} on the copy. That contrast changes two things, because BM25
over lemmas lowercases each word and keeps its diacritics before simplemma lemmatises it (so "Prahy" becomes "práh",
threshold, not "Praha"); against plain BM25, which keeps diacritics too, lemmas gain
{{czr.sqad32_test.bm25_lemma_minus_bm25.ndcg10}} on SQAD 3.2 and {{czr.copy_test.bm25_lemma_minus_bm25.ndcg10}} on the copy (a contrast added after the
rankings were read). EmbeddingGemma sits between BM25 and e5-small
({{czr.sqad32_test.embeddinggemma_minus_e5_small.ndcg10}} against e5-small on SQAD 3.2), and both granite models
below BM25 without diacritics ({{czr.sqad32_test.granite_278m_minus_bm25_nodiacritics.ndcg10}} for the larger).
No gain in the gold in the top 3 over e5-small is established and as large as e5-small's own lead over BM25
({{czr.rule.anchor}}), the bar a new retriever had to clear, by a rule fixed before the rankings were read, to answer
every set end to end; so none did. Timed on SQAD 3.2's
corpus on the same Mac, e5-base's gain over e5-small costs {{czr.cost.e5_base_minus_e5_small.gb}} GB of peak memory
and {{czr.cost.e5_base_minus_e5_small.p95_ms}} ms of p95 latency per query; BM25 over lemmas answers a query in
{{czr.cost.bm25_lemma.p95_ms}} ms at p95 with {{czr.cost.bm25_lemma.gb}} GB of peak memory, and e5-small in
{{czr.cost.e5_small.p95_ms}} ms with {{czr.cost.e5_small.gb}} GB. Every row is in
[docs/CROSSOVER.md](docs/CROSSOVER.md#ten-retrievers-on-sqad).

**End-to-end RAG** (k = {{runs.cz_rag.k}}). Five model families answer every set, two models each. Each model, with its
maker, the maker's country and the model's licence (in full in [THIRD_PARTY.md](THIRD_PARTY.md)):

<ul>
<li>Gemma 3n E2B: Google, United States; Gemma Terms of Use</li>
<li>Gemma 3n E4B: Google, United States; Gemma Terms of Use</li>
<li>Gemma 4 E2B: Google, United States; Apache 2.0</li>
<li>Gemma 4 E4B: Google, United States; Apache 2.0</li>
<li>Ministral 3 3B Instruct 2512: Mistral AI, France; Apache 2.0</li>
<li>Ministral 3 8B Instruct 2512: Mistral AI, France; Apache 2.0</li>
<li>Qwen2.5-1.5B-Instruct: Alibaba Cloud's Qwen team, China; Apache 2.0</li>
<li>Qwen2.5-3B-Instruct: Alibaba Cloud's Qwen team, China; Qwen Research License, non-commercial research and evaluation use only</li>
<li>Qwen3.5 2B: Alibaba Cloud's Qwen team, China; Apache 2.0</li>
<li>Qwen3.5 4B: Alibaba Cloud's Qwen team, China; Apache 2.0</li>
</ul>

Gemma 4, chosen by a benchmark of Western candidates on a separate dev set, and Qwen2.5 answered first. Gemma 3n, Ministral 3 and Qwen3.5 were added at the author's request after the earlier families'
results had been read, each run alone with the questions, rankings, prompt, token cap, seed and machine unchanged
([docs/LIMITATIONS.md](docs/LIMITATIONS.md#the-czech-and-english-families)). There are four sets, and none of them is the headline:
SQAD 3.2 debiased, SQAD 3.2 full, the released 200 of the SQAD 3.0 copy, and the copy debiased. The primary
metric is EM with yes and no mapped to ano and ne, because Qwen2.5 still answers many yes/no questions in
English. Raw EM and compliance stand beside it: compliance is, of the yes/no questions a model did not decline, the share
it answered with exactly ano or ne ([docs/METRICS.md](docs/METRICS.md)).

### Five families, model for model

Each family's smaller model is set against the other families' smaller models, and its larger against their
larger, all with e5-small's top 3 and paired on the same questions. The pairs are by place in the family,
not by size: at Q4_K_M the smaller models' files are {{cz.bench.gemma-4-e2b.gb}} GB (Gemma 4 E2B),
{{cz.bench.ministral-3-3b-unsloth.gb}} GB (Ministral 3 3B), {{cz.bench.qwen2.5-1.5b.gb}} GB (Qwen2.5 1.5B) and
{{cz.bench.qwen3.5-2b.gb}} GB (Qwen3.5 2B), and the larger models' {{cz.bench.gemma-4-e4b.gb}},
{{cz.bench.ministral-3-8b-unsloth.gb}}, {{cz.bench.qwen2.5-3b.gb}} and {{cz.bench.qwen3.5-4b.gb}} GB. Gemma 3n E2B and
E4B are Gemma 4's previous generation at the same two places, and Qwen3.5 is Qwen2.5's newer generation.

* Gemma 3n, Gemma 4, Ministral 3 and Qwen3.5 each answer better than Qwen2.5, at both sizes, on every set and in
  both metrics, every difference established. With yes and no mapped, Gemma 3n E2B scores above Qwen2.5 1.5B by
  {{cz.g3n.family.smaller.mapped_range}} and E4B above 3B by {{cz.g3n.family.larger.mapped_range}}; Gemma 4 E2B by
  {{cz.family.smaller.mapped_range}} and E4B by {{cz.family.larger.mapped_range}}; Ministral 3 3B by
  {{cz.m3.family.smaller.mapped_range}} and 8B by {{cz.m3.family.larger.mapped_range}}; Qwen3.5 2B by
  {{cz.q35.family.smaller.mapped_range}} and 4B by {{cz.q35.family.larger.mapped_range}}. The same {{cz.cross.n}}
  questions answered on each SQAD version's passages give {{cz.cross.g3n.family.mapped_range}} for Gemma 3n,
  {{cz.cross.family.mapped_range}} for Gemma 4, {{cz.cross.m3.family.mapped_range}} for Ministral 3 and
  {{cz.cross.q35.family.mapped_range}} for Qwen3.5.
* Among the smaller models, Gemma 4 E2B scores highest on every set. Its lead over Gemma 3n E2B is established
  on {{cz.holds.g4_minus_g3n.smaller.mapped_on}}, over Ministral 3 3B on {{cz.holds.m3_minus_g4.smaller.mapped_on}},
  and over Qwen3.5 2B on {{cz.holds.q35_minus_g4.smaller.mapped_on}}. Among the larger models, no model scores
  highest on every set, and on no set is the highest model's lead over the next established.
* Gemma 4 scores above its previous generation, Gemma 3n, on all four sets. Gemma 4 minus Gemma 3n is
  {{cz.g4_minus_g3n.smaller.mapped_sign}} at the smaller place ({{cz.g4_minus_g3n.smaller.mapped_range}}),
  established on {{cz.holds.g4_minus_g3n.smaller.mapped_on}}, and {{cz.g4_minus_g3n.larger.mapped_sign}} at the
  larger ({{cz.g4_minus_g3n.larger.mapped_range}}), established on {{cz.holds.g4_minus_g3n.larger.mapped_on}}. On the
  cross-version questions it is {{cz.cross.g4_minus_g3n.mapped_range}}.
* Gemma 4 scores above Ministral 3 on all four sets. Gemma 4 minus Ministral 3 is above zero on all four for the
  smaller models ({{cz.m3_minus_g4.smaller.mapped_range|neg}}) and for the larger
  ({{cz.m3_minus_g4.larger.mapped_range|neg}}), established on {{cz.holds.m3_minus_g4.smaller.mapped_on}} for E2B
  against 3B, and on {{cz.holds.m3_minus_g4.larger.mapped_on}} for E4B against 8B. On the cross-version questions it
  is {{cz.cross.m3_minus_g4.mapped_range|neg}}.
* Gemma 4 and Ministral 3 against Qwen3.5. Gemma 4 E2B scores above Qwen3.5 2B on all four sets, established on
  {{cz.holds.q35_minus_g4.smaller.mapped_on}}. Gemma 4 E4B and Qwen3.5 4B differ in both directions across the
  four, established on {{cz.holds.q35_minus_g4.larger.mapped_on}}. Ministral 3 3B and Qwen3.5 2B differ in both
  directions across the four, established on {{cz.holds.q35_minus_m3.smaller.mapped_on}}. Ministral 3 8B scores at
  or below Qwen3.5 4B on all four (level on one), by {{cz.q35_minus_m3.larger.mapped_range}},
  {{cz.holds.q35_minus_m3.larger.mapped}}.
* The larger model of each family answers better. For Gemma 3n this generator step, E4B over E2B, is
  established on every set, {{cz.g3n.range.generator_step.mapped}}. For Gemma 4 it is established on the three
  larger sets and not on the released 200 (below). For Ministral 3, 8B over 3B, it is established on every set,
  {{cz.m3.range.generator_step.mapped}}. For Qwen2.5 it is established on the three larger sets and not on the
  released 200, and for Qwen3.5, 4B over 2B, on every set, {{cz.q35.range.generator_step.mapped}}.
* Gemma 3n, Ministral 3 and Qwen3.5 answer in Czech. Gemma 3n gave {{cz.g3n.yes_no_in_english}} yes/no answers
  in English, so its two metrics are {{cz.g3n.metrics_agree}}. Ministral 3 gave {{cz.m3.yes_no_in_english}}, so its
  two are {{cz.m3.metrics_agree}}. Gemma 3n's compliance is {{cz.g3n.compliance_range}}, lower for E2B,
  which answers some yes/no questions with something other than ano or ne; Ministral 3's is
  {{cz.m3.compliance_range}}. Qwen2.5 gave {{cz.yes_no_in_english}} and Qwen3.5 {{cz.q35.yes_no_in_english}};
  Qwen3.5's compliance is {{cz.q35.compliance_range}}, lower for 2B for the same reason
  ([docs/CROSSOVER.md](docs/CROSSOVER.md#the-sets-and-the-answers)).
* Qwen3.5 answered without thinking. Its cards say it thinks before answering by default; through the
  files' own chat template it answered directly, and {{cz.q35.thinking.with_a_tag}} of its
  {{cz.q35.thinking.answer_rows}} answer rows on the test sets holds a thinking tag.

EM with yes and no mapped [95%], each model with e5-small's top 3, as [the figure at the top](#measurerag) draws it:

{{cz.table_families_levels_mapped|family_columns}}

The differences between the families, paired on the same questions, EM with yes and no mapped:

{{cz.table_families_differences_mapped|western_first}}

### The steps within each family

**What holds on all four sets.**

* e5-small answers better than retromae-cs, by {{cz.range.e5_minus_retromae_mapped}}, in every family
  on every set.
* For Gemma 3n, Ministral 3 and Qwen3.5, the generator step is established on every set in both metrics
  (above).
* With e5-small's top 3, Qwen2.5's generator step in raw EM (3B over 1.5B) is established
  on every set, {{cz.range.generator_step.em}}, and is larger than the retriever step on every set.

**What holds on the three larger sets.** At 80% power the released 200 can detect a retriever step of
{{cz.released.retriever_detects_observed}}, and SQAD 3.2 full one of {{cz.sqad32_full.retriever_detects_observed}},
so the released 200 establishes little on its own.

* **The generator step**, one model size up, with yes and no mapped: {{cz.g3n.range3.generator_step.mapped}} for
  Gemma 3n, {{cz.g4.range3.generator_step.mapped}} for Gemma 4, {{cz.m3.range3.generator_step.mapped}} for
  Ministral 3, {{cz.range3.generator_step.mapped}} for Qwen2.5 and {{cz.q35.range3.generator_step.mapped}} for
  Qwen3.5, each established.
* **The retriever step**, e5-small over BM25 with the smaller model: {{cz.g3n.range3.retriever_step.mapped}} for
  Gemma 3n, established on {{cz.holds.g3n.retriever_step.mapped_on}} only; {{cz.g4.range3.retriever_step.mapped}}
  for Gemma 4, established; {{cz.m3.range3.retriever_step.mapped}} for Ministral 3, established on
  {{cz.holds.m3.retriever_step.mapped_on}} only; {{cz.range3.retriever_step.mapped}} for Qwen2.5 and
  {{cz.q35.range3.retriever_step.mapped}} for Qwen3.5, each established.
* **Where the difference is established, the generator step is the larger.** On both SQAD 3.2 sets the generator step is the larger,
  established for Gemma 3n and Gemma 4, for Ministral 3 on
  {{cz.holds.m3.generator_step_minus_retriever_step.mapped_on}} only, and for Qwen2.5 and Qwen3.5. On the copy debiased the difference is
  established for Gemma 3n alone, whose generator step is the larger there,
  {{cz.copy_debiased.g3n.gap_steps_mapped}}; for Gemma 4 the retriever step is the larger point estimate, the
  generator step minus the retriever step being {{cz.copy_debiased.g4.gap_steps_mapped}}.

Each family's steps, EM with yes and no mapped:

{{cz.table_families_steps}}

### The sets, the prompt, the dial and the gap

*SQAD 3.0 here means the Hugging Face copy `{{czech_checks.copy_repo}}`, pinned at revision
`{{czech_checks.copy_revision}}`, not the whole archive. It holds
{{czech_checks.copy_rows}} of the archive's {{czech_checks.archive_records}} records, as
`scripts/sqad_copy_vs_archive.py` counts them in the LINDAT file (LINDAT's page states
{{czech_checks.archive_records_stated}}). Its card names no source and states no filter, but it is, row for row, the filtered
SQAD 3.0 that Štefánik et al. (2023) published as `fewshot-goes-multilingual/cs_squad-3.0`
(`scripts/sqad_copy_source.py` checks this). Comparing it with the archive shows it leaves out {{czech_checks.opens_dropped}} of the
{{czech_checks.opens_all}} records whose answer sentence opens the article
({{czech_checks.opens_dropped_pct}}) and keeps all {{czech_checks.later_kept}} records whose
answer sentence comes later. The licence is the one on LINDAT, LGPL-3.0
(http://hdl.handle.net/11234/1-3069), not the copy's own label; see
[THIRD_PARTY.md](THIRD_PARTY.md). SQAD 3.2 is the whole release on LINDAT, under CC BY-SA 3.0
(http://hdl.handle.net/11234/1-5019). A debiased set leaves out questions whose answer starts in the
first 50 characters of the context, or whose words overlap the context heavily (Jaccard above
{{czech_checks.threshold_stated}}). Štefánik et al. (2023) reported the first bias, about half of SQAD's questions having
their answer at the beginning of the article, and removed {{stefanik.share_removed}} of the questions whose answer starts in the
first {{stefanik.first_chars}} characters. The rule follows BenCzechMark's filter for SQAD 3.2 (Fajcik, Docekal,
Dolezal et al., https://arxiv.org/abs/2412.17933, Appendix B.7), which removes {{bcm.share_removed}} of
the pairs whose answer is in the first {{bcm.first_chars}} characters of the context, and
{{bcm.share_removed}} of the pairs whose question and context have a TF-IDF cosine similarity above
{{bcm.tfidf_cosine}}. This release differs in two ways: its overlap is the Jaccard similarity of the
question's and the context's sets of words, lowercased and without diacritics, above
{{czech_checks.threshold_stated}} (the 90th percentile on the whole SQAD 3.0 archive); and it leaves out
every flagged question, not {{bcm.share_removed}} of them. No SQAD question, context or document text is in this
repository outside the generated answers, which quote it: the longest extract is {{cz.quotes.longest}}
characters, and {{cz.quotes.repeated_rows}} answers repeat their own question
([THIRD_PARTY.md](THIRD_PARTY.md)).*

**Read differences within a set, not levels across sets.** The sets differ in which questions they keep,
and SQAD 3.2 full keeps the questions the debias rule flags (see the gap, below). A yes/no question's answer, ano
or ne, is not a span of its context, so the rule can flag it only by overlap, never by position; the debiased sets
therefore keep a larger share of yes/no questions ({{cz.sqad32_debiased.yes_no}} of {{cz.sqad32_debiased.n}} on
SQAD 3.2 debiased, against {{cz.sqad32_full.yes_no}} of {{cz.sqad32_full.n}} on SQAD 3.2 full).

**Compliance.** Gemma 3n's compliance is {{cz.g3n.compliance_range}} across models and sets, Gemma 4's
{{cz.g4.compliance_range}}, Ministral 3's {{cz.m3.compliance_range}}, Qwen2.5's {{cz.compliance_range}} and Qwen3.5's
{{cz.q35.compliance_range}}. For Gemma 3n the two metrics are {{cz.g3n.metrics_agree}}, for Gemma 4 they nearly
agree, and for Ministral 3 they are {{cz.m3.metrics_agree}}. Qwen2.5 3B answers ano or ne more
often than 1.5B, by {{cz.range.compliance_step}}, so part of Qwen2.5's generator step in raw EM is the
language of the answer: on SQAD 3.2 full it is {{cz.sqad32_full.generator_step}} in EM and
{{cz.sqad32_full.generator_step_mapped}} with yes and no mapped.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/answer_language_dark.svg">
  <img alt="Bar chart: for each of the ten models, the share of the yes/no questions of SQAD 3.2 full it answers with yes or no in English, although the prompt asks for ano or ne. Gemma 3n never does so, Gemma 4 almost never, Ministral 3 never, Qwen2.5 1.5B for most of them, Qwen2.5 3B for fewer and Qwen3.5 almost never." src="results/figures/answer_language_light.svg">
</picture>

*The yes/no questions of SQAD 3.2 full, e5-small's top 3. Every set and configuration, with the declined answers, is in
[docs/CROSSOVER.md](docs/CROSSOVER.md#the-sets-and-the-answers). Qwen2.5-3B-Instruct is under the Qwen Research License ([THIRD_PARTY.md](THIRD_PARTY.md)).*

**The prompt.** Czech questions are asked with the English template whose yes/no line asks for ano or ne, the
form of SQAD's gold; its other instructions, its labels and its decline word stay English. It replaced the same
template asking for yes or no, after the answers to the released questions showed that, under it, every yes/no
question scored 0. On the SQAD 3.0 copy's released {{cz.defect_mac.questions}} questions
({{cz.defect_mac.yes_no}} of them yes/no), answered on one machine under both, EM on the yes/no questions goes
from {{cz.defect_mac.e5.em_yes_no_v020}} under the yes-or-no line to {{cz.defect_mac.e5.em_yes_no_fixed}}
under the ano-or-ne line with e5-small + 1.5B, and to {{cz.defect_mac.e5_3b.em_yes_no_fixed}} with e5-small + 3B. The
fix is partial: on those questions 1.5B still answers {{cz.defect_mac.e5.english}} in English and 3B
{{cz.defect_mac.e5_3b.english}}. Three more prompts, written in Czech and tried on a separate dev set of
{{cz.dev.questions}} questions under rules fixed beforehand, did not make Qwen2.5 answer in Czech without
losing span answers, so the prompt was kept. Czech EM mostly under-counts: an inflected form of the right
answer is scored wrong
([docs/LIMITATIONS.md](docs/LIMITATIONS.md)).

**The Czech dial** (Gemma 4, e5-small's top {{czech.k}}), with yes and no mapped. Do its fits cross? On SQAD 3.2
debiased: {{cz.sqad32_debiased.g4.dial_crosses_mapped}}; on SQAD 3.2 full: {{cz.sqad32_full.g4.dial_crosses_mapped}}; on the
released 200: {{cz.released.g4.dial_crosses_mapped}}; on the copy debiased: {{cz.copy_debiased.g4.dial_crosses_mapped}}. The two
moves separate, their difference established, at {{cz.sqad32_debiased.g4.levels_established_mapped}} levels on SQAD 3.2
debiased, {{cz.sqad32_full.g4.levels_established_mapped}} on SQAD 3.2 full, {{cz.released.g4.levels_established_mapped}} on the
released 200 and {{cz.copy_debiased.g4.levels_established_mapped}} on the copy debiased. With every gold passage removed,
{{czech.leak_no_gold}} of the released 200's contexts still contain the answer in a non-gold passage, so the dial
moves gold more than it moves the answer. The dial for every family on every set is in an appendix of
[docs/CROSSOVER.md](docs/CROSSOVER.md), and in no headline or agreement statement.

**The gap between the SQAD versions**, decomposed as planned before the answers were generated, with yes and
no mapped. SQAD 3.2 full minus the released
200 is {{cz.planned.mapped.g1_range}} across configurations, established in {{cz.planned.mapped.g1_established}}
({{cz.planned.mapped.g1_established_sign}}). Where it is established, the questions the debias rule flags account for
{{cz.planned.mapped.rule_share_where_g1_established}} of it; across every configuration the share is
{{cz.planned.mapped.rule_share}}, because a gap near zero makes a share of it unstable. On the questions the rule passes the gap is established in {{cz.planned.mapped.g1_rule_established}}.
Between the two debiased sets the difference is {{cz.planned.mapped.g2_range}}, established in
{{cz.planned.mapped.g2_established}} ({{cz.planned.mapped.g2_established_sign}}). Changing only the passages
makes little difference: the same {{cz.cross.n}} questions, answered by the same models on each version's
passages, differ by {{cz.cross.mapped_range}}, established for {{cz.cross.mapped_established}} models; with this
many unadjusted intervals, about one excluding zero is what chance alone gives ([docs/CROSSOVER.md](docs/CROSSOVER.md)).

## How to reproduce

Requires Python 3.11 to 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/arytma/measurerag.git && cd measurerag
uv sync --extra dense --extra rag --extra dev
uv run measurerag doctor --fix      # macOS: one OpenMP runtime; re-run after any uv sync
```

Then the smoke test, on a small bundled synthetic fixture: no network, seconds,
a budget table. If it fails, nothing else in the kit will work.

```bash
# test
uv run measurerag smoke
```

A real run. `fetch` pre-downloads a dataset or the models, so the run itself can
be offline:

```bash
uv run measurerag fetch scifact && uv run measurerag fetch models
uv run measurerag run --config configs/en_small.yaml    # BEIR SciFact retrieval
uv run measurerag repro --config configs/en_small.yaml  # run twice, diff the manifests
```

Each run writes `results/<name>/`: quality beside cost as CSV and JSON, the
budget table, quality-against-memory and quality-against-latency curves with the
budget walls drawn in, and a `manifest.json` with the SHA-256 of every slice and
output. `repro` requires the metrics of two runs to match exactly (timings are
physics, not logic).

**The full experiment** is in [docs/REPRODUCE.md](docs/REPRODUCE.md), step by
step: rankings, the controlled dial, the timed sweeps, then the analysis
(`scripts/crossover.py` and `scripts/fill.py --all`, which regenerate
`results/crossover.json`, `results/stats.json` and every document from cached
data). **Hardware used:** one fanless laptop, {{machine.chip}},
{{machine.cores}} cores, {{machine.ram_gb}} GB of RAM, CPU only, macOS {{en_m3.budget.macos}} for every
Ministral 3 timing (every timed row records its OS version in [docs/BUDGETS.md](docs/BUDGETS.md)). The Czech answers, and the
English dials of Gemma 3n, Gemma 4 and Ministral 3, were generated on a second machine, an ASUS Ascent GX10
(NVIDIA GB10), with the generator on its GPU; no budget, latency or price comes from it ([docs/LIMITATIONS.md](docs/LIMITATIONS.md)).
**Expected run time:** the smoke test takes seconds. Of the full experiment,
Ministral 3's timed sweeps alone took {{en_m3.hygiene.wall_hours_accepted}} hours of
wall-clock, one batch at a time ({{en_m3.hygiene.n_wall_recorded}} batches); the dense rankings and the dial
generations come on top.

**Why `doctor` comes first.** On macOS the torch, faiss-cpu and scikit-learn
wheels each vendor their own `libomp.dylib`; with two OpenMP runtimes in one
process a dense retriever segfaults inside `__kmp_fork_barrier`, and it crashed
one of this kit's long runs four times. `doctor --fix` points every copy at
one file, and a real run refuses to start until it has. The kit does **not** set
`KMP_DUPLICATE_LIB_OK=TRUE`, which hides the diagnostic and keeps the second
runtime; libomp itself warns that duplicates can "cause incorrect results".

## What it measures

For every configuration (a retriever, optionally with a local generator) on
every dataset:

* **Quality:** nDCG@10, Recall@10, Recall@100; gold-in-context@k (did a relevant
  passage actually reach the generator?); end-to-end exact match and token F1.
* **Cost, measured rather than estimated:** peak resident memory of the whole
  process; index and model size on disk; retrieval latency p50/p95, one query at
  a time with encoding included; end-to-end latency p50/p95; generator
  throughput, prefill included.
* **Verdict:** fits or does not fit, for every budget in a
  `ram_gb × retrieval_ms × e2e_s` grid.

Each configuration runs in its own process, so it inherits no one else's
high-water mark, and latency is one query at a time, because that is what a
user waits for. Exact definitions are in [docs/METRICS.md](docs/METRICS.md).
There is no LLM-as-judge: it is not deterministic, and this kit promises
byte-identical reruns.

## Limitations

The most important; the full list is in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

* **Two English QA datasets; results are a guide, not a law.** The break-even
  bars are measured on HotpotQA and NQ with two generators of one family
  (Ministral 3 at 3B and 8B). The Czech appendix adds two SQAD versions and four
  more families, and is reported on its own.
* **The dial and the timings come from two machines.** Ministral 3's dial answered on the second
  machine's GPU and its natural sweeps on the laptop's CPU, so the check of the dial against real
  retrievers compares answers from two machines. On the same prompts the two machines' generator step
  differs by {{en_m3.hotpotqa.natural_k5_dial_own}} on HotpotQA and {{en_m3.nq.natural_k5_dial_own}} on NQ, an
  effect of the size of some of the differences the check reports
  ([docs/LIMITATIONS.md](docs/LIMITATIONS.md#the-czech-and-english-families)).
* **NQ's gold is a stated assumption, not an annotation.** A passage counts as
  gold only if BEIR judges it relevant and it contains an nq-open answer as a
  whole-token span; {{data.nq.mapped}} of {{data.nq.joined}} joined questions
  ({{data.nq.mapped_pct}}) have such a passage. At NQ's no-gold level
  {{en_m3.nq.leak_no_gold}} of the contexts still contain the answer in a non-gold
  passage, and on NQ {{en_m3.nq.established_text}}
  ([docs/METHOD.md](docs/METHOD.md#stated-assumption-the-nq-gold-mapping)).
* **The timed batches ran without the full measurement hygiene.**
  {{en_m3.hygiene.batches_ungated}} of the {{en_m3.hygiene.batches_accepted}} accepted timed
  batches ran without the calibration-probe gate and without rejection for
  contention, which the author switched off after two attempts at the first configuration under
  the gate were set aside (one measured against a baseline taken on a warm machine in use, one
  rejected for contention). Contention was still recorded: under the switched-off rule
  {{en_m3.hygiene.would_reject}} accepted batches would have been rejected, and batches more than a quarter
  contended: {{en_m3.hygiene.heavily_contended}}. Three HotpotQA configurations first timed in one night
  ({{en_m3.hygiene.night_utc}}), picked out after the run by their latencies, were timed again, one at a time on
  the same machine and macOS, from {{en_m3.hygiene.retimed_utc}}: {{en_m3.hygiene.retimed_list}} (peak RSS as measured, model file memory-mapped (library default);
  estimate without the double-counted mapped pages (probe; see [docs/LIMITATIONS.md](docs/LIMITATIONS.md#the-machine-and-the-timings)): {{en_m3.est.hygiene.retimed_peaks}}).
  The author chose to time them again after short probes had been read. The documents use the second timings,
  the first stay in the record, and both priced HotpotQA steps use these rows, in gigabytes and in milliseconds. Rows vary from batch to batch too, so
  every latency price carries this caveat
  ([docs/LIMITATIONS.md](docs/LIMITATIONS.md#the-machine-and-the-timings)). Re-measuring changes no EM
  number and no g\* in raw EM (on one machine and in one order, generation is deterministic); it changes
  timings and peak memory, and with them the prices in milliseconds and in gigabytes, as the three rows timed
  again show.
* **One machine for the timings.** Every English timing (Ministral 3's) comes from one fanless laptop, on
  one macOS version ({{en_m3.budget.macos}}), each row labelled in [docs/BUDGETS.md](docs/BUDGETS.md); the SQAD
  retrievers' timed rows ran on macOS {{czr.cost.macos}} with the hygiene on, and CsFEVER's on an earlier
  version.
* **Corpus slices, not full corpora.** HotpotQA is searched over {{data.hotpotqa.docs}} passages and NQ over
  {{data.nq.docs}}: every judged passage of a sampled question and a seeded sample of the rest. Real retrievers
  face far more distractors in the full collections, so g, EM, index sizes and latencies are these slices'
  ([docs/LIMITATIONS.md](docs/LIMITATIONS.md#the-corpora-and-the-samples)).
* **Exact match only.** EM is strict and is the only bootstrapped quality
  measure; F1 is reported beside it but not bootstrapped.
* **The e5 encoders have seen the training sets.** Their model cards list the
  NQ and HotpotQA training sets among their fine-tuning data; this kit
  evaluates on other splits, but they may look better on these two datasets
  than on unseen ones.

## Documentation

| document | what it holds |
|---|---|
| [README.cs.md](README.cs.md) | a one-page summary in Czech |
| [docs/CROSSOVER.md](docs/CROSSOVER.md) | the dial experiment: definitions, every table, the prices, the check against real retrievers, the Czech appendix |
| [docs/METHOD.md](docs/METHOD.md) | how the numbers are produced, from slice to bootstrap |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | the commands that reproduce every result |
| [docs/BUDGETS.md](docs/BUDGETS.md) | the English budget tables, timing hygiene, and the OS version of every English timed row |
| [docs/DATASETS_AND_LICENCES.md](docs/DATASETS_AND_LICENCES.md) | every model and dataset: source, revision, licence, restrictions |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | what the results do not show |
| [docs/METRICS.md](docs/METRICS.md) | the definition of every metric and cost |
| [CHANGELOG.md](CHANGELOG.md) | what each release contains |
| [THIRD_PARTY.md](THIRD_PARTY.md) | every dataset and model, with its licence and citation |
| [NOTICE](NOTICE) | the copyright and attribution notice |
| [CITATION.cff](CITATION.cff) | how to cite this work |

File and folder names that contain v020, v030, v040 or first_cut are build labels from before the first public release.

## Extending it

**A retriever** is a class with three methods and one registry line. The runner,
the budget bench and the plots pick it up.

```python
# src/measurerag/retrievers/mine.py
class MyRetriever:
    def __init__(self, **kwargs):
        self.name = "mine"

    def build(self, docs, workdir) -> BuildStats:
        """Index `docs` under `workdir`; index size on disk is measured."""

    def search_one(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """ONE query, encoding included: the latency numbers come from here."""

    def close(self) -> None: ...

# src/measurerag/retrievers/__init__.py
_REGISTRY["mine"] = ("measurerag.retrievers.mine", "MyRetriever")
```

In a config, list it under `retrievers:` as `{id: mine, kind: mine, some_param: 3}`;
everything after `kind` goes to `__init__` and into the manifest. A new *dense
encoder* needs only a line in `MODELS` in `retrievers/dense.py`. Get the
instruction prefixes right: e5 wants `query: ` and `passage: `, and silently
omitting them costs several nDCG points.

**A dataset** is a module exposing `load(cache_dir, **kwargs) -> Dataset` and one
line in `_REGISTRY` in `datasets/__init__.py`.

```python
from .base import Dataset, Doc, DatasetBlocked, DatasetUnavailable

def load(cache_dir=None, max_docs=None, max_queries=None, **_) -> Dataset:
    return Dataset(
        name="mine", language="cs",
        docs=[Doc(doc_id="d1", text="...", title="...")],
        queries={"q1": "..."},
        qrels={"q1": {"d1": 1}},
        answers={"q1": ["gold"]},      # only if you have gold answers
        license="...",                  # required: it ends up in the manifest
        source="...",
        notes=["anything a reader must know before citing a number from this"],
    )
```

Two rules are not optional. If the data is gated behind a login, a form or an
email, raise `DatasetBlocked` with instructions (as the CWRCzech loader does);
do not work around the gate. Cap a large corpus *while streaming*
(`datasets/csfever.py`); caps keep every judged-relevant document of a kept query.

**A budget** is a cross product in the config:

```yaml
budgets:
  ram_gb: [2, 4, 8, 16]
  retrieval_ms: [50, 200, 500]
  e2e_s: [3, 8]          # only used when the RAG stage is enabled
```

For a shape the grid cannot express (a disk ceiling, a throughput floor), add a
field to `Budget` and a check to `check()` in `budget.py`; the table and the
curves read the verdicts, so they follow.

## Licences

* **Code: Apache License 2.0** ([LICENSE](LICENSE)). It covers `src/`,
  `scripts/`, `tests/`, `configs/`, `fixtures/` and the build files
  (`pyproject.toml`, `uv.lock`).
* **Documentation and results: CC BY 4.0** (Creative Commons Attribution
  International; [LICENSE-docs](LICENSE-docs)). It covers the Markdown documents and their
  sources under `templates/`, and everything under `results/`: the numbers,
  tables and figures.
* **The generated Czech answers** (`answers/`): files under `answers/sqad32/` are licensed as a whole under
  CC BY-SA 3.0, the licence of SQAD 3.2; files under `answers/sqad30_copy/` under CC BY 4.0, except the spans
  of SQAD 3.0 an answer quotes, which keep LGPL-3.0. The answers quote SQAD, up to {{cz.quotes.longest}}
  characters of a passage, and some repeat their own question ([THIRD_PARTY.md](THIRD_PARTY.md)).
* **Third-party models and datasets keep their own licences**, and none is
  redistributed here: scripts download them when a run starts, except SQAD 3.2 and the SQAD 3.0 archive,
  which you download from LINDAT. Each one, with its licence and citation, is in
  [THIRD_PARTY.md](THIRD_PARTY.md). Two need attention: Gemma 3n and EmbeddingGemma are under Google's Gemma
  Terms of Use, with its Prohibited Use Policy, and Qwen2.5-3B-Instruct is under the Qwen Research License
  (non-commercial research and evaluation use only).

## How to cite

Use [CITATION.cff](CITATION.cff). In text:

> Laznicek, P. (2026). *measureRAG: a budget-aware retrieval and RAG benchmark
> in Czech and English, for a single machine with no network access* (version
> 0.4.5) [Computer software]. https://github.com/arytma/measurerag
