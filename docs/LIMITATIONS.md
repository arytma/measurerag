<!-- Generated from templates/docs/LIMITATIONS.md and results/stats.json by scripts/fill.py. Edit the template, not this file. -->
# Limitations: what to discount for

Everything below makes a result in this kit narrower, noisier or less general
than it might look. Each item says what the limitation is, what it does to the
numbers, and where it is discussed. The items are grouped, not ranked.

---

## The machine and the timings

**One machine.** Every memory figure and every timing comes from one
Apple M4 MacBook Air with 32 GB of memory. Nothing was
measured on another laptop, on a server CPU, on Linux or on a GPU. Peak RSS
depends on the allocator and the runtime as well as on the model and the index.
Latency depends on the core layout, the memory bandwidth and the thread count.
Both columns can therefore rank configurations differently on other hardware.
→ [BUDGETS.md](BUDGETS.md), §1.

**CPU only, for everything timed.** The generator runs on the CPU, with no layers offloaded to the
Apple GPU (every timed configuration sets `device: cpu`), using llama.cpp's default
thread count. A deployment on the same machine with GPU offload would see
different generation latencies, possibly in a different order.
→ `src/measurerag/rag/generator.py`.

**The Czech answers come from a second machine.** Every Czech answer behind the reported SQAD sets was generated on an ASUS
Ascent GX10 (NVIDIA GB10, Linux), with every layer of the generator on its GPU,
because an answer there took 2.1% of the Mac's time (0.15 s
against 7.0 s with 1.5B). Retrieval and every ranking stayed the Mac's. Two
machines and software stacks give different answers to the same prompt: on the
200 dev questions, with the frozen prompt, 169 of 200 answers of 1.5B and
179 of 200 of 3B were the same string on both, and on the English dial's own prompts Ministral 3's 🇫🇷
generator step differs between the two machines by -0.018 [-0.032, -0.004] on HotpotQA, an effect of the
size of the differences the documents compare ([CROSSOVER.md](CROSSOVER.md), §3). The dev set's choice of the prompt, and the
README's comparison of the two prompts, were answered on the Mac, each comparison on that one machine. One check
of the move failed. Of the bench's 6,585 rankings
(each question once per retriever), made again on the GX10, 2 differ from the
Mac's, each only at rank 10, between passages whose scores agree to six decimals.
No top 3 changed, so no prompt did, and no nDCG@10 moved. No budget, latency or price comes from the GX10:
its seconds per answer appear only in the dev benchmark, as a speed comparison, and the timings in
`results/cz_rag*/` are its own (Linux, `device: cuda`), so their budget verdicts do not apply to a CPU-only
machine. Every English
timing and CsFEVER's numbers are the Mac's. The English dials of Gemma 3n 🇺🇸, Gemma 4 🇺🇸 and Ministral 3 🇫🇷 were
generated on the GX10's GPU, and Ministral 3's 🇫🇷 dial is checked against its natural sweeps, timed on the Mac, across
the two machines ([below](#the-czech-and-english-families)); Qwen2.5's 🇨🇳 dial was generated on the Mac's CPU, so the
dials compare across two machines as well as families.

**A fanless laptop, heating under sustained load.** A long batch warms the
machine by itself. A configuration with a long index build or a slower generator
spends more of its run on a warmer machine than a short one does. With the hygiene
off, no probe ran around a batch; the chip's temperature at each batch's start and
end is in the batch record. → [BUDGETS.md](BUDGETS.md), §4.

**A laptop in use.** The machine was not dedicated to the benchmark. Other
processes were sampled every 30 s. While the hygiene rules were
on, a batch with too many contended samples was rejected and rerun
(1 rejected: contention: 1).
Contention that starts and ends between two samples can be missed.
→ [BUDGETS.md](BUDGETS.md), §4.

**Every timed batch ran with the hygiene switched off.** After the first
configuration's attempts under the gate were set aside, the author switched off the
calibration-probe gate and the contention rejection, and all
20 accepted batches ran one at a time, their contention
still recorded: 17 back to back, from 03:03 UTC on 12 Nov to 06:39 UTC on 14 Nov 2026, and the
three timed again in a later sitting (below). A batch may have started on a warmer or busier machine, so its
latencies, and any per-ms price that uses them, are less certain than gated ones
would be. Answers and EM are unaffected. Under the switched-off rule,
4 of 20 accepted batches would have been rejected for contention:
HotpotQA e5-base+3b (37 of 228); HotpotQA tfidf-svd+8b (18 of 252); NQ e5-small+3b (11 of 178); NQ bge-m3+8b (52 of 648). Among them are the lower row of HotpotQA's
generator step (e5-base with 3B) and the upper rung of NQ's priced retriever step
(e5-small with 3B). Batches in which more than a quarter of the samples
were contended: none. → [BUDGETS.md](BUDGETS.md), §4.

**Three rows first timed in one night were timed again.** Three HotpotQA configurations
were first timed in the same night, from 04:58 to 15:06 UTC on 12 Nov 2026: e5-base + 8B, bm25 + 3B and bm25 + 8B,
picked out after the run by their latencies; no recorded condition marks that span. After short probes had been
read, the author chose to time them again, by a rule fixed before the new batches ran, and the new batches ran from
21:28 UTC on 14 Nov to 07:41 UTC on 15 Nov 2026, one at a time on the same Mac and macOS: HotpotQA e5-base+8b (end-to-end p50 24.45 s when first timed, 19.58 s when timed again; peak RSS 11.80 GB, then 12.57 GB); HotpotQA bm25+3b (end-to-end p50 14.86 s when first timed, 11.07 s when timed again; peak RSS 5.65 GB, then 5.66 GB); HotpotQA bm25+8b (end-to-end p50 28.23 s when first timed, 23.75 s when timed again; peak RSS 10.77 GB, then 10.76 GB) (peak RSS as measured, model file memory-mapped (library default);
estimate without the double-counted mapped pages (probe; see [below](#the-machine-and-the-timings)):
HotpotQA e5-base+8b 8.10 GB, then 8.87 GB; HotpotQA bm25+3b 4.29 GB, then 4.29 GB; HotpotQA bm25+8b 7.07 GB, then 7.06 GB). All three
answered faster the second time. That does not show that the night ran slower: no row from outside the night was
timed again, rows picked out because they were slow tend to come out faster when timed again, and two of the three
second timings started on a cooler chip than most batches of the first sitting. With its second timing, bm25 + 3B is
still the slowest 3B configuration on HotpotQA. The documents use the second timings; the first stay in the record.
The rule asked for no other application to be open; another application, open during the second timings at the
author's request, departed from it. Both priced HotpotQA steps use these rows, in
gigabytes and in milliseconds: the retriever step's lower rung is bm25 + 3B, and the generator step's upper row, where
it is priced in front of e5-base, is e5-base + 8B. Each sets that row against one timed once, in the first sitting.
Rows vary from run to run too, so read every latency and memory price with this in view. Their EM, and every g\* in
raw EM, do not depend on it.
→ [BUDGETS.md](BUDGETS.md), §4.

**Timings on one macOS version.** Every English Ministral 3 🇫🇷 row, the timed sweeps of
[BUDGETS.md](BUDGETS.md) §2, ran on macOS 27.2.1, so any two of them are comparable as far as
the operating system goes. → [BUDGETS.md](BUDGETS.md), §2.

**The timed retrievers on SQAD: one later macOS, against an older probe baseline.** The ten retrievers'
memory and latency rows ran on macOS 27.2 with the hygiene on, one at a time. The probe's baseline was taken on macOS
27.0.1. On the newer system a cool, idle Mac probes more than 10% faster than that
baseline, so the gate held rows on an idle Mac as it holds them on a hot one, and each accepted row started
when the probe was back within 10%, about ten minutes after the previous row ended: the accepted rows'
probes drifted -9.8% to +9.3%. Rows that end before the first contention sample, at
30 s, have none: BM25 over lemmas, BM25 without diacritics and BM25. Compare these rows with each other, not with the
English tables, which ran with other corpora and without the gate.
→ [CROSSOVER.md](CROSSOVER.md#ten-retrievers-on-sqad).

**EM depends on more than whether gold is present.** The dial moves only the
gold; a real retriever change also changes which non-gold passages the
generator reads. The natural sweeps disagree with the dial on HotpotQA at
the generator step at bm25 (g 0.984): measured minus predicted -0.052 [-0.090, -0.013]; the generator step at e5-small (g 0.986): measured minus predicted -0.038 [-0.074, -0.002]; the generator step at e5-base (g 0.992): measured minus predicted -0.018 [-0.032, -0.004], and on NQ at tfidf-svd → bm25 (g 0.060 → 0.733): measured minus predicted +0.067 [+0.015, +0.120]. The
break-even gold gain assumes EM follows g, so it is a guide, not a law.
→ [CROSSOVER.md](CROSSOVER.md), §3.

**The dial leaks on NQ.** 27% of NQ's no-gold contexts still
contain an answer in a non-gold passage (HotpotQA: 5%),
so taking the gold away removes less than it seems and the dial understates
what gold is worth there. The only level where NQ's difference between the two
moves is established is that no-gold level.
→ [CROSSOVER.md](CROSSOVER.md), §4.

**The per-GB result rests on the generator step's price.** That price is a
difference of two process-wide peaks, and it depends on the retriever in front of
the generator. The per-GB crossover under every rung's price is in CROSSOVER.md.
As measured, under none of them is there a crossing. With the estimate without the double-counted mapped pages (below),
0 of 5 rungs cross on HotpotQA and 1 of 5 on NQ:
bge-m3, where the step costs 0.72 GB: g* = 0.51 [0.23, 0.82], the upper end censored; 1239 of 10000 resamples censored. That NQ step is the smallest of the ten as measured too, because NQ's bge-m3 + 3B
row peaked higher, against its 8B row, than any other pair; the per-GB answer at that rung rests on that one row.
→ [CROSSOVER.md](CROSSOVER.md), §3.

**Per-unit prices come from k = 5 only, from one priced step, as point values.**
Every per-GB and per-ms price is a difference between two rows of a k = 5 table.
The dial's synthetic retriever step is priced at the real step bm25 → e5-small;
other steps appear only as a sensitivity. Each cost is the difference of
two single runs and is not resampled. The intervals on the per-GB and per-ms g\*
carry question-sampling uncertainty only, not run-to-run variation in memory or
latency. → [BUDGETS.md](BUDGETS.md), §3.

**"End-to-end" latency does not include retrieval.** The kit's end-to-end clock
covers building the prompt from already-retrieved passages and generating the
answer. Retrieval is timed separately. A user waits for both, so the per-ms
price adds the two p50 changes: a sum of medians, not the median of a sum,
because the two are timed in separate passes. The budget verdicts' end-to-end
ceilings use the end-to-end p95 alone. All latencies are
single-query service latencies, one question at a time, with no concurrent
load. None of them is throughput. → [BUDGETS.md](BUDGETS.md), §1;
[METRICS.md](METRICS.md).

**Peak RSS includes building the index, and its "GB" is binary.** Each
configuration builds its own index inside the measured process, so a
deployment that loads a prebuilt index may peak differently. Peak RSS is in
units of `1024**3` bytes, while model and index sizes are decimal. tfidf-svd's
vocabulary and SVD components are in its peak RSS but in neither size column.
→ [BUDGETS.md](BUDGETS.md), §1.

**Peak RSS counts part of each English generator's weights twice.** llama-cpp-python maps a model file into memory by
default (`use_mmap`), and every timed configuration loaded its generator so. On the CPU, llama.cpp also copies most of
the weights into its own buffers, and with mapping on the file's pages stay resident beside that copy, so the peak RSS
of the whole process counts that part of the weights twice. A probe on the same Mac, `scripts/mmap_probe.py`, measured
by how much, from 16:19 to 16:32 UTC on 15 Nov 2026 on macOS 27.2.1 with llama-cpp-python 0.3.35: it loaded Ministral 3 3B 🇫🇷
and 8B alone, through the bench's own generator class and peak-RSS sampler, 5 times each with mapping on and
5 times off, each load in its own process, and answered one prompt of the bench's length each time. The median
peak RSS was 4.76 GB with mapping on and 3.40 GB off for 3B, and 9.87 GB and
6.17 GB for 8B, with 2.0G and 4.4G of mapped file resident with mapping on, as vmmap
prints it. The surplus, the median of each repeat's on-minus-off difference, is 1.36 GB for 3B (the
5 differences: all 1.361 GB) and 3.70 GB for 8B (3.696 to 3.701 GB). In the probe
every answer was the same string with mapping on and off, and the median answer took 23.2 s on and 21.9 s off with 3B,
45.9 s and 46.5 s with 8B. The Mac was not set aside for the probe; a process's peak RSS counts only its own pages, and at least
17.0 GB of memory was available before every load. The record, with the machine's load and free memory
before the first load and after the last, is `results/crossover/mmap_probe.jsonl`.

Every figure that rests on the English Ministral 3 🇫🇷 rows' peak RSS is therefore given twice: as measured, model file memory-mapped (library default), and beside it as an
estimate without the double-counted mapped pages (probe; see this section), each row's measured peak less its
generator's surplus, with everything derived from it computed again by `scripts/crossover.py`
(`results/crossover/mmap_estimate.json`). The sweeps' own records (each `results.json`, `results.csv`,
`results_table.txt` and the curve of quality against memory) keep the measured peaks. The estimate assumes a row peaked while its generator was loaded; the index
is built first, in the same process. In 20 of 20 rows the estimate is above the whole peak of the
same retriever's row with Qwen2.5 1.5B 🇨🇳 on the same slice, which bounds that build from above, so the build did not set
the estimated peak. The probe answered one prompt, and a sweep's row hundreds of questions: if a row read more of the
mapped file than one answer does, its surplus was larger and its estimate is a little high, but after one answer vmmap
already counted 2.0G of 2.1G and 4.4G of 4.5G of mapped file resident. What changes. The generator step costs 2.73 GB on HotpotQA and
2.77 GB on NQ in the estimate, against 5.07 GB and 5.11 GB as
measured, so the per-GB break-even bar is about 1.86 and 1.84 times as
large. The rows' peak memory runs from 4.3 to 9.6 GB in the estimate, against
5.7 to 13.3 GB as measured. The three rows timed again peaked, in the estimate,
HotpotQA e5-base+8b 8.10 GB, then 8.87 GB; HotpotQA bm25+3b 4.29 GB, then 4.29 GB; HotpotQA bm25+8b 7.07 GB, then 7.06 GB (first timing, then second). With the generator step priced in front of each
rung in turn, the per-GB fit crosses at 0 of 5 rungs on HotpotQA and
1 of 5 on NQ in the estimate (bge-m3, where the step costs 0.72 GB: g* = 0.51 [0.23, 0.82], the upper end censored; 1239 of 10000 resamples censored), against
0 of 5 and 0 of 5 as measured (above). What holds either way:

* **Per GB, no crossing, at the priced step.** With the costs estimated without the double-counted mapped pages, the
  per-GB fit in front of e5-base reads, on HotpotQA, no crossing inside g = 0.00 to 0.90 (censored above 0.90; 9999 of 10000 resamples have no root), and on NQ,
  no crossing inside g = 0.00 to 0.81 (censored above 0.81; 9903 of 10000 resamples have no root).
* **Every budget verdict.** Latency decides them: every row's end-to-end p95, 9.8 s to
  45.9 s, is above the grid's end-to-end ceilings (3 s and 8 s), so no row fits a budget
  either way (0 of 20 as measured, 0 of 20 in the estimate). By memory alone,
  rows under 8 GB: 8 of 20 as measured, 12 of 20 in the estimate; under
  4 GB: 0 of 20 and 0 of 20.
* **EM and latency.** The estimate changes no answer and no timing: the timings are as measured, with mapping on,
  and the probe's answers were the same string with mapping off.
* **The retriever step's GB**, 0.95 GB on HotpotQA and 0.83 GB on NQ: its two rows
  load the same generator, so the surplus cancels.
* **Every Czech number.** The Czech answers were generated on the GX10's GPU and no Czech generator's peak memory is
  reported; the SQAD retrievers' memory rows load no generator.

The next version measures the English Ministral 3 🇫🇷 rows again with mapping off and replaces these estimates.
→ [BUDGETS.md](BUDGETS.md), §1 and §2; [CROSSOVER.md](CROSSOVER.md), §3 and §4.

---

## The corpora and the samples

**Two English QA datasets; results are a guide, not a law.** The break-even
bars and every g\* statement come from HotpotQA and Natural Questions, with
two generators of one family. They say what happened on these two datasets,
and they are a guide for similar settings, not a constant to apply elsewhere.

**Corpus slices, not full corpora.** HotpotQA is searched over
100,000 passages and NQ over 100,000. Every judged
passage of a kept question is forced into the slice, and the rest is a seeded
sample. A retriever searching the full collection faces far more
distractors. The g and EM values here are for these slices, and index sizes and
retrieval latencies grow with the corpus, at different rates for different
retrievers. → [DATASETS_AND_LICENCES.md](DATASETS_AND_LICENCES.md).

**Sample sizes.** 500 HotpotQA questions and 300 NQ questions.
Every difference is a paired bootstrap over questions (10000 resamples)
with a 95% percentile interval, and at these sizes the intervals are wide. NQ's
g\* is 0.56 [0.29, 0.83], the upper end censored; 1847 of 10000 resamples censored. On HotpotQA, the retriever step's advantage is
established at 1 of 5 dial levels and the
generator step's at 0 of 5. On NQ, the counts are
1 of 5 and 0 of 5. → [CROSSOVER.md](CROSSOVER.md), §2.

**No p-values, by design, and no multiplicity correction.** "Established" means
only that a 95% interval excludes zero. Many intervals are reported (one per dial
level, per dataset, per check), and none is adjusted for how many there are.
Some may exclude zero by chance alone. The same holds for the Czech comparisons, which
report far more intervals (per set, family, model, step and retriever), none adjusted: a
Czech difference established on one set can be chance, and one established on several sets,
which share many questions, is firmer but not independent evidence.

**Censored intervals.** A bootstrap resample whose curve never crosses zero
inside the observed range of g is censored at the edge it fell off, not dropped.
The count is always reported. An interval with many censored resamples is
a bound, not an estimate. HotpotQA's headline: g* = 0.39 [0.16, 0.89], the upper end censored; 270 of 10000 resamples have no root.

**The NQ sample is a selected subset.** Questions are drawn only from those
that join to nq-open (2,255 of 3,452,
65.3%) and have a mapped gold passage (2,083 of
those, 92.4%). A question qualifies only if some judged passage
contains an nq-open answer as a whole-token span after normalisation, so the
sample leans towards questions whose evidence states the answer in the
answer's own words.

---

## What g\* and the dial can say

**g\* is conditional on step size.** On the dial, one retriever step is one
level, a change of 0.2 in the removal probability p. A
bigger step buys more EM and moves the crossing:

| a retriever step worth | g\* on HotpotQA [95%] | g\* on NQ [95%] |
|---|---|---|
| one dial level (the headline) | 0.39 [0.16, 0.89], the upper end censored; 270 of 10000 resamples censored | 0.56 [0.29, 0.83], the upper end censored; 1847 of 10000 resamples censored |
| two dial levels | no crossing in range | no crossing in range |

The step-free form is the **break-even gold gain**: how much gold-in-context a
retriever upgrade must add, from a given operating point, to match the
3B → 8B upgrade. Measure what your upgrade adds and compare it with
that. → [CROSSOVER.md](CROSSOVER.md), §1, §3 and §4.

**No crossing can be located outside the dial's range.** The crossing is sought
over the operating points a dial step can start from: g from 0.00 to 0.90
on HotpotQA and from 0.00 to 0.81 on NQ. The dial's top (p = 0) is its own
retriever's top k, at g = 0.992 (HotpotQA) and 0.960 (NQ).
No dial step starts above the range, so where the retriever wins at every level
the result is censored at the top, not located.

**The dial is a model of retrieval quality, not a retriever.** It holds the
corpus, k, the prompt and the amount of text fixed, and changes only whether the
gold passage is there. A real upgrade also changes which non-gold passages
appear, in what order, and how long they are (and with that, the generator's
prefill time). The natural-sweep checks are the evidence on how much that
matters. They are only as good as the range of g that real retrievers reach. On
HotpotQA at k = 5, bm25, e5-small, e5-base and bge-m3 all sit near the top of the g
axis. e5-base's top k is the dial's top, 0.992. The only real
retriever low on the axis is tfidf-svd, the instrument rung, which is not a
deployment candidate. The ladder leaves the middle of the axis empty, so the dial is
checked against reality mainly at its two ends. Where each rung lands is in the
natural-sweep tables of [CROSSOVER.md](CROSSOVER.md), §3 and §4.

**Both moves start from the 3B generator.** Δ_ret is a retriever step with the
3B model, and Δ_gen is the 3B → 8B upgrade with retrieval held fixed. A
retriever step taken with the 8B model, or both upgrades together, is not what
the crossover measures.

---

## Models and metrics

**Two generators of one family.** Ministral 3 3B 🇫🇷 and 8B (Instruct 2512),
both Q4_K_M, greedy decoding, one prompt template, one seed (20260101).
Nothing here says where a smaller or a larger model would sit. Nothing here covers
another model family, another quantisation, sampled decoding or a prompt tuned per
model. (Gemma 3n 🇺🇸, Gemma 4 🇺🇸 and Qwen2.5 🇨🇳 answer the English dial too, untimed in the
documents; Qwen2.5's 🇨🇳 English sweeps are timed and shipped but not tabulated, and Qwen3.5 🇨🇳
has no English result, [below](#the-czech-and-english-families); the Czech appendix
compares five families.)
→ [CROSSOVER.md](CROSSOVER.md), §6.

**max_tokens = 64.** Every answer is cut at 64 tokens. The
prompt asks for a short answer, but an answer that needed more would be cut off
and could be scored wrong. No other limit was compared for Ministral 3 🇫🇷. How many English answers reached the
cap, and how many declined to answer, is not recorded.

**Exact match only is bootstrapped.** F1 is reported beside EM but has no
interval. EM is strict: a correct answer wrapped in a sentence, or phrased
differently from every gold answer, scores zero ([METRICS.md](METRICS.md)).
Part of any EM difference between the two generators can be phrasing, not
knowledge.

**bge-m3 is scored dense-only**. The kit indexes one vector per passage,
so bge-m3 contributes only its dense head. Its published numbers combine dense,
sparse and multi-vector scoring. A multi-vector index would change its memory
as much as its quality. Its rows here are fair for a single-vector deployment,
and should not be quoted as "bge-m3" without that qualifier.

**The multilingual-e5 encoders were trained on these datasets' training sets.**
Both multilingual-e5 model cards list the Natural Questions and HotpotQA
training sets among their fine-tuning data. This kit evaluates on other splits,
but a dense retriever that has seen the training distribution may look better
on these two datasets than it would on unseen ones. That affects the e5 rungs of
the natural sweep and the dial, which is built on e5-base's ranking.
→ [DATASETS_AND_LICENCES.md](DATASETS_AND_LICENCES.md).

---

## The datasets

**HotpotQA is multi-hop.** Each question has two gold passages. "Any gold" (the
primary g) counts a question as covered when one of them arrives, so it is a
floor on the evidence, not full coverage. g on the all-gold axis is reported
beside it. → [CROSSOVER.md](CROSSOVER.md), §1.

**g does not mean quite the same thing on the two datasets.** HotpotQA's
500 questions have 1,000 gold passages
between them. NQ's 300 have 348, mostly one each.
Losing one gold passage on HotpotQA can leave the other, and on NQ it usually
leaves none.

**NQ's gold passages are mapped, not annotated: a stated assumption**. BEIR's NQ ships
judgements without answers, and nq-open ships answers without passages. A
passage is gold only if BEIR judges it relevant **and** its text contains an
nq-open answer as a whole-token span. A judged passage without the answer
string is not gold, and neither is an unjudged passage that contains it.
Coverage: 2,083 of 2,255 joined questions
(92.4%). The rule leaks: at NQ's no-gold level
27% of the contexts still contain the answer in a non-gold
passage, and that level is the only one where NQ's difference between the two moves
is established. The rule is stated as an
assumption in [METHOD.md](METHOD.md#stated-assumption-the-nq-gold-mapping) and
in the README; a different rule would give different NQ numbers.

**The Czech results are separate from the English result.** The Czech results in the README and the
appendix of [CROSSOVER.md](CROSSOVER.md) are in no headline, verdict or agreement statement. Read them with these
discounts, and with the order in which the families were added ([below](#the-czech-and-english-families)):

* **SQAD's non-gold passages leak answers**. With every gold passage removed, 36% of the
  contexts of the SQAD 3.0 copy's released 200 questions still contain the answer in a non-gold
  passage, against 5% on HotpotQA. Removing the gold removes less information than the
  dial assumes, which flattens Δ_ret.
* **Czech exact match mostly under-counts, but is not strictly a lower bound.** Normalisation keeps diacritics
  and does not lemmatise, so an inflected form of the right answer is scored wrong. Two of its steps can also
  credit a wrong answer: dropping English articles drops the Czech conjunction "a", and removing punctuation
  joins a decimal comma, so `3,5` matches `35` ([METRICS.md](METRICS.md#normalisation)). Gemma 4 🇺🇸 answers in Czech, and its under-count is
  1.5 to 3.6 points of EM across the four sets and configurations, nearly all declension by the proxy below.
  Gemma 3n's 🇺🇸 is 1.0 to 4.3 points. Ministral 3 🇫🇷 answers in Czech too: its under-count is
  1.5 to 3.7 points, of which yes and no in English cost 0.0. Qwen3.5's 🇨🇳 is
  1.5 to 3.5 points. The prompt asks Czech questions for ano or ne, and SQAD's gold is ano
  or ne, but Qwen2.5 🇨🇳 does not always follow that: its under-count is 4.0 to 12.3 points, of which yes
  and no in English cost 2.0 to 10.1 and declension 0.5 to 2.8.
  These are estimates, not bounds. The declension part is the match on the first five characters minus EM, and
  that proxy misses inflections that change a word within its first five characters, short words among them
  (Prahy against Praha), and credits different words that share them (Československá against Československo).
  EM with yes and no mapped, the primary Czech metric, removes the yes-and-no part; the labelled secondary
  metrics are in
  [CROSSOVER.md](CROSSOVER.md).
* **The fix moved a comparison, not only a level.** Qwen2.5 3B 🇨🇳 follows the ano-or-ne line more often than
  1.5B, so the fix raised Qwen2.5's 🇨🇳 generator step in EM, and much less with yes and no mapped. Read the two
  together ([README.md](../README.md#czech)).
* **The families are paired by place, not by size.** Gemma 3n E2B 🇺🇸, Gemma 4 E2B 🇺🇸, Ministral 3 3B 🇫🇷 and Qwen3.5 2B 🇨🇳
  are set against Qwen2.5 1.5B 🇨🇳, and E4B, 8B and 4B against 3B. The files differ. Gemma 3n E2B's 🇺🇸 file size is
  recorded in `results/cz_rag/results.json`, and E4B's is not recorded in this repository. At Q4_K_M the smaller
  models' files are 3.1 GB (Gemma 4 E2B 🇺🇸), 2.1 GB
  (Ministral 3B 🇫🇷), 1.1 GB (1.5B) and 1.3 GB (Qwen3.5 2B 🇨🇳). The larger
  models' are 5.0 GB (Gemma 4 E4B 🇺🇸), 5.2 GB (Ministral 8B 🇫🇷),
  2.1 GB (3B) and 2.7 GB (Qwen3.5 4B 🇨🇳). Gemma 3n 🇺🇸 is older than Gemma 4 🇺🇸,
  Ministral 3 🇫🇷 and Qwen3.5 🇨🇳, and so is Qwen2.5 🇨🇳. Three makers in three countries are no sample of a region's models. A difference
  between families is not a price per gigabyte, and no Czech generation is timed. It compares five published
  pairs of models on Czech at one prompt and one quantisation, not one maker's or one region's models against
  another's.
* **SQAD 3.0 here means a filtered copy.** The numbers come from the Hugging Face copy
  `Pehy/cs_sqad-3.0`, pinned at revision `f9e71384a8f7285412360cecabb7a852f49adbb9`. It
  leaves out most records whose answer sentence opens the article, so they are not the
  numbers of the whole of SQAD 3.0. The counts are in
  [DATASETS_AND_LICENCES.md](DATASETS_AND_LICENCES.md).
* **EM levels do not compare across SQAD versions.** SQAD 3.2 full scores above the released 200. With yes
  and no mapped, where that gap is established the questions the debias rule flags account for
  29% to 55% of it, and between the two debiased sets a gap is
  established in 2 of 15
  configurations ([CROSSOVER.md](CROSSOVER.md)). Compare differences within a set.
* **The debiased sets are not independent of the full sets.** They share many of their questions with them,
  so agreement between a full set and its debiased set is weak evidence.
* **An answer can depend on the prompt before it in the same process.** Near a tie, the model's first word can
  turn on what it answered just before, probably through llama.cpp reusing part of a shared prompt; that route
  was not isolated. Running the same questions in the same order gives the same strings, and the bench and the
  dial walk the questions in the same order. A different order can change a rare answer. Resetting the
  model's state between prompts is left for a future release.
* **The released 200 is small.** It can detect a retriever step of 0.074
  at 80% power, against 0.023 on SQAD 3.2 full, so it establishes
  little on its own. Every Czech set is read at k = 3, not at the English dial's k.

The author decided to publish the Czech numbers with the per-question records behind them. SQAD 3.0 is under LGPL-3.0 on LINDAT (http://hdl.handle.net/11234/1-3069) and SQAD 3.2
under CC BY-SA 3.0 (http://hdl.handle.net/11234/1-5019; [THIRD_PARTY.md](../THIRD_PARTY.md)). The records hold ids
and scores, no SQAD question, context or document text, and the generated answers, which quote SQAD: the longest
extract is 356 characters, and 15 answers repeat their own question
([THIRD_PARTY.md](../THIRD_PARTY.md)).

**The diacritics finding is CsFEVER's.** BM25 without diacritics leads on CsFEVER v2, which is
entity-heavy. The CsFEVER numbers are means with no interval, because no per-claim score ships, so that order is
not tested. On SQAD, e5-small leads it at large samples: +0.039 [+0.016, +0.061] in nDCG@10 on all
843 test questions of the SQAD 3.0 copy, and +0.043 [+0.034, +0.052] on all
4,771 test questions of SQAD 3.2.

**What can be recomputed, and how.** Every Czech number recomputes from this repository with
`scripts/rescore_czech.py`, which needs no model and no data. The copy's counts and its filter recompute from
the SQAD 3.0 archive you download from LINDAT, checked by a published MD5 hash, with `scripts/sqad_copy_vs_archive.py`
([REPRODUCE.md](REPRODUCE.md#10-checking-every-czech-number)). English and CsFEVER numbers are in results files
in this repository and recompute by running the shipped configs against the original downloads, which are
checked by published SHA-256 corpus hashes. Timings, the SQAD retrievers' among them, are a record of one machine,
and do not recompute on another.

## The Czech and English families

Qwen2.5 🇨🇳 was measured first, in English and in Czech. Every other family came later, and some of the choices
below were made after results had been read. They are listed here so that each result can be discounted for them.

* **Gemma 4 🇺🇸 was chosen on a separate dev set.** A benchmark of Western candidate models on 200
  dev questions, under a rule fixed before it ran, chose Gemma 4 🇺🇸 to answer beside Qwen2.5 🇨🇳
  ([CROSSOVER.md](CROSSOVER.md#the-sets-and-the-answers)). Qwen3.5 🇨🇳 answered there as a
  comparison only.
* **Gemma 3n 🇺🇸 was added last, after every other family's results were read and after an exploratory screen on the
  dev set.** The
  author asked for each Western maker's previous small generation; no rule chose it, and it joins whatever it
  scores. Its analysis was fixed before it answered any test question, knowing Gemma 3n E2B's 🇺🇸 dev-set score from
  a screen run outside the release. Mistral's previous small generation is not here, because it has no open
  model at Ministral 3 3B's 🇫🇷 place.
* **Ministral 3 🇫🇷 was added after the first two families' Czech results were read.** The author asked for it; no
  rule chose it, and it joins whatever it scores. Its analysis was fixed before it answered any question outside
  the dev set, where it had already run beside the benchmark. Its files are Unsloth's build of Mistral AI's
  weights, because Mistral AI's own GGUF files do not load in the release's llama.cpp build.
* **Ministral 3's 🇫🇷 results had been read, and Qwen3.5's 🇨🇳 own dev-set score was known, when Qwen3.5 🇨🇳 was added.** The author
  asked for Qwen's newer generation; no rule chose it, and it joins whatever it scores. It had answered the
  200 dev questions in the benchmark that chose Gemma 4 🇺🇸, as a comparison only, and its analysis
  was fixed knowing that result, before it answered any test question. Its cards say it thinks before answering
  by default; it answered directly here, through the files' own chat template, so these are its answers without
  thinking.
* **The Western families answered the English dial after Qwen2.5's 🇨🇳 English results were read.** Gemma 3n 🇺🇸,
  Gemma 4 🇺🇸 and Ministral 3 🇫🇷 answered it on the GX10's GPU, with the questions, rankings, prompt, token cap and seed
  of Qwen2.5's 🇨🇳 dial, which answered on the Mac's CPU.
* **Ministral 3 🇫🇷 became the English headline family after every English dial was read.** The author chose the
  family whose English sweeps were timed, and so the English headline, budgets and prices, after reading the
  dials; Ministral 3 3B 🇫🇷 and 8B then ran both main sweeps, HotpotQA and NQ at k = 5, on the Mac's CPU
  ([BUDGETS.md](BUDGETS.md)). Its dial's answers come from the GX10 and its sweeps' from the Mac, so the check of
  the dial against real retrievers compares answers from two machines, which differ on the same prompts (above).
* **Qwen3.5 🇨🇳 answered no English question.** It was added for the Czech comparison only and was not asked to answer
  the English dial or sweeps, so the English dials are four families' ([CROSSOVER.md](CROSSOVER.md#four-families-english-dials)).
* **Qwen2.5's 🇨🇳 English sweeps are shipped but not tabulated.** Qwen2.5 1.5B 🇨🇳 and 3B ran the same timed sweeps on
  the Mac, and HotpotQA at k = 3 and k = 10 as well. Their results are in `results/crossover/natural/` and
  `results/stats.json` and recompute as the others do, and the figures `results/crossover/crossover_*.png` draw that
  analysis (Qwen2.5 1.5B 🇨🇳 to 3B, the "generator 1.5B → 3B" of their legends), but no document tabulates or shows
  them; only Qwen2.5's 🇨🇳 English dial is shown ([CROSSOVER.md](CROSSOVER.md#four-families-english-dials)). That is a publication-scope decision, made after
  every English result was read, and it makes no claim about Qwen2.5's 🇨🇳 results. The documents' budget statements
  are Ministral 3's 🇫🇷 at 3B and 8B; generators of other sizes and families fit the grid differently.
* **The README's English dial figure was chosen after the dials were read.** Of the three Western families'
  English dials, it shows Ministral 3's 🇫🇷, the one in which the larger model gains most; the author chose it after
  reading all three. The four families' dials are in [CROSSOVER.md](CROSSOVER.md#four-families-english-dials) with their intervals.
* **One retriever contrast was added after the rankings were read.** BM25 over lemmas minus plain BM25 changes one
  factor, where the contrast fixed beforehand, against BM25 without diacritics, changes two. The author asked for it
  after every SQAD ranking had been read; it agrees in sign with the planned contrast, and no rule reads it
  ([CROSSOVER.md](CROSSOVER.md#ten-retrievers-on-sqad)).
* **The families figure's order and the social card were chosen after the results were read.** In each panel of
  the families figure the models stand in order of their score on the released 200, a tie keeping the makers'
  country order, and the sets stand with the released 200 first; the repository's social card shows the panel of
  the smaller models. The author chose these after reading the results. Every table keeps
  the families in a fixed order, by their makers' countries.

---

## Licences and the build

**Licences constrain use.** Qwen2.5-3B-Instruct 🇨🇳 is under the Qwen Research
License, which grants use "for research or evaluation purposes only". Every Qwen2.5 🇨🇳
3B number, in the Czech comparisons and on Qwen2.5's 🇨🇳 English dial, comes from that
model. The author decided to publish these results with the licence stated; the
weights are never redistributed. Gemma 3n 🇺🇸 and EmbeddingGemma are under Google's Gemma
Terms of Use, with its Prohibited Use Policy; Gemma 4 🇺🇸, Ministral 3 🇫🇷, Qwen3.5 🇨🇳 and Qwen2.5 🇨🇳
1.5B are under Apache 2.0 ([THIRD_PARTY.md](../THIRD_PARTY.md)).
Several dataset licences are share-alike or non-commercial, or disagree between
a card and the licence string a loader records. → [DATASETS_AND_LICENCES.md](DATASETS_AND_LICENCES.md).

**The network was limited to Hugging Face and PyPI during the build.** Some
upstream licences could not be verified as a result. SQAD 3.0's lives on
LINDAT, which the build could not reach, and was checked there afterwards,
on 8 Nov 2026. Disagreements between dataset cards were recorded as found, not resolved.

**The documentation's offline check cuts the network by environment variables
only.** `scripts/check_docs.py` runs each `# test` shell block with proxy
variables pointed at a closed port and Hugging Face and uv set offline. A tool
that ignores proxy variables and opens its own socket would still reach the
network. "Passes offline" therefore means "passes without the proxied network".
It does not prove the block needs no network.

---

*Generated from `results/stats.json`.
Costs and hygiene: [BUDGETS.md](BUDGETS.md). The crossover:
[CROSSOVER.md](CROSSOVER.md).*
