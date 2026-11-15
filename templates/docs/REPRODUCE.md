# Reproduce: from a clean clone to every result

There are two paths to the results. **Everything** (§2 to §5) downloads the
models and datasets, builds the rankings, generates every answer and re-times
every configuration; on one CPU laptop that runs to days. **The analysis only**
(§6) starts from the cached rankings, answers and bench predictions, and
needs no model, no network and no generation. §7 says what comes back byte for
byte and what cannot, and §8 is the clean-clone check. **Every Czech number**
recomputes from the tree alone: §10. How each number is produced is in
[METHOD.md](METHOD.md); this document is only the procedure.

Run every command from the repository root. File and folder names that contain v020, v030, v040 or first_cut are build labels from before the first public release.

---

## 1. Prerequisites

* **macOS or Linux, CPU for everything timed.** Every timed configuration runs on the CPU and no GPU
  is used. The Czech answers, and the English dials of Gemma 3n, Gemma 4 and Ministral 3, were generated
  on an NVIDIA GPU with llama-cpp-python built with CUDA, which the Czech queue and those dials ask for
  (Step 2). The Czech queue's phases took hours each on an NVIDIA GB10 (their records are
  `results/czech/complete/compute_gx10_*.json`); on a CPU laptop alone the full path takes weeks. The published measurements come from one machine
  ({{machine.chip}}, {{machine.ram_gb}} GB of memory); each configuration's peak RSS is in
  [BUDGETS.md](BUDGETS.md), and a machine needs at least the largest one you
  intend to run.
* **[uv](https://docs.astral.sh/uv/)** and **Python 3.11 to 3.13**
  (`requires-python` in `pyproject.toml`). uv installs a suitable Python if
  none is present. The build used Python 3.12.
* **A C and C++ compiler.** The `rag` extra's llama-cpp-python is locked as a
  source distribution and is compiled on install: the Xcode Command Line Tools
  on macOS, a GCC toolchain on Linux.
* **git**, and **network access to Hugging Face and PyPI** for setup only.
  After the models and datasets are fetched, everything runs offline.
* **Disk.** The generator files are {{cz.bench.ministral-3-3b-unsloth.gb}} GB and
  {{cz.bench.ministral-3-8b-unsloth.gb}} GB for Ministral 3 3B and 8B, {{models.qwen2.5-1.5b.gb}} GB and
  {{models.qwen2.5-3b.gb}} GB for Qwen2.5's two (the calibration probe needs the first), and
  {{models.e5-small.gb}} GB, {{models.e5-base.gb}} GB and {{models.bge-m3.gb}}
  GB for the three English encoders (decimal GB, as shipped). The Czech path adds the other families' generators
  (Gemma 3n E2B and E4B, Gemma 4 E2B {{cz.bench.gemma-4-e2b.gb}} GB and E4B {{cz.bench.gemma-4-e4b.gb}} GB, and Qwen3.5 2B
  {{cz.bench.qwen3.5-2b.gb}} GB and 4B {{cz.bench.qwen3.5-4b.gb}} GB) and the five Czech-split encoders. The dataset loaders
  download each dataset's full corpus before capping it to a slice, so the
  download cache under `data/` is far larger than the slices; each bench
  configuration also writes its own index under its sweep's `work/` directory.

## 2. Install

```bash
git clone https://github.com/arytma/measurerag.git measurerag
cd measurerag
uv sync --extra dense --extra rag --extra dev
uv run measurerag doctor --fix
```

`dense` brings torch, sentence-transformers and faiss-cpu; `rag` brings
llama-cpp-python; `dev` brings pytest. To pin the interpreter the build used,
add `--python 3.12` to `uv sync`. A later `uv sync` without the same extras
removes them.

**Why `doctor --fix`, and why again after every `uv sync`.** On macOS the
torch, faiss-cpu and scikit-learn wheels each ship their own copy of the OpenMP
runtime. Two copies in one process crashed the first unattended run, and
libomp's own diagnostic warns that duplicates can also produce wrong results.
`doctor --fix` points every copy at one file (torch's) and keeps each replaced
file as `.orig-backup`, so it is reversible. A `uv sync` reinstalls the wheels
and undoes the link. `measurerag run`, the ranking and generation scripts, the
calibration probe and the timed queue all run the same check first and refuse
to start while two runtimes are present. On Linux the check finds nothing to
repair.

**The offline smoke test.** It runs on a bundled synthetic fixture, needs no
model and no network, and prints a budget table. `scripts/check_docs.py` runs
this block with the network cut off:

```bash
# test
uv run measurerag smoke
```

If it fails, stop and fix it; nothing below will work.

## 3. Models

```bash
HF_HUB_CACHE=data/hf-hub uv run python scripts/fetch_verify_models.py --bge-m3
```

This fetches the dense encoders (multilingual-e5-small, multilingual-e5-base
and, with `--bge-m3`, bge-m3) and Qwen2.5's two generators (Qwen2.5-1.5B-Instruct and
Qwen2.5-3B-Instruct, Q4_K_M GGUF) into `data/hf-hub`, and verifies them against
the pinned records:

* each GGUF: `main` must resolve to the pinned revision, the file's
  SHA-256 must equal that revision's LFS object id, and its size must equal
  the recorded size;
* each encoder: fetched through sentence-transformers, so that only the files a
  load reads are present; `main` must resolve to the pinned commit, and the
  shipped byte count (the rule in [METHOD.md](METHOD.md) §1) must equal the
  recorded count exactly.

Every other generator, Ministral 3 among them, is downloaded on first use from the commit pinned in
`src/measurerag/rag/generator.py`, and each result records the file's path and size.

It writes `results/crossover/models_manifest.json` with a SHA-256 for every
file and exits non-zero on any mismatch. If one of those repositories' `main` has moved
since, the check fails instead of measuring a different model. retromae-cs loads the commit pinned in
`src/measurerag/retrievers/dense.py`, the one every result used. The other encoders (the two granite
models, EmbeddingGemma and simcse-cs-en) load their repository's `main`; the revisions read for their licences are listed in
[THIRD_PARTY.md](../THIRD_PARTY.md), and nothing checks them at fetch.

From here on, every command needs the same model cache. The path is relative,
which is one more reason to stay in the repository root:

```bash
export HF_HUB_CACHE=data/hf-hub
```

While the network is still on, fetch the other encoders into the same cache:

```bash
uv run measurerag fetch models
```

It loads every encoder in `src/measurerag/retrievers/dense.py`, retromae-cs at its pinned commit and the others at
`main`. EmbeddingGemma's repository is gated: run `hf auth login` first, from a Hugging Face account that has accepted
Google's Gemma terms, or the command reports it as FAILED.

## 4. Data

The loaders download each dataset from Hugging Face into `data/hf` (set
`MEASURERAG_CACHE` to move `data/`), at the commit pinned in each loader under `src/measurerag/datasets/`
(the commit every result used), read the corpus, and cap it to a seeded
slice that keeps every gold passage of every sampled question. A slice is
a directory of `corpus.jsonl`, `queries.jsonl`, `qrels.jsonl` and `meta.json`.
The analysis reads three:

```bash
uv run python scripts/make_slice.py --dataset hotpotqa-rag --max-docs 100000 --max-queries 500 --out results/en_rag/slice
uv run python scripts/make_slice.py --dataset nq-rag --max-docs 100000 --max-queries 300 --out results/crossover/slices/nq
uv run python scripts/make_slice.py --dataset sqad --max-docs 100000 --max-queries 200 --out results/cz_rag/slice
export HF_HUB_OFFLINE=1
```

* **HotpotQA**'s slice is `results/en_rag/slice`, the slice `configs/en_rag.yaml`
  writes. Every HotpotQA bench run writes the same slice into its own
  `<out_dir>/slice`, and the first command applies that config's dataset name and
  caps through the same loader.
* **Natural Questions** is built by the second command, with the caps of
  `configs/crossover/nq_k5.yaml`. Its gold passages
  are mapped rather than annotated ([CROSSOVER.md](CROSSOVER.md) §4).
* **The SQAD 3.0 copy** is needed only to regenerate the Czech answers, with the caps
  of `configs/cz_rag.yaml`. The Czech results are outside the headline, and their
  numbers recompute from the tree without it (§10). The licence is in
  [DATASETS_AND_LICENCES.md](DATASETS_AND_LICENCES.md). The loader pins the copy's revision.
* **SQAD 3.2 and the SQAD 3.0 archive** are downloaded by hand from LINDAT, and they stay in
  the git-ignored `.local/data/sqad_3.2/sqad_v3.2.tar.xz` and
  `.local/data/sqad_3.0/sqad_v3.tar.xz`: http://hdl.handle.net/11234/1-5019 (MD5
  `39c29a416d2542ecc924542ad79b4cfd`) and http://hdl.handle.net/11234/1-3069 (MD5
  `{{czech_checks.archive_md5}}`). `scripts/sqad32_slices.py` and
  `scripts/sqad32_archive_rule.py` read them, and `scripts/czech_plan.py --slices` builds the Czech question
  sets the queue and the records read (`results/crossover/slices/complete_*`).

**Every slice's corpus hash is checked.** Each slice's `meta.json` records the
SHA-256 of its corpus, queries and qrels. Compare the corpus hashes with the
ones the committed analysis used (`results/crossover.json`, one per dataset)
and, for HotpotQA, with the manifest in `results/en_rag/`:

```bash
grep -o '"corpus_sha256": "[0-9a-f]*"' results/en_rag/slice/meta.json results/crossover/slices/nq/meta.json results/cz_rag/slice/meta.json
grep -o '"corpus_sha256": "[0-9a-f]*"' results/crossover.json results/en_rag/manifest.json
```

After that, `scripts/crossover.py` checks it on every run: it refuses a cached
ranking or a bench run whose corpus hash differs from the slice it analyses.

## 5. Regenerating everything, in order

Run one heavy process at a time: two generators side by side are slower in
total than one after the other, and timings need a quiet machine. Every
step below is resumable; work already on disk is not redone.

### Step 1: rankings

One ranking per retriever and dataset, cached in
`results/crossover/cache/<dataset>/retrieval_<retriever>.json`:

```bash
H="--slice results/en_rag/slice --cache results/crossover/cache/hotpotqa"
N="--slice results/crossover/slices/nq --cache results/crossover/cache/nq"
for S in "$H" "$N"; do
    for R in tfidf-svd:tfidf-svd:256 bm25:bm25 e5-small:dense:e5-small e5-base:dense:e5-base bge-m3:dense:bge-m3; do
        uv run python scripts/retrieval_cache.py $S --retriever $R
    done
done
```

* `--retriever` is `id:kind[:model]`; for tfidf-svd the last field is the
  number of components, as in the sweep configurations.
* A ranking already on disk is never recomputed. A dense ranking embeds the
  whole corpus, which takes hours on a CPU. `--reuse-index <faiss_flat.index>`
  loads instead the flat index a bench configuration wrote into its `work/`
  directory; use it only when that configuration's slice is byte-identical to
  the one you rank, because the script checks only that the number of vectors
  matches (the timed queue compares the corpus files before reusing an index
  for bge-m3).
* The Czech sets need `bm25:bm25`, `bm25-nodiacritics:bm25`,
  `e5-small:dense:e5-small` and `retromae-cs:dense:retromae-cs`, over the whole test
  split of each version: `--slice results/crossover/slices/czech30_testall --cache
  results/crossover/cache/v030_copy` and `--slice
  results/crossover/slices/czech32_testall --cache results/crossover/cache/v030_32`.
* The six other retrievers rank the same two
  test splits into `results/crossover/cache/v040_copy` and `v040_32`: `e5-small:dense:e5-small`
  (encoded again, to check the Mac against the shipped rankings), `e5-base:dense:e5-base`,
  `granite-107m:dense:granite-107m`, `granite-278m:dense:granite-278m`,
  `embeddinggemma:dense:embeddinggemma`, `simcse-cs-en:dense:simcse-cs-en` and
  `bm25-lemma:bm25`. The encoders' revisions are in
  [DATASETS_AND_LICENCES.md](DATASETS_AND_LICENCES.md). EmbeddingGemma's repository is gated:
  log in with `hf auth login` from a Hugging Face account that has accepted Google's Gemma
  terms before the fetch in §3. Then `scripts/czech_retrievers.py --reproduce` compares e5-small's
  new ranking with the shipped one question for question, `--records` writes the new top 10s
  into `results/czech/records/rankings_v040_<split>.jsonl.gz`, and a plain run writes the
  readings.

### Step 2: the controlled dial

Each family's two generators, both datasets, on the dial's retriever. Gemma 3n's, Gemma 4's and Ministral 3's
dials were generated on a GPU (`--device cuda`, which needs llama-cpp-python built with CUDA, as below);
Qwen2.5's on the CPU:

```bash
H="--dataset hotpotqa --slice results/en_rag/slice --cache results/crossover/cache/hotpotqa"
N="--dataset nq --slice results/crossover/slices/nq --cache results/crossover/cache/nq"
for S in "$H" "$N"; do
    for G in ministral-3-3b ministral-3-8b gemma-4-e2b gemma-4-e4b gemma-3n-e2b gemma-3n-e4b; do
        uv run python scripts/dial_generate.py $S --retriever e5-base --k 5 --generator $G --device cuda
    done
    for G in qwen2.5-1.5b qwen2.5-3b; do
        uv run python scripts/dial_generate.py $S --retriever e5-base --k 5 --generator $G
    done
done
```

Other hardware gives different answers to the same prompt
([LIMITATIONS.md](LIMITATIONS.md#the-czech-and-english-families)).

* `--dataset` must be the dataset's key in `configs/crossover.yaml`
  (`hotpotqa`, `nq`, `czech`): the dial's nested draws are seeded by it.
* The defaults of `--levels`, `--max-tokens` and `--seed` are the values in
  `configs/crossover.yaml`. `--dry-run` prints how many answers are cached and
  how many remain, without loading a model.
* Answers go to `results/crossover/cache/<dataset>/gen_<generator>.jsonl`, one
  per distinct prompt; a killed run loses at most the answer in flight.
* **Czech:** `.venv/bin/python scripts/czech_complete_queue.py --phase main --machine gx10`
  runs Gemma 4's and Qwen2.5's Czech generations in the queue's order, one process at a time and each set from an
  empty cache: the bench, each dial at `--retriever e5-small --k 3` with both families, the
  natural top 3 of the smaller models, the check that the bench and the dial give the same
  strings, the cross-version questions and the sensitivity rows. It is restartable, writes
  `PROGRESS.md` in the git-ignored `logs/czech_progress/` (`--report-dir`), and stops after the current configuration when a
  file named `STOP` appears there. `--phase dev` answers the dev set that chose the prompt, and `--phase pilot`
  the dev set's repeat on the GX10 that `scripts/czech_pilot_check.py` reads.
  `--phase ministral` takes every stage the main phase ran for Gemma 4 and runs it for Ministral 3 alone, E2B's
  place taken by 3B and E4B's by 8B, each from an empty cache of its own generator, with its own record in
  `results/czech/complete/compute_gx10_ministral.json`. `--phase gemma3n` and `--phase qwen35` run Gemma 3n's
  and Qwen3.5's answers the same way, with their records in `compute_gx10_gemma3n.json` and
  `compute_gx10_qwen35.json`. `--machine gx10` generates on
  the GPU (`device: cuda`), so llama-cpp-python must be built with CUDA:
  `CMAKE_ARGS="-DGGML_CUDA=on" uv sync --frozen --all-extras --no-cache`. `uv.lock` pins only
  llama-cpp-python's source, so uv builds it, here with CUDA, and `--no-cache` keeps it from
  reusing a CPU build. The generator stops rather than fall back to the CPU. Other hardware
  gives different answers ([LIMITATIONS.md](LIMITATIONS.md)).
  `scripts/czech_records.py` then writes the records that ship; its set `released_v020` also reads an earlier
  bench's predictions under the yes-or-no line, which are not provided, so that set's records are read as shipped. `scripts/czech_redo_queue.py`
  stays as the record of how an earlier round of answers, used to size the sets, was made.
* **The same questions on both versions:** `caffeinate -dimsu .venv/bin/python
  scripts/czech_same_questions.py` answers the questions in both test splits on the other
  version's passages, in a cache of its own, and writes
  `results/czech/first_cut/same_questions_rewritten.jsonl.gz`; the files
  `answers/<source>/first_cut/same_questions.jsonl.gz` hold its rows, split by source.
* **Order matters a little.** The answer to a near tie can turn on the prompt before it
  in the same process. So keep a dial's cache empty until it has run, and do not run
  a dial twice with different caches and expect the same strings on every prompt.

**The token-cap check** uses the same cache. Its answers are part of the
cached rows that the generation counts in `results/stats.json` count, so a full
regeneration includes it:

```bash
for G in qwen2.5-1.5b qwen2.5-3b; do
    uv run python scripts/dial_generate.py --dataset hotpotqa --slice results/en_rag/slice \
        --cache results/crossover/cache/hotpotqa --retriever e5-small --k 5 --levels natural \
        --limit 100 --generator $G
done
uv run python scripts/max_tokens_check.py
```

`max_tokens_check.py` compares these answers with the prediction files in
`results/en_rag/`, generated at `max_tokens` = {{design.max_tokens_run1}}, and writes
`results/crossover/max_tokens_check.json`. The Czech record, `max_tokens_check_czech.json`,
was written by the same script with its options (`--slice`, `--cache`, `--run-1`, `--k`, `--n`,
`--generators`, `--out`) set for the Czech slice and `results/cz_rag`; the exact invocation
was not recorded, and it was written before the ano-or-ne line was adopted. For that
line, the check is `token_cap_check` in `results/czech/analysis.json`, recomputed
from the records.

### Step 3: the timed natural sweeps

**Ministral 3's sweeps**, as they were run, through the queue of [METHOD.md](METHOD.md) §4:

```bash
mv results/crossover/calibration_ministral/baseline.json results/crossover/calibration_ministral/baseline_build.json
mkdir -p logs/sweep_ministral
nohup .venv/bin/python scripts/sweep_queue.py --phase ministral --no-hygiene > logs/sweep_ministral/queue_stdout.log 2>&1 &
```

* **The baseline.** The committed `results/crossover/calibration_ministral/baseline.json` belongs to the build
  machine, so move it aside first: the queue takes a baseline of its own only when none is there. With the gate
  on (without `--no-hygiene`), every probe is compared with that baseline's median and with its answers. The probe
  needs the HotpotQA slice, the cached bm25 HotpotQA ranking and Qwen2.5-1.5B-Instruct.
* **`--no-hygiene`**, as every accepted Ministral 3 batch ran ([BUDGETS.md](BUDGETS.md), §4), runs the batches
  back to back with no probe, no gate and no rejection for contention, recording contention and running again a
  batch more than a quarter contended. Leave it out to run them under the gate.
* **What the queue runs** is the fixed list of the phase in the script: the five retrievers with Ministral 3
  3B and 8B, HotpotQA at k = 5 (`configs/crossover/hotpotqa_k5_ministral.yaml`), then NQ at k = 5
  (`configs/crossover/nq_k5_ministral.yaml`). Without `--phase`, it runs Qwen2.5's stages, whose results ship
  in `results/crossover/natural/` and are not tabulated.
* **What it does on its own:** runs `measurerag doctor` and stops if it
  complains; sets `HF_HUB_CACHE=data/hf-hub` and the offline variables for
  every child; calls `.venv/bin/python` and `.venv/bin/measurerag`, the
  environment `uv sync` created; **commits** each attempt's sweep directory and
  the calibration log to the checked-out branch, so run it on a branch you are
  willing to commit to.
* **Progress** is in `logs/sweep_ministral/queue.log`; every attempt is a row in
  `results/crossover/calibration_ministral/batches.jsonl`. That log is append-only and holds the record of the
  measured run that the documents describe, so compare a re-timed sweep's `results.json` with the committed one
  rather than refilling the documents from it. The queue is restartable: a configuration
  whose accepted batch is on record and whose result is on disk is skipped.
* **Anything heavy while it runs** (tests, the analysis) goes through
  `scripts/quiet.sh <command>`, which waits for the batch in flight and holds
  the queue until the command ends.

**Without the queue**, straight through the bench:

```bash
uv run measurerag run --config configs/crossover/hotpotqa_k5_ministral.yaml
uv run measurerag run --config configs/crossover/nq_k5_ministral.yaml
```

`--only` takes a comma-separated list of retriever ids or configurations (for
example `--only bm25+ministral-3-3b`); a configuration that already finished under
an unchanged spec is reused unless you pass `--no-resume`. Each sweep writes
`results.json`, `manifest.json` and `predictions_<config>.json` into its
`out_dir`, with the slice, the specs and the indexes beside them. Each
configuration builds its own index rather than reusing a cached one, so
every dense configuration embeds the whole corpus again. Qwen2.5's sweeps have
configs of their own in `configs/crossover/` (`hotpotqa_k3.yaml`, `hotpotqa_k5.yaml`,
`hotpotqa_k10.yaml`, `nq_k5.yaml`).

**The model file, mapped or not.** The generator loads its GGUF memory-mapped, llama-cpp-python's default and the
way every published row ran, and each result records the setting in its `generator` block (`use_mmap`). With
`use_mmap: false` under a config's `rag:`, the file is read into memory instead; the spec then names the setting, so
a row already finished is measured again. With mapping on, peak RSS counts part of the weights twice
([LIMITATIONS.md](LIMITATIONS.md#the-machine-and-the-timings)).

**The probe of the mapped model file**, on a quiet machine, in about a quarter of an hour:

```bash
uv run python scripts/mmap_probe.py
```

It loads Ministral 3 3B and 8B with mapping on and off, five times each, each load in its own process, and rewrites
`results/crossover/mmap_probe.jsonl` with the machine, a snapshot of the other processes and every load's peak RSS.

### Step 4: the analysis

```bash
uv run python scripts/crossover.py
uv run python scripts/crossover_figure.py
scripts/quiet.sh uv run python scripts/repro_check.py
uv run python scripts/crossover.py
uv run python scripts/fill.py --all
uv run python scripts/check_docs.py
```

1. `crossover.py` reads the slices, the caches and the sweeps' `results.json`
   and prediction files, and writes `results/crossover.json` and
   `results/stats.json`; from the probe's record it also writes
   `results/crossover/mmap_estimate.json`, the peak-RSS estimates without the
   double-counted mapped pages, and their tokens. A dataset with missing dial answers is recorded as
   incomplete; `--allow-partial` gives a progress look that is never
   committed.
2. `crossover_figure.py` draws `results/crossover/crossover_<dataset>.png` from
   `results/crossover.json` alone.
3. `repro_check.py` runs `crossover.py` twice more and requires both outputs to
   equal each other and the three files item 1 wrote; it writes
   `results/crossover/repro_check.json` and exits 1 on any difference. Through
   `scripts/quiet.sh` it waits for a running queue's batch; with no queue it
   runs at once.
4. `crossover.py` again, so that the `repro.*` tokens in `results/stats.json`
   carry the result of item 3.
5. `fill.py --all` regenerates every document from its template under
   `templates/`.
6. `check_docs.py` checks the documentation set, including running every
   shell block tagged `# test` (this document has three) with the network cut
   off; `--skip-exec` leaves those out.

### Inputs these steps do not regenerate

* **The first HotpotQA bench run** (`results/en_rag/`) and the other early runs are
  committed and read as they are. Their prediction files are gitignored and
  are needed by the token-cap check and by the `remeasure.*` tokens.
  `uv run measurerag run --config configs/en_rag.yaml --out <directory>`
  regenerates its answers (at `max_tokens` = {{design.max_tokens_run1}}) into
  that directory without touching the committed results; copy its prediction
  files into `results/en_rag/`.
* **A superseded e5-base pair** (`results/crossover/superseded/`) exists only
  as committed results and cached prediction files; without them, the
  `remeasure.*` tokens report its answers as not identical, because there is
  nothing to compare them with.
* **The calibration and batch logs** are append-only records of what happened.
* **The probe's record**, `results/crossover/mmap_probe.jsonl`, is a record of one machine; the probe above
  measures it again on yours.

---

## 6. Regenerating only the analysis from cached data

The analysis needs no model, no network and no generation, only these
gitignored inputs from the machine that ran the measurements:

* **the slices:** `results/en_rag/slice/`, `results/crossover/slices/nq/` and
  `results/cz_rag/slice/`;
* **the caches:** `results/crossover/cache/<dataset>/retrieval_*.json` and
  `gen_*.jsonl` for `hotpotqa`, `nq` and `czech` (the `work/` directories hold
  indexes and are not needed);
* **the bench predictions:** every `predictions_*.json` beside a bench
  `results.json` under `results/`: the natural sweeps', and the first HotpotQA
  run's and the superseded pair's for the `remeasure.*` tokens.

They are not in the repository: they are large and reproducible, and
they hold dataset text, which the repository does not redistribute. The Czech
answers and rankings are the exception: the Czech block of `results/crossover.json` reads
the records that ship, and `results/crossover/cache/czech` is needed only to count
the answers cached there. With them in place:

```bash
uv run python scripts/crossover.py
uv run python scripts/crossover_figure.py
uv run python scripts/fill.py --all
uv run python scripts/check_docs.py --skip-exec
```

Add `uv run python scripts/repro_check.py` and a second `crossover.py` (items 3
and 4 of step 4 in §5) to refresh the reproduction record.

Whether the committed documents are current needs no cached data at all: it
compares each generated document with its template filled from the committed
`results/stats.json`. `scripts/check_docs.py` runs this block offline:

```bash
# test
uv run python scripts/fill.py --check
```

---

## 7. What reproduces byte for byte, and what does not

| artefact | byte for byte? | how it is checked |
|---|---|---|
| slices | yes, from the same upstream files | corpus, query and qrels SHA-256 in `meta.json`, in each manifest and in `results/crossover.json` |
| rankings | yes, on the same machine and environment | the ranking SHA-256 in `results/crossover.json`; not checked across machines |
| answers | yes in the same order: greedy decoding, fixed seed, same prompt bytes. In another order, an answer that is a near tie can change | the token-cap check, the probe's answer hash before every timed batch, the re-measured ladder against the rows it replaced, the Czech bench against the Czech dial |
| EM, F1, g, every Δ, raw g\*, break-even gains | yes, computed from answers and rankings | `results/crossover.json` |
| `results/crossover.json`, `results/stats.json`, `results/crossover/mmap_estimate.json` | yes, from the same inputs | `scripts/repro_check.py`, `tests/test_crossover_determinism.py` |
| the generated documents | yes, filled from `results/stats.json` | `scripts/fill.py --check` |
| the crossover figures, `results/crossover/*.png` | drawn from `results/crossover.json` | not part of the byte-for-byte check |
| the release's figures, `results/figures/` | yes, from `results/crossover.json` and `results/czech/analysis.json`: SVG with its text as paths, a fixed hash salt and no date, and one PNG without metadata | `scripts/figures.py --check`, and the clean-clone test |
| retrieval and end-to-end latency, build and wall time | no | timings are physics, not logic |
| peak RSS | within noise only | compared row by row when a configuration is re-measured |
| budget verdicts, prices, and the per-GB and per-ms g\* | no: they are computed from timings and peak RSS | re-measuring a sweep changes them, while EM and g stay fixed |

`uv run measurerag repro --config <yaml>` runs one configuration file twice and
compares the two manifests with timings, peak RSS and verdicts removed
([METRICS.md](METRICS.md#reproducibility)). Byte identity also assumes the
locked environment, not only the seed: the slices' sampling and the bootstrap
draw from NumPy's random generators, NumPy does not promise the same stream
across versions, and `uv.lock` resolves a different NumPy for Python 3.11 than
for Python 3.12 and later.

---

## 8. The clean-clone check

It checks that a fresh environment built from the lockfile passes the tests
offline and turns the cached data into exactly the committed `stats.json` and
documents.

```bash
SRC="$PWD"                                  # the checkout that holds the cached data (§6)
WORK="$(mktemp -d)"
git clone "$SRC" "$WORK/measurerag"         # or the published repository's URL
cd "$WORK/measurerag"
uv sync --frozen --python 3.12 --extra dense --extra rag --extra dev
uv run measurerag doctor --fix
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1
uv run pytest

# point the clone at the cached data
for d in results/en_rag/slice results/cz_rag/slice results/crossover/slices/nq; do
    mkdir -p "$(dirname "$d")" && cp -R "$SRC/$d" "$d"
done
for ds in hotpotqa nq czech; do
    mkdir -p "results/crossover/cache/$ds"
    cp "$SRC/results/crossover/cache/$ds"/retrieval_*.json "$SRC/results/crossover/cache/$ds"/gen_*.jsonl \
        "results/crossover/cache/$ds/"
done
(cd "$SRC" && find results -name 'predictions_*.json') | while read -r f; do
    mkdir -p "$(dirname "$f")" && cp "$SRC/$f" "$f"
done

# regenerate, and compare with what the clone committed
uv run python scripts/crossover.py
git diff --exit-code -- results/crossover.json results/stats.json
uv run python scripts/fill.py --check
uv run python scripts/figures.py --check
```

* `uv sync --frozen` installs exactly what `uv.lock` records and never updates
  it; `--python 3.12` matches the build (§7 says why the interpreter matters).
* `doctor --fix` comes straight after the sync, for the reason in §2.
* The tests run with Hugging Face, transformers and uv forced offline.
* Run the check when no timed batch is running on the same machine: the
  clone's own `scripts/quiet.sh` watches the clone's `logs/`, not the source
  checkout's queue.
* `git clone` takes the committed state of the source checkout's branch, so the
  comparison is against what was committed, not against the source's working
  tree.
* `git diff --exit-code` fails if the regenerated `results/crossover.json` or
  `results/stats.json` differs from the committed one by a single byte.
  `fill.py --check` then fails if any committed document differs from its
  template filled from the regenerated `results/stats.json`.

---

## 9. Paths in the results

A bench run records the absolute path it loaded each GGUF from;
`uv run python scripts/scrub_paths.py` rewrites such paths (and any email
address under `results/`) in tracked result files, and `--check` lists any
that remain.

---

## 10. Checking every Czech number

Every number in the public text about Czech recomputes in one of three ways. The first
needs nothing but this tree, and the test suite and the clean-clone check run it.

```bash
# test
uv run python scripts/rescore_czech.py --check
```

| what | what it needs | command |
|---|---|---|
| every Czech table and number: EM, F1, retrieval at every sample size, the dial, the steps, the fix's measured size, the reweighting, the claims | this tree | `scripts/rescore_czech.py --check` |
| each stored EM and F1 in every answers file, against the gold | the pinned SQAD 3.0 copy and the SQAD 3.2 archive (§4) | `scripts/rescore_czech.py --verify-scores` |
| where the SQAD 3.0 copy comes from: row for row the filtered SQAD 3.0 of Štefánik et al. (2023) | the two pinned Hugging Face datasets, downloaded by the script | `scripts/sqad_copy_source.py` |
| the two record counts of SQAD 3.0, the copy's filter, the two overlap thresholds | the SQAD 3.0 archive (§4), whose MD5 the script checks first | `scripts/sqad_copy_vs_archive.py --archive <file>` |
| the prompt's selection on the dev set | this tree | `scripts/czech_dev_prompts.py --check` |
| the move to the second machine: its repeat, how often the two machines answer alike, its rankings | this tree | `scripts/czech_pilot_check.py --check` |
| the benchmark that chose the second family, and Ministral 3 beside it | this tree | `scripts/czech_model_bench.py --check`, `scripts/czech_model_bench_mistral.py --check` |
| the ten retrievers on SQAD's two test splits: nDCG@10, Recall@10, the gold in the top 3, every difference and the end-to-end rule | this tree | `scripts/czech_retrievers.py --check` |
| the release's figures, from `results/czech/analysis.json` and `results/crossover.json` alone, byte for byte | this tree | `scripts/figures.py --check` |
| how much SQAD the answers quote, and which repeat their question | the pinned SQAD 3.0 copy and the SQAD 3.2 archive (§4) | `scripts/czech_quote_scan.py --out results/czech/quote_scan.json` |
| the planned sizes and questions | this tree, on macOS: on Linux a few simulated powers differ in the third decimal | `scripts/czech_plan.py --check` |

`--check` recomputes `results/czech/analysis.json`, the Czech block of
`results/crossover.json` and every Czech token in `results/stats.json` from
`answers/` and `results/czech/records/`, and fails if one byte differs. These hold ids, scores and
each generated answer; [DATASETS_AND_LICENCES.md](DATASETS_AND_LICENCES.md) has their
licence. Gold answers do not ship, so EM and F1 are stored per row, and `--verify-scores`
recomputes them from each answer and the gold of the data you download.

**What this does not recompute.** Generating an answer again needs the models and takes
hours (§5). No per-question record of the English sweeps or of CsFEVER retrieval ships.
Their numbers trace to results files in the tree, and you recompute them by running the
shipped configs against the original downloads, with published SHA-256 corpus hashes (§4). Czech peak RSS and latency stay in
`results/cz_rag*/results.json` as recorded, and they are not quoted in the public text: they
measure one machine, it was busy, and no script can recompute them.

