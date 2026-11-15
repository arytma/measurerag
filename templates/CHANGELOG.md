# Changelog

All notable changes to measureRAG. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Results are not listed here. They live in the generated documents, filled from `results/stats.json`:
[README.md](README.md), [docs/CROSSOVER.md](docs/CROSSOVER.md) and [docs/BUDGETS.md](docs/BUDGETS.md).

## [0.4.5] - 2026-11-15

The first public release: a budget-aware benchmark of retrieval and retrieval-augmented generation (RAG) in Czech
and English, on one CPU-only machine with no network access. v0.4.4 was prepared but not released; its changes are
in v0.4.5.

### Added

- **The budget bench.** Every configuration, a retriever with or without a local generator, runs in its own process
  and is measured for quality (nDCG@10, Recall@10 and Recall@100, gold-in-context, exact match, token F1) and for
  cost (the peak resident memory of the whole process, index and model size on disk, retrieval and end-to-end
  latency at p50 and p95, generator throughput), then checked against a grid of memory and latency budgets. Each run
  writes a manifest with the SHA-256 of every slice and output, and `measurerag repro` runs a configuration twice
  and compares the two.
- **Retrievers.** BM25, as is, without diacritics and over Czech lemmas; TF-IDF with SVD; and dense encoders over a
  flat FAISS index: multilingual-e5 small and base, granite-embedding 107m and 278m multilingual,
  EmbeddingGemma 300m, simcse-cs-en and retromae-cs, and bge-m3, each with its pooling and prefixes; e5-small and e5-base
  are verified against a pinned commit at fetch, retromae-cs loads a pinned commit, and bge-m3 is verified against a
  pinned commit at fetch.
- **Generators.** Local GGUF files through llama.cpp, Q4_K_M, greedy decoding: Gemma 3n, Gemma 4, Ministral 3,
  Qwen2.5 and Qwen3.5, two sizes of each; Unsloth's files pinned to a commit, Qwen2.5's verified against a
  pinned revision at fetch.
- **The English experiment,** on HotpotQA and Natural Questions: a controlled dial that removes gold passages from
  the context with probability p, natural sweeps of five real retrievers with Ministral 3 3B and 8B timed on one
  laptop's CPU, paired bootstrap intervals over questions, the break-even gold gain, prices per GB of peak memory and
  per millisecond, and four families' dials.
- **Pinned data.** Every Hugging Face dataset the English and CsFEVER loaders read, and the SQAD 3.0 copy, is read
  at a pinned commit, and each slice records the SHA-256 of its corpus, queries and judgements.
- **Czech retrieval** on CsFEVER v2, and ten retrievers ranked on SQAD 3.2 and on a filtered copy of SQAD 3.0.
- **Czech RAG**, five model families on four SQAD sets, with the per-question records behind every number
  (`answers/`, `results/czech/records/`); `scripts/rescore_czech.py` recomputes every Czech number from them, with
  no model and no data.
- **Peak memory with and without the double-counted mapped pages.** The generator loads its model file
  memory-mapped, the library's default, and on the CPU also copies most weights into its own buffers, so peak
  memory counts that part of the weights twice. A `use_mmap` setting for the generator (on by default, recorded in
  each run's result), `scripts/mmap_probe.py` and its record, and, beside every affected figure of the English
  Ministral 3 rows, an estimate without the double count, computed from that record by `scripts/crossover.py`
  ([docs/LIMITATIONS.md](docs/LIMITATIONS.md#the-machine-and-the-timings)).
- **Measurement hygiene** for the timed sweeps: one timed batch at a time, a calibration probe, a sampler of every
  other process's CPU, and the chip's temperature, each recorded beside the batch.
- **Generated documents.** Every result in the documents is filled from `results/stats.json`.
  `scripts/check_docs.py` fails on a number that does not trace to it, and `scripts/number_audit.py` names the file
  or script that recomputes each one.
- **Figures** in light and dark, each SVG with a title and a description for screen readers, and every value they
  draw listed in `results/figures/figures.json`.
- **A smoke test** on a bundled synthetic fixture: no network, seconds, a budget table.
- **A Czech summary**, [README.cs.md](README.cs.md): what measureRAG is, the Czech data and results, how to run
  and cite it, and a note for theses, with links to the English documents.
