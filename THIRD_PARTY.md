# Third-party datasets and models

measureRAG redistributes **no model weights and no dataset**. The scripts
download them from Hugging Face when a run starts (`measurerag fetch`,
`scripts/fetch_verify_models.py`), except SQAD 3.2 and the SQAD 3.0 archive, which you
download from LINDAT yourself, and each run's `manifest.json` records the
SHA-256 of every slice it used. The repository holds aggregate numbers, hashes and
notes the kit wrote itself and, for Czech only, per-question records under
`answers/` and `results/czech/records/`: ids, scores and each generated answer. No question,
passage or gold answer from any dataset is committed, except as quoted inside a generated answer
under `answers/` (below).

Each item below keeps its own licence. The licences were read from each Hugging
Face card at the revisions in the tables below. Where two sources
disagree, both readings are given, as found.
[docs/DATASETS_AND_LICENCES.md](docs/DATASETS_AND_LICENCES.md) has the sizes
used and the full card wording.

## Models

| model | used as | Hugging Face repository @ revision | licence |
|---|---|---|---|
| Gemma 3n E2B, instruction-tuned (GGUF, Q4_K_M) | generator on every Czech set, beside Gemma 4 E2B and the other smaller models; on the English dial | `unsloth/gemma-3n-E2B-it-GGUF` @ `d7bd20d5`, a quantisation of `google/gemma-3n-E2B-it` | **Gemma Terms of Use** |
| Gemma 3n E4B, instruction-tuned (GGUF, Q4_K_M) | generator on every Czech set, beside Gemma 4 E4B and the other larger models; on the English dial | `unsloth/gemma-3n-E4B-it-GGUF` @ `90fa8b0e`, a quantisation of `google/gemma-3n-E4B-it` | **Gemma Terms of Use** |
| Gemma 4 E2B, instruction-tuned (GGUF, Q4_K_M) | generator on every Czech set, beside Qwen2.5-1.5B; on the English dial | `unsloth/gemma-4-E2B-it-GGUF` @ `0314792d`, a quantisation of `google/gemma-4-E2B-it` | Apache-2.0 |
| Gemma 4 E4B, instruction-tuned (GGUF, Q4_K_M) | generator on every Czech set, beside Qwen2.5-3B; on the English dial | `unsloth/gemma-4-E4B-it-GGUF` @ `bfc15c38`, a quantisation of `google/gemma-4-E4B-it` | Apache-2.0 |
| Ministral 3 3B Instruct 2512 (GGUF, Q4_K_M) | generator on every Czech set, beside the other smaller models; on the English dial and the timed English sweeps | `unsloth/Ministral-3-3B-Instruct-2512-GGUF` @ `7564922f`, a quantisation of `mistralai/Ministral-3-3B-Instruct-2512` | Apache-2.0 |
| Ministral 3 8B Instruct 2512 (GGUF, Q4_K_M) | generator on every Czech set, beside the other larger models; on the English dial and the timed English sweeps | `unsloth/Ministral-3-8B-Instruct-2512-GGUF` @ `3731507e`, a quantisation of `mistralai/Ministral-3-8B-Instruct-2512` | Apache-2.0 |
| Qwen2.5-1.5B-Instruct (GGUF, Q4_K_M) | generator on every Czech set; on the English dial and the English natural sweeps (shipped untabulated); the calibration probe's fixed generator | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` @ `91cad511` | Apache-2.0 |
| Qwen2.5-3B-Instruct (GGUF, Q4_K_M) | generator on every Czech set; on the English dial and the English natural sweeps (shipped untabulated) | `Qwen/Qwen2.5-3B-Instruct-GGUF` @ `7dabda4d` | **Qwen Research License** (non-commercial) |
| Qwen3.5 2B (GGUF, Q4_K_M) | generator on every Czech set, beside the other smaller models | `unsloth/Qwen3.5-2B-GGUF` @ `f6d5376b`, a quantisation of `Qwen/Qwen3.5-2B` | Apache-2.0 |
| Qwen3.5 4B (GGUF, Q4_K_M) | generator on every Czech set, beside the other larger models | `unsloth/Qwen3.5-4B-GGUF` @ `e87f1764`, a quantisation of `Qwen/Qwen3.5-4B` | Apache-2.0 |
| multilingual-e5-small | dense retriever | `intfloat/multilingual-e5-small` @ `614241f6` | MIT |
| multilingual-e5-base | dense retriever | `intfloat/multilingual-e5-base` @ `d1287505` | MIT |
| granite-embedding-107m-multilingual | dense retriever (the SQAD test splits) | `ibm-granite/granite-embedding-107m-multilingual` @ `d6cffd33` | Apache-2.0 |
| granite-embedding-278m-multilingual | dense retriever (the SQAD test splits) | `ibm-granite/granite-embedding-278m-multilingual` @ `a9cb5338` | Apache-2.0 |
| EmbeddingGemma 300m | dense retriever (the SQAD test splits) | `google/embeddinggemma-300m` @ `57c266a7` | **Gemma Terms of Use** |
| retromae-small-cs | dense retriever (Czech) | `Seznam/retromae-small-cs` @ `c9696fef` | CC BY 4.0 |
| simcse-dist-mpnet-paracrawl-cs-en | dense retriever (Czech, and the SQAD test splits) | `Seznam/simcse-dist-mpnet-paracrawl-cs-en` @ `9d379f86` | CC BY 4.0 |
| bge-m3 | dense retriever | `BAAI/bge-m3` @ `5617a9f6` | MIT |
| simplemma | the Czech lemmas of BM25 over lemmas, a Python package | PyPI `simplemma`, pinned in `uv.lock` | MIT |

*Qwen2.5-3B-Instruct is under the Qwen Research License (non-commercial research and evaluation use only; see the section below).*

**Gemma 4.** Google's Gemma 4 licence page (https://ai.google.dev/gemma/docs/gemma_4_license, read on
9 Nov 2026) is the Apache License 2.0, and both Hugging Face cards above say `apache-2.0`. Google publishes
no Q4_K_M file, so the release uses Unsloth's, pinned to the commits above (`src/measurerag/rag/generator.py`).

**Gemma 3n.** The cards of Unsloth's files, read at the revisions above on 11 Nov 2026, say `gemma`, as Google's cards of
`google/gemma-3n-E2B-it` and `google/gemma-3n-E4B-it` do: Google's Gemma Terms of Use (https://ai.google.dev/gemma/terms, last
modified on 1 Apr 2026, read on 11 Nov 2026), with its Prohibited Use Policy. Google's repositories are gated and Unsloth's are not.
This repository holds no Gemma 3n file. Its answers ship in `answers/`: the terms say that "Outputs are not deemed Model
Derivatives" and that "Google claims no rights in Outputs you generate using Gemma".

**EmbeddingGemma.** The card of `google/embeddinggemma-300m`, read at the revision above on 10 Nov 2026, says `gemma`: Google's
Gemma Terms of Use. The repository is gated, so downloading it needs a Hugging Face account that has accepted those terms. This
repository holds no EmbeddingGemma file; of its rankings of SQAD's passages, only each question's top 10 passage ids ship, in
`results/czech/records/rankings_v040_*.jsonl.gz`.

**Ministral 3.** Mistral AI's cards for `mistralai/Ministral-3-3B-Instruct-2512` and `mistralai/Ministral-3-8B-Instruct-2512`,
and Unsloth's for the files above, read on 9 Nov 2026, say `apache-2.0`. Mistral AI's own GGUF files do not load in the release's
llama.cpp build, so the release uses Unsloth's files of the same weights, pinned to the commits above.

**Qwen3.5.** The Qwen Team's cards for `Qwen/Qwen3.5-2B` and `Qwen/Qwen3.5-4B`, and Unsloth's for the files above, read on
9 Nov 2026, say `apache-2.0`. The release uses Unsloth's Q4_K_M files, pinned to the commits above.

**Qwen2.5-3B-Instruct: the Qwen Research License.** The model is licensed
under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All
Rights Reserved. The licence grants use for non-commercial purposes only,
defined as "research or evaluation purposes only"; commercial use needs a
licence from Alibaba Cloud. measureRAG downloads the model and measures it as
an evaluation. It does not redistribute the weights and does not train on the
model's outputs. Every table in the documents with a Qwen2.5 3B row (the Czech comparisons and
Qwen2.5's English dial) carries this licence next to it. Qwen2.5's English sweeps, shipped
untabulated under `results/crossover/natural/`, are results of it too.

**The dev-set benchmark of model families.** These models answered the 200 questions of the SQAD 3.2 dev
set once each, to choose the family that joins Qwen2.5. Their answers ship
in `answers/sqad32/`. Of the models below, Granite 4.0, Mistral AI's own Ministral 3 files and Ministral 3
14B answered no test question; the others are generators of the table above.
Every card, read at the revision below on 9 Nov 2026, says `apache-2.0`.

| model | Hugging Face repository @ revision | licence |
|---|---|---|
| Granite 4.0 1B and Granite 4.0 Micro (GGUF, Q4_K_M) | `ibm-granite/granite-4.0-1b-GGUF` @ `b27c2fe3`, `ibm-granite/granite-4.0-micro-GGUF` @ `ec48475f` | Apache-2.0 |
| Gemma 4 E2B and E4B | as in the table above | as above |
| Ministral 3 3B and 8B Instruct, Mistral AI's files (GGUF, Q4_K_M) | `mistralai/Ministral-3-3B-Instruct-2512-GGUF` @ `eb599d40`, `mistralai/Ministral-3-8B-Instruct-2512-GGUF` @ `0102285a` | Apache-2.0; these files do not load in the release's llama.cpp build, so they answered nothing |
| Ministral 3 3B, 8B and 14B Instruct, Unsloth's files (GGUF, Q4_K_M) | `unsloth/Ministral-3-3B-Instruct-2512-GGUF` @ `7564922f`, `unsloth/Ministral-3-8B-Instruct-2512-GGUF` @ `3731507e`, `unsloth/Ministral-3-14B-Instruct-2512-GGUF` @ `9b948841` | Apache-2.0 |
| Qwen2.5 1.5B and 3B | as in the table above | as above |
| Qwen3.5 2B and 4B (GGUF, Q4_K_M) | `unsloth/Qwen3.5-2B-GGUF` @ `f6d5376b`, `unsloth/Qwen3.5-4B-GGUF` @ `e87f1764`, quantisations of `Qwen/Qwen3.5-2B` and `Qwen/Qwen3.5-4B` | Apache-2.0 |

*Qwen2.5-3B-Instruct is under the Qwen Research License (non-commercial research and evaluation use only; see the section above).*

**Citations.**
* Gemma 4: Google DeepMind, as published on the model cards `google/gemma-4-E2B-it` and
  `google/gemma-4-E4B-it`. The GGUF files: Unsloth, as published on their cards.
* Gemma 3n: Google DeepMind, as published on the model cards `google/gemma-3n-E2B-it` and
  `google/gemma-3n-E4B-it`. The GGUF files: Unsloth, as published on their cards.
* EmbeddingGemma: Google DeepMind, as published on the model card `google/embeddinggemma-300m`.
* Granite 4.0: IBM, as published on the Hugging Face model cards.
* granite-embedding-107m-multilingual and granite-embedding-278m-multilingual: IBM, as published on
  their Hugging Face model cards.
* multilingual-e5: Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan
  Majumder and Furu Wei (2024). *Multilingual E5 Text Embeddings: A Technical
  Report*. arXiv:2402.05672.
* Ministral 3: Mistral AI, as published on the Hugging Face model cards.
* retromae-small-cs and simcse-dist-mpnet-paracrawl-cs-en: Jiří Bednář, Jakub Náplava, Petra
  Barančíková and Ondřej Lisický (2023). *Some Like It Small: Czech Semantic Embedding Models for
  Industry Applications*. arXiv:2311.13921. The models are Seznam's, as published on their Hugging
  Face model cards.
* bge-m3: Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian and
  Zheng Liu (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality,
  Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*.
  arXiv:2402.03216.
* Qwen2.5, as its model cards ask: Qwen Team (2024). *Qwen2.5: A Party of Foundation Models*.
  https://qwenlm.github.io/blog/qwen2.5/, September 2024; and An Yang, Baosong Yang, Binyuan Hui et al.
  (2024). *Qwen2 Technical Report*. arXiv:2407.10671. The later *Qwen2.5 Technical Report* is
  arXiv:2412.15115.
* Qwen3.5: the Qwen Team, as published on the Hugging Face model cards.
* simplemma: Adrien Barbaresi, as published on PyPI and in the package's repository.

## Datasets

| dataset | used for | Hugging Face source | licence as found |
|---|---|---|---|
| HotpotQA | the dial and the natural sweeps; the English RAG bench | `BeIR/hotpotqa` @ `a7e8bab2`, `BeIR/hotpotqa-qrels` @ `b15429e9`, `hotpotqa/hotpot_qa` @ `1908d6af` | CC BY-SA 4.0. The BeIR qrels card: a CC BY-SA 4.0 tag over "[Needs More Information]" |
| Natural Questions (BEIR nq) and NQ-open | the dial and the natural sweep, second dataset | `BeIR/nq` @ `b7253e6c`, `BeIR/nq-qrels` @ `519acd4e`, `google-research-datasets/nq_open` @ `5dd9790a` | CC BY-SA 3.0 on the nq-open card; CC BY-SA 4.0 on the BeIR mirror |
| SQAD 3.0 | the Czech RAG and the Czech dial | `Pehy/cs_sqad-3.0` at revision `f9e71384a8f7285412360cecabb7a852f49adbb9` (a filtered copy: 7,663 of the archive's 13,476 records as `scripts/sqad_copy_vs_archive.py` counts them; LINDAT's page states 13,477) | **LGPL-3.0** on LINDAT (http://hdl.handle.net/11234/1-3069); the copy's card says only `mit` and names no source; row for row it is `fewshot-goes-multilingual/cs_squad-3.0`, the filtered SQAD 3.0 of Štefánik et al. (2023), whose card states LGPL-3.0 |
| SQAD 3.2 | the Czech RAG and the Czech dial, beside the SQAD 3.0 copy | the archive on LINDAT, not on Hugging Face | **CC BY-SA 3.0** on LINDAT (http://hdl.handle.net/11234/1-5019) |
| SciFact | English retrieval | `BeIR/scifact` @ `b3b53356`, `BeIR/scifact-qrels` @ `2938d17d` | CC BY-NC 2.0 on the authors' release (`allenai/scifact`); CC BY-SA 4.0 on the BeIR card; the bench's records say CC BY-NC 4.0, a string the loader set, not one it read |
| NFCorpus | English retrieval | `BeIR/nfcorpus` @ `b5026a0e`, `BeIR/nfcorpus-qrels` @ `a451b3b2` | CC BY-SA 4.0 (BeIR card tag) |
| FiQA-2018 | English retrieval | `BeIR/fiqa` @ `979c07a7`, `BeIR/fiqa-qrels` @ `252958f2` | "FiQA-2018 task terms" in one record; CC BY-SA 4.0 on the BeIR card |
| CsFEVER v2 | Czech retrieval | `ctu-aic/csfever_v2` @ `20507f3f` | CC BY-SA 3.0 (Czech Wikipedia) |
| CWRCzech | not used: the loader stops, because access is by request only | not on Hugging Face | Seznam's terms, after a request |
| DaReCzech | not used: a loader only, never fetched | not on Hugging Face | Seznam's terms, after a request |
| toy fixture | `measurerag smoke` and the tests | bundled in `fixtures/toy/` | synthetic, generated by `scripts/make_fixtures.py`; this repository's licence |

**SQAD 3.0's licence.** LINDAT/CLARIAH-CZ's record for SQAD 3.0
(http://hdl.handle.net/11234/1-3069), read on 8 Nov 2026, says "GNU Library or "Lesser"
General Public License 3.0 (LGPL-3.0)". The build read the dataset through a
Hugging Face mirror whose card carries only an `mit` tag; the upstream licence
governs. The build first recorded CC BY-NC-SA 3.0, before the licence could be
checked on LINDAT; that was wrong, and `scripts/scrub_paths.py` replaces that
string in the run records. This repository includes no SQAD file.

**SQAD 3.2's licence.** LINDAT's record (http://hdl.handle.net/11234/1-5019), read on
8 Nov 2026, says "Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)". This repository
includes no SQAD 3.2 file.

**SQAD 3.0's copy is filtered.** The Hugging Face copy leaves out 5,798 of the 6,442
records whose answer sentence opens the article, and keeps all 7,019 whose answer sentence
comes later, as read by comparing it with the LINDAT archive on 8 Nov 2026. Its card states
no filter. The copy's own label does not change the licence above. The loader pins this
revision.

**The generated Czech answers.** `answers/` holds each answer the models generated on SQAD, as data apart from the code, so that the Czech numbers can be checked. An answer is model output, and it can quote the passages it was shown or repeat its question. The longest extract is 356 characters, on SQAD 3.2 (293 on the SQAD 3.0 copy). 42 answers copy a passage past 200 characters, all scoring 0: 27 of Gemma 3n E2B, two of Gemma 3n E4B, two of Gemma 4 E4B and 11 of Qwen3.5 2B; and 15 answers repeat their own question (Qwen2.5 1.5B, also all scoring 0; three SQAD 3.2 questions of four or five words). Two answers quote the e-mail addresses that a passage on XMPP gives as its fictional examples, Romeo's and Juliet's. Every answer stays as the model wrote it, so that its score recomputes from it; `scripts/czech_quote_scan.py`, which needs the data, records these counts by id in `results/czech/quote_scan.json`. `answers/sqad32/` holds the answers built on SQAD 3.2; those files are licensed as a whole under CC BY-SA 3.0, with attribution, and CC BY 4.0 does not relicense them. `answers/sqad30_copy/` holds the answers built on SQAD 3.0; those files are licensed under CC BY 4.0, except the spans of SQAD 3.0 an answer quotes, which keep LGPL-3.0 (its text and attribution are in the folder). The answers of
Qwen2.5-3B-Instruct are outputs of a model under the Qwen Research License, published as an
evaluation; the author does not train on them, and the licence's §4(b) asks for "Built with Qwen" or
"Improved using Qwen" in the documentation of a distributed model created or trained with its outputs. The questions, the passages and the gold answers
are not committed. Gemma 4's, Ministral 3's and Qwen3.5's answers, and the benchmark's in `answers/sqad32/`, are
outputs of models under Apache-2.0, which places no condition on outputs; Gemma 3n's are outputs of models under the Gemma Terms of Use, which claim no rights in them, though a model trained on Gemma outputs to perform like Gemma is a Model Derivative under those terms (§1.1(e)), and Gemma's use is subject to its Prohibited Use Policy.

**Share-alike and non-commercial terms.** Several datasets are share-alike, and
SciFact has non-commercial terms in at least one record. Those terms bind
redistributed or adapted content. This repository redistributes none of these datasets. Apart from
the spans above, its results are aggregate statistics and ids, licensed CC BY 4.0
([LICENSE-docs](LICENSE-docs)).

**NQ's gold passages are this kit's mapping, not an annotation.** See
[docs/METHOD.md](docs/METHOD.md#stated-assumption-the-nq-gold-mapping).

**Citations.**
* BEIR: Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava and
  Iryna Gurevych (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot
  Evaluation of Information Retrieval Models*. NeurIPS Datasets and Benchmarks.
* HotpotQA: Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W.
  Cohen, Ruslan Salakhutdinov and Christopher D. Manning (2018). *HotpotQA: A
  Dataset for Diverse, Explainable Multi-hop Question Answering*. EMNLP.
* Natural Questions: Tom Kwiatkowski et al. (2019). *Natural Questions: A
  Benchmark for Question Answering Research*. Transactions of the ACL.
* NQ-open: Kenton Lee, Ming-Wei Chang and Kristina Toutanova (2019). *Latent
  Retrieval for Weakly Supervised Open Domain Question Answering*. ACL.
* SQAD 3.0: Marek Medveď and Aleš Horák (2019). *sqad 3.0*. Masaryk University,
  NLP Centre; LINDAT/CLARIAH-CZ, http://hdl.handle.net/11234/1-3069.
* SQAD 3.2: Marek Medveď (2022). *SQAD 3.2*. Masaryk University, NLP Centre;
  LINDAT/CLARIAH-CZ, http://hdl.handle.net/11234/1-5019.
* Štefánik et al. (2023), who reported that about half of SQAD's questions have their answer at the
  beginning of the article, removed 90% of those whose answer starts in the first 50 characters, and
  published that filtered SQAD 3.0 as `fewshot-goes-multilingual/cs_squad-3.0`, which the copy this
  release reads matches row for row: Michal Štefánik, Marek Kadlčík, Piotr Gramacki, and Petr Sojka.
  2023. Resources and few-shot learners for in-context learning in Slavic languages. In Proceedings of
  the 9th Workshop on Slavic Natural Language Processing 2023 (SlavicNLP 2023), pages 94-105,
  Dubrovnik, Croatia. Association for Computational Linguistics.
* BenCzechMark, whose filter for SQAD 3.2 the debias rule follows: Martin Fajcik, Martin Docekal,
  Jan Dolezal et al. (2024). *BenCzechMark: A Czech-centric Multitask and Multimetric Benchmark for
  Large Language Models with Duel Scoring Mechanism*. arXiv:2412.17933, v2 of 22 May 2025, Appendix B.7.
  It removes 90% of the pairs whose answer is in the first 50 characters of the context and 90% of the
  pairs whose question and context have a TF-IDF cosine similarity above 0.9. This release's rule uses the
  Jaccard similarity of word sets above 0.138 in place of the cosine, and leaves out every flagged question.
* SciFact: David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van
  Zuylen, Arman Cohan and Hannaneh Hajishirzi (2020). *Fact or Fiction:
  Verifying Scientific Claims*. EMNLP.
* NFCorpus: Vera Boteva, Demian Gholipour, Artem Sokolov and Stefan Riezler
  (2016). *A Full-Text Learning to Rank Dataset for Medical Information
  Retrieval*. ECIR.
* FiQA-2018: Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross
  McDermott, Manel Zarrouk and Alexandra Balahur (2018). *WWW'18 Open
  Challenge: Financial Opinion Mining and Question Answering*. Companion
  Proceedings of The Web Conference.
* CsFEVER: Herbert Ullrich, Jan Drchal, Martin Rýpar, Hana Vincourová and
  Václav Moravec (2023). *CsFEVER and CTKFacts: acquiring Czech data for fact
  verification*. Language Resources and Evaluation.

## Software

Python packages are installed from PyPI as pinned in `uv.lock` and are not
vendored. Their licences are permissive (BSD, MIT, Apache-2.0, ISC, PSF-2.0, MIT-CMU and the PSF-based
licence of matplotlib, among others), except certifi (MPL-2.0) and tqdm (MPL-2.0 AND MIT). On Linux, torch
also brings NVIDIA's CUDA libraries as wheels (`nvidia-*` and the `cuda-toolkit` meta-package). Most are under
NVIDIA's own licence terms, not an open-source licence, and their metadata say NVIDIA proprietary where they
name a licence; `nvidia-nvtx` ships the Apache License 2.0, and `nvidia-nccl-cu13` ships a BSD-style licence
file although its metadata say NVIDIA proprietary. `cuda-bindings` and `cuda-pathfinder` are Apache-2.0 and
`triton` is MIT. They are installed from PyPI on Linux only and are not redistributed here. llama.cpp is used through llama-cpp-python (MIT). For the Czech
generations and the English dials of Gemma 3n, Gemma 4 and Ministral 3, the same version of llama-cpp-python was built from source with CUDA on the
machine that ran them. NVIDIA's CUDA toolkit is not redistributed.
