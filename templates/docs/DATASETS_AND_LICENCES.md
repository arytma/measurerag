# Datasets, models and licences

**What this repository contains and under which licence.** The code is
Apache-2.0 ([LICENSE](../LICENSE)). The documentation and the results (every
file under `results/`, the figures, the tables) are CC BY 4.0
([LICENSE-docs](../LICENSE-docs)). The generated Czech answers in `answers/`: files under
`answers/sqad32/` are licensed as a whole under CC BY-SA 3.0, and files under `answers/sqad30_copy/`
under CC BY 4.0, except the spans of SQAD 3.0 an answer quotes, which keep LGPL-3.0 (see below). Every third-party dataset and model, with
its licence and citation, is listed in [THIRD_PARTY.md](../THIRD_PARTY.md);
the copyright notice is in [NOTICE](../NOTICE).

**What it does not contain.** No model weights and no dataset. Scripts download
both from Hugging Face at a run's start (`measurerag fetch`,
`scripts/fetch_verify_models.py`), except SQAD 3.2 and the SQAD 3.0 archive, which
you download from LINDAT yourself. Every slice's SHA-256 is in its run's
`manifest.json`. Tracked files hold aggregate numbers, hashes and notes the kit
wrote itself and, for Czech only, per-question records under
`answers/` and `results/czech/records/`: ids, scores and each generated answer. No question,
passage or gold answer from any dataset is committed, except as quoted inside a generated answer under
`answers/` (see below).

**Licences were read from each Hugging Face card**, metadata and text, at the
revisions below. Where a card and the licence string a loader records
disagree, both are shown and left as found; nothing is resolved by guessing.

## Models

| model | role | Hugging Face repo @ revision | licence (card) | restrictions |
|---|---|---|---|---|
| Gemma 3n E2B, instruction-tuned, Q4_K_M | Czech generator beside Gemma 4 E2B and the other smaller models, and on the English dial | `unsloth/gemma-3n-E2B-it-GGUF` @ `d7bd20d510eff26f3f95260c50f032fee8931a35` | **Gemma Terms of Use** (`gemma`) | **Google's Prohibited Use Policy; Google's own repository is gated** |
| Gemma 3n E4B, instruction-tuned, Q4_K_M | Czech generator beside Gemma 4 E4B and the other larger models, and on the English dial | `unsloth/gemma-3n-E4B-it-GGUF` @ `90fa8b0e431faeae50c305828bc260d6f71720e1` | **Gemma Terms of Use** (`gemma`) | **Google's Prohibited Use Policy; Google's own repository is gated** |
| Gemma 4 E2B, instruction-tuned, Q4_K_M | Czech generator beside 1.5B; English dial | `unsloth/gemma-4-E2B-it-GGUF` @ `0314792d7f1f7e229411f620751375812bb9faf2` | Apache-2.0 | none beyond Apache-2.0 |
| Gemma 4 E4B, instruction-tuned, Q4_K_M | Czech generator beside 3B; English dial | `unsloth/gemma-4-E4B-it-GGUF` @ `bfc15c382204943c3a8fff0c750b94ae2364d7a3` | Apache-2.0 | none beyond Apache-2.0 |
| Ministral 3 3B Instruct 2512, Q4_K_M | Czech generator beside the other smaller models; English dial; the English sweeps' smaller generator | `unsloth/Ministral-3-3B-Instruct-2512-GGUF` @ `7564922f37fa5bbb62b87f09a55c12f1f91d7a6a` | Apache-2.0 | none beyond Apache-2.0 |
| Ministral 3 8B Instruct 2512, Q4_K_M | Czech generator beside the other larger models; English dial; the English sweeps' larger generator | `unsloth/Ministral-3-8B-Instruct-2512-GGUF` @ `3731507ec3e867db16d620f73e14d689125758f4` | Apache-2.0 | none beyond Apache-2.0 |
| Qwen2.5-1.5B-Instruct, Q4_K_M | Czech generator; English dial; the calibration probe's fixed generator | `{{models.qwen2.5-1.5b.repo}}` @ `{{models.qwen2.5-1.5b.revision}}` | Apache-2.0 | none beyond Apache-2.0 |
| Qwen2.5-3B-Instruct, Q4_K_M | Czech generator; English dial | `{{models.qwen2.5-3b.repo}}` @ `{{models.qwen2.5-3b.revision}}` | **Qwen Research License** (`qwen-research`) | **non-commercial: "research or evaluation purposes only"** |
| Qwen3.5 2B, Q4_K_M | Czech generator beside the other smaller models | `unsloth/Qwen3.5-2B-GGUF` @ `f6d5376be1edb4d416d56da11e5397a961aca8ae` | Apache-2.0 | none beyond Apache-2.0 |
| Qwen3.5 4B, Q4_K_M | Czech generator beside the other larger models | `unsloth/Qwen3.5-4B-GGUF` @ `e87f176479d0855a907a41277aca2f8ee7a09523` | Apache-2.0 | none beyond Apache-2.0 |
| multilingual-e5-small | dense retriever | `{{models.e5-small.repo}}` @ `{{models.e5-small.revision}}` | MIT | none stated |
| multilingual-e5-base | dense retriever | `{{models.e5-base.repo}}` @ `{{models.e5-base.revision}}` | MIT | none stated |
| bge-m3 | dense retriever | `{{models.bge-m3.repo}}` @ `{{models.bge-m3.revision}}` | MIT | none stated |
| retromae-small-cs | dense retriever (Czech) | `Seznam/retromae-small-cs` @ `c9696feffb314a7f104b4be71e46dc197c193232` | CC BY 4.0 | attribution |
| simcse-dist-mpnet-paracrawl-cs-en | dense retriever (Czech, and the SQAD test splits) | `Seznam/simcse-dist-mpnet-paracrawl-cs-en` @ `9d379f864f29ede6a76167632ac1366efe0119be` | CC BY 4.0 | attribution |
| granite-embedding-107m-multilingual | dense retriever (the SQAD test splits) | `ibm-granite/granite-embedding-107m-multilingual` @ `d6cffd338414d6a1c1f5decfad5fec62eebc90d5` | Apache-2.0 | none beyond Apache-2.0 |
| granite-embedding-278m-multilingual | dense retriever (the SQAD test splits) | `ibm-granite/granite-embedding-278m-multilingual` @ `a9cb5338491faf32b73dd17b714a31821c021bbf` | Apache-2.0 | none beyond Apache-2.0 |
| EmbeddingGemma 300m | dense retriever (the SQAD test splits) | `google/embeddinggemma-300m` @ `57c266a740f537b4dc058e1b0cda161fd15afa75` | **Gemma Terms of Use** (`gemma`) | **gated: the download needs a Hugging Face account that has accepted Google's usage licence; Gemma's Prohibited Use Policy applies** |

**Gemma 4's licence.** Both cards say `apache-2.0` and link Google's Gemma 4 licence page,
which is the Apache License 2.0 (read on 9 Nov 2026). Google publishes no Q4_K_M file, so the
release uses Unsloth's quantisations of `google/gemma-4-E2B-it` and `google/gemma-4-E4B-it`, pinned
to the commits above in `src/measurerag/rag/generator.py`; a commit fixes each file's LFS SHA-256.

**Ministral 3's licence.** Mistral AI's cards for Ministral 3 3B and 8B Instruct 2512, and Unsloth's
GGUF cards at the revisions above, say `apache-2.0`. Mistral AI's own GGUF files do not load in the
release's llama.cpp build, so the release uses Unsloth's Q4_K_M files of the same weights.

**Qwen3.5's licence.** The Qwen Team's cards for Qwen3.5 2B and 4B, and Unsloth's GGUF cards at the
revisions above, say `apache-2.0` (read on 9 Nov 2026). The release uses Unsloth's Q4_K_M files.

**Gemma 3n's licence.** Unsloth's GGUF cards at the revisions above, and Google's cards for
`google/gemma-3n-E2B-it` and `google/gemma-3n-E4B-it`, say `gemma`: Google's Gemma Terms of Use
(https://ai.google.dev/gemma/terms, last modified on 1 Apr 2026), with its Prohibited Use Policy (read on
11 Nov 2026). Google's repositories are gated; Unsloth's are not. This repository holds no Gemma 3n file.
Its generated answers ship in `answers/`: under the terms, "Outputs are not deemed Model Derivatives",
and Google claims no rights in them.

**EmbeddingGemma's licence.** The card says `gemma`, Google's Gemma Terms of Use, and the repository
is gated (read on 10 Nov 2026). This repository holds no EmbeddingGemma file; of its rankings of SQAD's
passages, only each question's top 10 passage ids ship. The granite-embedding cards say `apache-2.0`. BM25 over
lemmas takes its Czech lemmas from simplemma, a Python package under MIT.

The models of the dev-set benchmark that answered no test question (Granite 4.0, Mistral AI's own
Ministral 3 files and Ministral 3 14B) are listed with their revisions in
[THIRD_PARTY.md](../THIRD_PARTY.md); every card says Apache-2.0.

**Qwen2.5-3B-Instruct's licence.** The Qwen2.5-3B-Instruct card (and its GGUF
repository) is tagged `qwen-research`, and its LICENSE says:

> "Non-Commercial" shall mean for research or evaluation purposes only.
>
> You are granted a non-exclusive, worldwide, non-transferable and royalty-free
> limited license under Alibaba Cloud's intellectual property or other rights
> owned by us embodied in the Materials to use, reproduce, distribute, copy,
> create derivative works of, and make modifications to the Materials FOR
> NON-COMMERCIAL PURPOSES ONLY.
>
> If you are commercially using the Materials, you shall request a license from us.

This kit downloads the model and measures it; it does not redistribute it and
does not train on its outputs. The author decided to publish the
measurements as research results, with the licence stated next to every table
that includes the 3B model. The 1.5B model is Apache-2.0 and carries no such restriction.

The files of five models were verified against the records of the runs that first used them: revision and
shipped byte count for multilingual-e5-small, multilingual-e5-base and bge-m3, and revision, LFS SHA-256 and byte count
for Qwen2.5's two GGUF files (all verified: {{models.all_verified}}; `results/crossover/models_manifest.json` lists a
SHA-256 per file). The other GGUF files load the commit pinned in `src/measurerag/rag/generator.py`, which fixes each
file's LFS SHA-256, and retromae-small-cs loads the commit pinned in `src/measurerag/retrievers/dense.py`. The granite
models, EmbeddingGemma and simcse-cs-en load their repository's `main` ([REPRODUCE.md](REPRODUCE.md), §3).

## Datasets

| dataset | used for | Hugging Face source | size used | licence (card) | recorded by this kit | open question |
|---|---|---|---|---|---|---|
| HotpotQA | the crossover (dial and natural sweep) | `BeIR/hotpotqa` @ `a7e8bab2`, `BeIR/hotpotqa-qrels` @ `b15429e9`, `hotpotqa/hotpot_qa` @ `1908d6af` | {{data.hotpotqa.docs}} passages, {{hotpotqa.n}} questions | CC BY-SA 4.0 ("HotpotQA is distributed under a CC BY-SA 4.0 License") | {{runs.en_rag.licence_note}} | qrels card text says "[Needs More Information]" |
| Natural Questions | the crossover's second dataset | `BeIR/nq` @ `b7253e6c`, `BeIR/nq-qrels` @ `519acd4e`, `google-research-datasets/nq_open` @ `5dd9790a` | {{data.nq.docs}} passages, {{nq.n}} questions | BeIR: CC BY-SA 4.0 (tag); nq-open: "All of the Natural Questions data is released under the CC BY-SA 3.0 license" | {{runs.nq_k5.licence_note}} | CC BY-SA 3.0 or CC BY-SA 4.0 |
| SQAD 3.0 | the Czech results, on a filtered copy; the Czech appendix of the crossover | `{{czech_checks.copy_repo}}` at revision `{{czech_checks.copy_revision}}` (a filtered copy: {{czech_checks.copy_rows}} of the archive's {{czech_checks.archive_records}} records; LINDAT's page states {{czech_checks.archive_records_stated}}) | {{runs.cz_rag.docs}} passages, {{runs.cz_rag.queries}} questions | `mit` tag only; the card has no text and names no upstream | {{runs.cz_rag.licence_note}} | none: LINDAT's record states the licence |
| SQAD 3.2 | the Czech results, beside the SQAD 3.0 copy | LINDAT, not Hugging Face: http://hdl.handle.net/11234/1-5019, archive MD5 `39c29a416d2542ecc924542ad79b4cfd` | {{runs.cz_rag_32.docs}} passages, {{runs.cz_rag_32.queries}} questions | no card: LINDAT's record states CC BY-SA 3.0 | {{runs.cz_rag_32.licence_note}} | none: LINDAT's record states the licence |
| SciFact | English retrieval | `BeIR/scifact` @ `b3b53356`, `BeIR/scifact-qrels` @ `2938d17d` | {{runs.en_small.docs}} abstracts, {{runs.en_small.queries}} claims | CC BY-SA 4.0 (tag only) | CC BY-NC 2.0 on the authors' release (`allenai/scifact`); the bench recorded {{runs.en_small.licence_note}}, a string its loader set | NC or SA |
| NFCorpus | English retrieval | `BeIR/nfcorpus` @ `b5026a0e`, `BeIR/nfcorpus-qrels` @ `a451b3b2` | {{runs.en_nfcorpus.docs}} documents, {{runs.en_nfcorpus.queries}} queries | CC BY-SA 4.0 (tag only) | {{runs.en_nfcorpus.licence_note}} | upstream not stated on the card |
| FiQA-2018 | English retrieval | `BeIR/fiqa` @ `979c07a7`, `BeIR/fiqa-qrels` @ `252958f2` | {{runs.en_fiqa.docs}} documents (seeded sample), {{runs.en_fiqa.queries}} queries | CC BY-SA 4.0 (tag only) | {{runs.en_fiqa.licence_note}} | which terms govern |
| CsFEVER v2 | Czech retrieval | `ctu-aic/csfever_v2` @ `20507f3f` | {{runs.cz_small.docs}} and {{runs.cz_full.docs}} pages, {{runs.cz_small.queries}} claims | CC BY-SA 3.0 (tag) | {{runs.cz_small.licence_note}} | the loader script's own licence field is empty |
| toy fixture | `measurerag smoke`, tests | bundled, `fixtures/toy/` | synthetic | this repository's licence | generated by `scripts/make_fixtures.py` | none |

Every Hugging Face dataset above is read at a pinned commit, set in its loader under `src/measurerag/datasets/`:
the commit the build's cache was made from, which is also each repository's `main` as checked on 14 Nov 2026.
None of these repositories has a commit after 9 Apr 2026, before the first measurement, so every result read the
same files. Each slice's corpus hash (`meta.json`, and `results/crossover.json`) checks this again.

**The SQAD 3.0 copy is filtered.** Its card states no filter, so this was read off by
comparing the copy with the LINDAT archive on 8 Nov 2026. Every row of the copy has the
same id and question in the archive. Of the {{czech_checks.opens_all}} records whose answer
sentence opens the article it leaves out {{czech_checks.opens_dropped}}
({{czech_checks.opens_dropped_pct}}); it keeps all {{czech_checks.later_kept}} records whose
answer sentence comes later, and leaves out the {{czech_checks.not_located}} whose sentence
could not be located.

**Where the copy comes from.** Its card names no source; the account `Pehy` uploaded it on 18 Jun 2024
and holds no other dataset. But it is, row for row, `fewshot-goes-multilingual/cs_squad-3.0` at revision
`e2a26bd1bc6d1ff15a33ab9d2ed118054151a784`, published on 20 Dec 2022: every split has the same ids, and
each id the same question, context and answers (`scripts/sqad_copy_source.py` downloads both pinned
revisions and recomputes `results/czech/copy_source.json`). That organization publishes the templates and
models of Štefánik, Kadlčík, Gramacki and Sojka (2023), whose paper reports that about half of SQAD's
questions have their answer at the beginning of the article and removes {{stefanik.share_removed}} of the
questions whose answer starts in the first {{stefanik.first_chars}} characters. Its card describes removing the bias of
answers in the article's first sentence, and states LGPL-3.0, the licence on LINDAT; the comparison with the archive
matches the card's description, with every removed record one whose answer sentence opens the article. So the copy's filter is theirs, and the copy is cited as their
work ([THIRD_PARTY.md](../THIRD_PARTY.md)). Michal Štefánik is also an author of BenCzechMark, whose filter
for SQAD 3.2 keeps that rule and adds one on question and context overlap
(https://arxiv.org/abs/2412.17933, Appendix B.7). The archive holds
{{czech_checks.archive_records}} records as `scripts/sqad_copy_vs_archive.py` counts them; LINDAT's page for
the item states {{czech_checks.archive_records_stated}}. The licence is the one on LINDAT, LGPL-3.0, not the
copy's own label.
The loader pins this revision, and a slice's corpus hash shows that the data still matches
([REPRODUCE.md](REPRODUCE.md)). `scripts/sqad_copy_vs_archive.py` recomputes the two record
counts, this filter and the two overlap thresholds from the archive you download from LINDAT
(http://hdl.handle.net/11234/1-3069, MD5 `{{czech_checks.archive_md5}}`, which the script
checks first) and the pinned copy, and compares them with `results/czech_checks.json`.

**The generated Czech answers.** `answers/` holds each answer the models generated on SQAD,
as data apart from the code, so that the Czech numbers can be checked. An answer is model
output, and it can quote the passages it was shown and repeat its question. The longest extract is {{cz.quotes.longest}} characters. {{cz.quotes.over_line}} answers of {{cz.quotes.over_line_by}} copy a passage past {{cz.quotes.line}} characters, {{cz.quotes.over_line_scoring}}, and {{cz.quotes.repeated_rows}} answers repeat their own question ({{cz.quotes.repeated_by}}, also {{cz.quotes.repeated_scoring}}).
`answers/sqad30_copy/` holds
the answers built on SQAD 3.0, with the LGPL-3.0 text and attribution. `answers/sqad32/` holds
the answers built on SQAD 3.2; those files are under CC BY-SA 3.0, with attribution
([THIRD_PARTY.md](../THIRD_PARTY.md)). The answers of the
3B model are outputs of a model under the Qwen Research License, published as an evaluation
and not for training another model.

**NQ's gold passages are mapped, not annotated.** BEIR's `nq` ships passages
and relevance judgements without answers; `nq_open` ships answers without
passages. A passage counts as gold only if BEIR judges it relevant **and** it
contains an nq-open answer as a whole-token span. Coverage:
{{data.nq.beir_queries}} BEIR test questions with judgements;
{{data.nq.joined}} ({{data.nq.joined_pct}}) join to nq-open on normalised
question text; {{data.nq.mapped}} of those ({{data.nq.mapped_pct}}) have a
mapped gold passage; {{nq.n}} are sampled with a fixed seed.

**Share-alike.** HotpotQA, Natural Questions, CsFEVER and the BeIR corpora are
share-alike, and so is SQAD 3.2. Share-alike binds redistributed or adapted
content. This repository redistributes none of these datasets, and apart from the
spans above its results are aggregate statistics and ids, which is the reading
under which the results are licensed CC BY 4.0.

**Not a licence matter, but it bears on the results.** The multilingual-e5
model cards list the Natural Questions and HotpotQA *training* sets among their
fine-tuning data. This kit evaluates on the other splits, but a retriever that
has seen the training distribution may look better on these two datasets than
it would on unseen ones ([LIMITATIONS.md](LIMITATIONS.md)).

## Citations the sources ask for

* **BEIR**: Thakur et al. 2021, "BEIR: A Heterogeneous Benchmark for Zero-shot
  Evaluation of Information Retrieval Models" (every `BeIR/*` card).
* **HotpotQA**: Yang et al. 2018, "HotpotQA: A Dataset for Diverse, Explainable
  Multi-hop Question Answering".
* **Natural Questions**: Kwiatkowski et al. 2019, "Natural Questions: A
  Benchmark for Question Answering Research"; **NQ-open**: Lee et al. 2019,
  "Latent Retrieval for Weakly Supervised Open Domain Question Answering".
* **Qwen2.5**: "Qwen2.5: A Party of Foundation Models", and the Qwen2
  Technical Report.
* **multilingual-e5**: Wang et al. 2024, "Multilingual E5 Text Embeddings: A
  Technical Report".
* **bge-m3**: "BGE M3-Embedding: Multi-Lingual, Multi-Functionality,
  Multi-Granularity Text Embeddings Through Self-Knowledge Distillation".
* **CsFEVER**: Ullrich et al. 2023, "CsFEVER and CTKFacts: acquiring Czech data
  for fact verification".
* **SciFact** (Wadden et al. 2020), **NFCorpus** (Boteva et al. 2016),
  **FiQA-2018** (Maia et al. 2018): as recorded in the run manifests.

## Software

Nothing is vendored; `uv.lock` pins every package. All direct dependencies are
permissive (BSD, MIT, Apache-2.0, PSF-based for matplotlib), except tqdm, which is
MPL-2.0 AND MIT, a file-level copyleft that matters only if its source were vendored.
[THIRD_PARTY.md](../THIRD_PARTY.md) lists the licences of the packages they bring in.
