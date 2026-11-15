# The generated Czech answers

Every answer the models generated on SQAD for this release, so that the Czech numbers can be checked. They are data, apart from the code.

| folder | built on | licence |
|---|---|---|
| `sqad30_copy/` | the SQAD 3.0 copy | the files CC BY 4.0, except the quoted spans of SQAD 3.0, which keep LGPL-3.0 (text in `sqad30_copy/LICENSE`, with the GPL-3.0 it incorporates in `sqad30_copy/GPL-3.0.txt`; attribution in `sqad30_copy/NOTICE.md`) |
| `sqad32/` | SQAD 3.2 | the files as a whole CC BY-SA 3.0 (text in `sqad32/LICENSE`, attribution in `sqad32/NOTICE.md`) |

An answer is model output, and it can quote the passages the model was shown or repeat its question: the longest extract is 356 characters, and 15 answers repeat their own question ([THIRD_PARTY.md](../THIRD_PARTY.md)). Every answer is as the model wrote it. No question, passage or gold answer is here beyond what an answer quotes. A row holds ids, the answer and its scores, and `scripts/rescore_czech.py --verify-scores` recomputes the scores against a copy of SQAD you download yourself.

- `answers_<set>.jsonl.gz`: the answers behind the reported numbers, one row per configuration, dial level and question.
- `answers_cross_version.jsonl.gz`: one question on each version's passages.
- `answers_dev.jsonl.gz`: the dev set, used only to choose the prompt.
- `answers_dev_gx10.jsonl.gz`: the same dev set answered again on the second machine, before it generated the answers behind the reported numbers.
- `answers_dev_gx10_repeat.jsonl.gz`: V1's dev answers on that machine a second time, from an empty cache, to show that it repeats itself.
- `answers_model_bench.jsonl.gz` and `answers_model_bench_mistral.jsonl.gz`: the dev set answered by each model of the benchmark that chose the second family, and by Ministral 3 beside it.
- `answers_released_v020.jsonl.gz`: the released questions answered on the laptop under the earlier instruction to answer yes or no in English, one side of the prompt comparison.
- `first_cut/`: an earlier round of answers under the ano-or-ne line, on the laptop: the other side of the prompt comparison, and the answers the sets were sized from.

`scripts/rescore_czech.py` recomputes every Czech number from these files and from `results/czech/records/`.

File and folder names that contain v020, v030, v040 or first_cut are build labels from before the first public release.
