"""A generation cache keyed on the generator, the token cap, the seed and the prompt.

The dial's levels share most of their prompts (a question whose gold passage
survives to p = 0.6 has the same context at 0, 0.2 and 0.4), the dial at p = 0
shares its prompts with the natural sweep, and a second statistics pass must
not regenerate anything. So an answer is stored once per (generator, max_tokens,
seed, prompt) and looked up by the SHA-256 of that tuple.

Thread count and wall-clock are recorded beside each answer but are not part of
the key. That is safe only because it was checked: the cached answers reproduce run 1's harness strings.

The key is not quite everything that decides an answer: a near tie can also turn
on the prompt before it in the same process. So a cache holds
one answer per prompt, from whichever order reached that prompt first.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..hashing import sha256_json


def gen_key(generator: str, max_tokens: int, seed: int, prompt: str) -> str:
    return sha256_json(
        {"generator": generator, "max_tokens": int(max_tokens), "seed": int(seed), "prompt": prompt}
    )


class GenCache:
    """Append-only JSONL. A torn last line (a killed process) is ignored."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entries: dict[str, dict] = {}
        self.last_of_stage: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.entries[row["key"]] = row
                    if row.get("stage"):
                        self.last_of_stage[row["stage"]] = row

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def put(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
        self.entries[row["key"]] = row
        if row.get("stage"):
            self.last_of_stage[row["stage"]] = row
