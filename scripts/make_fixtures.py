"""Generate the bundled toy fixtures: 200 documents, 20 queries, gold answers.

Run once; the output is committed so that `measurerag smoke` is fully offline.
The corpus is synthetic on purpose - no licence, no PII, no download.

The design matters. Each of the 40 entities gets five documents that all mention
the entity and differ only in *which fact* they carry, and each question is
worded with synonyms rather than the document's own words. That makes the task
lexically hard enough that BM25 does not score 1.0, so the smoke budget table
shows real spread instead of a wall of perfect scores - which is the whole point
of a fixture a newcomer learns the tool from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from measurerag.seeds import rng  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "toy"

EN_ENTITIES = [
    "Kestrel gantry crane", "Vardo ice core", "Calder alloy works", "Onymacris beetle reserve",
    "Tolvan radio array", "Brakken seed vault", "Marlow river ferry", "Ferrand clock tower",
    "Suldal kelp farm", "Astyanax cave pool", "Ashcombe tidal mill", "Fenwick peat bog",
    "Corrigan lighthouse", "Halstead orchard", "Wrenfield bell foundry", "Pellow salt marsh",
    "Draycott wool mill", "Ormskirk canal lock", "Tarnbrook moth station", "Grendel slate mine",
]
CS_ENTITIES = [
    "Zbirozska drazka", "sklarna Lhotka", "pivovar Radhost", "lom Trebonin",
    "hnizdiste Sycek", "papirna Otava", "susarna Bystrina", "vozovna Kunratice",
    "pramen Solanka", "turbina Vetrnik", "rybnik Kamenec", "hrad Roupov",
    "mlyn Podhradi", "jeskyne Bozkov", "rozhledna Kleta", "kovarna Zelezny Brod",
    "prehrada Vranov", "zahrada Cernosice", "solnice Mestec", "vapenka Kotoun",
]

# (fact sentence, question worded differently, unit for the answer)
EN_ASPECTS = [
    ("The structure of {e} was measured at {v} tonnes during the last survey.",
     "What is the total weight of {e}?", "{v} tonnes"),
    ("{e} entered regular service in {v} after a long delay.",
     "In which year did {e} open?", "{v}"),
    ("Annual upkeep of {e} is billed at {v} thousand crowns.",
     "What does it cost each year to maintain {e}?", "{v} thousand crowns"),
    ("{e} is looked after by a team of {v} people, most of them volunteers.",
     "How large is the staff of {e}?", "{v}"),
    ("{e} lies {v} kilometres from the nearest district town.",
     "How far away is {e} from the closest town?", "{v} kilometres"),
]
CS_ASPECTS = [
    ("Konstrukce {e} byla pri posledni prohlidce zmerena na {v} tun.",
     "Jakou celkovou hmotnost ma {e}?", "{v} tun"),
    ("{e} zacala pravidelne slouzit v roce {v} po dlouhem odkladu.",
     "Ve kterem roce byla otevrena {e}?", "{v}"),
    ("Rocni udrzba {e} se uctuje na {v} tisic korun.",
     "Kolik stoji kazdy rok provoz {e}?", "{v} tisic korun"),
    ("O {e} se stara skupina {v} lidi, vetsinou dobrovolniku.",
     "Jak velky tym ma na starosti {e}?", "{v}"),
    ("{e} lezi {v} kilometru od nejblizsiho okresniho mesta.",
     "Jak daleko je {e} od nejblizsiho mesta?", "{v} kilometru"),
]

EN_FRAMES = [
    "Field notes on {e} were collected over three seasons by the regional survey.",
    "A short technical review of {e} appeared in the county engineering digest.",
    "Maintenance records for {e} are kept in the district archive.",
    "Visitors to {e} follow the posted safety instructions at all times.",
    "The operating history of {e} is summarised here for new staff.",
]
CS_FRAMES = [
    "Terenni poznamky o {e} sbiral krajsky tym po tri sezony.",
    "Kratky technicky prehled o {e} vysel v okresnim sborniku.",
    "Zaznamy o udrzbe {e} jsou ulozeny v okresnim archivu.",
    "Navstevnici {e} se ridi vyvesenymi bezpecnostnimi pokyny.",
    "Provozni historie {e} je zde shrnuta pro nove zamestnance.",
]

EN_TAIL = "The entry is reviewed once a year by the regional committee."
CS_TAIL = "Zaznam jednou rocne kontroluje krajska komise."


def main() -> None:
    r = rng("toy-fixture-v2")
    docs: list[dict] = []
    queries: list[dict] = []
    qrels: list[tuple[str, str, int]] = []

    entities = [(e, "en") for e in EN_ENTITIES] + [(e, "cs") for e in CS_ENTITIES]
    # ten English and ten Czech entities carry a query; the other twenty are the
    # distractor mass that stops a retriever winning by luck
    with_query = set(
        r.choice(len(EN_ENTITIES), size=10, replace=False).tolist()
    ) | {
        len(EN_ENTITIES) + i
        for i in r.choice(len(CS_ENTITIES), size=10, replace=False).tolist()
    }

    for e_idx, (entity, lang) in enumerate(entities):
        aspects = EN_ASPECTS if lang == "en" else CS_ASPECTS
        frames = EN_FRAMES if lang == "en" else CS_FRAMES
        tail = EN_TAIL if lang == "en" else CS_TAIL
        key = f"e{e_idx:02d}"

        values = [
            str(int(r.integers(40, 990))),          # tonnes
            str(int(r.integers(1890, 2020))),        # year
            str(int(r.integers(11, 480))),           # cost
            str(int(r.integers(2, 60))),             # staff
            str(int(r.integers(3, 90))),             # distance
        ]
        # shuffle which frame wraps which aspect so position carries no signal
        order = r.permutation(5).tolist()
        ask = int(r.integers(0, 5)) if e_idx in with_query else None

        for slot, aspect_idx in enumerate(order):
            fact, question, answer_tpl = aspects[aspect_idx]
            did = f"{key}-{slot}"
            text = " ".join(
                [
                    frames[slot].format(e=entity),
                    fact.format(e=entity, v=values[aspect_idx]),
                    tail,
                ]
            )
            docs.append({"doc_id": did, "title": entity, "text": text, "lang": lang})
            if ask is not None and aspect_idx == ask:
                qid = f"q-{key}"
                queries.append(
                    {
                        "query_id": qid,
                        "text": question.format(e=entity),
                        "lang": lang,
                        "answers": [answer_tpl.format(v=values[aspect_idx])],
                    }
                )
                qrels.append((qid, did, 1))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for d in sorted(docs, key=lambda x: x["doc_id"]):
            fh.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")
    with open(OUT / "queries.jsonl", "w", encoding="utf-8") as fh:
        for q in sorted(queries, key=lambda x: x["query_id"]):
            fh.write(json.dumps(q, ensure_ascii=False, sort_keys=True) + "\n")
    with open(OUT / "qrels.tsv", "w", encoding="utf-8") as fh:
        fh.write("query_id\tdoc_id\trelevance\n")
        for qid, did, rel in sorted(qrels):
            fh.write(f"{qid}\t{did}\t{rel}\n")
    print(f"{len(docs)} docs, {len(queries)} queries -> {OUT}")


if __name__ == "__main__":
    main()
