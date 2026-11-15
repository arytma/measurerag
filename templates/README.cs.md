[English](README.md) | Čeština

# measureRAG

**Benchmark vyhledávání a RAG s ohledem na rozpočet, v češtině a angličtině, pro jeden počítač bez připojení
k síti.**

Toto je jednostránkové české shrnutí. Podrobnosti, všechny tabulky a omezení jsou v anglických dokumentech, na které
odkazuje. Čísla zde mají desetinnou čárku; v anglických dokumentech a v souborech výsledků mají desetinnou tečku.

## Co je measureRAG

measureRAG měří vyhledávání (retrieval) a generování odpovědí s vyhledáváním (RAG) v podmínkách nasazení na jednom
počítači jen s procesorem, s několika gigabajty paměti a bez sítě. Rozpočet je na prvním místě: konfigurace se buď
vejde do limitů paměti, latence vyhledávání a celkové latence, nebo ne, a kvalita odpovědí se uvádí uvnitř limitů, do
kterých se vejde. Vše běží lokálně: žádné klíče k API, žádné cloudové modely, žádná telemetrie.

Hlavní otázka zní: kam na daném stroji vložit další gigabajt paměti, aby přinesl co nejvíc kvality odpovědí, do
silnějšího vyhledávače, nebo do většího generátoru? Na anglických sadách HotpotQA a Natural Questions je odpovědí
tabulka, ne jeden bod překřížení: kolik
pasáží se správnou odpovědí musí lepší vyhledávač navíc dostat do kontextu, aby vyvážil krok k většímu generátoru
([README.md](README.md#result-the-break-even-gold-gain)).

## České datové sady

* **CsFEVER v2** pro vyhledávání: {{runs.cz_small.queries|cs}} tvrzení nad {{runs.cz_small.docs|cs}} stránkami
  Wikipedie a totéž nad {{runs.cz_full.docs|cs}} dokumenty.
* **SQAD 3.2**, celé vydání z LINDAT pod licencí CC BY-SA 3.0 (http://hdl.handle.net/11234/1-5019). Stáhnete si ho
  sami.
* **Kopie SQAD 3.0** na Hugging Face, `{{czech_checks.copy_repo}}`, pevně na jedné revizi: filtrovaná podmnožina
  vydání SQAD 3.0 z LINDAT. Platí licence z LINDAT, LGPL-3.0 (http://hdl.handle.net/11234/1-3069), ne štítek kopie.

Odpovědi se hodnotí na čtyřech sadách SQAD a žádná z nich není hlavní: SQAD 3.2 očištěná (*debiased*), SQAD 3.2
celá (*full*), vydaných {{cz.released.n|cs}} otázek kopie SQAD 3.0 (*released 200*) a očištěná kopie (*copy
debiased*). Očištěná sada vynechává otázky, jejichž odpověď začíná v prvních {{bcm.first_chars|cs}} znacích kontextu
nebo jejichž slova se s kontextem silně překrývají. Sady se liší v tom, které otázky ponechávají, proto porovnávejte
rozdíly uvnitř jedné sady, ne úrovně mezi sadami.

Hlavní metrikou je přesná shoda (EM) s mapováním ano/ne: anglické odpovědi yes a no se počítají jako ano a ne, protože
Qwen2.5 stále odpovídá na mnoho otázek typu ano/ne anglicky. Surová EM a dodržení formátu (*compliance*) stojí vedle
ní ([docs/METRICS.md](docs/METRICS.md)). České odpovědi vznikly na GPU druhého stroje, takže žádný český generátor
nemá latenci na procesoru ani verdikt rozpočtu.

## Výsledky v češtině

**Vyhledávání.**

* **CsFEVER v2:** BM25 bez diakritiky má nDCG@10 {{runs.cz_small.bm25-nodiacritics.ndcg10|cs}}, BM25
  {{runs.cz_small.bm25.ndcg10|cs}} a multilingual-e5-small {{runs.cz_small.e5-small.ndcg10|cs}}; nad
  {{runs.cz_full.docs|cs}} dokumenty platí stejné pořadí ({{runs.cz_full.bm25-nodiacritics.ndcg10|cs}},
  {{runs.cz_full.bm25.ndcg10|cs}} a {{runs.cz_full.e5-small.ndcg10|cs}}). Jde o průměry přes tvrzení bez intervalů
  spolehlivosti, takže pořadí není statisticky testováno. V CsFEVER převažují pojmenované entity, což zvýhodňuje
  lexikální vyhledávání.
* **SQAD:** e5-small vede před BM25 bez diakritiky o {{cz.large.copy_test.lead_nd|cs}} v nDCG@10 na všech
  {{cz.large.copy_test.n|cs}} testovacích otázkách kopie SQAD 3.0 a o {{cz.large.sqad32_test.lead_nd|cs}} na všech
  {{cz.large.sqad32_test.n|cs}} testovacích otázkách SQAD 3.2 (v hranatých závorkách 95% intervaly). Žádný ze šesti
  dalších vyhledávačů nevede před e5-small o tolik, o kolik e5-small vede před BM25: e5-base, granite-embedding 107m a
  278m multilingual, EmbeddingGemma 300m, simcse-cs-en a BM25 nad lemmaty
  ([docs/CROSSOVER.md](docs/CROSSOVER.md#ten-retrievers-on-sqad)).

**RAG.** Pět rodin modelů, každá ve dvou velikostech, odpovídá na každou sadu, vždy s prvními
{{runs.cz_rag.k|cs}} pasážemi od e5-small. Každý model s výrobcem, zemí výrobce a licencí modelu (celé licence jsou v
[THIRD_PARTY.md](THIRD_PARTY.md)):

<ul>
<li>Gemma 3n E2B: Google, Spojené státy; Gemma Terms of Use</li>
<li>Gemma 3n E4B: Google, Spojené státy; Gemma Terms of Use</li>
<li>Gemma 4 E2B: Google, Spojené státy; Apache 2.0</li>
<li>Gemma 4 E4B: Google, Spojené státy; Apache 2.0</li>
<li>Ministral 3 3B Instruct 2512: Mistral AI, Francie; Apache 2.0</li>
<li>Ministral 3 8B Instruct 2512: Mistral AI, Francie; Apache 2.0</li>
<li>Qwen2.5-1.5B-Instruct: tým Qwen společnosti Alibaba Cloud, Čína; Apache 2.0</li>
<li>Qwen2.5-3B-Instruct: tým Qwen společnosti Alibaba Cloud, Čína; Qwen Research License, jen nekomerční výzkum a hodnocení</li>
<li>Qwen3.5 2B: tým Qwen společnosti Alibaba Cloud, Čína; Apache 2.0</li>
<li>Qwen3.5 4B: tým Qwen společnosti Alibaba Cloud, Čína; Apache 2.0</li>
</ul>

Jako první odpovídaly Gemma 4 a Qwen2.5; Gemma 4 byla vybrána srovnávacím testem západních kandidátů na samostatné
vývojové sadě, aby odpovídala vedle Qwen2.5. Gemma 3n, Ministral 3 a Qwen3.5 byly přidány na žádost autora až po přečtení
výsledků dřívějších rodin, každá se stejnými otázkami, pořadími pasáží, promptem, limitem tokenů, semínkem a strojem
([docs/LIMITATIONS.md](docs/LIMITATIONS.md#the-czech-and-english-families)). Všechny rozdíly níže jsou EM s mapováním
ano/ne, párově na stejných otázkách.

* Gemma 3n, Gemma 4, Ministral 3 a Qwen3.5 odpovídají lépe než Qwen2.5, v obou velikostech, na každé sadě a v obou
  metrikách, a každý z těchto rozdílů je prokázán. Gemma 3n E2B je nad Qwen2.5 1.5B o
  {{cz.g3n.family.smaller.mapped_range|cs}} a E4B nad 3B o {{cz.g3n.family.larger.mapped_range|cs}}; Gemma 4 E2B o
  {{cz.family.smaller.mapped_range|cs}} a E4B o {{cz.family.larger.mapped_range|cs}}; Ministral 3 3B o
  {{cz.m3.family.smaller.mapped_range|cs}} a 8B o {{cz.m3.family.larger.mapped_range|cs}}; Qwen3.5 2B o
  {{cz.q35.family.smaller.mapped_range|cs}} a 4B o {{cz.q35.family.larger.mapped_range|cs}}.
* Z menších modelů má Gemma 4 E2B nejvyšší skóre na každé sadě. Gemma 4 E2B minus Gemma 3n E2B je
  {{cz.g4_minus_g3n.smaller.mapped_range|cs}}, prokázaně na třech sadách; Gemma 4 E2B minus Ministral 3 3B je
  {{cz.m3_minus_g4.smaller.mapped_range|neg|cs}}, prokázaně na dvou: na SQAD 3.2 celé a na očištěné kopii. Z větších
  modelů nemá nejvyšší skóre na každé sadě žádný a na žádné sadě není náskok nejvyššího modelu před dalším prokázán.
* Gemma 4 a Ministral 3 proti Qwen3.5. Gemma 4 E2B je nad Qwen3.5 2B na všech čtyřech sadách, prokázaně na třech.
  Gemma 4 E4B a Qwen3.5 4B se prokazatelně neliší na žádné sadě. Ministral 3 3B a Qwen3.5 2B se prokazatelně neliší
  na žádné sadě. Ministral 3 8B není nad Qwen3.5 4B na žádné sadě (na jedné mají stejné skóre), rozdíl je
  {{cz.q35_minus_m3.larger.mapped_range|cs}} ve prospěch Qwen3.5 4B, prokázaný na dvou: na SQAD 3.2 očištěné a celé.
* Větší model každé rodiny odpovídá lépe. U Gemma 3n je tento krok generátoru prokázán na každé sadě
  ({{cz.g3n.range.generator_step.mapped|cs}}), u Gemma 4 na třech větších sadách
  ({{cz.g4.range3.generator_step.mapped|cs}}) a na vydaných {{cz.released.n|cs}} ne, u Ministral 3 na každé sadě
  ({{cz.m3.range.generator_step.mapped|cs}}), u Qwen2.5 na třech větších sadách ({{cz.range3.generator_step.mapped|cs}})
  a na vydaných {{cz.released.n|cs}} ne a u Qwen3.5 na každé sadě ({{cz.q35.range.generator_step.mapped|cs}}).
* e5-small odpovídá lépe než retromae-cs, o {{cz.range.e5_minus_retromae_mapped|cs}}, v každé rodině na každé
  sadě.
* Gemma 3n, Ministral 3 a Qwen3.5 odpovídají česky. Odpovědí ano/ne v angličtině: Gemma 3n
  {{cz.g3n.yes_no_in_english|cs}}, Ministral 3 {{cz.m3.yes_no_in_english|cs}}, Qwen2.5 {{cz.yes_no_in_english|cs}}
  a Qwen3.5 {{cz.q35.yes_no_in_english|cs}}.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/families_dark.svg">
  <img alt="Tečkový graf ve dvou panelech, vlevo menší model každé rodiny, vpravo větší, se čtyřmi českými sadami SQAD na vodorovné ose, vydanými 200 otázkami kopie SQAD 3.0 jako první, a s přesnou shodou při mapování ano/ne na svislé ose; každá značka má svůj 95% interval. V každém panelu stojí modely od nejlepšího podle skóre na vydaných 200 a každé jméno nese vlajku svého výrobce. Na každé sadě a v obou velikostech mají Gemma 3n, Gemma 4, Ministral 3 i Qwen3.5 vyšší skóre než Qwen2.5. Z menších modelů má Gemma 4 E2B nejvyšší skóre na každé sadě; z větších má Gemma 4 E4B nejvyšší skóre na vydaných 200." src="results/figures/families_light.svg">
</picture>

*Český RAG, pět rodin modelů, každý model s prvními {{runs.cz_rag.k|cs}} pasážemi od e5-small: EM s mapováním ano/ne
a 95% bootstrapový interval. Popisky grafu jsou anglicky; čísla a tabulky jsou v
[README.md](README.md#five-families-model-for-model). Qwen2.5-3B-Instruct je pod licencí Qwen Research License
([THIRD_PARTY.md](THIRD_PARTY.md)).*

## Jak to spustit

Potřebujete Python 3.11 až 3.13 a [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/arytma/measurerag.git && cd measurerag
uv sync --extra dense --extra rag --extra dev
uv run measurerag doctor --fix      # macOS: jedna knihovna OpenMP; spusťte znovu po každém uv sync
```

Kouřový test (*smoke test*) na malé přibalené syntetické sadě: bez sítě, za několik sekund, s tabulkou rozpočtů.

```bash
# test
uv run measurerag smoke
```

Každé české číslo se přepočítá jen z tohoto repozitáře, bez GPU a bez modelů, z odpovědí v `answers/` a záznamů v
`results/czech/records/`:

```bash
uv run python scripts/rescore_czech.py --check
```

Celý experiment krok za krokem popisuje [docs/REPRODUCE.md](docs/REPRODUCE.md); co ověřuje který příkaz pro češtinu,
je v jeho [oddílu 10](docs/REPRODUCE.md#10-checking-every-czech-number).

## Pro bakalářské a diplomové práce

measureRAG může sloužit jako hotové měřicí prostředí pro závěrečnou práci o českém vyhledávání nebo RAG na omezeném
hardwaru.

* **Co lze rozšířit.** Nový vyhledávač je třída se třemi metodami a jedním řádkem v registru; běh, měření rozpočtu a
  grafy ho převezmou ([README.md](README.md#extending-it)). Rozpočty paměti a latence se nastavují v konfiguraci.
* **Co ověříte bez výpočetního výkonu.** Každé české číslo se přepočítá z odpovědí, které repozitář obsahuje (výše).
  Nové generování odpovědí potřebuje modely a hodiny výpočtu.
* **Na co dát pozor.** Porovnávejte rozdíly uvnitř jedné sady, ne úrovně mezi sadami. Česká EM většinou podhodnocuje,
  i když není přísně dolní mezí: skloněný tvar správné odpovědi se počítá jako chyba. Intervaly nejsou upraveny pro
  vícenásobné srovnání. České odpovědi vznikly na GPU, takže české generátory nemají latenci ani paměť změřenou na
  procesoru. Vše ostatní je v [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
* **Licence.** Gemma 3n a EmbeddingGemma jsou pod podmínkami Gemma Terms of Use; Qwen2.5-3B-Instruct je pod licencí
  Qwen Research License (jen nekomerční výzkum a hodnocení). SQAD 3.2 a archiv SQAD 3.0 si stáhnete z LINDAT sami
  ([THIRD_PARTY.md](THIRD_PARTY.md)).
* **Citace.** Uveďte verzi a commit, ze kterého vycházíte.

## Dokumentace

Podrobnosti jsou v angličtině:

| dokument | co obsahuje |
|---|---|
| [README.md](README.md) | celý přehled: výsledek, česká část s tabulkami, reprodukce, rozšíření, licence |
| [docs/CROSSOVER.md](docs/CROSSOVER.md) | řízený experiment (*dial*), všechny tabulky, ceny a českou přílohu |
| [docs/METHOD.md](docs/METHOD.md) | jak čísla vznikají, od výřezu dat po bootstrap |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | příkazy, které reprodukují každý výsledek |
| [docs/BUDGETS.md](docs/BUDGETS.md) | anglické tabulky rozpočtů a hygiena měření času |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | co výsledky neukazují |
| [docs/METRICS.md](docs/METRICS.md) | definice každé metriky a každého nákladu |
| [docs/DATASETS_AND_LICENCES.md](docs/DATASETS_AND_LICENCES.md) | každý model a datová sada: zdroj, revize, licence |
| [THIRD_PARTY.md](THIRD_PARTY.md) | licence a citace modelů a dat třetích stran |

## Licence

Kód je pod licencí Apache 2.0 ([LICENSE](LICENSE)), dokumentace a výsledky pod CC BY 4.0
([LICENSE-docs](LICENSE-docs)). Vygenerované české odpovědi v `answers/` mají licenci podle sady, ze které
vycházejí ([NOTICE](NOTICE)).

## Jak citovat

Použijte [CITATION.cff](CITATION.cff). V textu:

> Laznicek, P. (2026). *measureRAG: a budget-aware retrieval and RAG benchmark
> in Czech and English, for a single machine with no network access* (version
> 0.4.5) [Computer software]. https://github.com/arytma/measurerag
