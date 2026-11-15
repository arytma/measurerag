"""Run 2: locating the retriever/generator crossover along gold-in-context.

Definitions live in docs/CROSSOVER.md and nowhere else; this package implements
them. `dial` moves gold-in-context with everything else fixed, `retrieval`
caches one ranking per retriever x dataset, `generate` caches one answer per
distinct prompt, and `stats` turns paired per-question outcomes into intervals.
"""
