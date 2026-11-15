"""One place to fix every source of randomness.

Every sampling decision in this kit goes through `rng()` or `seed_everything()`
so that a second identical run produces byte-identical metrics.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

GLOBAL_SEED = 20260101


def seed_everything(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is an optional extra
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - torch absent
        pass


def rng(*tags: object, seed: int = GLOBAL_SEED) -> np.random.Generator:
    """A generator derived from the global seed plus a stable string tag.

    Using a tag means adding a new dataset cannot shift the sample drawn for an
    existing one - each slice has its own independent, reproducible stream.
    """
    material = "|".join([str(seed), *(str(t) for t in tags)]).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def stable_sample(items: list, n: int, *tags: object) -> list:
    """Deterministic subset of `items`, order preserved."""
    if n >= len(items):
        return list(items)
    idx = rng(*tags).choice(len(items), size=n, replace=False)
    return [items[i] for i in sorted(idx.tolist())]
