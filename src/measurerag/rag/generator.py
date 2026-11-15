"""Local generators only, via llama-cpp-python.

No API keys, no cloud, nothing leaves the box - that is the premise of the
whole kit, so a generator that needs a key does not get built. Decoding is
greedy with a fixed seed and a hard token cap (`max_tokens`: 256 by default; the
dial and the crossover's sweeps set 64 in their configurations) so that a second
run, in the same order, produces the same strings and therefore the same EM/F1. Another order
can change an answer that was a near tie.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompt_template.txt"


@dataclass(frozen=True)
class GeneratorSpec:
    key: str
    repo_id: str
    filename: str
    context_window: int = 4096
    note: str = ""
    revision: str | None = None      # a commit of the repo, for a file not published by the model's own makers


GENERATORS: dict[str, GeneratorSpec] = {
    "qwen2.5-1.5b": GeneratorSpec(
        "qwen2.5-1.5b",
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-1.5b-instruct-q4_k_m*.gguf",
        context_window=4096,
        note="~1.0 GB on disk at Q4_K_M. The candidate for a 4 GB box.",
    ),
    "qwen2.5-3b": GeneratorSpec(
        "qwen2.5-3b",
        "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "qwen2.5-3b-instruct-q4_k_m*.gguf",
        context_window=4096,
        note="~1.9 GB on disk at Q4_K_M.",
    ),
    # Google's Gemma 4, chosen on the dev set beside Qwen2.5 for the Czech runs.
    # Google publishes no Q4_K_M; these are Unsloth's, pinned to a commit.
    "gemma-4-e2b": GeneratorSpec(
        "gemma-4-e2b",
        "unsloth/gemma-4-E2B-it-GGUF",
        "gemma-4-E2B-it-Q4_K_M.gguf",
        context_window=4096,
        note="~3.1 GB on disk at Q4_K_M. Apache 2.0.",
        revision="0314792d7f1f7e229411f620751375812bb9faf2",
    ),
    "gemma-4-e4b": GeneratorSpec(
        "gemma-4-e4b",
        "unsloth/gemma-4-E4B-it-GGUF",
        "gemma-4-E4B-it-Q4_K_M.gguf",
        context_window=4096,
        note="~5.0 GB on disk at Q4_K_M. Apache 2.0.",
        revision="bfc15c382204943c3a8fff0c750b94ae2364d7a3",
    ),
    # Mistral AI's Ministral 3, the third Czech family of v0.3.1. Mistral AI's own GGUF files do not
    # load in this llama.cpp build; these are Unsloth's Q4_K_M files of the same weights, pinned to a commit.
    "ministral-3-3b": GeneratorSpec(
        "ministral-3-3b",
        "unsloth/Ministral-3-3B-Instruct-2512-GGUF",
        "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
        context_window=4096,
        note="~2.1 GB on disk at Q4_K_M. Apache 2.0.",
        revision="7564922f37fa5bbb62b87f09a55c12f1f91d7a6a",
    ),
    "ministral-3-8b": GeneratorSpec(
        "ministral-3-8b",
        "unsloth/Ministral-3-8B-Instruct-2512-GGUF",
        "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
        context_window=4096,
        note="~5.2 GB on disk at Q4_K_M. Apache 2.0.",
        revision="3731507ec3e867db16d620f73e14d689125758f4",
    ),
    # Qwen3.5, Qwen's current generation and the fourth Czech family of v0.4.0: Unsloth's Q4_K_M files,
    # pinned to the commits the dev-set benchmark used.
    "qwen3.5-2b": GeneratorSpec(
        "qwen3.5-2b",
        "unsloth/Qwen3.5-2B-GGUF",
        "Qwen3.5-2B-Q4_K_M.gguf",
        context_window=4096,
        note="~1.3 GB on disk at Q4_K_M. Apache 2.0.",
        revision="f6d5376be1edb4d416d56da11e5397a961aca8ae",
    ),
    "qwen3.5-4b": GeneratorSpec(
        "qwen3.5-4b",
        "unsloth/Qwen3.5-4B-GGUF",
        "Qwen3.5-4B-Q4_K_M.gguf",
        context_window=4096,
        note="~2.7 GB on disk at Q4_K_M. Apache 2.0.",
        revision="e87f176479d0855a907a41277aca2f8ee7a09523",
    ),
    # v0.4.2: Gemma 4's previous generation at the same places
    "gemma-3n-e2b": GeneratorSpec(
        "gemma-3n-e2b",
        "unsloth/gemma-3n-E2B-it-GGUF",
        "gemma-3n-E2B-it-Q4_K_M.gguf",
        context_window=4096,
        note="~3.0 GB on disk at Q4_K_M. Gemma Terms of Use.",
        revision="d7bd20d510eff26f3f95260c50f032fee8931a35",
    ),
    "gemma-3n-e4b": GeneratorSpec(
        "gemma-3n-e4b",
        "unsloth/gemma-3n-E4B-it-GGUF",
        "gemma-3n-E4B-it-Q4_K_M.gguf",
        context_window=4096,
        note="~4.5 GB on disk at Q4_K_M. Gemma Terms of Use.",
        revision="90fa8b0e431faeae50c305828bc260d6f71720e1",
    ),
}

_FALLBACK_REPOS = {
    # Community requantisations, used only if the official repo layout moves.
    "qwen2.5-1.5b": ("bartowski/Qwen2.5-1.5B-Instruct-GGUF", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"),
    "qwen2.5-3b": ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q4_K_M.gguf"),
}


class GeneratorUnavailable(RuntimeError):
    pass


# The one instruction that depends on the language. SQAD's gold for a yes/no question
# is ano or ne, and exact match compares strings, so asking for yes or no made every
# such question score 0. Every other language keeps the template as
# it stands, byte for byte.
YES_NO_INSTRUCTION = "answer exactly yes or no"
YES_NO_BY_LANGUAGE = {"cs": "answer exactly ano or ne"}

# The Czech prompt's variants. v1 is the English template with that one instruction swapped; v2 and v3
# are written in Czech. One is chosen on the dev set by a rule fixed before it was read, and frozen here
# (results/czech/dev/selection.json).
CZECH_VARIANTS = {"v1": None, "v2": "prompt_template_cs_v2.txt", "v3": "prompt_template_cs_v3.txt", "v4": "prompt_template_cs_v4.txt"}
CZECH_PROMPT = "v1"


def load_prompt_template(language: str = "en", variant: str | None = None) -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    swap = YES_NO_BY_LANGUAGE.get(language)
    if swap is None:
        if variant is not None:
            raise ValueError(f"prompt variants are Czech only, not {language!r}")
        return text
    file = CZECH_VARIANTS[variant or CZECH_PROMPT]
    if file is not None:
        return (PROMPT_PATH.parent / file).read_text(encoding="utf-8")
    if text.count(YES_NO_INSTRUCTION) != 1:
        raise ValueError(f"{PROMPT_PATH.name} no longer holds the yes/no instruction exactly once")
    return text.replace(YES_NO_INSTRUCTION, swap)


def build_prompt(question: str, passages: Sequence[str], max_chars_per_passage: int = 1200, language: str = "en", variant: str | None = None) -> str:
    """One template, every configuration; one language-dependent instruction.

    Passages are numbered and truncated to a fixed character budget so that a
    retriever cannot win by handing the generator more text than its rival.
    """
    blocks = [
        f"[{i + 1}] {p[:max_chars_per_passage].strip()}" for i, p in enumerate(passages)
    ]
    context = "\n\n".join(blocks) if blocks else "(no passages retrieved)"
    return load_prompt_template(language, variant).format(context=context, question=question)


def resolve_model_path(key: str, cache_dir: str | Path | None = None) -> Path:
    """Fetch the GGUF from Hugging Face (or reuse the cached copy)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    if key not in GENERATORS:
        raise KeyError(f"unknown generator {key!r}; known: {', '.join(GENERATORS)}")
    spec = GENERATORS[key]
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if spec.revision:
        try:
            return Path(hf_hub_download(repo_id=spec.repo_id, filename=spec.filename, revision=spec.revision, cache_dir=str(cache_dir) if cache_dir else None))
        except Exception as exc:
            raise GeneratorUnavailable(f"no usable GGUF for {key}: {spec.repo_id}/{spec.filename}@{spec.revision}: {exc}") from exc

    candidates: list[tuple[str, str]] = []
    try:
        import fnmatch

        files = list_repo_files(spec.repo_id)
        matches = sorted(f for f in files if fnmatch.fnmatch(f.lower(), spec.filename.lower()))
        # a split gguf ("-00001-of-00002") needs llama.cpp's own merge; take the
        # single-file quant when one exists and skip the repo otherwise
        single = [m for m in matches if "-of-" not in m]
        candidates += [(spec.repo_id, m) for m in single]
    except Exception:
        pass
    # Offline, the repo cannot be listed to expand the glob; the exact name is
    # what the official repos ship, and it resolves from the local cache.
    exact = (spec.repo_id, spec.filename.replace("*", ""))
    if exact not in candidates:
        candidates.append(exact)
    if key in _FALLBACK_REPOS:
        candidates.append(_FALLBACK_REPOS[key])

    errors = []
    for repo_id, filename in candidates:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            return Path(path)
        except Exception as exc:
            errors.append(f"{repo_id}/{filename}: {exc}")
    raise GeneratorUnavailable(f"no usable GGUF for {key}: " + " | ".join(errors) or key)


class LlamaGenerator:
    """Deterministic wrapper: temperature 0, fixed seed, at most `max_tokens` new tokens (256 by default; the dial and
    the crossover's sweeps pass 64).

    `use_mmap` is llama-cpp-python's own setting and its default, on: every published run loaded its GGUF that way.
    On the CPU, llama.cpp also copies most weights into its own buffers, so with mapping on the file's pages stay
    resident beside that copy and peak RSS counts that part of the weights twice (docs/LIMITATIONS.md). Off, the file is
    read into memory once. The answers do not depend on it (scripts/mmap_probe.py records both)."""

    def __init__(
        self,
        key: str,
        device: str = "cpu",
        seed: int = 20260101,
        max_tokens: int = 256,
        n_ctx: int | None = None,
        n_threads: int | None = None,
        cache_dir: str | Path | None = None,
        use_mmap: bool = True,
    ) -> None:
        self.key = key
        self.spec = GENERATORS[key]
        self.device = device
        self.seed = seed
        self.max_tokens = max_tokens
        self.use_mmap = bool(use_mmap)
        self.model_path = resolve_model_path(key, cache_dir)
        self.model_bytes = self.model_path.stat().st_size

        try:
            from llama_cpp import Llama
        except Exception as exc:  # pragma: no cover - optional extra
            raise GeneratorUnavailable(
                "llama-cpp-python is not installed; `uv sync --extra rag`"
            ) from exc

        # CPU by default. Metal or CUDA only when a run asks for it, and then every layer. A CUDA run on a build
        # that cannot offload stops, rather than running on the CPU unseen.
        if device == "cuda":
            import llama_cpp

            if not llama_cpp.llama_supports_gpu_offload():
                raise GeneratorUnavailable("device cuda: this llama-cpp-python was built without GPU offload")
        n_gpu_layers = -1 if device in ("mps", "cuda") else 0
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx or self.spec.context_window,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            n_threads=n_threads,
            logits_all=False,
            verbose=False,
            use_mmap=self.use_mmap,
        )

    def generate(self, prompt: str) -> tuple[str, dict]:
        t0 = time.perf_counter()
        out = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            seed=self.seed,
            max_tokens=self.max_tokens,
        )
        elapsed = time.perf_counter() - t0
        text = out["choices"][0]["message"]["content"] or ""
        usage = out.get("usage", {})
        completion_tokens = int(usage.get("completion_tokens", 0))
        return text.strip(), {
            "seconds": elapsed,
            "completion_tokens": completion_tokens,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "tokens_per_s": (completion_tokens / elapsed) if elapsed > 0 else 0.0,
        }

    def close(self) -> None:
        self.llm = None
