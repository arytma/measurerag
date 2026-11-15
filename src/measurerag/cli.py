"""measurerag command line.

    measurerag smoke                     offline sanity run on bundled fixtures
    measurerag run --config c.yaml        the real thing
    measurerag repro --config c.yaml      run twice, diff the manifests
    measurerag list                       what is registered
    measurerag doctor                     check the environment is safe to measure on
    measurerag info                       what this machine is
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import typer

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_config(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    alt = REPO_ROOT / path
    if alt.exists():
        return alt
    raise typer.BadParameter(f"config not found: {path}")


@app.command()
def info() -> None:
    """Print the machine, the device options and the package versions."""
    from .hardware import describe_machine, is_apple_silicon, resolve_device
    from .manifest import package_versions

    m = describe_machine()
    typer.echo("machine:")
    for k, v in m.as_dict().items():
        typer.echo(f"  {k:16s} {v}")
    typer.echo(f"  apple_silicon    {is_apple_silicon()}")
    typer.echo(f"  device 'mps' ->  {resolve_device('mps')}")
    typer.echo("  default device   cpu  (the target box is CPU-only)")
    typer.echo("\npackages:")
    for k, v in package_versions().items():
        typer.echo(f"  {k:24s} {v}")


@app.command("list")
def list_components() -> None:
    """List registered datasets, retrievers, dense models and generators."""
    from . import datasets, retrievers
    from .rag.generator import GENERATORS
    from .retrievers.dense import MODELS

    typer.echo("datasets:   " + ", ".join(datasets.available()))
    typer.echo("retrievers: " + ", ".join(retrievers.available()))
    typer.echo("\ndense models:")
    for k, spec in sorted(MODELS.items()):
        typer.echo(f"  {k:14s} {spec.hf_id:46s} {spec.note}")
    typer.echo("\ngenerators (GGUF, local):")
    for k, spec in sorted(GENERATORS.items()):
        typer.echo(f"  {k:14s} {spec.repo_id:40s} {spec.note}")
    typer.echo("\nconfigs:")
    for p in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        typer.echo(f"  configs/{p.name}")


@app.command()
def doctor(
    fix: bool = typer.Option(False, "--fix", help="repair what can be repaired"),
) -> None:
    """Check the environment can produce trustworthy measurements.

    Today that means one question: how many OpenMP runtimes are installed? Two
    of them in one process is what killed the first unattended run, and libomp
    warns it can also corrupt results rather than crashing outright.
    """
    from . import openmp

    report = openmp.check()
    typer.echo(f"OpenMP runtimes: {len(report.runtimes)} copies, "
               f"{len(report.images)} distinct image(s)")
    for r in report.runtimes:
        mark = "->" if r.is_link else "  "
        rel = r.path.relative_to(openmp.site_packages())
        typer.echo(f"  {mark} {r.package:10s} {rel}" + (f"  (shares {r.real.parent.parent.name})" if r.is_link else ""))

    if not report.conflicted:
        typer.echo("\ndoctor: OK - a single OpenMP runtime, safe to measure on")
        return

    typer.echo("\ndoctor: CONFLICT - two or more OpenMP runtimes would load into one process.")
    actions = openmp.unify(apply=fix)
    if not fix:
        typer.echo("Run `measurerag doctor --fix` to point these at one copy:")
        for a in actions:
            typer.echo(f"  {a}")
        raise typer.Exit(1)
    for a in actions:
        typer.echo(f"  linked {a}")
    after = openmp.check()
    if after.conflicted:
        typer.echo("doctor: FAILED - still conflicted", err=True)
        raise typer.Exit(1)
    typer.echo("doctor: repaired. The replaced files are kept as .orig-backup.")


@app.command()
def smoke(
    out: str = typer.Option("results/smoke", help="where to write the smoke results"),
    device: str = typer.Option("cpu", help="cpu | mps"),
    keep: bool = typer.Option(False, help="keep the previous smoke output"),
) -> None:
    """Offline sanity run: 200 bundled documents, 20 queries, a budget table.

    This is the first thing a new user runs. It must never need the network
    and must finish well inside three minutes.
    """
    from .config import load_config
    from .runner import run as run_cfg

    t0 = time.perf_counter()
    cfg = load_config(_resolve_config("configs/smoke.yaml"))
    cfg["device"] = device
    out_path = Path(out)
    if out_path.exists() and not keep:
        shutil.rmtree(out_path)

    result = run_cfg(cfg, config_path="configs/smoke.yaml", out_dir=out_path)

    elapsed = time.perf_counter() - t0
    ok = [r for r in result["results"] if r.get("status") == "ok"]
    typer.echo(f"smoke: {len(ok)}/{len(result['results'])} configurations ok in {elapsed:.1f}s")
    if not ok:
        typer.echo("smoke: FAILED - no configuration completed", err=True)
        raise typer.Exit(1)
    if elapsed > 180:
        typer.echo(f"smoke: WARNING - took {elapsed:.0f}s, over the 180s target", err=True)
    typer.echo("smoke: OK")


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="path to a run config YAML"),
    out: str = typer.Option(None, help="override the output directory"),
    device: str = typer.Option(None, help="cpu | mps  (default from the config, which is cpu)"),
    only: str = typer.Option(None, help="comma-separated retriever ids to run"),
    max_docs: int = typer.Option(None, help="override the corpus cap"),
    max_queries: int = typer.Option(None, help="override the query cap"),
    resume: bool = typer.Option(
        True, help="reuse results already on disk whose spec is unchanged"
    ),
) -> None:
    """Run one config: retrieval, optional RAG, budget verdicts, curves."""
    from .config import load_config
    from .runner import run as run_cfg

    path = _resolve_config(config)
    cfg = load_config(path)
    if device:
        cfg["device"] = device
    if max_docs is not None:
        cfg["dataset"]["max_docs"] = max_docs
    if max_queries is not None:
        cfg["dataset"]["max_queries"] = max_queries
    ids = [s.strip() for s in only.split(",")] if only else None

    result = run_cfg(cfg, config_path=str(path), out_dir=out, only=ids, resume=resume)
    failed = [r for r in result["results"] if r.get("status") != "ok"]
    raise typer.Exit(1 if not result["results"] or len(failed) == len(result["results"]) else 0)


@app.command()
def repro(
    config: str = typer.Option(..., "--config", "-c"),
    out: str = typer.Option(None, help="base output dir; two subdirs are written"),
) -> None:
    """Run the same config twice and diff the manifests.

    The claim this kit makes is that metrics are byte-identical across runs.
    This is the command that checks it; timings are excluded because they are
    physics, not logic.
    """
    from .config import load_config
    from .manifest import compare_manifests
    from .runner import run as run_cfg

    path = _resolve_config(config)
    cfg = load_config(path)
    base = Path(out or cfg["out_dir"]).parent / f"{cfg['name']}_repro"
    # resume is off on purpose: the claim under test is that two independent
    # runs agree, and reusing a result from disk would test nothing at all.
    a = run_cfg(cfg, config_path=str(path), out_dir=base / "run_a", resume=False)
    b = run_cfg(load_config(path), config_path=str(path), out_dir=base / "run_b", resume=False)

    cmp = compare_manifests(a["manifest"], b["manifest"])
    (base / "repro_check.json").write_text(
        json.dumps(cmp, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo("")
    typer.echo(f"metrics identical:            {cmp['metrics_identical']}")
    typer.echo(f"manifest identical (no times): {cmp['identical_ignoring_timings']}")
    if cmp["differences"]:
        typer.echo("differences:")
        for d in cmp["differences"][:20]:
            typer.echo(f"  {d}")
    typer.echo(f"wrote {base}/repro_check.json")
    raise typer.Exit(0 if cmp["metrics_identical"] else 1)


@app.command()
def fetch(
    what: str = typer.Argument(..., help="dataset name, 'models', or 'generators'"),
    cache: str = typer.Option("data", help="cache directory"),
) -> None:
    """Pre-download a dataset or the model weights, so a later run is offline."""
    from .datasets import load_dataset

    if what == "models":
        from sentence_transformers import SentenceTransformer

        from .retrievers.dense import MODELS

        for key, spec in sorted(MODELS.items()):
            typer.echo(f"fetching {spec.hf_id} ...")
            try:
                SentenceTransformer(spec.hf_id, device="cpu", revision=spec.revision)   # a pinned commit where there is one
                typer.echo(f"  ok {key}")
            except Exception as exc:
                typer.echo(f"  FAILED {key}: {exc}")
        return
    if what == "generators":
        from .rag.generator import GENERATORS, resolve_model_path

        for key in sorted(GENERATORS):
            typer.echo(f"fetching {key} ...")
            try:
                p = resolve_model_path(key, cache_dir=None)
                typer.echo(f"  ok {p} ({p.stat().st_size / 1024**2:.0f} MB)")
            except Exception as exc:
                typer.echo(f"  FAILED {key}: {exc}")
        return

    ds = load_dataset(what, cache_dir=Path(cache))
    typer.echo(json.dumps(ds.fingerprint(), indent=2, ensure_ascii=False))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
