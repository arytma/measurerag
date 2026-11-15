"""Config resolution, device resolution and the shipped config files."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from measurerag.config import DEFAULTS, load_config
from measurerag.hardware import describe_machine, resolve_device

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "configs").glob("*.yaml"))


def test_there_are_configs_to_ship():
    assert CONFIGS


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    cfg = load_config(path)
    assert cfg["name"]
    assert cfg["retrievers"]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_defaults_to_cpu(path):
    # the target box is CPU-only; a run must never silently become a GPU run
    assert load_config(path)["device"] == "cpu"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_pins_a_seed(path):
    assert isinstance(load_config(path)["seed"], int)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_names_unique_retriever_ids(path):
    ids = [r["id"] for r in load_config(path)["retrievers"]]
    assert len(set(ids)) == len(ids), path.name


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_uses_registered_components(path):
    from measurerag import retrievers
    from measurerag.datasets import available as datasets_available
    from measurerag.rag.generator import GENERATORS
    from measurerag.retrievers.dense import MODELS

    cfg = load_config(path)
    known_datasets = set(datasets_available())
    assert cfg["dataset"]["name"] in known_datasets
    for fb in cfg["dataset"].get("fallbacks", []):
        assert fb in known_datasets
    for r in cfg["retrievers"]:
        assert r["kind"] in retrievers.available()
        if r["kind"] == "dense":
            assert r["model"] in MODELS
    for g in (cfg.get("rag") or {}).get("generators", []):
        assert g in GENERATORS


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_respects_the_scale_caps(path):
    cfg = load_config(path)
    assert cfg["dataset"]["max_docs"] <= 100_000
    assert cfg["dataset"]["max_queries"] <= 2_000


def test_defaults_are_conservative():
    assert DEFAULTS["device"] == "cpu"
    assert DEFAULTS["query_threads"] == 1
    assert DEFAULTS["rag"]["enabled"] is False


def test_config_merges_nested_sections_rather_than_replacing_them(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"name": "x", "retrieval": {"top_k": 5}}), encoding="utf-8")
    cfg = load_config(p)
    assert cfg["retrieval"]["top_k"] == 5
    assert cfg["retrieval"]["ks"] == DEFAULTS["retrieval"]["ks"]  # not lost


def test_config_without_a_name_takes_the_filename(tmp_path):
    p = tmp_path / "my_run.yaml"
    p.write_text("device: cpu\n", encoding="utf-8")
    assert load_config(p)["name"] == "my_run"
    assert load_config(p)["out_dir"] == "results/my_run"


def test_cpu_is_always_resolvable():
    assert resolve_device("cpu") == "cpu"


def test_an_unavailable_or_nonsense_device_falls_back_to_cpu():
    assert resolve_device("cuda") in {"cuda", "cpu"}
    assert resolve_device("tpu") == "cpu"
    assert resolve_device("") == "cpu"


def test_the_machine_description_is_complete():
    m = describe_machine().as_dict()
    for key in ("chip", "arch", "cpu_count", "ram_gb", "os_name", "os_version",
                "python_version"):
        assert m[key], key
