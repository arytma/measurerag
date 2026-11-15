"""Machine facts recorded with every result row.

A budget verdict is meaningless without the box it was measured on, so the
chip, the RAM and the OS travel with the numbers.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass(frozen=True)
class Machine:
    chip: str
    arch: str
    cpu_count: int
    ram_gb: float
    os_name: str
    os_version: str
    python_version: str

    def as_dict(self) -> dict:
        return asdict(self)


def describe_machine() -> Machine:
    import psutil

    chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown"
    os_version = platform.mac_ver()[0] or platform.release()
    return Machine(
        chip=chip,
        arch=platform.machine(),
        cpu_count=psutil.cpu_count(logical=True) or 0,
        ram_gb=round(psutil.virtual_memory().total / 1024**3, 2),
        os_name=platform.system(),
        os_version=os_version,
        python_version=sys.version.split()[0],
    )


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def resolve_device(requested: str) -> str:
    """`cpu` is the default everywhere; `mps` is opt-in and validated.

    The target deployment is a customer's CPU-only machine, so a run must never
    silently become a GPU run.
    """
    requested = (requested or "cpu").lower()
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    if requested == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return "cpu"
