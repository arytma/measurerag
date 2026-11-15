"""Detects and repairs duplicate OpenMP runtimes in the environment.

On macOS the wheels for torch, faiss-cpu and scikit-learn each vendor their own
copy of `libomp.dylib`. Loading more than one into a single process is not a
performance footnote - it is a crash. The first unattended run of this kit died
four times inside `__kmp_fork_barrier` (SIGSEGV) partway through the Czech
retrieval stage, after the worker alternated torch parallel regions (encoding a
query) with faiss parallel regions (searching the index) in one process.

libomp's own diagnostic says it plainly: multiple copies "can degrade
performance or cause incorrect results". A kit whose entire output is a
measurement cannot accept "or cause incorrect results", so this module refuses
to run rather than hoping.

The repair is to make every copy the same file. All the vendored copies are
LLVM libomp with the same 5.0.0 dylib ABI, and each is loaded through a
`@loader_path`-relative path, so pointing the extras at one canonical copy
leaves dyld with a single image.

    measurerag doctor          report
    measurerag doctor --fix    repair

The alternative everyone reaches for, `KMP_DUPLICATE_LIB_OK=TRUE`, only silences
the warning; it is documented as unsafe and is exactly the configuration that
segfaulted here. This kit does not set it.
"""

from __future__ import annotations

import platform
import sysconfig
from dataclasses import dataclass
from pathlib import Path

# Vendored runtime filenames worth looking at. libgomp (GCC) and libiomp5
# (Intel) are listed because mixing *those* with LLVM's libomp is the same bug
# wearing a different name.
_RUNTIME_GLOBS = ("libomp*.dylib", "libiomp5*.dylib", "libgomp*.dylib")

# Preferred owner of the canonical copy, most-authoritative first. torch ships
# the newest build and the most code depends on its threading behaviour, so it
# wins by default.
_CANONICAL_ORDER = ("torch", "faiss", "sklearn", "scipy", "numpy")


def site_packages() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


@dataclass(frozen=True)
class Runtime:
    path: Path       # where the file sits
    real: Path       # what it resolves to; symlinked copies share this
    package: str     # the top-level package that vendored it

    @property
    def is_link(self) -> bool:
        return self.path.is_symlink()


def find_runtimes(root: Path | None = None) -> list[Runtime]:
    """Every vendored OpenMP runtime in the environment, sorted by package."""
    root = root or site_packages()
    found: list[Runtime] = []
    seen: set[Path] = set()
    for pattern in _RUNTIME_GLOBS:
        for p in root.rglob(pattern):
            if not p.is_file() or p.name.endswith(".orig-backup") or p in seen:
                continue
            seen.add(p)
            try:
                package = p.relative_to(root).parts[0]
            except ValueError:  # pragma: no cover - defensive
                package = "?"
            found.append(Runtime(path=p, real=p.resolve(), package=package))
    return sorted(found, key=lambda r: (r.package, r.path.name))


def distinct_images(runtimes: list[Runtime]) -> list[Path]:
    """The number of images dyld would actually map - symlinks collapse here."""
    return sorted({r.real for r in runtimes})


def _canonical(runtimes: list[Runtime]) -> Runtime | None:
    for pkg in _CANONICAL_ORDER:
        for r in runtimes:
            if r.package == pkg and not r.is_link:
                return r
    return next((r for r in runtimes if not r.is_link), None)


@dataclass(frozen=True)
class Report:
    runtimes: list[Runtime]
    images: list[Path]
    platform_affected: bool

    @property
    def conflicted(self) -> bool:
        return self.platform_affected and len(self.images) > 1

    def as_dict(self) -> dict:
        return {
            "openmp_copies": [
                {"package": r.package, "path": str(r.path), "resolves_to": str(r.real),
                 "symlink": r.is_link}
                for r in self.runtimes
            ],
            "distinct_images": [str(p) for p in self.images],
            "conflicted": self.conflicted,
        }


def check(root: Path | None = None) -> Report:
    runtimes = find_runtimes(root)
    return Report(
        runtimes=runtimes,
        images=distinct_images(runtimes),
        platform_affected=platform.system() == "Darwin",
    )


def unify(root: Path | None = None, apply: bool = False) -> list[str]:
    """Point every vendored copy at one canonical file.

    Returns the actions taken (or that would be taken, when `apply` is False).
    The replaced file is kept next to the symlink as `.orig-backup` so the
    change is reversible without reinstalling.
    """
    root = root or site_packages()
    runtimes = find_runtimes(root)
    canon = _canonical(runtimes)
    actions: list[str] = []
    if canon is None or len(distinct_images(runtimes)) <= 1:
        return actions

    for r in runtimes:
        if r.real == canon.real:
            continue
        actions.append(f"{r.path.relative_to(root)} -> {canon.package}/…/{canon.path.name}")
        if not apply:
            continue
        backup = r.path.with_suffix(r.path.suffix + ".orig-backup")
        if not backup.exists() and not r.is_link:
            r.path.replace(backup)
        elif r.path.exists() or r.path.is_symlink():
            r.path.unlink()
        r.path.symlink_to(canon.path)
    return actions


class OpenMPConflict(RuntimeError):
    """Raised before a run rather than segfaulting in the middle of one."""


def preflight(root: Path | None = None) -> Report:
    """Called before any measurement. Refuses a run that would be unsafe."""
    report = check(root)
    if report.conflicted:
        listing = "\n".join(f"    {r.package:10s} {r.path}" for r in report.runtimes if not r.is_link)
        raise OpenMPConflict(
            f"{len(report.images)} separate OpenMP runtimes are installed:\n{listing}\n"
            "  Loading two of them into one process segfaults inside libomp, and the\n"
            "  libomp documentation warns it can also produce silently wrong results.\n"
            "  Repair the environment (reversible, keeps a .orig-backup) with:\n"
            "      measurerag doctor --fix"
        )
    return report
