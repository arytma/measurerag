"""Peak resident memory, measured rather than estimated.

The number that decides a RAM verdict is the high-water mark of the whole
process - weights, index, tokenizer, runtime, arena fragmentation and all -
because that is what makes a customer's 8 GB box swap. Each configuration is
therefore measured in its own subprocess (see runner.py); this module is what
the child uses on itself.
"""

from __future__ import annotations

import resource
import sys
import threading
import time


def _rusage_peak_bytes() -> int:
    """ru_maxrss is bytes on macOS and kilobytes on Linux."""
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(val) if sys.platform == "darwin" else int(val) * 1024


class PeakRSS:
    """Samples our own RSS while the work runs.

    The sampler catches transient spikes (a big batch of embeddings held twice
    during a copy) that ru_maxrss also catches, and the two are combined with
    max(); the sampler alone can miss a spike between ticks, ru_maxrss alone is
    coarse on some platforms.
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self.peak_bytes:
                    self.peak_bytes = rss
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "PeakRSS":
        import psutil

        self._proc = psutil.Process()
        self.peak_bytes = max(self.peak_bytes, self._proc.memory_info().rss)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.peak_bytes = max(self.peak_bytes, self._proc.memory_info().rss)
        except Exception:
            pass
        self.peak_bytes = max(self.peak_bytes, _rusage_peak_bytes())

    @property
    def peak_gb(self) -> float:
        return self.peak_bytes / 1024**3


def current_rss_gb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1024**3


class Stopwatch:
    """Perf-counter timer used for every latency figure in the kit."""

    def __init__(self) -> None:
        self.elapsed = 0.0
        self._t0 = 0.0

    def __enter__(self) -> "Stopwatch":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._t0
