"""The budget harness - the point of the whole kit.

A configuration does not get a quality score and a footnote about memory. It
gets a verdict: under this budget it either FITS or it does not, and the
quality number is only reported inside the budgets it fits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Budget:
    """One deployment envelope. All limits are hard ceilings."""

    name: str
    ram_gb: float
    retrieval_ms: float  # p95 per query
    e2e_s: float | None = None  # p95 per question, RAG stage only

    @staticmethod
    def from_dict(d: dict) -> "Budget":
        return Budget(
            name=d["name"],
            ram_gb=float(d["ram_gb"]),
            retrieval_ms=float(d["retrieval_ms"]),
            e2e_s=None if d.get("e2e_s") is None else float(d["e2e_s"]),
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Measurement:
    """What a configuration actually cost, as measured, not as advertised."""

    peak_rss_gb: float = 0.0
    index_bytes: int = 0
    model_bytes: int = 0
    retrieval_ms_p50: float = 0.0
    retrieval_ms_p95: float = 0.0
    e2e_s_p50: float | None = None
    e2e_s_p95: float | None = None
    gen_tokens_per_s: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    budget: str
    fits: bool
    reasons: list[str] = field(default_factory=list)  # why it did not fit

    def as_dict(self) -> dict:
        return {"budget": self.budget, "fits": self.fits, "reasons": list(self.reasons)}


def check(measurement: Measurement, budget: Budget) -> Verdict:
    """FITS only if every applicable ceiling holds.

    Peak RSS is the whole process at its worst moment - index, model weights and
    the runtime together - because that is what the customer's box has to
    survive. A budget with an `e2e_s` limit is not applicable to a
    retrieval-only configuration, and such a configuration cannot fit it.
    """
    reasons: list[str] = []

    if measurement.peak_rss_gb > budget.ram_gb:
        reasons.append(
            f"peak RSS {measurement.peak_rss_gb:.2f} GB > {budget.ram_gb:g} GB"
        )
    if measurement.retrieval_ms_p95 > budget.retrieval_ms:
        reasons.append(
            f"retrieval p95 {measurement.retrieval_ms_p95:.1f} ms > {budget.retrieval_ms:g} ms"
        )
    if budget.e2e_s is not None:
        if measurement.e2e_s_p95 is None:
            reasons.append("budget requires end-to-end latency; configuration has no generator")
        elif measurement.e2e_s_p95 > budget.e2e_s:
            reasons.append(
                f"end-to-end p95 {measurement.e2e_s_p95:.2f} s > {budget.e2e_s:g} s"
            )

    return Verdict(budget=budget.name, fits=not reasons, reasons=reasons)


def check_all(measurement: Measurement, budgets: list[Budget]) -> list[Verdict]:
    return [check(measurement, b) for b in budgets]


def budget_grid(
    ram_gb: list[float], retrieval_ms: list[float], e2e_s: list[float] | None = None
) -> list[Budget]:
    """The cross product the specification asks for: ram x retrieval x (optionally) e2e."""
    out: list[Budget] = []
    for r in ram_gb:
        for lat in retrieval_ms:
            if e2e_s:
                for e in e2e_s:
                    out.append(
                        Budget(
                            name=f"ram{r:g}g_ret{lat:g}ms_e2e{e:g}s",
                            ram_gb=r,
                            retrieval_ms=lat,
                            e2e_s=e,
                        )
                    )
            else:
                out.append(
                    Budget(name=f"ram{r:g}g_ret{lat:g}ms", ram_gb=r, retrieval_ms=lat)
                )
    return out
