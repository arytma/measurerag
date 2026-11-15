"""The specification's determinism check: crossover.py from cached generations, twice.

    scripts/quiet.sh .venv/bin/python scripts/repro_check.py

Runs scripts/crossover.py twice into logs/sweep/repro/ and requires both runs
to equal each other and the committed results/crossover.json,
results/stats.json and results/crossover/mmap_estimate.json byte for byte. Writes results/crossover/repro_check.json (no
timestamps, so the record itself is reproducible). Exit 1 on any difference.

Run it after scripts/crossover.py has written the committed outputs; then run
crossover.py once more so the repro.* tokens in stats.json carry the result.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "crossover" / "repro_check.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    work = REPO / "logs" / "sweep" / "repro"
    work.mkdir(parents=True, exist_ok=True)
    runs = []
    for tag in ("a", "b"):
        c, s, e = work / f"crossover_{tag}.json", work / f"stats_{tag}.json", work / f"mmap_estimate_{tag}.json"
        p = subprocess.run([sys.executable, "scripts/crossover.py", "--out", str(c), "--stats", str(s), "--estimate", str(e)],
                           cwd=REPO, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout + p.stderr)
            return 1
        runs.append((c, s, e))
    (ca, sa, ea), (cb, sb, eb) = runs
    committed_e = REPO / "results" / "crossover" / "mmap_estimate.json"
    committed_c, committed_s = REPO / "results" / "crossover.json", REPO / "results" / "stats.json"
    # stats.json carries the repro.* tokens read from this file, so compare it
    # with those tokens taken out; crossover.json is compared whole
    # and without "source", which names the output path each run was given
    def stats_wo_repro(p: Path) -> dict:
        d = json.loads(p.read_text(encoding="utf-8"))
        d.pop("source", None)
        d["tokens"] = {k: v for k, v in d["tokens"].items() if not k.startswith("repro.")}
        return d

    rec = {
        "crossover_runs_identical": ca.read_bytes() == cb.read_bytes(),
        "crossover_matches_committed": committed_c.exists() and ca.read_bytes() == committed_c.read_bytes(),
        "stats_runs_identical": stats_wo_repro(sa) == stats_wo_repro(sb),
        "stats_matches_committed": committed_s.exists() and stats_wo_repro(sa) == stats_wo_repro(committed_s),
        "crossover_sha256": sha(ca),
        # the estimate without the double-counted mapped pages, written beside the measured analysis
        "estimate_runs_identical": ea.read_bytes() == eb.read_bytes(),
        "estimate_matches_committed": committed_e.exists() and ea.read_bytes() == committed_e.read_bytes(),
    }
    rec["crossover_identical"] = rec["crossover_runs_identical"] and rec["crossover_matches_committed"]
    rec["stats_identical"] = rec["stats_runs_identical"] and rec["stats_matches_committed"]
    rec["estimate_identical"] = rec["estimate_runs_identical"] and rec["estimate_matches_committed"]
    OUT.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0 if rec["crossover_identical"] and rec["stats_identical"] and rec["estimate_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
