"""Remove the names of other processes and applications from the shipped run records, keeping every count.

    .venv/bin/python scripts/redact_records.py            # rewrite the records in place
    .venv/bin/python scripts/redact_records.py --check    # exit 1 if a record still names a process

The contention sampler (scripts/sweep_queue.py) writes the name of the busiest other process beside each batch's
counts, and scripts/mmap_probe.py's snapshots once listed what else was open. Those names describe the machine's
owner, not the measurement, so the shipped records replace each with "another process" and drop the probe's lists;
every sample count, share of a core and free-memory figure stays as recorded. Run it after a sweep or a probe, before
publishing, as scripts/scrub_paths.py is run for paths.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NEUTRAL = "another process"
RECORDS = ("results/crossover/calibration/batches.jsonl", "results/crossover/calibration_ministral/batches.jsonl",
           "results/czech/v040/timed/batches.jsonl", "results/crossover/mmap_probe.jsonl")
_MAX = re.compile(r'"max_comm": "(?:[^"\\]|\\.)*"')
_LIST = re.compile(r'"contended_comms": \[(?:"(?:[^"\\]|\\.)*"(?:, )?)+\]')
# a token's "mostly <name>" after a share of samples (scripts/stats_tokens.py wrote it so before names were removed)
_MOSTLY = re.compile(r"(\(\d+% of samples), mostly [^;]*\)")
_PROBE_DROP = ("applications_open", "remote_desktop_processes", "cpu_by_process")


def redact_line(line: str) -> str:
    """One JSON line: the contention names replaced in place, byte for byte elsewhere; a probe snapshot without its
    lists of names, its busiest other process's share kept as a number."""
    line = _LIST.sub(f'"contended_comms": ["{NEUTRAL}"]', _MAX.sub(f'"max_comm": "{NEUTRAL}"', line))
    if '"record": "snapshot"' in line and any(f'"{k}"' in line for k in _PROBE_DROP):
        d = json.loads(line)
        shares = [v for _, v in d.get("cpu_by_process", [])]
        d = {k: v for k, v in d.items() if k not in _PROBE_DROP}
        d["cpu_busiest_other_pct_of_a_core"] = max(shares, default=0.0)
        line = json.dumps(d, ensure_ascii=False)
    return line


def redact_text(text: str) -> str:
    return "\n".join(redact_line(x) if x.strip() else x for x in text.split("\n"))


def redact_token(value: str) -> str:
    return _MOSTLY.sub(r"\1)", value)


def names_left(text: str) -> list[str]:
    left = [m.group(0) for m in _MAX.finditer(text) if m.group(0) != f'"max_comm": "{NEUTRAL}"']
    left += [m.group(0) for m in _LIST.finditer(text) if m.group(0) != f'"contended_comms": ["{NEUTRAL}"]']
    left += [k for k in _PROBE_DROP if f'"{k}"' in text]
    return left


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    bad = 0
    for rel in RECORDS:
        p = REPO / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        if args.check:
            left = names_left(text)
            if left:
                bad += 1
                print(f"redact_records: {rel} still names {len(left)} processes or lists")
        else:
            new = redact_text(text)
            if new != text:
                p.write_text(new, encoding="utf-8")
                print(f"redact_records: rewrote {rel}")
    if args.check:
        print("redact_records:", "FAILED" if bad else "OK: no shipped record names another process")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
