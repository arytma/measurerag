"""Remove machine-specific absolute paths from committed results, for publication.

    uv run python scripts/scrub_paths.py            # rewrite in place
    uv run python scripts/scrub_paths.py --check    # exit 1 if any tracked file still has one

The harness records where it loaded each GGUF from, which on the build machine
is an absolute path under the user's home directory. Nothing the analysis reads
depends on it. The scrub rewrites
  <anything>/models--ORG--NAME/snapshots/REV/FILE  ->  hf://ORG/NAME@REV/FILE
  <repo root>/X                                    ->  X
  /Users/<name>/X or /home/<name>/X                ->  ~/X
in every tracked results file. In files under results/ it also replaces email
addresses (a loader's error message once quoted a dataset's request address):
  and email <address>    ->  and request access by email at the address given there
  any other <address>    ->  <email address removed>
Run 1 recorded SQAD 3.0's licence before it could be checked; the two strings it
wrote are replaced by the line LINDAT states (RECORDED below). It then re-hashes
the files a manifest lists and notes the scrub in it. A manifest's
results_stable_sha256 covers metrics and identities only, so it does not change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from measurerag.datasets.sqad import LICENCE, LICENCE_NOTE  # noqa: E402

HF = re.compile(r"[^\s\"']*?/models--([^/\"']+?)--([^/\"']+)/snapshots/([0-9a-f]{40})/([^\s\"']+)")
HOME = re.compile(r"/(?:Users|home)/[^/\s\"']+/")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
EMAIL_SENTENCE = re.compile(r"\band email " + EMAIL.pattern)
EMAIL_NOTE = "email addresses replaced by scripts/scrub_paths.py (no result changed)"
NOTE = ("absolute paths replaced by scripts/scrub_paths.py: hf://ORG/NAME@REV/FILE for Hugging Face "
        "cache files, repository-relative paths, ~/ for anything else under a home directory")
RECORDED = {
    "CC BY-NC-SA 3.0 (LINDAT upstream governs; the HF mirror's `mit` tag is [VERIFY])": LICENCE,
    "[VERIFY] the licence: the mirror claims MIT, upstream LINDAT is CC BY-NC-SA 3.0. "
    "Non-commercial research use assumed here.": LICENCE_NOTE,
}
LICENCE_NOTE_MAN = "SQAD 3.0's licence strings corrected by scripts/scrub_paths.py after the LINDAT check (no result changed)"


def scrub_text(s: str, root: str) -> str:
    s = HF.sub(lambda m: f"hf://{m.group(1)}/{m.group(2)}@{m.group(3)}/{m.group(4)}", s)
    s = s.replace(root.rstrip("/") + "/", "")
    return HOME.sub("~/", s)


def scrub_emails(s: str) -> str:
    s = EMAIL_SENTENCE.sub("and request access by email at the address given there", s)
    return EMAIL.sub("<email address removed>", s)


def has_email(p: Path, text: str) -> bool:
    return p.relative_to(REPO).parts[0] == "results" and bool(EMAIL.search(text))


def has_recorded_licence(p: Path, text: str) -> bool:
    return p.relative_to(REPO).parts[0] == "results" and any(old in text for old in RECORDED)


def fix_licence(s: str) -> str:
    for old, new in RECORDED.items():
        s = s.replace(old, new)
    return s


def tracked(paths: list[str]) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--", *paths], cwd=REPO, capture_output=True, text=True).stdout
    return [REPO / p for p in out.splitlines() if p]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    files = [p for p in tracked(["results", "configs", "docs", "templates", "*.md", "*.cff"])
             if p.suffix in {".json", ".jsonl", ".csv", ".txt", ".md", ".yaml", ".cff"}]
    texts = {p: p.read_text(encoding="utf-8", errors="replace") for p in files}
    homes = {p for p, s in texts.items() if HOME.search(s)}
    emails = {p for p, s in texts.items() if has_email(p, s)}
    licences = {p for p, s in texts.items() if has_recorded_licence(p, s)}
    dirty = sorted(homes | emails | licences)
    if args.check:
        for p in sorted(homes):
            print(f"scrub: {p.relative_to(REPO)} has an absolute home path")
        for p in sorted(emails):
            print(f"scrub: {p.relative_to(REPO)} has an email address")
        for p in sorted(licences):
            print(f"scrub: {p.relative_to(REPO)} has SQAD 3.0's unchecked licence string")
        return 1 if dirty else 0
    root = str(REPO)
    changed: set[Path] = set()
    for p in dirty:
        if p.name == "manifest.json":
            continue
        text = scrub_text(p.read_text(encoding="utf-8"), root)
        p.write_text(fix_licence(scrub_emails(text) if p in emails else text), encoding="utf-8")
        changed.add(p)
        print(f"scrub: {p.relative_to(REPO)}")
    for m in sorted({p.parent / "manifest.json" for p in changed} | {p for p in dirty if p.name == "manifest.json"}):
        if not m.exists():
            continue
        mail = m in emails or any(p in emails for p in changed if p.parent == m.parent)
        text = scrub_text(m.read_text(encoding="utf-8"), root)
        lic = m in licences or any(p in licences for p in changed if p.parent == m.parent)
        man = json.loads(fix_licence(scrub_emails(text) if mail else text))
        for name, entry in (man.get("files") or {}).items():
            f = m.parent / name
            if f.exists():
                entry["bytes"] = f.stat().st_size
                entry["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
        if m in homes or any(p in homes for p in changed if p.parent == m.parent):
            man["scrubbed"] = NOTE
        if mail:
            man["scrubbed_emails"] = EMAIL_NOTE
        if lic:
            man["licence_corrected"] = LICENCE_NOTE_MAN
        m.write_text(json.dumps(man, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"scrub: re-hashed {m.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
