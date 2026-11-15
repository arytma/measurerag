#!/bin/bash
# The clean-clone test (the specification's Rule 7, item 7).
#
#     scripts/quiet.sh scripts/clean_clone_test.sh [<ref>]     # default ref: HEAD
#
# 1. clone this repository at <ref> into logs/clean_clone/repo (gitignored);
# 2. build a fresh virtual environment there from uv.lock (uv sync --frozen),
#    with uv's cache also under logs/clean_clone, then measurerag doctor --fix;
# 3. run the whole test suite offline in the clone, then recompute every Czech number
#    from the shipped records (scripts/rescore_czech.py --check);
# 4. link in the cached data the analysis reads (gitignored here: rankings,
#    answers, slices, harness prediction files) - nothing else;
# 5. regenerate results/crossover.json, results/stats.json and every generated
#    document in the clone, and require them to be byte-identical to <ref>'s;
# 6. draw every figure again from the regenerated analysis (scripts/figures.py --check)
#    and require each file under results/figures/ to be byte-identical to <ref>'s.
# Writes logs/clean_clone/result.json and exits 1 on any difference.
set -uo pipefail
cd "$(dirname "$0")/.."
SRC=$(pwd)
REF=${1:-HEAD}
W="$SRC/logs/clean_clone"
C="$W/repo"
export UV_CACHE_DIR="$W/uv-cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_CACHE="$SRC/data/hf-hub"
rm -rf "$C"; mkdir -p "$W"
step() { echo "clean-clone: $*"; }
result() { printf '{"ref": "%s", "commit": "%s", "%s": %s}\n' "$REF" "$(git rev-parse "$REF")" "$1" "$2" >> "$W/steps.jsonl"; }
: > "$W/steps.jsonl"

step "cloning $REF"
git clone -q --no-hardlinks "$SRC" "$C" && git -C "$C" checkout -q "$(git rev-parse "$REF")" || { result clone false; exit 1; }
result clone true

step "building a fresh environment from uv.lock"
# uv.lock resolves a different numpy below Python 3.12; the build used 3.12
( cd "$C" && uv sync --frozen --python 3.12 --extra dense --extra rag --extra dev ) > "$W/uv_sync.log" 2>&1
ok=$?; result uv_sync_frozen "$([ $ok -eq 0 ] && echo true || echo false)"; [ $ok -eq 0 ] || exit 1
( cd "$C" && .venv/bin/measurerag doctor --fix ) > "$W/doctor.log" 2>&1
ok=$?; result doctor "$([ $ok -eq 0 ] && echo true || echo false)"; [ $ok -eq 0 ] || exit 1

step "running the test suite offline"
( cd "$C" && .venv/bin/python -m pytest -q -p no:cacheprovider ) > "$W/pytest.log" 2>&1
ok=$?; result tests_pass "$([ $ok -eq 0 ] && echo true || echo false)"
tail -3 "$W/pytest.log"

# before any cache is linked: every Czech number has to recompute from the tree alone
step "recomputing every Czech number from the shipped records, with no cache"
( cd "$C" && .venv/bin/python scripts/rescore_czech.py --check ) > "$W/rescore.log" 2>&1
ok=$?; result rescore_from_the_tree "$([ $ok -eq 0 ] && echo true || echo false)"
tail -1 "$W/rescore.log"

step "linking the cached data the analysis reads"
link() { # link <path relative to the repo>, if it exists here
    [ -e "$SRC/$1" ] || return 0
    mkdir -p "$(dirname "$C/$1")"; rm -rf "${C:?}/$1"; ln -s "$SRC/$1" "$C/$1"
}
for p in results/crossover/cache results/crossover/slices results/en_rag/slice results/cz_rag/slice; do link "$p"; done
for f in $(cd "$SRC" && ls results/crossover/natural/*/predictions_*.json results/en_rag/predictions_*.json \
           results/crossover/superseded/*/predictions_*.json 2>/dev/null); do link "$f"; done

step "regenerating the analysis and the documents"
( cd "$C" && .venv/bin/python scripts/crossover.py && .venv/bin/python scripts/fill.py --all ) > "$W/regen.log" 2>&1
ok=$?; result regenerate "$([ $ok -eq 0 ] && echo true || echo false)"; [ $ok -eq 0 ] || exit 1

same=true
docs=$(cd "$C" && .venv/bin/python -c "
import sys; sys.path.insert(0, 'scripts')
from pathlib import Path; from fill import render_all
root = Path('.').resolve()
out, missing = render_all(root)
assert not missing and out, (missing, len(out))
print(' '.join(str(p.relative_to(root)) for p in sorted(out)))") || { step "could not list the generated documents"; result byte_identical false; exit 1; }
step "comparing $(echo $docs | wc -w | tr -d ' ') generated documents and the two analysis files"
for f in results/crossover.json results/stats.json $docs; do
    if cmp -s "$C/$f" <(git show "$REF:$f"); then step "identical: $f"; else step "DIFFERS: $f"; same=false; fi
done
result byte_identical "$same"

step "drawing every figure again and comparing it byte for byte"
( cd "$C" && .venv/bin/python scripts/figures.py --check ) > "$W/figures.log" 2>&1
ok=$?; result figures_byte_identical "$([ $ok -eq 0 ] && echo true || echo false)"
tail -1 "$W/figures.log"
step "done"
python3 - "$W" <<'EOF'
import json, sys
from pathlib import Path
w = Path(sys.argv[1])
rec = {}
for line in (w / "steps.jsonl").read_text().splitlines():
    rec.update(json.loads(line))
(w / "result.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
print(json.dumps(rec, indent=1))
EOF
[ "$same" = true ] && grep -q '"tests_pass": true' "$W/steps.jsonl" && grep -q '"rescore_from_the_tree": true' "$W/steps.jsonl" \
    && grep -q '"figures_byte_identical": true' "$W/steps.jsonl"
