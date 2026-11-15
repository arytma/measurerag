#!/bin/bash
# Run a heavy command only while no timed batch runs (the specification's Rule 6).
#
#     scripts/quiet.sh .venv/bin/python -m pytest
#
# Takes a hold on every phase of scripts/sweep_queue.py (logs/sweep for Qwen2.5's stages,
# logs/sweep_ministral for Ministral 3's: neither starts another batch), waits for the batch
# in flight to finish, runs the command, releases the hold. Several holders may overlap;
# each holds by its own pid. Without a queue running it runs the command at once.
set -u
cd "$(dirname "$0")/.."
DIRS="logs/sweep logs/sweep_ministral"
for L in $DIRS; do
    mkdir -p "$L/hold.d"
    echo "$(date -u +%FT%TZ) $*" > "$L/hold.d/$$"
done
trap 'for L in $DIRS; do rm -f "$L/hold.d/$$"; done' EXIT
waited=0
for L in $DIRS; do
    while [ -f "$L/RUNNING" ]; do
        pid=$(sed -n 's/.*"pid": *\([0-9]*\).*/\1/p' "$L/RUNNING")
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            echo "quiet: removing a stale RUNNING marker in $L (pid $pid is gone)"; rm -f "$L/RUNNING"; break
        fi
        [ $waited -eq 0 ] && echo "quiet: a timed batch is running ($L); waiting for it to finish"
        waited=1
        sleep 20
    done
done
"$@"
