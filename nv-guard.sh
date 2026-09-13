#!/usr/bin/env bash
# Nirvana guard for this project's graphs. For a SCHEDULER — the exit code is the signal.
#
#   0  nothing new
#   1  a NEW crit or failure; the alert text is on stdout
#   2  could not look (bad token, API down, nvt missing)
#
# 2 IS NOT 0. A scheduler that treats "could not look" as "all clear" turns an outage into
# silence, which is worse than having no guard, so the two are kept distinct and the
# could-not-look case says so in words as well.
#
# This is a script rather than a Makefile target because `make` collapses every recipe
# failure to its own exit 2 — a target that runs `exit 1` makes `make` exit 2, so the
# three-way contract cannot survive being wrapped in one. `make nv-guard` calls this and
# is only a convenience; a scheduler should call this file directly.
#
# `nvt --check` is STATEFUL AND CONSUMES THE ALERT: it reports each problem once (keyed
# instance + severity, re-arming after 24h) so a 15-minute guard does not re-page last
# night's failure 96 times. Running it "just to see" therefore eats the alert the next
# scheduled run would have fired on. To look without consuming, use `make nv-status`.
# Pass --repeat to re-report everything.
#
# Do not schedule tighter than ~1/min: a poll is ~0.4s warm, ~2.5s with the failures query.

set -uo pipefail

NVT="${NVT:-$HOME/arcadia/junk/alexkay28/tools/nvt/nvt}"
NV_FILTER="${NV_FILTER:-%DDI%}"

if [ ! -x "$NVT" ]; then
    echo "GUARD COULD NOT LOOK: nvt not found or not executable at $NVT"
    echo "  (override with NVT=<path>) — this is NOT an all-clear"
    exit 2
fi

out="$("$NVT" --check -f "$NV_FILTER" "$@" 2>&1)"
rc=$?

case "$rc" in
    0)
        exit 0
        ;;
    1)
        echo "$out"
        exit 1
        ;;
    *)
        echo "GUARD COULD NOT LOOK (nvt exit $rc) — this is NOT an all-clear:"
        echo "$out" | sed 's/^/  /'
        exit 2
        ;;
esac
