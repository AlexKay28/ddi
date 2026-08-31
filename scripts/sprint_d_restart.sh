#!/usr/bin/env bash
# Archived Sprint D (VIBEQUEUE-28) runner. Its recorded results used the
# evaluator that omitted cached-decode position_ids and are not evidence of a
# restart gap or RoPE sensitivity. A new sweep needs a corrected, preregistered
# protocol; see CHANGELOG.md (VIBEQUEUE-39).
set -euo pipefail

printf '%s\n' 'Sprint D is archived; do not run its invalid legacy protocol.' >&2
exit 1
