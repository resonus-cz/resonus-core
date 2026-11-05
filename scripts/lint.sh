#!/usr/bin/env bash
# Lint + type check for the given Python files (also run after every edit).
# Usage: scripts/lint.sh [--check] file.py...   (--check: report only, don't modify)
set -uo pipefail
cd "$(dirname "$0")/.."
py=.venv/bin/python

fix=1
[[ "${1:-}" == "--check" ]] && { fix=0; shift; }
files=()
for f in "$@"; do [[ "$f" == *.py ]] && files+=("$f"); done
[[ ${#files[@]} -eq 0 ]] && exit 0

status=0
if (( fix )); then
  $py -m ruff format -q "${files[@]}"
  $py -m ruff check -q --fix "${files[@]}" || status=1
else
  $py -m ruff format -q --check "${files[@]}" || status=1
  $py -m ruff check -q "${files[@]}" || status=1
fi
$py -m mypy --no-error-summary "${files[@]}" || status=1
exit $status
