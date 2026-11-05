#!/usr/bin/env bash
# Full quality gate: tests, lint, format check, type check.
# Usage: scripts/check.sh [paths...]   (default: whole project)
set -uo pipefail
cd "$(dirname "$0")/.."
py=.venv/bin/python
status=0
$py -m pytest -q || status=1
$py -m ruff check "${@:-.}" || status=1
$py -m ruff format --check "${@:-.}" || status=1
$py -m mypy "$@" || status=1   # paths from [tool.mypy] files
exit $status
