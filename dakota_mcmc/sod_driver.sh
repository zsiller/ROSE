#!/usr/bin/env bash
# Dakota fork-interface wrapper: run sod_driver.py under the project venv so
# numpy / the Sod solvers are importable. Dakota appends the params and results
# filenames, which we forward verbatim via "$@".
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

# Use the venv python without needing the venv activated in Dakota's env.
PY="$ROOT/rose_env/bin/python"

exec "$PY" "$HERE/sod_driver.py" "$@"
