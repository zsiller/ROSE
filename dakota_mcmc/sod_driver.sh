#!/usr/bin/env bash
# Dakota fork wrapper: run sod_driver.py with the project venv's Python (so numpy
# and the Sod solvers import without activating the venv in Dakota's env).
# Dakota appends <params_file> <results_file>, which we forward via "$@".
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/../rose_env/bin/python" "$HERE/sod_driver.py" "$@"
