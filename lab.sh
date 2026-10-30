#!/bin/sh
set -eu
# Install the notebook extra in your active environment before running this script.
exec python -m jupyterlab --notebook-dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/notebooks"
