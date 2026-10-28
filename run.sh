#!/usr/bin/env bash
# iCreate Music - one-command start (macOS / Linux)
set -e
cd "$(dirname "$0")"

PY=python3
command -v $PY >/dev/null 2>&1 || PY=python

# numpy is the only hard requirement
$PY -c "import numpy" 2>/dev/null || {
  echo "Installing numpy (one time)..."
  $PY -m pip install --quiet numpy
}

exec $PY -m icreate_music "$@"
