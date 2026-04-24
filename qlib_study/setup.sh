#!/usr/bin/env bash
# qlib_study/setup.sh — one-shot env + data setup
# Usage:
#   conda create -n qlib python=3.10 -y && conda activate qlib
#   bash qlib_study/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==> installing pyqlib + deps"
pip install -r "$SCRIPT_DIR/requirements.txt"

QLIB_DATA_DIR="${QLIB_DATA_DIR:-$HOME/.qlib/qlib_data/us_data}"
if [ ! -d "$QLIB_DATA_DIR" ]; then
    echo "==> downloading Qlib US daily data → $QLIB_DATA_DIR"
    python -m qlib.run.get_data qlib_data \
        --target_dir "$QLIB_DATA_DIR" \
        --region us
else
    echo "==> qlib us_data already exists at $QLIB_DATA_DIR (skip download)"
fi

echo "==> smoke-testing qlib import"
python -c "import qlib; qlib.init(provider_uri='$QLIB_DATA_DIR', region='us'); \
from qlib.data import D; print('instruments:', len(D.instruments('sp500').__dict__.get('market', 'sp500')))"

echo "==> done. Next:"
echo "   python qlib_study/run.py baseline"
echo "   python qlib_study/run.py enhanced"
