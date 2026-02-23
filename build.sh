#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Init / update submodules ---
echo "=== Initializing submodules ==="
git submodule update --init --recursive

# --- Create venv if it doesn't exist ---
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "=== Creating virtual environment ==="
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "=== Installing build dependencies ==="
pip install --upgrade pip
pip install scikit-build-core pybind11 numpy pytest

# --- Build wheel ---
echo "=== Building wheel ==="
CC="${CC:-gcc}" CXX="${CXX:-g++}" pip wheel . --no-build-isolation -w dist/

# --- Install wheel ---
echo "=== Installing wheel ==="
pip install --force-reinstall --no-deps dist/chszlablib-*.whl

# --- Copy .so extensions into source tree (so tests work from project root) ---
SITE_PKG="$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/chszlablib"
cp "$SITE_PKG"/_*.so "$SCRIPT_DIR/chszlablib/"

# --- Run tests ---
echo "=== Running tests ==="
pytest tests/ -v

echo ""
echo "=== All done ==="
echo "Wheel(s) in dist/:"
ls -lh dist/chszlablib-*.whl
