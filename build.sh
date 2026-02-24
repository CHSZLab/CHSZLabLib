#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Init / update submodules ---
echo "=== Initializing submodules ==="

# Init top-level submodules (non-recursive to avoid broken nested refs)
git submodule update --init

# Recursively init nested submodules, tolerating failures from commits
# that no longer exist on remotes (e.g. fpt-max-cut/solvers/MQLib).
# For those, fetch and checkout the remote default branch as a fallback.
for top in $(git submodule foreach --quiet 'echo $sm_path'); do
    if [ -f "$top/.gitmodules" ]; then
        if ! (cd "$top" && git submodule update --init --recursive 2>/dev/null); then
            echo "  -> recursive init failed in $top, fixing broken nested submodules"
            # Register nested submodules (creates .git file + module dir)
            (cd "$top" && git submodule init 2>/dev/null || true)
            for nested in $(cd "$top" && git config --file .gitmodules -l \
                            | grep '\.path=' | sed 's/.*\.path=//'); do
                target="$top/$nested"
                # Skip if already populated (has files beyond just .git)
                file_count=$(find "$target" -maxdepth 1 ! -name .git ! -name . 2>/dev/null | wc -l)
                [ "$file_count" -gt 0 ] && continue
                echo "  -> fixing $target (pinned commit unavailable, using remote HEAD)"
                # The submodule dir has a .git pointer; fetch and checkout default branch
                (cd "$target" && git fetch origin 2>/dev/null \
                    && branch=$(git remote show origin 2>/dev/null | sed -n 's/.*HEAD branch: //p') \
                    && git checkout "origin/$branch" -- . ) \
                || {
                    # Fallback: replace with a fresh shallow clone
                    url=$(cd "$top" && git config --file .gitmodules "submodule.$nested.url")
                    rm -rf "$target"
                    git clone --depth 1 "$url" "$target"
                }
            done
        fi
    fi
done

# Ensure submodule working trees are fully checked out
git submodule foreach --recursive 'git checkout HEAD -- . 2>/dev/null || true'

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
rm -f dist/chszlablib-*.whl
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
