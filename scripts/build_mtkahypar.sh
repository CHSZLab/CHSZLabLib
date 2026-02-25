#!/usr/bin/env bash
# Build libmtkahypar.so from source for HeiCut integration.
# Clones mt-kahypar at a pinned commit, builds the shared library,
# and installs it to HeiCut's extern/mt-kahypar-library/ directory.
#
# System dependencies: libtbb-dev, libboost-program-options-dev, libhwloc-dev
#
# Based on HeiCut's install_mtkahypar.sh.
set -euo pipefail

# ---- Config (override via env vars) -----------------------------------
MTK_REPO="${MTK_REPO:-https://github.com/kahypar/mt-kahypar.git}"
MTK_COMMIT="${MTK_COMMIT:-0ef674a}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
KAHYPAR_DOWNLOAD_TBB="${KAHYPAR_DOWNLOAD_TBB:-OFF}"

# ---- Paths ------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

HEICUT_DIR="${REPO_ROOT}/external_repositories/HeiCut"
DEST_DIR="${HEICUT_DIR}/extern/mt-kahypar-library"

WORK_DIR="${REPO_ROOT}/build-mtkahypar"
SRC_DIR="${WORK_DIR}/mt-kahypar"
BLD_DIR="${SRC_DIR}/build"
STAGE_DIR="${WORK_DIR}/_install"

# ---- Skip if already built --------------------------------------------
if [ -f "${DEST_DIR}/libmtkahypar.so" ]; then
    echo ">>> libmtkahypar.so already exists at ${DEST_DIR}, skipping build."
    echo "    To rebuild, delete it first: rm ${DEST_DIR}/libmtkahypar.so"
    exit 0
fi

echo ">>> Building Mt-KaHyPar (libmtkahypar.so)"
echo "    repo   : ${MTK_REPO}"
echo "    commit : ${MTK_COMMIT}"
echo "    build  : ${BUILD_TYPE}"
echo "    dest   : ${DEST_DIR}"
echo

# ---- Clone at pinned commit -------------------------------------------
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}" "${DEST_DIR}"

git clone "${MTK_REPO}" "${SRC_DIR}"
git -C "${SRC_DIR}" checkout "${MTK_COMMIT}"
git -C "${SRC_DIR}" submodule update --init --recursive

test -d "${SRC_DIR}/external_tools" || {
    echo "!!! external_tools/ missing. Submodules not fetched correctly."
    exit 1
}

# ---- Configure & build ------------------------------------------------
mkdir -p "${BLD_DIR}"

cmake -S "${SRC_DIR}" -B "${BLD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DKAHYPAR_DOWNLOAD_TBB="${KAHYPAR_DOWNLOAD_TBB}" \
    -DKAHYPAR_DOWNLOAD_BOOST=OFF \
    -DKAHYPAR_ENFORCE_MINIMUM_TBB_VERSION=OFF \
    -DKAHYPAR_PYTHON=OFF \
    -DMT_KAHYPAR_DISABLE_BOOST=OFF

cmake --build "${BLD_DIR}" --target mtkahypar -j"$(nproc)"

# ---- Find and install the .so -----------------------------------------
cmake --install "${BLD_DIR}" --prefix "${STAGE_DIR}" || true

CANDIDATES=(
    "${STAGE_DIR}/lib/libmtkahypar.so"
    "${STAGE_DIR}/lib64/libmtkahypar.so"
    "${STAGE_DIR}/lib/x86_64-linux-gnu/libmtkahypar.so"
    "${BLD_DIR}/libmtkahypar.so"
    "${BLD_DIR}/lib/libmtkahypar.so"
    "${BLD_DIR}/mt-kahypar/libmtkahypar.so"
)

FOUND=""
for f in "${CANDIDATES[@]}"; do
    [[ -f "$f" ]] && { FOUND="$f"; break; }
done

if [[ -z "${FOUND}" ]]; then
    FOUND="$(find "${BLD_DIR}" -maxdepth 6 -type f -name 'libmtkahypar.so' | head -n1 || true)"
fi

if [[ -z "${FOUND}" ]]; then
    echo "!!! Could not locate libmtkahypar.so after build."
    echo "    Ensure hwloc & TBB dev packages are installed:"
    echo "      sudo apt-get install libtbb-dev libhwloc-dev libboost-program-options-dev"
    exit 1
fi

install -m 644 "${FOUND}" "${DEST_DIR}/libmtkahypar.so"
echo ">>> Installed: ${DEST_DIR}/libmtkahypar.so"

# ---- Clean up build directory -----------------------------------------
rm -rf "${WORK_DIR}"
echo ">>> Build directory cleaned up."
