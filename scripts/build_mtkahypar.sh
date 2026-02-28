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
TBB_URL="https://github.com/oneapi-src/oneTBB/releases/download/v2021.7.0/oneapi-tbb-2021.7.0-lin.tgz"
BOOST_URL="https://sourceforge.net/projects/boost/files/boost/1.69.0/boost_1_69_0.tar.bz2/download"

# ---- Paths ------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

HEICUT_DIR="${REPO_ROOT}/external_repositories/HeiCut"
DEST_DIR="${HEICUT_DIR}/extern/mt-kahypar-library"

WORK_DIR="${REPO_ROOT}/build-mtkahypar"
SRC_DIR="${WORK_DIR}/mt-kahypar"
BLD_DIR="${SRC_DIR}/build"
STAGE_DIR="${WORK_DIR}/_install"

# ---- Platform-specific shared library extension -----------------------
case "$(uname -s)" in
    Darwin*) LIB_EXT="dylib" ;;
    *)       LIB_EXT="so" ;;
esac

# ---- Skip if already built --------------------------------------------
if [ -f "${DEST_DIR}/libmtkahypar.${LIB_EXT}" ]; then
    echo ">>> libmtkahypar.${LIB_EXT} already exists at ${DEST_DIR}, skipping build."
    echo "    To rebuild, delete it first: rm ${DEST_DIR}/libmtkahypar.${LIB_EXT}"
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

# ---- Download TBB & Boost into source tree ----------------------------
# mt-KaHyPar's execute_process(cmake -P download_*.cmake) uses
# FetchContent with relative SOURCE_DIR, which extracts to / inside
# containers instead of the source tree. We download both ourselves.
TBB_DIR="${SRC_DIR}/external_tools/tbb"
BOOST_DIR="${SRC_DIR}/external_tools/boost"

echo ">>> Downloading TBB..."
mkdir -p "${TBB_DIR}"
curl -sL "${TBB_URL}" | tar xz --strip-components=1 -C "${TBB_DIR}"

echo ">>> Downloading Boost..."
mkdir -p "${BOOST_DIR}"
curl -sL "${BOOST_URL}" | tar xj --strip-components=1 -C "${BOOST_DIR}"

# ---- Patch CMakeLists: fix download paths ----------------------------
# Replace BINARY_DIR with SOURCE_DIR so cmake finds the files we
# downloaded above, and disable the execute_process download calls.
python3 -c "
import pathlib
f = pathlib.Path('${SRC_DIR}/CMakeLists.txt')
txt = f.read_text()
# Fix Boost paths: glob and include use BINARY_DIR but files are in SOURCE_DIR
txt = txt.replace(
    '\${CMAKE_CURRENT_BINARY_DIR}/external_tools/boost/',
    '\${CMAKE_CURRENT_SOURCE_DIR}/external_tools/boost/')
# Fix TBB path
txt = txt.replace(
    'set(TBB_ROOT \${CMAKE_CURRENT_BINARY_DIR}/external_tools/tbb)',
    'set(TBB_ROOT \${CMAKE_CURRENT_SOURCE_DIR}/external_tools/tbb)')
# Disable the broken execute_process download calls (files already exist)
txt = txt.replace(
    'execute_process(COMMAND cmake -P \${CMAKE_CURRENT_SOURCE_DIR}/scripts/download_boost.cmake)',
    '# download_boost.cmake skipped (pre-downloaded)')
txt = txt.replace(
    'execute_process(COMMAND cmake -P \${CMAKE_CURRENT_SOURCE_DIR}/scripts/download_tbb_linux.cmake)',
    '# download_tbb_linux.cmake skipped (pre-downloaded)')
# Fix mini_boost: needs -fPIC to link into shared library
txt = txt.replace(
    'set_target_properties(mini_boost PROPERTIES LINKER_LANGUAGE CXX)',
    'set_target_properties(mini_boost PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE ON)')
f.write_text(txt)
"

# ---- Configure & build ------------------------------------------------
mkdir -p "${BLD_DIR}"

cmake -S "${SRC_DIR}" -B "${BLD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DKAHYPAR_DOWNLOAD_TBB=ON \
    -DKAHYPAR_DOWNLOAD_BOOST=ON \
    -DKAHYPAR_ENFORCE_MINIMUM_TBB_VERSION=OFF \
    -DKAHYPAR_PYTHON=OFF \
    -DMT_KAHYPAR_DISABLE_BOOST=OFF

cmake --build "${BLD_DIR}" --target mtkahypar -j"$(nproc)"

# ---- Find and install the .so -----------------------------------------
cmake --install "${BLD_DIR}" --prefix "${STAGE_DIR}" || true

CANDIDATES=(
    "${STAGE_DIR}/lib/libmtkahypar.${LIB_EXT}"
    "${STAGE_DIR}/lib64/libmtkahypar.${LIB_EXT}"
    "${STAGE_DIR}/lib/x86_64-linux-gnu/libmtkahypar.${LIB_EXT}"
    "${BLD_DIR}/libmtkahypar.${LIB_EXT}"
    "${BLD_DIR}/lib/libmtkahypar.${LIB_EXT}"
    "${BLD_DIR}/mt-kahypar/libmtkahypar.${LIB_EXT}"
)

FOUND=""
for f in "${CANDIDATES[@]}"; do
    [[ -f "$f" ]] && { FOUND="$f"; break; }
done

if [[ -z "${FOUND}" ]]; then
    FOUND="$(find "${BLD_DIR}" -maxdepth 6 -type f -name "libmtkahypar.${LIB_EXT}" | head -n1 || true)"
fi

if [[ -z "${FOUND}" ]]; then
    echo "!!! Could not locate libmtkahypar.${LIB_EXT} after build."
    echo "    Ensure hwloc & TBB dev packages are installed:"
    echo "      Linux:  yum install -y hwloc-devel  (or apt-get install libhwloc-dev libtbb-dev)"
    echo "      macOS:  brew install hwloc tbb boost"
    exit 1
fi

install -m 644 "${FOUND}" "${DEST_DIR}/libmtkahypar.${LIB_EXT}"
echo ">>> Installed: ${DEST_DIR}/libmtkahypar.${LIB_EXT}"

# ---- Clean up build directory -----------------------------------------
rm -rf "${WORK_DIR}"
echo ">>> Build directory cleaned up."
