#!/usr/bin/env bash
# Build libmtkahypar.so from source for HeiCut integration.
# Clones mt-kahypar at a pinned commit, builds the shared library,
# and installs it to HeiCut's extern/mt-kahypar-library/ directory.
#
# System dependencies (macOS): brew install hwloc tbb boost
# System dependencies (Linux): hwloc-devel (TBB & Boost downloaded automatically)
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

# ---- Platform detection -----------------------------------------------
case "$(uname -s)" in
    Darwin*)
        LIB_EXT="dylib"
        NPROC="$(sysctl -n hw.ncpu)"
        IS_LINUX=false
        ;;
    *)
        LIB_EXT="so"
        NPROC="$(nproc)"
        IS_LINUX=true
        ;;
esac

# ---- Skip if already built --------------------------------------------
if [ -f "${DEST_DIR}/libmtkahypar.${LIB_EXT}" ]; then
    echo ">>> libmtkahypar.${LIB_EXT} already exists at ${DEST_DIR}, skipping build."
    echo "    To rebuild, delete it first: rm ${DEST_DIR}/libmtkahypar.${LIB_EXT}"
    exit 0
fi

echo ">>> Building Mt-KaHyPar (libmtkahypar.${LIB_EXT})"
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

# ---- Platform-specific dependency setup & cmake flags -----------------
if $IS_LINUX; then
    # On Linux (manylinux containers), system TBB/Boost are too old.
    # mt-KaHyPar's cmake download scripts are broken in containers
    # (FetchContent extracts to / instead of source dir).
    # We download both ourselves and patch cmake to find them.

    TBB_DIR="${SRC_DIR}/external_tools/tbb"
    BOOST_DIR="${SRC_DIR}/external_tools/boost"

    echo ">>> Downloading TBB..."
    mkdir -p "${TBB_DIR}"
    curl -sL "${TBB_URL}" | tar xz --strip-components=1 -C "${TBB_DIR}"

    echo ">>> Downloading Boost..."
    mkdir -p "${BOOST_DIR}"
    curl -sL "${BOOST_URL}" | tar xj --strip-components=1 -C "${BOOST_DIR}"

    # Patch CMakeLists: fix paths and add -fPIC for mini_boost
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

    CMAKE_EXTRA_FLAGS=(
        -DKAHYPAR_DOWNLOAD_TBB=ON
        -DKAHYPAR_DOWNLOAD_BOOST=ON
        -DMT_KAHYPAR_DISABLE_BOOST=OFF
    )
else
    # On macOS, use system TBB and Boost from Homebrew.
    CMAKE_EXTRA_FLAGS=(
        -DKAHYPAR_DOWNLOAD_TBB="${KAHYPAR_DOWNLOAD_TBB:-OFF}"
        -DKAHYPAR_DOWNLOAD_BOOST="${KAHYPAR_DOWNLOAD_BOOST:-OFF}"
        -DMT_KAHYPAR_DISABLE_BOOST=OFF
    )
fi

# ---- Configure & build ------------------------------------------------
mkdir -p "${BLD_DIR}"

cmake -S "${SRC_DIR}" -B "${BLD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DKAHYPAR_ENFORCE_MINIMUM_TBB_VERSION=OFF \
    -DKAHYPAR_PYTHON=OFF \
    "${CMAKE_EXTRA_FLAGS[@]}"

cmake --build "${BLD_DIR}" --target mtkahypar -j"${NPROC}"

# ---- Find and install the library -------------------------------------
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

# ---- Copy TBB/Boost headers for HeiCut compilation --------------------
if $IS_LINUX; then
    # TBB headers are needed by mt-kahypar headers included by HeiCut
    if [ -d "${SRC_DIR}/external_tools/tbb/include" ]; then
        cp -r "${SRC_DIR}/external_tools/tbb/include" "${DEST_DIR}/tbb_include"
        echo ">>> Installed TBB headers to ${DEST_DIR}/tbb_include"
    fi
    # TBB shared libraries needed at link time
    if [ -d "${SRC_DIR}/external_tools/tbb/lib" ]; then
        cp -r "${SRC_DIR}/external_tools/tbb/lib" "${DEST_DIR}/tbb_lib"
        echo ">>> Installed TBB libraries to ${DEST_DIR}/tbb_lib"
    fi
    # Boost headers (for program_options used by HeiCut)
    if [ -d "${SRC_DIR}/external_tools/boost/boost" ]; then
        mkdir -p "${DEST_DIR}/boost_include"
        cp -r "${SRC_DIR}/external_tools/boost/boost" "${DEST_DIR}/boost_include/boost"
        echo ">>> Installed Boost headers to ${DEST_DIR}/boost_include"
    fi
fi

# ---- Copy mt-kahypar submodule headers needed by HeiCut ----------------
MTKAHYPAR_SUBMOD="${HEICUT_DIR}/extern/mt-kahypar"
# kahypar-shared-resources (provides kahypar-resources/macros.h etc.)
if [ -d "${SRC_DIR}/external_tools/kahypar-shared-resources" ] && \
   [ ! -d "${MTKAHYPAR_SUBMOD}/external_tools/kahypar-shared-resources" ]; then
    mkdir -p "${MTKAHYPAR_SUBMOD}/external_tools"
    cp -r "${SRC_DIR}/external_tools/kahypar-shared-resources" \
          "${MTKAHYPAR_SUBMOD}/external_tools/kahypar-shared-resources"
    echo ">>> Installed kahypar-shared-resources headers"
fi
# growt (provides growt hash tables)
if [ -d "${SRC_DIR}/external_tools/growt" ] && \
   [ ! -d "${MTKAHYPAR_SUBMOD}/external_tools/growt" ]; then
    mkdir -p "${MTKAHYPAR_SUBMOD}/external_tools"
    cp -r "${SRC_DIR}/external_tools/growt" \
          "${MTKAHYPAR_SUBMOD}/external_tools/growt"
    echo ">>> Installed growt headers"
fi

# ---- Clean up build directory -----------------------------------------
rm -rf "${WORK_DIR}"
echo ">>> Build directory cleaned up."
