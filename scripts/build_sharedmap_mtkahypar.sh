#!/usr/bin/env bash
# Build libmtkahypar.so (v1.5.3) from source for SharedMap integration.
# Clones mt-kahypar at tag v1.5.3, builds the C library,
# and installs it to SharedMap's extern/local/mt-kahypar/ directory.
#
# System dependencies (macOS): brew install hwloc tbb boost
# System dependencies (Linux): hwloc-devel (TBB & Boost downloaded automatically)
#
# Based on SharedMap's build.sh and scripts/build_mtkahypar.sh (HeiCut).
set -euo pipefail

# ---- Config (override via env vars) -----------------------------------
MTK_REPO="${MTK_REPO:-https://github.com/kahypar/mt-kahypar.git}"
MTK_TAG="${MTK_TAG:-v1.5.3}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
TBB_URL="https://github.com/oneapi-src/oneTBB/releases/download/v2021.7.0/oneapi-tbb-2021.7.0-lin.tgz"
BOOST_URL="https://sourceforge.net/projects/boost/files/boost/1.69.0/boost_1_69_0.tar.bz2/download"

# ---- Paths ------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

SHAREDMAP_DIR="${REPO_ROOT}/external_repositories/SharedMap"
DEST_DIR="${SHAREDMAP_DIR}/extern/local/mt-kahypar"

WORK_DIR="${REPO_ROOT}/build-sharedmap-mtkahypar"
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
# Check both lib/ and lib64/ — install location varies by platform
for _dir in "${DEST_DIR}/lib" "${DEST_DIR}/lib64"; do
    if [ -f "${_dir}/libmtkahypar.${LIB_EXT}" ]; then
        echo ">>> libmtkahypar.${LIB_EXT} already exists at ${_dir}, skipping build."
        echo "    To rebuild, delete it first: rm ${_dir}/libmtkahypar.${LIB_EXT}"
        exit 0
    fi
done

echo ">>> Building Mt-KaHyPar v1.5.3 for SharedMap (libmtkahypar.${LIB_EXT})"
echo "    repo   : ${MTK_REPO}"
echo "    tag    : ${MTK_TAG}"
echo "    build  : ${BUILD_TYPE}"
echo "    dest   : ${DEST_DIR}"
echo

# ---- Clone at pinned tag ----------------------------------------------
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}" "${DEST_DIR}"

git clone --recursive --branch "${MTK_TAG}" "${MTK_REPO}" "${SRC_DIR}"

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
    -DCMAKE_INSTALL_PREFIX="${STAGE_DIR}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DKAHYPAR_USE_64_BIT_IDS=ON \
    -DKAHYPAR_ENABLE_THREAD_PINNING=OFF \
    -DKAHYPAR_DISABLE_ASSERTIONS=ON \
    -DKAHYPAR_ENFORCE_MINIMUM_TBB_VERSION=OFF \
    -DKAHYPAR_PYTHON=OFF \
    "${CMAKE_EXTRA_FLAGS[@]}"

cmake --build "${BLD_DIR}" --target install-mtkahypar -j"${NPROC}"

# ---- Find installed library directory ----------------------------------
if [ -d "${STAGE_DIR}/lib64" ]; then
    STAGE_LIBDIR="${STAGE_DIR}/lib64"
else
    STAGE_LIBDIR="${STAGE_DIR}/lib"
fi

# Verify install produced the library
if [ ! -f "${STAGE_LIBDIR}/libmtkahypar.${LIB_EXT}" ] && \
   ! ls "${STAGE_LIBDIR}"/libmtkahypar.${LIB_EXT}* 2>/dev/null | head -n1 | grep -q .; then
    echo "!!! Could not locate libmtkahypar.${LIB_EXT} after install."
    echo "    Searched: ${STAGE_LIBDIR}/"
    echo "    Contents:"
    ls -la "${STAGE_LIBDIR}/" 2>/dev/null || echo "    (directory does not exist)"
    exit 1
fi

# ---- Install to SharedMap's expected location --------------------------
# Copy include files
mkdir -p "${DEST_DIR}/include"
if [ -d "${STAGE_DIR}/include" ]; then
    cp -r "${STAGE_DIR}/include/"* "${DEST_DIR}/include/"
fi
echo ">>> Installed headers to ${DEST_DIR}/include/"

# Determine destination lib directory (match what cmake installed)
DEST_LIBDIR="${DEST_DIR}/lib"
mkdir -p "${DEST_LIBDIR}"

if $IS_LINUX; then
    # Copy versioned .so and create symlinks (same as SharedMap's build.sh)
    cp -a "${STAGE_LIBDIR}"/libmtkahypar.so* "${DEST_LIBDIR}/"

    # Bundle downloaded TBB + Boost runtime deps
    for pattern in libtbb.so* libtbbmalloc.so* libboost_program_options.so* \
                   libboost_system.so* libboost_filesystem.so* \
                   libboost_thread.so* libboost_container.so*; do
        found=$(find "${BLD_DIR}" -type f -name "${pattern}" 2>/dev/null | head -n1)
        if [ -n "${found}" ]; then
            cp -a "$(dirname "${found}")"/${pattern} "${DEST_LIBDIR}/" 2>/dev/null || true
        fi
    done

    # Patch RPATH so libmtkahypar finds bundled deps next to itself
    if command -v patchelf >/dev/null 2>&1; then
        for f in "${DEST_LIBDIR}"/libmtkahypar.so*; do
            [ -f "$f" ] && [ ! -L "$f" ] && patchelf --set-rpath '$ORIGIN' "$f" || true
        done
    fi

    echo ">>> Bundled libs in ${DEST_LIBDIR}:"
    ls -1 "${DEST_LIBDIR}" | grep -E 'mtkahypar|libboost_|libtbb' || true
else
    # macOS: copy .dylib (no versioned soname scheme)
    cp -a "${STAGE_LIBDIR}"/libmtkahypar.${LIB_EXT}* "${DEST_LIBDIR}/"

    # Fix install name to use @rpath for relocatable wheels
    for f in "${DEST_LIBDIR}"/libmtkahypar.${LIB_EXT}*; do
        [ -f "$f" ] && [ ! -L "$f" ] && \
            install_name_tool -id "@rpath/$(basename "$f")" "$f" 2>/dev/null || true
    done

    echo ">>> Installed: ${DEST_LIBDIR}/libmtkahypar.${LIB_EXT}"
fi

# ---- Clean up build directory -----------------------------------------
rm -rf "${WORK_DIR}"
echo ">>> Build directory cleaned up."
echo ">>> SharedMap Mt-KaHyPar v1.5.3 build complete."
