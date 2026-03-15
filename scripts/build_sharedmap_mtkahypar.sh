#!/usr/bin/env bash
# Build libmtkahypar.so (v1.5.3) from source for SharedMap integration.
# Clones mt-kahypar at tag v1.5.3, builds the C library,
# and installs it to SharedMap's extern/local/mt-kahypar/ directory.
#
# v1.5.3 uses CMake FetchContent for dependencies (no git submodules).
#
# System dependencies (macOS): brew install hwloc tbb boost
# System dependencies (Linux): hwloc-devel (TBB & Boost downloaded via FetchContent)
set -euo pipefail

# ---- Config (override via env vars) -----------------------------------
MTK_REPO="${MTK_REPO:-https://github.com/kahypar/mt-kahypar.git}"
MTK_TAG="${MTK_TAG:-v1.5.3}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

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
# On macOS, the library is renamed to libmtkahypar_sm.dylib to avoid
# collision with HeiCut's libmtkahypar.dylib.
for _dir in "${DEST_DIR}/lib" "${DEST_DIR}/lib64"; do
    for _name in "libmtkahypar.${LIB_EXT}" "libmtkahypar_sm.${LIB_EXT}"; do
        if [ -f "${_dir}/${_name}" ]; then
            echo ">>> ${_name} already exists at ${_dir}, skipping build."
            echo "    To rebuild, delete it first: rm ${_dir}/${_name}"
            exit 0
        fi
    done
done

echo ">>> Building Mt-KaHyPar v1.5.3 for SharedMap (libmtkahypar.${LIB_EXT})"
echo "    repo   : ${MTK_REPO}"
echo "    tag    : ${MTK_TAG}"
echo "    build  : ${BUILD_TYPE}"
echo "    dest   : ${DEST_DIR}"
echo

# ---- Clone at pinned tag ----------------------------------------------
# v1.5.3 has no git submodules; dependencies are fetched via CMake FetchContent.
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}" "${DEST_DIR}"

git clone --branch "${MTK_TAG}" --depth 1 "${MTK_REPO}" "${SRC_DIR}"

# ---- Platform-specific cmake flags ------------------------------------
USE_64BIT_IDS=ON
if $IS_LINUX; then
    # On Linux (manylinux containers), use FetchContent to download TBB & Boost.
    CMAKE_EXTRA_FLAGS=(
        -DKAHYPAR_DOWNLOAD_TBB=ON
        -DKAHYPAR_DOWNLOAD_BOOST=ON
    )
else
    # On macOS, use system TBB and Boost from Homebrew.
    # Disable 64-bit IDs on macOS: unsigned long is already 8 bytes on arm64,
    # but KAHYPAR_USE_64_BIT_IDS changes typedefs to unsigned long long which
    # causes type mismatches with size_t (unsigned long) in std::max etc.
    USE_64BIT_IDS=OFF
    CMAKE_EXTRA_FLAGS=(
        -DKAHYPAR_DOWNLOAD_TBB="${KAHYPAR_DOWNLOAD_TBB:-OFF}"
        -DKAHYPAR_DOWNLOAD_BOOST="${KAHYPAR_DOWNLOAD_BOOST:-OFF}"
    )
fi

# ---- Configure & build ------------------------------------------------
mkdir -p "${BLD_DIR}"

cmake -S "${SRC_DIR}" -B "${BLD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="${STAGE_DIR}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DKAHYPAR_USE_64_BIT_IDS="${USE_64BIT_IDS}" \
    -DKAHYPAR_ENABLE_THREAD_PINNING=OFF \
    -DKAHYPAR_DISABLE_ASSERTIONS=ON \
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

# Determine destination lib directory
DEST_LIBDIR="${DEST_DIR}/lib"
mkdir -p "${DEST_LIBDIR}"

if $IS_LINUX; then
    # Copy versioned .so and create symlinks
    cp -a "${STAGE_LIBDIR}"/libmtkahypar.so* "${DEST_LIBDIR}/"

    # Bundle FetchContent-downloaded TBB + Boost runtime deps
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
    # macOS: copy and rename to avoid collision with HeiCut's libmtkahypar.dylib
    cp "${STAGE_LIBDIR}/libmtkahypar.${LIB_EXT}" "${DEST_LIBDIR}/libmtkahypar_sm.${LIB_EXT}"

    # Fix install name to use @rpath for relocatable wheels
    install_name_tool -id "@rpath/libmtkahypar_sm.${LIB_EXT}" \
        "${DEST_LIBDIR}/libmtkahypar_sm.${LIB_EXT}" 2>/dev/null || true

    echo ">>> Installed: ${DEST_LIBDIR}/libmtkahypar_sm.${LIB_EXT}"
fi

# ---- Clean up build directory -----------------------------------------
rm -rf "${WORK_DIR}"
echo ">>> Build directory cleaned up."
echo ">>> SharedMap Mt-KaHyPar v1.5.3 build complete."
