#!/usr/bin/env bash
#
# setup.sh — One-click dependency setup for Sherpa-ONNX C ASR project
#
# Downloads:
#   1. Sherpa-ONNX pre-built Windows x64 shared library (v1.13.4)
#   2. Chinese streaming Zipformer model (int8, 2025-06-30)
#   3. Chinese-English punctuation model (ct-transformer, int8)
#
# Usage: bash setup.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${SCRIPT_DIR}/deps"
MODELS_DIR="${SCRIPT_DIR}/models"
LIB_DIR="${SCRIPT_DIR}/lib"

SHERPA_VER="v1.13.4"
SHERPA_PKG="sherpa-onnx-${SHERPA_VER}-win-x64-shared-MD-Release-no-tts"
SHERPA_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/${SHERPA_VER}/${SHERPA_PKG}.tar.bz2"

ASR_MODEL="sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30"
ASR_MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${ASR_MODEL}.tar.bz2"

PUNCT_MODEL="sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8"
PUNCT_MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/${PUNCT_MODEL}.tar.bz2"

echo "============================================"
echo " Sherpa-ONNX C ASR — Dependency Setup"
echo "============================================"
echo ""

# ---- Check for required tools ----
check_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo "[ERROR] Required tool not found: $1"
        echo "  Please install $1 and try again."
        exit 1
    fi
}

check_tool curl
check_tool tar

# ---- Step 1: Download Sherpa-ONNX library ----
echo "[1/3] Sherpa-ONNX pre-built library..."
SHERPA_DEST="${DEPS_DIR}/sherpa-onnx"

if [ -f "${SHERPA_DEST}/include/sherpa-onnx/c-api/c-api.h" ] && \
   [ -f "${SHERPA_DEST}/lib/sherpa-onnx-c-api.dll" ]; then
    echo "  -> Already exists, skipping."
else
    echo "  Downloading ${SHERPA_PKG}.tar.bz2 ..."
    mkdir -p "${DEPS_DIR}"
    curl -L --progress-bar -o "${DEPS_DIR}/${SHERPA_PKG}.tar.bz2" "${SHERPA_URL}"

    echo "  Extracting..."
    mkdir -p "${SHERPA_DEST}"
    tar -xf "${DEPS_DIR}/${SHERPA_PKG}.tar.bz2" -C "${DEPS_DIR}"

    # Move contents from extracted dir into sherpa-onnx/
    if [ -d "${DEPS_DIR}/${SHERPA_PKG}" ]; then
        cp -r "${DEPS_DIR}/${SHERPA_PKG}/"* "${SHERPA_DEST}/"
        rm -rf "${DEPS_DIR}/${SHERPA_PKG}"
    fi

    rm -f "${DEPS_DIR}/${SHERPA_PKG}.tar.bz2"
    echo "  -> Done."
fi

# ---- Step 1b: Generate MinGW-compatible import library ----
echo "  Generating MinGW import library..."
SHERPA_DLL="${SHERPA_DEST}/lib/sherpa-onnx-c-api.dll"
SHERPA_LIB_A="${SHERPA_DEST}/lib/libsherpa-onnx-c-api.a"

if [ -f "${SHERPA_LIB_A}" ]; then
    echo "  -> MinGW .a already exists, skipping."
else
    # Check if gendef and dlltool are available
    GENDEF_CMD=""
    DLLTOOL_CMD=""

    # Try CP Editor MinGW
    if [ -f "D:/CP Editor/cpeditor/mingw64/bin/gendef.exe" ]; then
        GENDEF_CMD="D:/CP Editor/cpeditor/mingw64/bin/gendef.exe"
        DLLTOOL_CMD="D:/CP Editor/cpeditor/mingw64/bin/dlltool.exe"
    fi

    # Try system PATH
    if [ -z "${GENDEF_CMD}" ] && command -v gendef &>/dev/null; then
        GENDEF_CMD="gendef"
        DLLTOOL_CMD="dlltool"
    fi

    if [ -n "${GENDEF_CMD}" ] && [ -f "${SHERPA_DLL}" ]; then
        cd "${SHERPA_DEST}/lib"
        "${GENDEF_CMD}" sherpa-onnx-c-api.dll 2>/dev/null || true
        if [ -f sherpa-onnx-c-api.def ]; then
            "${DLLTOOL_CMD}" -d sherpa-onnx-c-api.def \
                -l libsherpa-onnx-c-api.a \
                -D sherpa-onnx-c-api.dll 2>/dev/null || true
            echo "  -> MinGW import library generated."
        else
            echo "  [WARN] gendef failed to produce .def file."
            echo "  [WARN] Will try direct linking at build time."
        fi
        cd "${SCRIPT_DIR}"
    else
        echo "  [WARN] gendef/dlltool not found."
        echo "  [WARN] You may need to link directly against the .dll."
        echo "  [WARN] The Makefile will use '-lsherpa-onnx-c-api' which needs the .a file."
    fi
fi

# ---- Step 1c: Copy DLLs to lib/ for runtime ----
mkdir -p "${LIB_DIR}"
cp -u "${SHERPA_DEST}/lib/"*.dll "${LIB_DIR}/" 2>/dev/null || true
echo "  -> Runtime DLLs copied to lib/"

# ---- Step 2: Download ASR model ----
echo ""
echo "[2/3] Chinese streaming ASR model..."
ASR_DEST="${MODELS_DIR}/asr"

if [ -f "${ASR_DEST}/encoder.onnx" ] && \
   [ -f "${ASR_DEST}/decoder.onnx" ] && \
   [ -f "${ASR_DEST}/joiner.onnx" ] && \
   [ -f "${ASR_DEST}/tokens.txt" ]; then
    echo "  -> Already exists, skipping."
else
    echo "  Downloading ${ASR_MODEL}.tar.bz2 ..."
    mkdir -p "${MODELS_DIR}"
    curl -L --progress-bar -o "${MODELS_DIR}/${ASR_MODEL}.tar.bz2" "${ASR_MODEL_URL}"

    echo "  Extracting..."
    tar -xf "${MODELS_DIR}/${ASR_MODEL}.tar.bz2" -C "${MODELS_DIR}"

    # Move contents from model dir into models/asr/
    mkdir -p "${ASR_DEST}"
    if [ -d "${MODELS_DIR}/${ASR_MODEL}" ]; then
        cp -r "${MODELS_DIR}/${ASR_MODEL}/"* "${ASR_DEST}/"
        rm -rf "${MODELS_DIR}/${ASR_MODEL}"
    fi

    rm -f "${MODELS_DIR}/${ASR_MODEL}.tar.bz2"
    echo "  -> Done."
fi

# ---- Step 3: Download punctuation model ----
echo ""
echo "[3/3] Chinese-English punctuation model..."
PUNCT_DEST="${MODELS_DIR}/punct"

if [ -f "${PUNCT_DEST}/model.int8.onnx" ]; then
    echo "  -> Already exists, skipping."
else
    echo "  Downloading ${PUNCT_MODEL}.tar.bz2 ..."
    mkdir -p "${MODELS_DIR}"
    curl -L --progress-bar -o "${MODELS_DIR}/${PUNCT_MODEL}.tar.bz2" "${PUNCT_MODEL_URL}"

    echo "  Extracting..."
    tar -xf "${MODELS_DIR}/${PUNCT_MODEL}.tar.bz2" -C "${MODELS_DIR}"

    # Move contents from model dir into models/punct/
    mkdir -p "${PUNCT_DEST}"
    if [ -d "${MODELS_DIR}/${PUNCT_MODEL}" ]; then
        cp -r "${MODELS_DIR}/${PUNCT_MODEL}/"* "${PUNCT_DEST}/"
        rm -rf "${MODELS_DIR}/${PUNCT_MODEL}"
    fi

    rm -f "${MODELS_DIR}/${PUNCT_MODEL}.tar.bz2"
    echo "  -> Done."
fi

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " Directory layout:"
echo "   deps/sherpa-onnx/  — Sherpa-ONNX library + headers"
echo "   models/asr/        — Streaming ASR model"
echo "   models/punct/      — Punctuation model"
echo "   lib/               — Runtime DLLs"
echo ""
echo " Next step: make"
echo "============================================"
