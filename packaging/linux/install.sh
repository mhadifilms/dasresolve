#!/usr/bin/env bash
# Copyright 2026 The DasGrain OFX Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Build + install DasGrain.ofx.bundle to the system OFX plugin location.
# Requires sudo for the final install step.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
PLUGIN_DIR="/usr/OFX/Plugins"

cd "${REPO_ROOT}"

if [ ! -f "third_party/openfx/include/ofxImageEffect.h" ]; then
    echo "OpenFX submodule missing. Initialising..."
    git submodule update --init --recursive
fi

cmake -B "${BUILD_DIR}" -S . \
      -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j "$(nproc)"

echo "Installing to ${PLUGIN_DIR} (sudo)..."
sudo mkdir -p "${PLUGIN_DIR}"
sudo cp -R "${BUILD_DIR}/DasGrain.ofx.bundle" "${PLUGIN_DIR}/"

echo "Done. Restart DaVinci Resolve to load the plugin."
