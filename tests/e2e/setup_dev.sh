#!/usr/bin/env bash
# One-time dev setup. Run ONCE; afterwards plugin rebuilds are picked up
# automatically (just rebuild + restart Resolve).
#
# What this does:
#   1. removes any existing /Library/OFX/Plugins/DasGrain.ofx.bundle
#      (whether a real directory or a stale symlink)
#   2. creates a symlink there pointing at this repo's
#      build/DasGrain.ofx.bundle
#
# After this, a typical dev loop is:
#
#   cmake --build build -j        # no sudo, refreshes the bundle in place
#   bash tests/e2e/devloop.sh     # auto-restarts Resolve and re-tests
#
# Re-run this script only if you delete the build/ tree or move the repo.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO=$(pwd -P)
BUILD_BUNDLE="$REPO/build/DasGrain.ofx.bundle"
INSTALL_DIR="/Library/OFX/Plugins"
LINK_TARGET="$INSTALL_DIR/DasGrain.ofx.bundle"

if [[ ! -d "$BUILD_BUNDLE" ]]; then
  echo "ERROR: $BUILD_BUNDLE doesn't exist. Run 'cmake --build build -j' first." >&2
  exit 1
fi

echo "About to symlink:"
echo "    $LINK_TARGET"
echo "  ->"
echo "    $BUILD_BUNDLE"
echo
echo "This is a ONE-TIME sudo prompt. After this, just run cmake --build"
echo "(no sudo) and the new bundle is picked up automatically on Resolve"
echo "restart."
echo
sudo mkdir -p "$INSTALL_DIR"
sudo rm -rf "$LINK_TARGET"
sudo ln -s "$BUILD_BUNDLE" "$LINK_TARGET"

ls -la "$LINK_TARGET"
echo
echo "OK — dev symlink is in place."
