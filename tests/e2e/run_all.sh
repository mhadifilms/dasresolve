#!/usr/bin/env bash
# DasGrain end-to-end test driver.
#
# Steps performed (each one optional, see flags):
#   1. (re)build the OFX plugin
#   2. install it to /Library/OFX/Plugins (sudo)
#   3. wait for the user to restart DaVinci Resolve so the new plugin is
#      picked up by Resolve's OFX cache
#   4. (re)generate the synthetic test fixtures
#   5. probe Resolve via the scripting API
#   6. run the full Resolve pipeline harness
#
# Usage:
#   bash tests/e2e/run_all.sh                   # default: do everything
#   bash tests/e2e/run_all.sh --skip-build      # skip cmake build
#   bash tests/e2e/run_all.sh --skip-install    # skip sudo install
#   bash tests/e2e/run_all.sh --cleanup         # delete the test project at the end
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO=$(pwd)

DO_BUILD=1
DO_INSTALL=1
DO_FIXTURES=1
DO_PROBE=1
DO_HARNESS=1
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --skip-build)    DO_BUILD=0 ;;
    --skip-install)  DO_INSTALL=0 ;;
    --skip-fixtures) DO_FIXTURES=0 ;;
    --skip-probe)    DO_PROBE=0 ;;
    --skip-harness)  DO_HARNESS=0 ;;
    --cleanup)       EXTRA_ARGS+=(--cleanup) ;;
    --keep-current-project) EXTRA_ARGS+=(--keep-current-project) ;;
    *)               echo "unknown arg: $arg" ; exit 2 ;;
  esac
done

export RESOLVE_SCRIPT_API="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
export RESOLVE_SCRIPT_LIB="/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
export PYTHONPATH="${PYTHONPATH:-}:$RESOLVE_SCRIPT_API/Modules"

if [[ "$DO_BUILD" == 1 ]]; then
  echo "==> [1/6] cmake build"
  cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build build -j
fi

if [[ "$DO_INSTALL" == 1 ]]; then
  echo "==> [2/6] sudo install (will prompt for your password)"
  sudo cmake --install build
  ls -la /Library/OFX/Plugins/DasGrain.ofx.bundle/Contents/MacOS/
fi

if [[ "$DO_INSTALL" == 1 ]]; then
  echo
  echo "==> [3/6] Please QUIT and RELAUNCH DaVinci Resolve so it re-scans"
  echo "          /Library/OFX/Plugins for the new DasGrain bundle."
  echo "          Press <Return> here once Resolve is back up."
  read -r _
fi

if [[ "$DO_FIXTURES" == 1 ]]; then
  echo "==> [4/6] generate synthetic fixtures"
  python3 tests/e2e/gen_fixtures.py
fi

if [[ "$DO_PROBE" == 1 ]]; then
  echo "==> [5/6] probe Resolve"
  python3 tests/e2e/probe_resolve.py
fi

if [[ "$DO_HARNESS" == 1 ]]; then
  echo "==> [6/6] run end-to-end harness"
  python3 tests/e2e/run_e2e.py "${EXTRA_ARGS[@]}"
fi

echo
echo "OK"
