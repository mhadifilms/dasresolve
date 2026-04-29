#!/usr/bin/env bash
# Hands-free dev loop: rebuild the plugin, ask Resolve to quit cleanly,
# relaunch it, wait for scripting to come back up, optionally reload the
# previously-open project so you don't lose your place.
#
# Prereq: you've run tests/e2e/setup_dev.sh once.
#
# Usage:
#   bash tests/e2e/devloop.sh
#   bash tests/e2e/devloop.sh --no-build       # skip cmake build
#   bash tests/e2e/devloop.sh --no-restart     # skip Resolve restart
#   bash tests/e2e/devloop.sh --probe          # also run probe_resolve.py
#   bash tests/e2e/devloop.sh --reload-project Meridian_IMF_Spanish
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO=$(pwd -P)

DO_BUILD=1
DO_RESTART=1
DO_PROBE=0
RELOAD_PROJECT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build)        DO_BUILD=0; shift ;;
    --no-restart)      DO_RESTART=0; shift ;;
    --probe)           DO_PROBE=1; shift ;;
    --reload-project)  RELOAD_PROJECT=${2:-}; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

export RESOLVE_SCRIPT_API="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
export RESOLVE_SCRIPT_LIB="/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
export PYTHONPATH="${PYTHONPATH:-}:$RESOLVE_SCRIPT_API/Modules"

if [[ "$DO_BUILD" == 1 ]]; then
  echo "==> cmake build"
  cmake --build build -j
fi

if [[ "$DO_RESTART" == 1 ]]; then
  echo "==> restart Resolve"
  python3 - <<PY
import os, sys, time, subprocess
sys.path.insert(0, os.path.join(os.environ["RESOLVE_SCRIPT_API"], "Modules"))
import DaVinciResolveScript as bmd

remembered = "$RELOAD_PROJECT"

resolve = bmd.scriptapp("Resolve")
if resolve is not None:
    pm = resolve.GetProjectManager()
    cur = pm.GetCurrentProject() if pm else None
    if cur and not remembered:
        remembered = cur.GetName()
        print(f"  remembering open project: {remembered}")
        pm.SaveProject()
    print("  asking Resolve to quit ...")
    try:
        resolve.Quit()
    except Exception as exc:
        print(f"  Quit() raised (often harmless): {exc}")
else:
    print("  Resolve isn't running, nothing to quit")

# Wait for the process to actually disappear (Quit is async).
import shutil
deadline = time.time() + 60
while time.time() < deadline:
    rc = subprocess.run(
        ["pgrep", "-x", "Resolve"], stdout=subprocess.DEVNULL,
    ).returncode
    if rc != 0:
        break
    time.sleep(1)
print("  Resolve fully quit.")

# Relaunch via 'open' (so it shows up in the Dock as usual).
print("  relaunching Resolve ...")
subprocess.Popen(["open", "-a", "DaVinci Resolve"])

# Wait until scripting answers again.
deadline = time.time() + 120
resolve = None
while time.time() < deadline:
    resolve = bmd.scriptapp("Resolve")
    if resolve is not None:
        break
    time.sleep(2)
if resolve is None:
    print("  ERROR: Resolve did not come back up", file=sys.stderr)
    raise SystemExit(3)
print(f"  back up: {resolve.GetProductName()} v{resolve.GetVersionString()}")

if remembered:
    pm = resolve.GetProjectManager()
    print(f"  reopening project '{remembered}' ...")
    proj = pm.LoadProject(remembered)
    print(f"  reopened: {proj.GetName() if proj else None}")
PY
fi

if [[ "$DO_PROBE" == 1 ]]; then
  echo "==> probe Resolve"
  python3 tests/e2e/probe_resolve.py || true
fi

echo
echo "OK"
