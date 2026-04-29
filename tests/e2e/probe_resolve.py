#!/usr/bin/env python3
"""Lightweight probes to confirm DasGrain is installed and Resolve sees it.

Run as::

    python3 tests/e2e/probe_resolve.py

Checks performed (in order):

1. The OFX bundle exists at ``/Library/OFX/Plugins/DasGrain.ofx.bundle``.
2. Resolve scripting is reachable.
3. The currently running Resolve has cached the plugin (looks at
   ``~/Library/Application Support/Blackmagic Design/DaVinci Resolve/OFXPluginCacheV2.xml``).

If any probe fails, the script prints actionable advice and exits non-zero.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

OFX_BUNDLE = Path("/Library/OFX/Plugins/DasGrain.ofx.bundle")
OFX_CACHE  = Path.home() / (
    "Library/Application Support/Blackmagic Design/DaVinci Resolve/OFXPluginCacheV2.xml"
)


def step(idx: int, label: str) -> None:
    print(f"[{idx}] {label}")


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}", file=sys.stderr)


def ok(msg: str) -> None:
    print(f"  ok: {msg}")


def main() -> int:
    rc = 0

    step(1, "check installed bundle")
    if OFX_BUNDLE.is_dir():
        ok(f"{OFX_BUNDLE} exists")
        binary = OFX_BUNDLE / "Contents" / "MacOS" / "DasGrain.ofx"
        if binary.exists():
            ok(f"binary found at {binary}")
        else:
            fail(f"binary missing inside bundle: {binary}")
            rc |= 1
    else:
        fail(f"{OFX_BUNDLE} not found. Run: sudo cmake --install build")
        rc |= 1

    step(2, "connect to Resolve scripting")
    try:
        os.environ.setdefault(
            "RESOLVE_SCRIPT_API",
            "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting",
        )
        os.environ.setdefault(
            "RESOLVE_SCRIPT_LIB",
            "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so",
        )
        modules = os.path.join(os.environ["RESOLVE_SCRIPT_API"], "Modules")
        if modules not in sys.path:
            sys.path.insert(0, modules)
        import DaVinciResolveScript as bmd  # type: ignore
        resolve = bmd.scriptapp("Resolve")
        if resolve is None:
            fail("scriptapp('Resolve') returned None — Resolve not running, "
                 "or external scripting is disabled (Preferences > System > "
                 "General > External scripting using → Local).")
            rc |= 2
            resolve = None
        else:
            ok(f"connected to {resolve.GetProductName()} "
               f"v{resolve.GetVersionString()} (page={resolve.GetCurrentPage()})")
    except Exception as exc:
        fail(f"could not import DaVinciResolveScript: {exc!r}")
        rc |= 2

    step(3, "check OFX plugin cache")
    if OFX_CACHE.exists():
        text = OFX_CACHE.read_text(encoding="utf-8", errors="ignore")
        if "DasGrain" in text or "com.dasgrain.ofx.DasGrain" in text:
            ok(f"DasGrain entry present in {OFX_CACHE.name}")
        else:
            fail(f"DasGrain not yet in OFX cache. Quit and relaunch DaVinci "
                 f"Resolve so it re-scans /Library/OFX/Plugins.")
            rc |= 4
    else:
        fail(f"{OFX_CACHE} does not exist yet (Resolve has never been run).")
        rc |= 4

    if rc == 0:
        print("\nall probes ok — proceed to tests/e2e/run_e2e.py")
    else:
        print("\none or more probes failed (rc=%d)" % rc, file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
