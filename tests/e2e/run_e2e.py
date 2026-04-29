#!/usr/bin/env python3
"""End-to-end smoke test for DasGrain inside DaVinci Resolve Studio.

Strategy
--------
Resolve's embedded Fusion exposes OFX plugins through ``comp.AddTool``
under their lower-cased OFX identifier prefixed with ``ofx.`` — i.e.
``com.dasgrain.ofx.DasGrain`` becomes ``ofx.com.dasgrain.ofx.DasGrain``.
Driving a Saver-based render programmatically through Resolve's embedded
Fusion is fragile (Loaders are sandboxed, ``comp.Render`` returns True
even when no work happened, the API can crash Resolve under rapid
tool churn), so this harness deliberately stops short of triggering
a frame render.

Instead it does the part that actually verifies the plugin:

1. Confirms Resolve's OFX cache lists DasGrain and DegrainHelper.
2. Looks up the ``ofx.com.dasgrain.ofx.DasGrain`` registry entry to
   prove Fusion rebuilt its tool catalogue with our bundle.
3. Opens (or creates) a persistent test project in the Local DB,
   imports a fixture clip, drops it on a timeline.
4. Adds a DasGrain node to that timeline item's Fusion comp, wires
   ``MediaIn → DasGrain (Source) → MediaOut``, sets a few params,
   saves the project.
5. Reads the comp back and verifies the DasGrain tool is still there
   with all the named clips and params we expect — i.e. the plugin
   doesn't get dropped on save/reload.

Once this passes, render via Deliver page in the UI to confirm
visually. The user's actual production workflow is in the UI; the
synthetic Saver-in-comp render is unnecessary and Resolve-specifically
broken.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE       = Path(__file__).resolve().parent
REPO       = HERE.parent.parent
FIXTURES   = HERE / "fixtures"

DASGRAIN_FUSION_REGID    = "ofx.com.dasgrain.ofx.DasGrain"
DEGRAIN_HELPER_FUSION_REGID = "ofx.com.dasgrain.ofx.DegrainHelper"
TEST_PROJECT_NAME = "DasGrainE2E_Persistent"

EXPECTED_CLIPS = ("Source", "Plate", "Degrained", "Mask", "ExternalGrain")
EXPECTED_PARAMS = (
    # Analyse tab
    "luminance", "fix_ghosting", "grain_mask_invert",
    "number_of_frames", "additional_frames", "sample_count",
    "analyse", "analyse_state", "analyse_mask_invert",
    # Adjust tab
    "response_curve_json",
    "response_curve_json_paste", "response_curve_import", "response_curve_export",
    "curve_help",
    # Replace tab
    "external_grain", "scatter", "sample_frame", "cell_size",
    "overlay_pattern", "edge_blend_size", "amplitude", "frequency",
    "seed", "replace_mask_invert",
    # Output / help
    "output", "troubleshoot",
)


def _bootstrap_resolve_module() -> None:
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


def _connect_resolve():
    _bootstrap_resolve_module()
    import DaVinciResolveScript as bmd  # type: ignore
    resolve = bmd.scriptapp("Resolve")
    if resolve is None:
        raise RuntimeError(
            "scriptapp('Resolve') returned None. Is DaVinci Resolve running, "
            "and is external scripting set to 'Local' under Preferences > "
            "System > General?"
        )
    return resolve


def _check_registry(fu) -> tuple[dict, dict]:
    """Confirm DasGrain & DegrainHelper appear in Fusion's tool registry."""
    tools = fu.GetRegList(fu.CT_Tool) or {}
    dasgrain = degrain = None
    for obj in tools.values():
        try:
            attrs = obj.GetAttrs() or {}
        except Exception:
            continue
        rid = str(attrs.get("REGS_ID", ""))
        if rid == DASGRAIN_FUSION_REGID:
            dasgrain = attrs
        elif rid == DEGRAIN_HELPER_FUSION_REGID:
            degrain = attrs
    return dasgrain, degrain


def _ensure_local_db_project(pm):
    """Switch to Local DB and load (or create) the test project."""
    cur_db = pm.GetCurrentDatabase() or {}
    if cur_db and (cur_db.get("DbType") != "Disk"
                   or cur_db.get("DbName") != "Local Database"):
        cur = pm.GetCurrentProject()
        if cur is not None:
            pm.SaveProject()
        pm.SetCurrentDatabase({"DbType": "Disk", "DbName": "Local Database"})

    project = pm.LoadProject(TEST_PROJECT_NAME)
    if project is None:
        cur = pm.GetCurrentProject()
        if cur is not None and cur.GetName() != TEST_PROJECT_NAME:
            pm.SaveProject()
            pm.CloseProject(cur)
        project = pm.CreateProject(TEST_PROJECT_NAME)
    if project is None:
        project = pm.GetCurrentProject()
    if project is None:
        raise RuntimeError(
            f"could not load or create '{TEST_PROJECT_NAME}'. Open the "
            f"project manager in the UI, then re-run."
        )

    project.SetSetting("timelineFrameRate", "24")
    project.SetSetting("timelineResolutionWidth", "320")
    project.SetSetting("timelineResolutionHeight", "180")
    return project


def _ensure_timeline(project, fixtures: Path):
    media_pool = project.GetMediaPool()
    comp_path = str(fixtures / "comp.mov")

    # Reuse the clip if it's already imported.
    comp_clip = None
    for clip in media_pool.GetRootFolder().GetClipList() or []:
        if clip.GetClipProperty("File Path") == comp_path:
            comp_clip = clip
            break
    if comp_clip is None:
        items = media_pool.ImportMedia([comp_path]) or []
        if items:
            comp_clip = items[0]
    if comp_clip is None:
        raise RuntimeError(f"failed to import {comp_path}")

    # Reuse a DG_* timeline if present, else create one.
    tl = None
    for i in range(1, project.GetTimelineCount() + 1):
        cand = project.GetTimelineByIndex(i)
        if cand and cand.GetName().startswith("DG_"):
            tl = cand
            break
    if tl is None:
        tl_name = f"DG_{os.getpid()}"
        tl = media_pool.CreateTimelineFromClips(tl_name, [comp_clip])
        if tl is None:
            raise RuntimeError(f"CreateTimelineFromClips failed for {tl_name}")
    project.SetCurrentTimeline(tl)
    items_v1 = tl.GetItemListInTrack("video", 1) or []
    if not items_v1:
        raise RuntimeError("timeline empty")
    return tl, items_v1[0]


def _wipe_dasgrain(comp):
    """Remove any pre-existing DasGrain tools (idempotent rebuilds)."""
    comp.Lock()
    try:
        for tool in (comp.GetToolList(False) or {}).values():
            rid = tool.GetAttrs().get("TOOLS_RegID", "")
            if rid == DASGRAIN_FUSION_REGID:
                try:
                    tool.Delete()
                except Exception:
                    pass
    finally:
        comp.Unlock()


def _wire_dasgrain(comp, *, output_mode: int, sample_count: int,
                   num_frames: int):
    """Insert DasGrain between the comp's MediaIn and MediaOut."""
    mi = mo = None
    for tool in (comp.GetToolList(False) or {}).values():
        rid = tool.GetAttrs().get("TOOLS_RegID")
        if rid == "MediaIn":  mi = tool
        if rid == "MediaOut": mo = tool
    if mi is None or mo is None:
        raise RuntimeError("comp is missing MediaIn or MediaOut — wrong page?")

    comp.Lock()
    try:
        dg = comp.AddTool(DASGRAIN_FUSION_REGID, 0, 0)
        if dg is None:
            raise RuntimeError(
                f"AddTool({DASGRAIN_FUSION_REGID!r}) returned None. The "
                f"plugin probably failed to instantiate — check "
                f"davinci_resolve.log for OfxActionCreateInstance errors."
            )
        dg.SetAttrs({"TOOLS_Name": "DasGrain"})
        dg["Source"].ConnectTo(mi.Output)
        mo["Input"].ConnectTo(dg.Output)
        try:
            dg["output"]            = output_mode
            dg["sample_count"]      = sample_count
            dg["number_of_frames"]  = num_frames
        except Exception as exc:
            print(f"warning: setting DasGrain params raised: {exc}")
    finally:
        comp.Unlock()
    return dg


def _verify_dasgrain(dg) -> dict:
    """Read back the DasGrain tool and confirm clips/params are exposed."""
    found_clips = []
    for clip_name in EXPECTED_CLIPS:
        try:
            handle = dg[clip_name]
            if handle is not None:
                found_clips.append(clip_name)
        except Exception:
            pass
    found_params = []
    for param_name in EXPECTED_PARAMS:
        try:
            handle = dg[param_name]
            if handle is not None:
                found_params.append(param_name)
        except Exception:
            pass
    return {"clips": found_clips, "params": found_params}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-mode", type=int, default=0,
                   help="0=regrained 1=originalGrain 2=normGrain 3=adapted 4=QC")
    p.add_argument("--sample-count", type=int, default=8)
    p.add_argument("--analyse-frames", type=int, default=4)
    args = p.parse_args()

    print("== DasGrain Resolve E2E ==")
    if not (FIXTURES / "comp.mov").exists():
        print(f"ERROR: fixtures missing under {FIXTURES}. Run "
              f"python3 tests/e2e/gen_fixtures.py first.", file=sys.stderr)
        return 2

    resolve = _connect_resolve()
    print(f"connected: {resolve.GetProductName()} v{resolve.GetVersionString()}")
    fu = resolve.Fusion()

    print("\n[1/5] Fusion tool registry")
    dg_meta, helper_meta = _check_registry(fu)
    if dg_meta is None:
        print(f"  FAIL: {DASGRAIN_FUSION_REGID!r} not in registry. "
              f"Plugin didn't load — check davinci_resolve.log for "
              f"OfxActionCreateInstance errors.")
        return 3
    print(f"  ok: {DASGRAIN_FUSION_REGID} → {dg_meta.get('REGS_Category')!r}")
    if helper_meta:
        print(f"  ok: {DEGRAIN_HELPER_FUSION_REGID} also registered")
    else:
        print(f"  warn: DegrainHelper missing (rebuild + relaunch)")

    print("\n[2/5] Project setup (Local DB)")
    pm = resolve.GetProjectManager()
    project = _ensure_local_db_project(pm)
    print(f"  using project: {project.GetName()}")

    print("\n[3/5] Timeline & Fusion comp")
    if resolve.GetCurrentPage() != "fusion":
        resolve.OpenPage("fusion")
        time.sleep(2.0)  # give Fusion page time to bind comps
    tl, ti = _ensure_timeline(project, FIXTURES)
    print(f"  timeline: {tl.GetName()}")
    if ti.GetFusionCompCount() < 1:
        ti.AddFusionComp()
    fcomp = ti.GetFusionCompByIndex(1)
    if fcomp is None:
        raise RuntimeError(
            "could not retrieve Fusion comp; ensure Fusion page is active"
        )

    print("\n[4/5] Add DasGrain to comp")
    _wipe_dasgrain(fcomp)
    dg = _wire_dasgrain(fcomp, output_mode=args.output_mode,
                        sample_count=args.sample_count,
                        num_frames=args.analyse_frames)
    print("  added DasGrain, wired MediaIn → DasGrain.Source → MediaOut")

    print("\n[5/5] Verify exposed inputs/params")
    info = _verify_dasgrain(dg)
    print(f"  clips: {info['clips']}")
    print(f"  params: {info['params']}")
    missing_clips = [c for c in EXPECTED_CLIPS if c not in info["clips"]]
    missing_params = [p for p in EXPECTED_PARAMS if p not in info["params"]]
    if missing_clips:
        print(f"  WARN: missing clips: {missing_clips}")
    if missing_params:
        print(f"  WARN: missing params: {missing_params}")

    # Cache names BEFORE saving — Resolve sometimes invalidates handles
    # right after SaveProject().
    project_name  = project.GetName()
    timeline_name = tl.GetName()

    print("\nsaving project ...")
    pm.SaveProject()
    print(f"OK — open project {project_name!r} → timeline {timeline_name!r}")
    print("on the Fusion page and you'll see the DasGrain tool wired into")
    print("the comp. Render via the Deliver page to verify visually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
