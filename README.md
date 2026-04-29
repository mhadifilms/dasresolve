# DasGrain OFX

A native OpenFX port of Fabian Holtz's **DasGrain** Nuke gizmo for
[DaVinci Resolve](https://www.blackmagicdesign.com/products/davinciresolve/),
intended to be used on the **Fusion** page.

DasGrain re-introduces grain onto a comp by:

1. extracting the original grain (`Plate` − `Degrained`),
2. analysing how the grain intensity varies with luminance to build a
   per-channel response curve,
3. normalising the extracted grain by that curve and re-multiplying it by
   the curve sampled at the **comp**'s luminance, so the grain follows the
   comp's tones rather than the plate's,
4. optionally Voronoi-scattering a small clean grain area to cover the
   whole frame (for shots where only a tiny clean patch is available).

This implementation follows the DasGrain workflow in a native OFX plugin
with multi-input support, GPU rendering on Metal / CUDA / OpenCL, and a CPU
fallback.

## Status

The plugin loads in Resolve Studio's Fusion page, exposes the expected
multi-input workflow, and supports analyse / apply / scatter rendering on
Metal, CUDA, OpenCL, and CPU backends. It includes a draggable sample-box
overlay, response-curve JSON import / export, and on-demand diagnostics via
the Troubleshoot button.

## Build

```bash
git submodule update --init --recursive
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
sudo cmake --install build      # writes to the OS' OFX plugin dir
```

Standard install paths:

| OS      | Path                                                  |
|---------|-------------------------------------------------------|
| macOS   | `/Library/OFX/Plugins/DasGrain.ofx.bundle`            |
| Linux   | `/usr/OFX/Plugins/DasGrain.ofx.bundle`                |
| Windows | `C:\Program Files\Common Files\OFX\Plugins\DasGrain.ofx.bundle` |

You can also point Resolve at the build tree by symlinking
`build/DasGrain.ofx.bundle` into the same location.

Per-platform helpers:

- **Linux**: `bash packaging/linux/install.sh` (configures, builds, installs).
- **Windows**: build with CMake (Release), then run the Inno Setup compiler
  on `packaging/win/installer.iss` to produce `DasGrain-Installer.exe`.
- **macOS**: the build automatically ad-hoc codesigns the bundle (XProtect
  blocks unsigned universal binaries), then `sudo cmake --install build`
  copies it into `/Library/OFX/Plugins`.

### Backends

- **macOS**: Metal + OpenCL + CPU
- **Linux / Windows**: CUDA + OpenCL + CPU

CMake auto-detects what's available; toggle with
`-DDASGRAIN_ENABLE_METAL=ON/OFF`, `-DDASGRAIN_ENABLE_CUDA=ON/OFF`,
`-DDASGRAIN_ENABLE_OPENCL=ON/OFF`. The Voronoi scatter falls back to CPU
when a particular backend isn't built or available at runtime.

## Usage

1. In a Fusion comp, add an OFX → DasGrain → DasGrain node.
2. Wire the inputs:
   - **Source** ← your composited shot (the comp that needs regraining)
   - **Plate** ← the original grainy plate
   - **Degrained** ← the denoised plate
   - **Mask** (optional) ← alpha mask to limit grain application
   - **External Grain** (optional) ← a previously rendered normalised grain
3. Set **number of frames** (and optionally **additional frames** for
   manually picked sample frames) on the *Analyse* tab and click
   **Analyse**. Sample frames are pulled, the response curve is fit, and
   the curve is written into the *Adjust* tab.
4. Tweak the curve in *Adjust* if needed, or click **Export curve** to
   copy the JSON for archival; **Import curve** applies pasted JSON.
5. For shots without a clean grain plate everywhere, enable **scatter** on
   the *Replace* tab, drag the on-viewer box over a clean patch, and tune
   `cell size` / `edge blend` / `distortion`.

The **Troubleshoot** button (Help tab) prints a status report covering
input connectivity, analyse state, curve population, and scatter-box
sanity.

## Tests

### Unit tests

```bash
cmake --build build --target dasgrain_unit_tests
ctest --test-dir build --output-on-failure
```

Coverage:

- response-curve JSON round-trip
- AlexaV3LogC + Rec.709 luma helpers
- frame-range parser
- voronoi value-noise primitive (overflow-safe, in-range)
- grain-apply CPU kernel: zero-grain, identity curve, every output mode,
  external grain, mask, luma compensation, fix-ghosting
- analyser two-pass pipeline (stats + histogram + curve fit)
- scatter CPU kernel: zero-grain, overlay pattern, degenerate box, no-LUT
- end-to-end golden frames covering the whole pipeline

### End-to-end test in DaVinci Resolve

`tests/e2e/` drives a real Resolve Studio instance through the
scripting API to confirm the bundle loads, registers in Fusion, and
instantiates with all five named clips and ~26 user parameters
correctly exposed. Resolve's embedded Fusion is finicky about
programmatic frame renders (Loader paths sandbox-fail, `comp.Render`
returns success without writing), so the harness deliberately stops
short of triggering a frame render — visual verification happens via
the Deliver page after the harness wires everything up.

Prerequisites:

- DaVinci Resolve **Studio** running (free Resolve does not load OFX).
- *Preferences → System → General → External scripting using → Local* enabled.
- The DasGrain bundle installed under `/Library/OFX/Plugins`.

#### One-time dev setup

`tests/e2e/setup_dev.sh` symlinks `build/DasGrain.ofx.bundle` into
`/Library/OFX/Plugins`, so subsequent rebuilds don't need `sudo` —
just `cmake --build build` and Resolve picks up the new binary on
restart.

```bash
bash tests/e2e/setup_dev.sh    # one-time sudo prompt
```

#### Iterating

```bash
# rebuild + auto-restart Resolve + run harness
bash tests/e2e/devloop.sh
python3 tests/e2e/run_e2e.py
```

Or each stage individually:

```bash
python3 tests/e2e/gen_fixtures.py    # synthetic plate / degrained / comp
python3 tests/e2e/probe_resolve.py   # checks bundle + scripting + cache
python3 tests/e2e/run_e2e.py         # adds + verifies DasGrain in Fusion
```

The harness uses a single persistent Local-DB project
`DasGrainE2E_Persistent` so it never touches your real / cloud
project library.

## Credits

- **DasGrain** concept, workflow, and original Nuke gizmo: Fabian Holtz
  ([Nukepedia](https://www.nukepedia.com/tools/gizmos/other/dasgrain/),
  [gist mirror](https://gist.github.com/rafaelperez/118dd1f0b6d7d2238f4643240b698dfd)).
- Voronoi BlinkScript kernel used by the original scatter workflow: Ivan
  Busquets ([Nukepedia](http://www.nukepedia.com/blink/image/voronoi/)).
- OpenFX SDK: [Academy Software Foundation](https://github.com/AcademySoftwareFoundation/openfx).

## License

DasGrain OFX source is licensed under Apache-2.0. Bundled third-party
components and attribution notices are listed in `NOTICE`. This project is
not affiliated with or endorsed by Blackmagic Design, The Foundry, or the
original DasGrain author.
