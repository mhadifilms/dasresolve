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
overlay, Resolve-friendly response shaping controls, optional advanced
response-curve JSON import / export, and on-demand diagnostics via the
Troubleshoot button.

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
4. Tweak the result in *Adjust* with **grain amount**, **shadow grain**,
   **midtone grain**, **highlight grain**, **curve contrast**, **curve
   pivot**, and the RGB grain trims.
5. For shots without a clean grain plate everywhere, enable **scatter** on
   the *Replace* tab, drag the on-viewer box over a clean patch, and tune
   `cell size` / `edge blend` / `distortion`.

The **Troubleshoot** button (Help tab) prints a status report covering
input connectivity, analyse state, curve population, and scatter-box
sanity.

## Detailed Guide

### What DasGrain Does

DasGrain extracts grain from the original plate, learns how that grain changes
with image brightness, and applies matching grain back onto a comp:

1. `original grain = Plate - Degrained`
2. analysis builds an RGB response curve from the plate/degrained pair
3. extracted grain is normalised by that curve
4. normalised grain is adapted to Source luminance and added to Source

The Source input is always the comp that needs regraining. Plate and Degrained
must be the original grainy plate and its denoised version.

### Basic Workflow

1. In Fusion, add `OFX -> DasGrain -> DasGrain`.
2. Connect `Source` to the comp that needs grain.
3. Connect `Plate` to the original grainy plate.
4. Connect `Degrained` to a denoised version of the same original plate.
5. Optionally connect `Mask` to limit where grain is added.
6. On the Analyse tab, set sample settings and click `Analyse`.
7. On the Adjust tab, tune grain amount, tone controls, and RGB trims.
8. Use Output modes or `Troubleshoot` to inspect setup problems.

### Inputs

- `Source`: The comp that needs grain. The regrained result is Source plus the
  adapted grain.
- `Plate`: The original grainy plate.
- `Degrained`: A denoised copy of Plate. This should match Plate geometry and
  timing as closely as possible.
- `Mask`: Optional alpha mask for final grain application and, when requested,
  analysis.
- `External Grain`: Optional normalised grain input. This replaces the inline
  `Plate - Degrained` grain source when `use external grain` is enabled.

If Source is connected but Plate or Degrained is missing, DasGrain passes Source
through instead of rendering black.

### Analyse Tab

- `luminance`: Controls luminance compensation for grain left in the degrained
  plate. `1.0` disables compensation. Lower values increase compensation.
- `fix ghosting`: Disables luminance compensation when it would bake plate
  detail into Source. Use this for retimed, repo'd, lipsync-modified, or
  geometry-changed shots.
- `invert grain level mask`: Inverts the Mask alpha used for final grain
  application.
- `number of frames`: Evenly distributed frames sampled during analysis. More
  frames usually give a more stable response curve.
- `additional frames`: Extra explicit frames or frame ranges. Examples:
  `1001`, `1001,1010,1040`, `1001-1040`, `1001-1120x4`.
- `sample count`: Number of luminance buckets used to fit the RGB response
  curve. Higher values preserve more curve detail but make analysis slower.
- `Analyse`: Runs analysis and stores the resulting response curve internally.
- `invert analysis mask`: Inverts the Mask alpha only during the analysis pass.

### Adjust Tab

These controls are live render controls. They do not require running Analyse
again.

- `grain amount`: Global grain gain. `0` returns Source only, `1` is normal
  DasGrain, and values above `1` exaggerate grain.
- `shadow grain`: Scales adapted grain in dark Source values.
- `midtone grain`: Scales adapted grain around `curve pivot`.
- `highlight grain`: Scales adapted grain in bright Source values.
- `curve contrast`: Strength of the shadow/midtone/highlight shaping. `0`
  ignores tone shaping; `1` is normal.
- `curve pivot`: Source luminance treated as the midtone center.
- `red grain`: Red-channel grain gain.
- `green grain`: Green-channel grain gain.
- `blue grain`: Blue-channel grain gain.
- `What am I looking at?`: Opens a Resolve message explaining the response
  controls.
- `show curve JSON tools`: Enables advanced raw curve import/export controls.
  Leave off for normal work.
- `Curve JSON`: Advanced single-line response-curve JSON field.
- `Import curve`: Parses `Curve JSON` and stores it as the active response
  curve.
- `Export curve`: Copies the internal response curve into `Curve JSON` and a
  message dialog.

Resolve does not support the original Nuke gizmo's interactive parametric
curve widget. The tone and RGB controls are the normal Resolve-facing curve UI;
JSON exists only for debugging or transferring analysed curves between shots.

### Replace Tab

- `use external grain`: Uses the `External Grain` input as normalised grain.
- `activate scatter`: Builds normalised grain from a sampled clean area and
  Voronoi-tiles it across the whole frame.
- `sample box (low)`: Lower-left sample-box coordinate in normalized image
  space.
- `sample box (high)`: Upper-right sample-box coordinate in normalized image
  space.
- `sample frame`: Frame used as the scatter source.
- `overlay cell pattern`: Debug overlay for Voronoi cells.
- `cell size`: Voronoi cell size in pixels.
- `edge blend size`: Blends cell edges to hide repetition. Higher values cost
  more.
- `distortion amplitude`: Pixel displacement amount used to warp scatter
  cells.
- `distortion frequency`: Frequency of the scatter distortion pattern.
- `voronoi seed`: Random seed for the scatter pattern.
- `invert replace mask`: Inverts the Mask alpha for replacement/scatter logic.

Use scatter when the plate/degrained pair has only a small clean region that
can provide usable grain.

### Output Modes

- `regrained comp`: Normal final output.
- `original grain`: Shows `Plate - Degrained`.
- `normalised grain`: Shows extracted grain after response-curve
  normalisation.
- `adapted grain`: Shows grain after adapting it to Source luminance and live
  Adjust controls.
- `grain QC`: High-pass style view for checking whether the final grain sits
  naturally.

### Troubleshooting

Use the `Troubleshoot` button first. It checks required input wiring, format
mismatches, frame-range mismatches, empty response curves, luminance settings,
and scatter-box sanity.

If MediaOut looks like Plate, check whether Source and Degrained are the same
image. DasGrain's math is `Source + (Plate - Degrained)` after adaptation; when
Source equals Degrained, the result naturally trends toward Plate.

If there is ghosting, check whether Source has geometry changes that Plate and
Degrained do not have. For lipsync or face replacement shots, enable
`fix ghosting`, reduce affected tone gains, or use a Mask to exclude modified
regions.

If sliders appear inactive, make sure analysis has been run and Output is set
to `regrained comp` or `adapted grain`. Analyse controls affect the stored
curve only when `Analyse` runs; Adjust controls are live.

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
