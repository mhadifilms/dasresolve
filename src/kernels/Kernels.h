// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Cross-backend kernel surface used by the OFX render entry point. Each
// backend implements the symbols declared here for its compute API.
//
// Phase 1 ships only the CPU backend (and a no-op host glue); Phases 2/4/5
// add the actual GrainApply, Voronoi and Analyse kernels.

#pragma once

#include <cstdint>

namespace dasgrain {
namespace kernels {

// Pixel format we operate on inside the kernels. OFX hands us interleaved
// float RGBA planes; that's the format every kernel signature assumes.
struct ImageView {
    float*  data    = nullptr;       // RGBA float32, row-major, interleaved
    int32_t width   = 0;
    int32_t height  = 0;
    int32_t stride  = 0;             // bytes between consecutive rows
};

struct ImageViewConst {
    const float* data   = nullptr;
    int32_t      width  = 0;
    int32_t      height = 0;
    int32_t      stride = 0;
};

// Per-channel response curve, sampled into a flat 1-D LUT of `size` floats
// per channel. Layout: [R0..R_{n-1}, G0..G_{n-1}, B0..B_{n-1}].
//
// Sampling is in linear space — `x` is mapped to the LUT index via
// `t = clamp((x - minX) / (maxX - minX), 0, 1)`.
struct CurveLUT {
    const float* data = nullptr;     // 3 * size floats
    int32_t      size = 0;
    float        minX = 0.0f;
    float        maxX = 1.0f;
};

// Numerical stability epsilon used when normalising grain by the response
// curve. Matches the `target = 0.01` constant in the original Nuke gizmo's
// Expression node.
inline constexpr float kCurveEpsilon = 0.01f;

// Parameters for the fused grain-apply kernel.
struct GrainApplyParams {
    float    luminance         = 1.0f;   // luminance compensation factor (0..1)
    int32_t  fixGhosting       = 0;      // bool
    int32_t  externalGrain     = 0;      // bool: use ExternalGrain input as the normalised grain
    int32_t  outputMode        = 0;      // OutputMode enum (Params.h)
    int32_t  hasMask           = 0;
    int32_t  invertMask        = 0;
    CurveLUT curve;
};

// CPU entry. Defined in src/kernels/cpu/GrainApplyCPU.cpp.
//
// All views are interleaved RGBA float32. Optional inputs (`mask`,
// `externalGrain`) may have `data == nullptr`.
//
// The kernel only writes pixels inside the destination's bounds; it does not
// touch dst outside that.
void runGrainApplyCPU(const GrainApplyParams& p,
                      ImageViewConst comp,
                      ImageViewConst plate,
                      ImageViewConst degrained,
                      ImageViewConst mask,
                      ImageViewConst externalGrain,
                      ImageView dst);

#if DASGRAIN_HAS_METAL
void runGrainApplyMetal(void* metalCmdQueue,
                        const GrainApplyParams& p,
                        ImageViewConst comp,
                        ImageViewConst plate,
                        ImageViewConst degrained,
                        ImageViewConst mask,
                        ImageViewConst externalGrain,
                        ImageView dst);
#endif

#if DASGRAIN_HAS_CUDA
void runGrainApplyCuda(void* cudaStream,
                       const GrainApplyParams& p,
                       ImageViewConst comp,
                       ImageViewConst plate,
                       ImageViewConst degrained,
                       ImageViewConst mask,
                       ImageViewConst externalGrain,
                       ImageView dst);
#endif

#if DASGRAIN_HAS_OPENCL
void runGrainApplyOpenCL(void* clCmdQueue,
                         bool isImage,
                         const GrainApplyParams& p,
                         ImageViewConst comp,
                         ImageViewConst plate,
                         ImageViewConst degrained,
                         ImageViewConst mask,
                         ImageViewConst externalGrain,
                         ImageView dst);
#endif

// ---------------------------------------------------------------------------
// Voronoi scatter
// ---------------------------------------------------------------------------
//
// Replicates the gizmo's BlinkScript-driven scatter: per-output-pixel, a
// Voronoi cell centre UV is computed in normalised space, wrapped into the
// user-defined `box`, optionally distorted by a 2D noise field, and used as
// the sample point in `plateSample` / `degSample`. The result is the
// **normalised grain** (matches the input expected by the externalGrain path
// of GrainApply).
//
// `plateSample` and `degSample` are FULL-frame views of the plate/degrained
// clips at `sample_frame` (FrameHold equivalent). They MUST share the same
// width/height; the OFX layer copies them into flat buffers before calling.
// The kernel uses bilinear sampling and clamps out-of-bounds reads to edge.
//
// Output `dst` is in the renderWindow coordinate system; `dstOffsetX/Y`
// translate it to the same pixel-domain as the sample images (i.e. the
// renderWindow's lower-left corner relative to the sample images' origin).

struct ScatterParams {
    // Sample box in PIXEL coordinates relative to the sample image's
    // origin. Matches the gizmo's `box {x y r t}` knob exactly.
    float boxX       = 0.0f;     // left
    float boxY       = 0.0f;     // bottom
    float boxR       = 1.0f;     // right (exclusive in pixels)
    float boxT       = 1.0f;     // top   (exclusive in pixels)

    // Voronoi cell parameters. `cellSize` is expressed in pixels (matches
    // the gizmo's `cell_size` knob); we convert to a frequency internally.
    float cellSize   = 32.0f;
    int32_t seed     = 0;
    int32_t overlayPattern = 0;     // 1 == debug colour pattern

    // UV distortion (pre-STMap noise). `distortAmplitude` is in PIXELS in
    // the sample image (matches the gizmo's IDistort.uv_scale = amplitude).
    // `distortFrequency` is the Noise1/Noise2 size knob in pixels per cycle.
    float distortAmplitude = 0.0f;
    float distortFrequency = 1.0f;

    // Sub-pixel edge blend (TimeBlur trick): N => (N+1)^2 samples averaged.
    // 0 => single sample.
    int32_t edgeBlendSize = 0;

    // Translation of dst's (0,0) pixel into the sample-image coordinate
    // space. Lets us render a sub-region without copying the full sample
    // images into a renderWindow-sized buffer.
    int32_t dstOffsetX = 0;
    int32_t dstOffsetY = 0;

    CurveLUT curve;
};

void runScatterCPU(const ScatterParams& p,
                   ImageViewConst plateSample,
                   ImageViewConst degSample,
                   ImageView      dst);

#if DASGRAIN_HAS_METAL
void runScatterMetal(void* metalCmdQueue,
                     const ScatterParams& p,
                     ImageViewConst plateSample,
                     ImageViewConst degSample,
                     ImageView      dst);
#endif

#if DASGRAIN_HAS_CUDA
void runScatterCuda(void* cudaStream,
                    const ScatterParams& p,
                    ImageViewConst plateSample,
                    ImageViewConst degSample,
                    ImageView      dst);
#endif

#if DASGRAIN_HAS_OPENCL
void runScatterOpenCL(void* clCmdQueue,
                      const ScatterParams& p,
                      ImageViewConst plateSample,
                      ImageViewConst degSample,
                      ImageView      dst);
#endif

}  // namespace kernels
}  // namespace dasgrain
