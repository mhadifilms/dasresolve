// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Thin facade that the OFX render entry calls. Picks the active GPU backend
// from the OFX RenderArguments (CUDA / Metal / OpenCL) and falls back to the
// CPU implementation. Kernel shapes that aren't implemented yet for a given
// backend will route through the CPU path.

#pragma once

#include "Kernels.h"

namespace dasgrain {
namespace kernels {

enum class Backend {
    kCPU,
    kCUDA,
    kMetal,
    kOpenCL,
};

struct GpuContext {
    Backend  backend     = Backend::kCPU;
    void*    cudaStream  = nullptr;     // cudaStream_t
    void*    metalCmdQ   = nullptr;     // id<MTLCommandQueue>
    void*    openCLCmdQ  = nullptr;     // cl_command_queue
    bool     openCLImages = false;
};

// Phase-1 helper: simple passthrough copy of `src` into `dst`. Used as a
// fallback by the OFX render entry when no Plate/Degrained inputs are wired
// up yet (so the timeline doesn't go black).
void runPassThrough(const GpuContext& ctx,
                    ImageViewConst src,
                    ImageView dst);

// Phase-2 entry: dispatches the grain-apply kernel onto the active backend.
// Falls back to the CPU implementation when the active backend doesn't yet
// have a kernel built or available at link time.
void dispatchGrainApply(const GpuContext& ctx,
                        const GrainApplyParams& params,
                        ImageViewConst comp,
                        ImageViewConst plate,
                        ImageViewConst degrained,
                        ImageViewConst mask,
                        ImageViewConst externalGrain,
                        ImageView dst);

// Phase-5 entry: dispatches the Voronoi scatter kernel. Produces the
// normalised grain (matching the externalGrain input expected by
// dispatchGrainApply).
void dispatchScatter(const GpuContext& ctx,
                     const ScatterParams& params,
                     ImageViewConst plateSample,
                     ImageViewConst degSample,
                     ImageView      dst);

}  // namespace kernels
}  // namespace dasgrain
