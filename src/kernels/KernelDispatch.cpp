// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0

#include "KernelDispatch.h"

#include <algorithm>
#include <cstring>

namespace dasgrain {
namespace kernels {

void runPassThrough(const GpuContext& /*ctx*/,
                    ImageViewConst src,
                    ImageView dst) {
    if (!src.data || !dst.data) return;
    const int w = std::min(src.width,  dst.width);
    const int h = std::min(src.height, dst.height);
    const size_t rowBytes = static_cast<size_t>(w) * 4 * sizeof(float);
    for (int y = 0; y < h; ++y) {
        const auto*  srow = reinterpret_cast<const uint8_t*>(src.data) + y * src.stride;
        auto*        drow = reinterpret_cast<uint8_t*>(dst.data)       + y * dst.stride;
        std::memcpy(drow, srow, rowBytes);
    }
}

void dispatchGrainApply(const GpuContext& ctx,
                        const GrainApplyParams& params,
                        ImageViewConst comp,
                        ImageViewConst plate,
                        ImageViewConst degrained,
                        ImageViewConst mask,
                        ImageViewConst externalGrain,
                        ImageView dst) {
    switch (ctx.backend) {
#if DASGRAIN_HAS_METAL
        case Backend::kMetal:
            runGrainApplyMetal(ctx.metalCmdQ, params,
                               comp, plate, degrained, mask, externalGrain, dst);
            return;
#endif
#if DASGRAIN_HAS_CUDA
        case Backend::kCUDA:
            runGrainApplyCuda(ctx.cudaStream, params,
                              comp, plate, degrained, mask, externalGrain, dst);
            return;
#endif
#if DASGRAIN_HAS_OPENCL
        case Backend::kOpenCL:
            runGrainApplyOpenCL(ctx.openCLCmdQ, ctx.openCLImages, params,
                                comp, plate, degrained, mask, externalGrain, dst);
            return;
#endif
        case Backend::kCPU:
        default:
            break;
    }
    runGrainApplyCPU(params, comp, plate, degrained, mask, externalGrain, dst);
}

void dispatchScatter(const GpuContext& ctx,
                     const ScatterParams& params,
                     ImageViewConst plateSample,
                     ImageViewConst degSample,
                     ImageView      dst) {
    switch (ctx.backend) {
#if DASGRAIN_HAS_METAL
        case Backend::kMetal:
            runScatterMetal(ctx.metalCmdQ, params, plateSample, degSample, dst);
            return;
#endif
#if DASGRAIN_HAS_CUDA
        case Backend::kCUDA:
            runScatterCuda(ctx.cudaStream, params, plateSample, degSample, dst);
            return;
#endif
#if DASGRAIN_HAS_OPENCL
        case Backend::kOpenCL:
            runScatterOpenCL(ctx.openCLCmdQ, params, plateSample, degSample, dst);
            return;
#endif
        case Backend::kCPU:
        default:
            break;
    }
    runScatterCPU(params, plateSample, degSample, dst);
}

}  // namespace kernels
}  // namespace dasgrain
