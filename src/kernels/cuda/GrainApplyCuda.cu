// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// CUDA grain-apply kernel. Mirrors the CPU reference and Metal/OpenCL paths.

#include "../Kernels.h"

#if DASGRAIN_HAS_CUDA

#include <cuda_runtime.h>

#include <cstdio>

namespace dasgrain {
namespace kernels {

namespace {

constexpr float kEps = 0.01f;

__device__ inline float sampleCurve(const float* curve,
                                    int   curveSize,
                                    int   ch,
                                    float minX,
                                    float maxX,
                                    float x) {
    if (curveSize <= 0) return x;
    float t = fmaxf(0.0f, fminf((x - minX) / (maxX - minX), 1.0f));
    float fpos = t * float(curveSize - 1);
    int   i0   = int(floorf(fpos));
    int   i1   = min(i0 + 1, curveSize - 1);
    float fr   = fpos - float(i0);
    int   off  = ch * curveSize;
    return curve[off + i0] * (1.0f - fr) + curve[off + i1] * fr;
}

__device__ inline float luma709(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

struct GrainParamsGPU;

__device__ inline float toneGain(const GrainParamsGPU& gp, const float C[3]);

struct GrainParamsGPU {
    float luminance;
    float grainAmount;
    float shadowGrain;
    float midtoneGrain;
    float highlightGrain;
    float curveContrast;
    float curvePivot;
    float redGrain;
    float greenGrain;
    float blueGrain;
    int   fixGhosting;
    int   externalGrain;
    int   outputMode;
    int   hasMask;
    int   invertMask;
    int   curveSize;
    float minX;
    float maxX;
    int   width;
    int   height;
    int   compStrideF;
    int   plateStrideF;
    int   degrStrideF;
    int   maskStrideF;
    int   extStrideF;
    int   dstStrideF;
    int   hasComp;
    int   hasPlate;
    int   hasDegr;
    int   hasMaskBuf;
    int   hasExt;
};

// GPU copy of the CPU tone-region shaping. Keep behavior matched with
// GrainApplyCPU.cpp so Resolve can switch backends without changing output.
__device__ inline float toneGain(const GrainParamsGPU& gp, const float C[3]) {
    float pivot = fmaxf(0.001f, fminf(gp.curvePivot, 0.999f));
    float y = fmaxf(0.0f, fminf(luma709(C[0], C[1], C[2]), 1.0f));
    float shadow = fmaxf(0.0f, fminf((pivot - y) / pivot, 1.0f));
    float highlight = fmaxf(0.0f, fminf((y - pivot) / (1.0f - pivot), 1.0f));
    float midtone = fmaxf(0.0f, 1.0f - fmaxf(shadow, highlight));
    float shaped = gp.shadowGrain * shadow
                 + gp.midtoneGrain * midtone
                 + gp.highlightGrain * highlight;
    return fmaxf(0.0f, 1.0f + (shaped - 1.0f) * fmaxf(gp.curveContrast, 0.0f));
}

__global__ void grainApplyKernel(const float* __restrict__ comp,
                                 const float* __restrict__ plate,
                                 const float* __restrict__ degr,
                                 const float* __restrict__ mask,
                                 const float* __restrict__ ext,
                                 float*       __restrict__ dst,
                                 const float* __restrict__ curve,
                                 GrainParamsGPU gp) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= gp.width || y >= gp.height) return;

    float C[3] = {0,0,0}, P[3] = {0,0,0}, D[3] = {0,0,0};
    float A = 1.0f;
    if (gp.hasComp) {
        const float* px = comp + y * gp.compStrideF + x * 4;
        C[0] = px[0]; C[1] = px[1]; C[2] = px[2]; A = px[3];
    }
    if (gp.hasPlate) {
        const float* px = plate + y * gp.plateStrideF + x * 4;
        P[0] = px[0]; P[1] = px[1]; P[2] = px[2];
    }
    if (gp.hasDegr) {
        const float* px = degr + y * gp.degrStrideF + x * 4;
        D[0] = px[0]; D[1] = px[1]; D[2] = px[2];
    }

    float gO[3] = {P[0]-D[0], P[1]-D[1], P[2]-D[2]};
    float lf = (gp.fixGhosting != 0 || gp.luminance >= 1.0f || gp.luminance <= 0.0f)
               ? 0.0f : (1.0f / gp.luminance - 1.0f);
    float yG  = luma709(gO[0], gO[1], gO[2]);
    float lyG = yG * lf;
    float gLc[3]   = {gO[0]+lyG, gO[1]+lyG, gO[2]+lyG};
    float clean[3] = {D[0]-lyG,  D[1]-lyG,  D[2]-lyG};

    float c0 = sampleCurve(curve, gp.curveSize, 0, gp.minX, gp.maxX, clean[0]);
    float c1 = sampleCurve(curve, gp.curveSize, 1, gp.minX, gp.maxX, clean[1]);
    float c2 = sampleCurve(curve, gp.curveSize, 2, gp.minX, gp.maxX, clean[2]);
    float gN[3] = {
        gLc[0] * kEps / fmaxf(c0, kEps),
        gLc[1] * kEps / fmaxf(c1, kEps),
        gLc[2] * kEps / fmaxf(c2, kEps),
    };

    float gS[3];
    if (gp.externalGrain && gp.hasExt) {
        const float* px = ext + y * gp.extStrideF + x * 4;
        gS[0] = px[0]; gS[1] = px[1]; gS[2] = px[2];
    } else { gS[0] = gN[0]; gS[1] = gN[1]; gS[2] = gN[2]; }

    float cC0 = sampleCurve(curve, gp.curveSize, 0, gp.minX, gp.maxX, C[0]);
    float cC1 = sampleCurve(curve, gp.curveSize, 1, gp.minX, gp.maxX, C[1]);
    float cC2 = sampleCurve(curve, gp.curveSize, 2, gp.minX, gp.maxX, C[2]);
    float tone = toneGain(gp, C);
    float liveGain = fmaxf(gp.grainAmount, 0.0f);
    float channelGain[3] = {
        fmaxf(gp.redGrain, 0.0f),
        fmaxf(gp.greenGrain, 0.0f),
        fmaxf(gp.blueGrain, 0.0f),
    };
    float gA[3] = {
        gS[0] * cC0 / kEps * tone * channelGain[0] * liveGain,
        gS[1] * cC1 / kEps * tone * channelGain[1] * liveGain,
        gS[2] * cC2 / kEps * tone * channelGain[2] * liveGain,
    };

    float m = 1.0f;
    if (gp.hasMask && gp.hasMaskBuf) {
        const float* px = mask + y * gp.maskStrideF + x * 4;
        m = fmaxf(0.0f, fminf(px[3], 1.0f));
        if (gp.invertMask) m = 1.0f - m;
    }

    float r[3] = {
        C[0]+gA[0]*m,
        C[1]+gA[1]*m,
        C[2]+gA[2]*m,
    };
    float o[3];
    if      (gp.outputMode == 1) { o[0]=gO[0]; o[1]=gO[1]; o[2]=gO[2]; }
    else if (gp.outputMode == 2) { o[0]=gN[0]; o[1]=gN[1]; o[2]=gN[2]; }
    else if (gp.outputMode == 3) { o[0]=gA[0]; o[1]=gA[1]; o[2]=gA[2]; }
    else                          { o[0]=r[0];  o[1]=r[1];  o[2]=r[2];  }

    float* px = dst + y * gp.dstStrideF + x * 4;
    px[0] = o[0]; px[1] = o[1]; px[2] = o[2]; px[3] = A;
}

}  // namespace

void runGrainApplyCuda(void* cudaStream,
                       const GrainApplyParams& p,
                       ImageViewConst comp,
                       ImageViewConst plate,
                       ImageViewConst degrained,
                       ImageViewConst mask,
                       ImageViewConst externalGrain,
                       ImageView dst) {
    if (!dst.data) {
        runGrainApplyCPU(p, comp, plate, degrained, mask, externalGrain, dst);
        return;
    }
    cudaStream_t stream = static_cast<cudaStream_t>(cudaStream);

    // Curve buffer.
    float* dCurve = nullptr;
    const size_t curveBytes = size_t(p.curve.size) * 3 * sizeof(float);
    if (p.curve.data && p.curve.size > 0) {
        cudaMalloc(reinterpret_cast<void**>(&dCurve), curveBytes);
        cudaMemcpyAsync(dCurve, p.curve.data, curveBytes,
                        cudaMemcpyHostToDevice, stream);
    }

    auto sf = [](int strideBytes) { return strideBytes / int(sizeof(float)); };
    GrainParamsGPU gp{};
    gp.luminance     = p.luminance;
    gp.grainAmount   = p.grainAmount;
    gp.shadowGrain   = p.shadowGrain;
    gp.midtoneGrain  = p.midtoneGrain;
    gp.highlightGrain = p.highlightGrain;
    gp.curveContrast = p.curveContrast;
    gp.curvePivot    = p.curvePivot;
    gp.redGrain      = p.redGrain;
    gp.greenGrain    = p.greenGrain;
    gp.blueGrain     = p.blueGrain;
    gp.fixGhosting   = p.fixGhosting;
    gp.externalGrain = p.externalGrain;
    gp.outputMode    = p.outputMode;
    gp.hasMask       = p.hasMask;
    gp.invertMask    = p.invertMask;
    gp.curveSize     = p.curve.size;
    gp.minX          = p.curve.minX;
    gp.maxX          = p.curve.maxX;
    gp.width         = dst.width;
    gp.height        = dst.height;
    gp.compStrideF   = sf(comp.stride);
    gp.plateStrideF  = sf(plate.stride);
    gp.degrStrideF   = sf(degrained.stride);
    gp.maskStrideF   = sf(mask.stride);
    gp.extStrideF    = sf(externalGrain.stride);
    gp.dstStrideF    = sf(dst.stride);
    gp.hasComp       = comp.data       ? 1 : 0;
    gp.hasPlate      = plate.data      ? 1 : 0;
    gp.hasDegr       = degrained.data  ? 1 : 0;
    gp.hasMaskBuf    = mask.data       ? 1 : 0;
    gp.hasExt        = externalGrain.data ? 1 : 0;

    dim3 block(16, 16);
    dim3 grid((dst.width  + block.x - 1) / block.x,
              (dst.height + block.y - 1) / block.y);
    grainApplyKernel<<<grid, block, 0, stream>>>(
        comp.data, plate.data, degrained.data, mask.data, externalGrain.data,
        dst.data, dCurve, gp);

    if (dCurve) {
        // Free after the kernel completes; we don't synchronise here so
        // attach the free to the same stream via cudaFreeAsync if available.
#if CUDART_VERSION >= 11020
        cudaFreeAsync(dCurve, stream);
#else
        cudaStreamSynchronize(stream);
        cudaFree(dCurve);
#endif
    }
}

}  // namespace kernels
}  // namespace dasgrain

#endif  // DASGRAIN_HAS_CUDA
