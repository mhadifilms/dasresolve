// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// OpenCL grain-apply kernel. Buffers backend only for now (the simpler one
// to ship and the one Resolve uses by default for OFX). Mirrors the CPU
// reference pixel-for-pixel.

#include "../Kernels.h"

#if DASGRAIN_HAS_OPENCL

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace dasgrain {
namespace kernels {

namespace {

const char* const kSrc = R"CL(
#define EPS 0.01f

inline float sampleCurve(__global const float* curve,
                         int   curveSize,
                         int   ch,
                         float minX,
                         float maxX,
                         float x) {
    if (curveSize <= 0) return x;
    float t = clamp((x - minX) / (maxX - minX), 0.0f, 1.0f);
    float fpos = t * (float)(curveSize - 1);
    int   i0   = (int)floor(fpos);
    int   i1   = min(i0 + 1, curveSize - 1);
    float fr   = fpos - (float)i0;
    int   off  = ch * curveSize;
    return curve[off + i0] * (1.0f - fr) + curve[off + i1] * fr;
}

inline float luma709(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// GPU copy of the CPU tone-region shaping. Keep behavior matched with
// GrainApplyCPU.cpp so Resolve can switch backends without changing output.
inline float toneGain(float shadowGrain,
                      float midtoneGrain,
                      float highlightGrain,
                      float curveContrast,
                      float curvePivot,
                      float C0,
                      float C1,
                      float C2) {
    float pivot = clamp(curvePivot, 0.001f, 0.999f);
    float y = clamp(luma709(C0, C1, C2), 0.0f, 1.0f);
    float shadow = clamp((pivot - y) / pivot, 0.0f, 1.0f);
    float highlight = clamp((y - pivot) / (1.0f - pivot), 0.0f, 1.0f);
    float midtone = fmax(0.0f, 1.0f - fmax(shadow, highlight));
    float shaped = shadowGrain * shadow
                 + midtoneGrain * midtone
                 + highlightGrain * highlight;
    return fmax(0.0f, 1.0f + (shaped - 1.0f) * fmax(curveContrast, 0.0f));
}

__kernel void grainApply(__global const float* comp,
                         __global const float* plate,
                         __global const float* degr,
                         __global const float* mask,
                         __global const float* ext,
                         __global       float* dst,
                         __global const float* curve,
                         float luminance,
                         float grainAmount,
                         float shadowGrain,
                         float midtoneGrain,
                         float highlightGrain,
                         float curveContrast,
                         float curvePivot,
                         float redGrain,
                         float greenGrain,
                         float blueGrain,
                         int   fixGhosting,
                         int   externalGrain,
                         int   outputMode,
                         int   hasMask,
                         int   invertMask,
                         int   curveSize,
                         float minX,
                         float maxX,
                         int   width,
                         int   height,
                         int   compStrideF,
                         int   plateStrideF,
                         int   degrStrideF,
                         int   maskStrideF,
                         int   extStrideF,
                         int   dstStrideF,
                         int   hasComp,
                         int   hasPlate,
                         int   hasDegr,
                         int   hasMaskBuf,
                         int   hasExt) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float C[3] = {0,0,0}, P[3] = {0,0,0}, D[3] = {0,0,0};
    float A = 1.0f;
    if (hasComp)  { __global const float* px = comp  + y * compStrideF  + x * 4;
        C[0]=px[0]; C[1]=px[1]; C[2]=px[2]; A=px[3]; }
    if (hasPlate) { __global const float* px = plate + y * plateStrideF + x * 4;
        P[0]=px[0]; P[1]=px[1]; P[2]=px[2]; }
    if (hasDegr)  { __global const float* px = degr  + y * degrStrideF  + x * 4;
        D[0]=px[0]; D[1]=px[1]; D[2]=px[2]; }

    float gO[3] = {P[0]-D[0], P[1]-D[1], P[2]-D[2]};
    float lf = (fixGhosting != 0 || luminance >= 1.0f || luminance <= 0.0f)
               ? 0.0f : (1.0f / luminance - 1.0f);
    float yG  = luma709(gO[0], gO[1], gO[2]);
    float lyG = yG * lf;
    float gLc[3]   = {gO[0]+lyG, gO[1]+lyG, gO[2]+lyG};
    float clean[3] = {D[0]-lyG,  D[1]-lyG,  D[2]-lyG};

    float cClean[3] = {
        sampleCurve(curve, curveSize, 0, minX, maxX, clean[0]),
        sampleCurve(curve, curveSize, 1, minX, maxX, clean[1]),
        sampleCurve(curve, curveSize, 2, minX, maxX, clean[2]),
    };
    float gN[3] = {
        gLc[0] * EPS / fmax(cClean[0], EPS),
        gLc[1] * EPS / fmax(cClean[1], EPS),
        gLc[2] * EPS / fmax(cClean[2], EPS),
    };

    float gS[3];
    if (externalGrain && hasExt) {
        __global const float* px = ext + y * extStrideF + x * 4;
        gS[0]=px[0]; gS[1]=px[1]; gS[2]=px[2];
    } else { gS[0]=gN[0]; gS[1]=gN[1]; gS[2]=gN[2]; }

    float cC[3] = {
        sampleCurve(curve, curveSize, 0, minX, maxX, C[0]),
        sampleCurve(curve, curveSize, 1, minX, maxX, C[1]),
        sampleCurve(curve, curveSize, 2, minX, maxX, C[2]),
    };
    float tone = toneGain(shadowGrain, midtoneGrain, highlightGrain,
                          curveContrast, curvePivot, C[0], C[1], C[2]);
    float liveGain = fmax(grainAmount, 0.0f);
    float channelGain[3] = {
        fmax(redGrain, 0.0f),
        fmax(greenGrain, 0.0f),
        fmax(blueGrain, 0.0f),
    };
    float gA[3] = {
        gS[0] * cC[0] / EPS * tone * channelGain[0] * liveGain,
        gS[1] * cC[1] / EPS * tone * channelGain[1] * liveGain,
        gS[2] * cC[2] / EPS * tone * channelGain[2] * liveGain,
    };

    float m = 1.0f;
    if (hasMask && hasMaskBuf) {
        __global const float* px = mask + y * maskStrideF + x * 4;
        m = clamp(px[3], 0.0f, 1.0f);
        if (invertMask) m = 1.0f - m;
    }

    float r[3] = {
        C[0]+gA[0]*m,
        C[1]+gA[1]*m,
        C[2]+gA[2]*m,
    };
    float o[3];
    if      (outputMode == 1) { o[0]=gO[0]; o[1]=gO[1]; o[2]=gO[2]; }
    else if (outputMode == 2) { o[0]=gN[0]; o[1]=gN[1]; o[2]=gN[2]; }
    else if (outputMode == 3) { o[0]=gA[0]; o[1]=gA[1]; o[2]=gA[2]; }
    else                      { o[0]=r[0];  o[1]=r[1];  o[2]=r[2];  }

    __global float* px = dst + y * dstStrideF + x * 4;
    px[0]=o[0]; px[1]=o[1]; px[2]=o[2]; px[3]=A;
}
)CL";

struct CachedKernel {
    cl_program program = nullptr;
    cl_kernel  kernel  = nullptr;
};

std::mutex& cacheMutex() { static std::mutex m; return m; }

CachedKernel& getKernel(cl_command_queue cmdQ) {
    std::lock_guard<std::mutex> lk(cacheMutex());
    static std::map<cl_command_queue, CachedKernel> cache;
    auto it = cache.find(cmdQ);
    if (it != cache.end()) return it->second;
    CachedKernel k;

    cl_context ctx = nullptr;
    cl_device_id dev = nullptr;
    clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);
    clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr);

    cl_int err;
    const char* src = kSrc;
    size_t      len = std::strlen(kSrc);
    k.program = clCreateProgramWithSource(ctx, 1, &src, &len, &err);
    if (err == CL_SUCCESS) {
        err = clBuildProgram(k.program, 1, &dev, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            char buf[8192]{};
            clGetProgramBuildInfo(k.program, dev, CL_PROGRAM_BUILD_LOG,
                                  sizeof(buf), buf, nullptr);
            std::fprintf(stderr, "DasGrain OpenCL build failed: %s\n", buf);
        }
    }
    if (k.program) {
        k.kernel = clCreateKernel(k.program, "grainApply", &err);
    }
    cache[cmdQ] = k;
    return cache[cmdQ];
}

}  // namespace

void runGrainApplyOpenCL(void* clCmdQueue,
                         bool /*isImage*/,
                         const GrainApplyParams& p,
                         ImageViewConst comp,
                         ImageViewConst plate,
                         ImageViewConst degrained,
                         ImageViewConst mask,
                         ImageViewConst externalGrain,
                         ImageView dst) {
    if (!clCmdQueue || !dst.data) {
        runGrainApplyCPU(p, comp, plate, degrained, mask, externalGrain, dst);
        return;
    }
    cl_command_queue cmdQ = static_cast<cl_command_queue>(clCmdQueue);
    auto& k = getKernel(cmdQ);
    if (!k.kernel) {
        runGrainApplyCPU(p, comp, plate, degrained, mask, externalGrain, dst);
        return;
    }

    cl_context ctx = nullptr;
    clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);

    cl_int err;
    cl_mem curveBuf = nullptr;
    if (p.curve.data && p.curve.size > 0) {
        curveBuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  size_t(p.curve.size) * 3 * sizeof(float),
                                  const_cast<float*>(p.curve.data), &err);
    }

    auto sf = [](int strideBytes) { return strideBytes / int(sizeof(float)); };
    int compStrideF  = sf(comp.stride);
    int plateStrideF = sf(plate.stride);
    int degrStrideF  = sf(degrained.stride);
    int maskStrideF  = sf(mask.stride);
    int extStrideF   = sf(externalGrain.stride);
    int dstStrideF   = sf(dst.stride);

    int hasComp  = (comp.data       != nullptr) ? 1 : 0;
    int hasPlate = (plate.data      != nullptr) ? 1 : 0;
    int hasDegr  = (degrained.data  != nullptr) ? 1 : 0;
    int hasMaskB = (mask.data       != nullptr) ? 1 : 0;
    int hasExt   = (externalGrain.data != nullptr) ? 1 : 0;

    int idx = 0;
    err  = clSetKernelArg(k.kernel, idx++, sizeof(cl_mem), &comp.data);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(cl_mem), &plate.data);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(cl_mem), &degrained.data);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(cl_mem), &mask.data);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(cl_mem), &externalGrain.data);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(cl_mem), &dst.data);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(cl_mem), &curveBuf);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(float), &p.luminance);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(float), &p.grainAmount);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(float), &p.shadowGrain);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(float), &p.midtoneGrain);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(float), &p.highlightGrain);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(float), &p.curveContrast);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(float), &p.curvePivot);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(float), &p.redGrain);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(float), &p.greenGrain);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(float), &p.blueGrain);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &p.fixGhosting);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &p.externalGrain);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &p.outputMode);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &p.hasMask);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &p.invertMask);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &p.curve.size);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(float), &p.curve.minX);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(float), &p.curve.maxX);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &dst.width);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &dst.height);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &compStrideF);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &plateStrideF);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &degrStrideF);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &maskStrideF);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &extStrideF);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &dstStrideF);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &hasComp);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &hasPlate);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &hasDegr);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &hasMaskB);
    err |= clSetKernelArg(k.kernel, idx++, sizeof(int),   &hasExt);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "DasGrain OpenCL setKernelArg err=%d\n", err);
        if (curveBuf) clReleaseMemObject(curveBuf);
        runGrainApplyCPU(p, comp, plate, degrained, mask, externalGrain, dst);
        return;
    }

    size_t global[2] = { static_cast<size_t>(dst.width),
                         static_cast<size_t>(dst.height) };
    err = clEnqueueNDRangeKernel(cmdQ, k.kernel, 2, nullptr, global,
                                 nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "DasGrain OpenCL enqueue err=%d\n", err);
    }

    if (curveBuf) clReleaseMemObject(curveBuf);
}

}  // namespace kernels
}  // namespace dasgrain

#endif  // DASGRAIN_HAS_OPENCL
