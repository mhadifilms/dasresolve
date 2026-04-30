// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Metal grain-apply kernel.
//
// Resolve hands us either CPU pointers or Metal MTLBuffer handles. When
// Metal is "enabled" the data pointers in our ImageView{Const}s are
// id<MTLBuffer> casts; we wrap them, dispatch a compute kernel that
// implements the exact CPU pixel formula, and the host inserts our work
// into the supplied command queue.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <mutex>
#include <string>
#include <unordered_map>

#include "../Kernels.h"

namespace dasgrain {
namespace kernels {

namespace {

// Kernel mirrors src/kernels/cpu/GrainApplyCPU.cpp. Buffer indices line up
// with the slots in the dispatch wrapper below.
const char* const kKernelSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct GrainParams {
    float    luminance;
    float    grainAmount;
    float    shadowGrain;
    float    midtoneGrain;
    float    highlightGrain;
    float    curveContrast;
    float    curvePivot;
    float    redGrain;
    float    greenGrain;
    float    blueGrain;
    int      fixGhosting;
    int      externalGrain;
    int      outputMode;
    int      hasMask;
    int      invertMask;
    int      curveSize;
    float    minX;
    float    maxX;
    int      width;
    int      height;
    int      compStrideF;
    int      plateStrideF;
    int      degrStrideF;
    int      maskStrideF;
    int      extStrideF;
    int      dstStrideF;
    int      hasComp;
    int      hasPlate;
    int      hasDegr;
    int      hasMaskBuf;
    int      hasExt;
};

constant float kEps = 0.01f;

inline float sampleCurve(device const float* curve,
                         int   curveSize,
                         int   ch,
                         float minX,
                         float maxX,
                         float x) {
    if (curveSize <= 0) return x;
    float t = clamp((x - minX) / (maxX - minX), 0.0f, 1.0f);
    float fpos = t * float(curveSize - 1);
    int   i0   = int(floor(fpos));
    int   i1   = min(i0 + 1, curveSize - 1);
    float fr   = fpos - float(i0);
    int   off  = ch * curveSize;
    return curve[off + i0] * (1.0f - fr) + curve[off + i1] * fr;
}

inline float luma709(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// GPU copy of the CPU tone-region shaping. Keep behavior matched with
// GrainApplyCPU.cpp so Resolve can switch backends without changing output.
inline float toneGain(constant GrainParams& gp, float3 C) {
    float pivot = clamp(gp.curvePivot, 0.001f, 0.999f);
    float y = clamp(luma709(C.x, C.y, C.z), 0.0f, 1.0f);
    float shadow = clamp((pivot - y) / pivot, 0.0f, 1.0f);
    float highlight = clamp((y - pivot) / (1.0f - pivot), 0.0f, 1.0f);
    float midtone = max(0.0f, 1.0f - max(shadow, highlight));
    float shaped = gp.shadowGrain * shadow
                 + gp.midtoneGrain * midtone
                 + gp.highlightGrain * highlight;
    return max(0.0f, 1.0f + (shaped - 1.0f) * max(gp.curveContrast, 0.0f));
}

kernel void grainApply(device const float*       comp       [[buffer(0)]],
                       device const float*       plate      [[buffer(1)]],
                       device const float*       degr       [[buffer(2)]],
                       device const float*       mask       [[buffer(3)]],
                       device const float*       ext        [[buffer(4)]],
                       device       float*       dst        [[buffer(5)]],
                       device const float*       curve      [[buffer(6)]],
                       constant     GrainParams& gp         [[buffer(7)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= gp.width || int(gid.y) >= gp.height) return;

    int  x = int(gid.x);
    int  y = int(gid.y);

    float C[3] = {0,0,0};
    float P[3] = {0,0,0};
    float D[3] = {0,0,0};
    float A    = 1.0f;

    if (gp.hasComp) {
        device const float* px = comp + y * gp.compStrideF + x * 4;
        C[0] = px[0]; C[1] = px[1]; C[2] = px[2]; A = px[3];
    }
    if (gp.hasPlate) {
        device const float* px = plate + y * gp.plateStrideF + x * 4;
        P[0] = px[0]; P[1] = px[1]; P[2] = px[2];
    }
    if (gp.hasDegr) {
        device const float* px = degr + y * gp.degrStrideF + x * 4;
        D[0] = px[0]; D[1] = px[1]; D[2] = px[2];
    }

    float gO[3] = {P[0]-D[0], P[1]-D[1], P[2]-D[2]};

    float lf = (gp.fixGhosting != 0 || gp.luminance >= 1.0f || gp.luminance <= 0.0f)
               ? 0.0f : (1.0f / gp.luminance - 1.0f);
    float yG  = luma709(gO[0], gO[1], gO[2]);
    float lyG = yG * lf;
    float gLc[3]  = {gO[0]+lyG, gO[1]+lyG, gO[2]+lyG};
    float clean[3] = {D[0]-lyG, D[1]-lyG, D[2]-lyG};

    float cClean[3] = {
        sampleCurve(curve, gp.curveSize, 0, gp.minX, gp.maxX, clean[0]),
        sampleCurve(curve, gp.curveSize, 1, gp.minX, gp.maxX, clean[1]),
        sampleCurve(curve, gp.curveSize, 2, gp.minX, gp.maxX, clean[2]),
    };
    float gN[3] = {
        gLc[0] * kEps / max(cClean[0], kEps),
        gLc[1] * kEps / max(cClean[1], kEps),
        gLc[2] * kEps / max(cClean[2], kEps),
    };

    float gS[3];
    if (gp.externalGrain && gp.hasExt) {
        device const float* px = ext + y * gp.extStrideF + x * 4;
        gS[0] = px[0]; gS[1] = px[1]; gS[2] = px[2];
    } else {
        gS[0] = gN[0]; gS[1] = gN[1]; gS[2] = gN[2];
    }

    float cC[3] = {
        sampleCurve(curve, gp.curveSize, 0, gp.minX, gp.maxX, C[0]),
        sampleCurve(curve, gp.curveSize, 1, gp.minX, gp.maxX, C[1]),
        sampleCurve(curve, gp.curveSize, 2, gp.minX, gp.maxX, C[2]),
    };
    float tone = toneGain(gp, float3(C[0], C[1], C[2]));
    float liveGain = max(gp.grainAmount, 0.0f);
    float channelGain[3] = {
        max(gp.redGrain, 0.0f),
        max(gp.greenGrain, 0.0f),
        max(gp.blueGrain, 0.0f),
    };
    float gA[3] = {
        gS[0] * cC[0] / kEps * tone * channelGain[0] * liveGain,
        gS[1] * cC[1] / kEps * tone * channelGain[1] * liveGain,
        gS[2] * cC[2] / kEps * tone * channelGain[2] * liveGain,
    };

    float m = 1.0f;
    if (gp.hasMask && gp.hasMaskBuf) {
        device const float* px = mask + y * gp.maskStrideF + x * 4;
        m = clamp(px[3], 0.0f, 1.0f);
        if (gp.invertMask) m = 1.0f - m;
    }

    float r[3] = {
        C[0] + gA[0] * m,
        C[1] + gA[1] * m,
        C[2] + gA[2] * m,
    };
    float o[3];
    int mode = gp.outputMode;
    if (mode == 1)      { o[0]=gO[0]; o[1]=gO[1]; o[2]=gO[2]; }
    else if (mode == 2) { o[0]=gN[0]; o[1]=gN[1]; o[2]=gN[2]; }
    else if (mode == 3) { o[0]=gA[0]; o[1]=gA[1]; o[2]=gA[2]; }
    else                { o[0]=r[0];  o[1]=r[1];  o[2]=r[2];  }   // 0 or 4

    device float* px = dst + y * gp.dstStrideF + x * 4;
    px[0] = o[0]; px[1] = o[1]; px[2] = o[2]; px[3] = A;
}
)METAL";

struct CachedPipeline {
    id<MTLLibrary>              library{nil};
    id<MTLFunction>             function{nil};
    id<MTLComputePipelineState> pipeline{nil};
};

std::mutex& pipelineMutex() {
    static std::mutex m;
    return m;
}
std::unordered_map<id<MTLDevice>, CachedPipeline>& pipelineCache() {
    static std::unordered_map<id<MTLDevice>, CachedPipeline> c;
    return c;
}

CachedPipeline& getPipeline(id<MTLDevice> device) {
    std::lock_guard<std::mutex> lock(pipelineMutex());
    auto& cache = pipelineCache();
    auto it = cache.find(device);
    if (it != cache.end()) return it->second;
    CachedPipeline p;
    NSError* err = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.fastMathEnabled = YES;
    p.library = [device newLibraryWithSource:@(kKernelSrc) options:opts error:&err];
    if (!p.library) {
        NSLog(@"DasGrain Metal compile failed: %@", err.localizedDescription);
        return cache[device] = p;
    }
    p.function = [p.library newFunctionWithName:@"grainApply"];
    p.pipeline = [device newComputePipelineStateWithFunction:p.function error:&err];
    if (!p.pipeline) {
        NSLog(@"DasGrain Metal pipeline create failed: %@", err.localizedDescription);
    }
    cache[device] = p;
    return cache[device];
}

#pragma pack(push, 4)
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
    int fixGhosting;
    int externalGrain;
    int outputMode;
    int hasMask;
    int invertMask;
    int curveSize;
    float minX;
    float maxX;
    int width;
    int height;
    int compStrideF;
    int plateStrideF;
    int degrStrideF;
    int maskStrideF;
    int extStrideF;
    int dstStrideF;
    int hasComp;
    int hasPlate;
    int hasDegr;
    int hasMaskBuf;
    int hasExt;
};
#pragma pack(pop)

}  // namespace

void runGrainApplyMetal(void* metalCmdQueue,
                        const GrainApplyParams& p,
                        ImageViewConst comp,
                        ImageViewConst plate,
                        ImageViewConst degrained,
                        ImageViewConst mask,
                        ImageViewConst externalGrain,
                        ImageView dst) {
    if (!metalCmdQueue || !dst.data) {
        runGrainApplyCPU(p, comp, plate, degrained, mask, externalGrain, dst);
        return;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metalCmdQueue;
    id<MTLDevice>       device = queue.device;
    auto& pipe = getPipeline(device);
    if (!pipe.pipeline) {
        runGrainApplyCPU(p, comp, plate, degrained, mask, externalGrain, dst);
        return;
    }

    auto asBuffer = [](const void* ptr) -> id<MTLBuffer> {
        return (__bridge id<MTLBuffer>)const_cast<void*>(ptr);
    };

    // Image data pointers are MTLBuffer handles when Metal is enabled.
    id<MTLBuffer> compBuf  = asBuffer(comp.data);
    id<MTLBuffer> plateBuf = asBuffer(plate.data);
    id<MTLBuffer> degrBuf  = asBuffer(degrained.data);
    id<MTLBuffer> maskBuf  = asBuffer(mask.data);
    id<MTLBuffer> extBuf   = asBuffer(externalGrain.data);
    id<MTLBuffer> dstBuf   = asBuffer(dst.data);

    if (!dstBuf) {
        runGrainApplyCPU(p, comp, plate, degrained, mask, externalGrain, dst);
        return;
    }

    // Curve buffer (small — 3 * curveSize floats).
    id<MTLBuffer> curveBuf = nil;
    if (p.curve.data && p.curve.size > 0) {
        curveBuf = [device newBufferWithBytes:p.curve.data
                                       length:size_t(p.curve.size) * 3 * sizeof(float)
                                      options:MTLResourceStorageModeShared];
    }

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
    auto sf = [](int strideBytes) { return strideBytes / int(sizeof(float)); };
    gp.compStrideF   = sf(comp.stride);
    gp.plateStrideF  = sf(plate.stride);
    gp.degrStrideF   = sf(degrained.stride);
    gp.maskStrideF   = sf(mask.stride);
    gp.extStrideF    = sf(externalGrain.stride);
    gp.dstStrideF    = sf(dst.stride);
    gp.hasComp       = (compBuf != nil) ? 1 : 0;
    gp.hasPlate      = (plateBuf != nil) ? 1 : 0;
    gp.hasDegr       = (degrBuf  != nil) ? 1 : 0;
    gp.hasMaskBuf    = (maskBuf  != nil) ? 1 : 0;
    gp.hasExt        = (extBuf   != nil) ? 1 : 0;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    cmdBuf.label = @"DasGrain.grainApply";

    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pipe.pipeline];

    [enc setBuffer:compBuf  offset:0 atIndex:0];
    [enc setBuffer:plateBuf offset:0 atIndex:1];
    [enc setBuffer:degrBuf  offset:0 atIndex:2];
    [enc setBuffer:maskBuf  offset:0 atIndex:3];
    [enc setBuffer:extBuf   offset:0 atIndex:4];
    [enc setBuffer:dstBuf   offset:0 atIndex:5];
    [enc setBuffer:curveBuf offset:0 atIndex:6];
    [enc setBytes:&gp length:sizeof(gp) atIndex:7];

    NSUInteger w = pipe.pipeline.threadExecutionWidth;
    NSUInteger h = pipe.pipeline.maxTotalThreadsPerThreadgroup / w;
    MTLSize tgCount = MTLSizeMake(w, h, 1);
    MTLSize gridCount = MTLSizeMake(NSUInteger(dst.width), NSUInteger(dst.height), 1);
    [enc dispatchThreads:gridCount threadsPerThreadgroup:tgCount];
    [enc endEncoding];
    [cmdBuf commit];
}

}  // namespace kernels
}  // namespace dasgrain
