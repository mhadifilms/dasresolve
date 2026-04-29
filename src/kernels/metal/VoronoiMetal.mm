// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Metal Voronoi scatter kernel. Mirrors the CPU reference in
// src/kernels/cpu/VoronoiCPU.cpp; same Voronoi cell, same per-cell random
// offset within the shrunken box, same UV distortion, same edge-blend trick.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "../Kernels.h"

namespace dasgrain {
namespace kernels {

namespace {

constexpr const char* kKernelSrc = R"(
#include <metal_stdlib>
using namespace metal;

struct Params {
    float boxX, boxY, boxR, boxT;
    float cellSize;
    int   seed;
    int   overlayPattern;
    float distortAmplitude;
    float distortFrequency;
    int   edgeBlendSize;
    int   dstOffsetX, dstOffsetY;
    int   curveSize;
    float curveMinX, curveMaxX;
    int   plateW, plateH;
    int   plateRowStride;
    int   degRowStride;
    int   dstRowStride;
    int   dstW, dstH;
};

constant int kXNoise = 1619;
constant int kYNoise = 31337;
constant int kZNoise = 6971;
constant int kSeedNoise = 1013;

inline int intValueNoise3D(int x, int y, int z, int seed) {
    uint n = (uint)((kXNoise * x + kYNoise * y + kZNoise * z + kSeedNoise * seed) & 0x7fffffff);
    n = (n >> 13) ^ n;
    return (int)((n * (n * n * 60493 + 19990303) + 1376312589) & 0x7fffffff);
}

inline float valueNoise3D(int x, int y, int z, int seed) {
    return 1.0f - (float(intValueNoise3D(x, y, z, seed)) / 1073741824.0f);
}

struct Cell { float u; float v; };

inline Cell cellAt(int px, int py, int W, int H, float aspect, float frequency, int seed) {
    float x = float(px) * aspect * frequency / float(W);
    float y = float(py) * frequency / float(W);
    int xInt = (x > 0.0f) ? int(x) : int(x) - 1;
    int yInt = (y > 0.0f) ? int(y) : int(y) - 1;
    float minDist = 2147483647.0f;
    float xCand = 0.0f, yCand = 0.0f;
    const float randomness = 0.5f;
    for (int yCur = yInt - 2; yCur <= yInt + 2; ++yCur) {
        for (int xCur = xInt - 2; xCur <= xInt + 2; ++xCur) {
            float xPos = float(xCur) + (valueNoise3D(xCur, yCur, 0, seed)     + 1.0f) * randomness + (1.0f - randomness) - 1.0f;
            float yPos = float(yCur) + (valueNoise3D(xCur, yCur, 0, seed + 1) + 1.0f) * randomness + (1.0f - randomness) - 1.0f;
            float dx = xPos - x, dy = yPos - y;
            float d = dx * dx + dy * dy;
            if (d < minDist) { minDist = d; xCand = xPos; yCand = yPos; }
        }
    }
    Cell c;
    c.u = xCand / aspect / frequency;
    c.v = yCand / float(H) * float(W) / frequency;
    return c;
}

inline float sampleCurve(device const float* lut, int size, float minX, float maxX,
                         int channel, float v) {
    if (size <= 0) return max(v, 0.01f);
    float t  = (v - minX) / max(maxX - minX, 1e-12f);
    float ti = clamp(t, 0.0f, 1.0f) * float(size - 1);
    int   i0 = int(floor(ti));
    int   i1 = min(i0 + 1, size - 1);
    float f  = ti - float(i0);
    device const float* row = lut + channel * size;
    return mix(row[i0], row[i1], f);
}

inline float4 fetchPixel(device const float* img, int rowFloats, int W, int H, int x, int y) {
    x = clamp(x, 0, W - 1);
    y = clamp(y, 0, H - 1);
    device const float* p = img + y * rowFloats + x * 4;
    return float4(p[0], p[1], p[2], p[3]);
}

inline float4 sampleBilinear(device const float* img, int rowFloats, int W, int H, float u, float v) {
    float fx = clamp(u, 0.0f, float(W - 1));
    float fy = clamp(v, 0.0f, float(H - 1));
    int   x0 = int(floor(fx));
    int   y0 = int(floor(fy));
    int   x1 = min(x0 + 1, W - 1);
    int   y1 = min(y0 + 1, H - 1);
    float dx = fx - float(x0);
    float dy = fy - float(y0);
    float4 p00 = fetchPixel(img, rowFloats, W, H, x0, y0);
    float4 p10 = fetchPixel(img, rowFloats, W, H, x1, y0);
    float4 p01 = fetchPixel(img, rowFloats, W, H, x0, y1);
    float4 p11 = fetchPixel(img, rowFloats, W, H, x1, y1);
    float4 a   = mix(p00, p10, dx);
    float4 b   = mix(p01, p11, dx);
    return mix(a, b, dy);
}

inline float distortNoise(float u, float v, int seed, float freq) {
    float fx = u * freq;
    float fy = v * freq;
    int xi = int(floor(fx));
    int yi = int(floor(fy));
    float tx = fx - float(xi);
    float ty = fy - float(yi);
    float v00 = valueNoise3D(xi,     yi,     0, seed);
    float v10 = valueNoise3D(xi + 1, yi,     0, seed);
    float v01 = valueNoise3D(xi,     yi + 1, 0, seed);
    float v11 = valueNoise3D(xi + 1, yi + 1, 0, seed);
    return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

inline uint hashU32(uint x) {
    x = (x ^ 0x6c50b47cu) * 0xb82f1e52u;
    x = (x ^ (x >> 13))  * 0x7c4a02e9u;
    x ^= x >> 17;
    return x;
}

inline float2 cellRandom01(float cu, float cv) {
    uint hx = hashU32(uint(int(cu * 1.0e6f)));
    uint hy = hashU32(uint(int(cv * 1.0e6f) ^ 0x9e3779b9u));
    return float2(float(hx) * (1.0f / 4294967296.0f),
                  float(hy) * (1.0f / 4294967296.0f));
}

kernel void scatter(device const float* plate    [[buffer(0)]],
                    device const float* deg      [[buffer(1)]],
                    device float*       dst      [[buffer(2)]],
                    device const float* curve    [[buffer(3)]],
                    constant Params&    p        [[buffer(4)]],
                    uint2               gid      [[thread_position_in_grid]]) {
    if (int(gid.x) >= p.dstW || int(gid.y) >= p.dstH) return;
    const int W = p.plateW;
    const int H = p.plateH;
    const float cellSize  = max(p.cellSize, 1.0f);
    const float frequency = float(W) / cellSize;
    const float aspect    = float(W) / float(max(H, 1));

    const float bxRaw0 = min(p.boxX, p.boxR);
    const float byRaw0 = min(p.boxY, p.boxT);
    const float bxRaw1 = max(p.boxX, p.boxR);
    const float byRaw1 = max(p.boxY, p.boxT);
    const float shrink = cellSize * 0.5f
                       + ceil(float(max(p.edgeBlendSize, 0)) * 0.5f)
                       + p.distortAmplitude * 0.5f;
    const float bxLo = bxRaw0 + shrink;
    const float byLo = byRaw0 + shrink;
    const float bxHi = bxRaw1 - shrink;
    const float byHi = byRaw1 - shrink;
    const bool  boxValid = (bxHi > bxLo) && (byHi > byLo);

    const int blend = max(p.edgeBlendSize, 0);
    const int subN  = (blend <= 0) ? 1 : (blend + 1);
    const float subStep = 1.0f / float(subN);
    const int sx_pix = int(gid.x) + p.dstOffsetX;
    const int sy_pix = int(gid.y) + p.dstOffsetY;

    float4 accum = float4(0);
    int    accumN = 0;
    for (int by = 0; by < subN; ++by) {
        for (int bx = 0; bx < subN; ++bx) {
            float ox = (subN > 1) ? (float(bx) + 0.5f) * subStep - 0.5f : 0.0f;
            float oy = (subN > 1) ? (float(by) + 0.5f) * subStep - 0.5f : 0.0f;
            int qx = int(round(float(sx_pix) + ox));
            int qy = int(round(float(sy_pix) + oy));

            Cell cr = cellAt(qx, qy, W, H, aspect, frequency, p.seed);
            float cellPx = cr.u;
            float cellPy = cr.v;
            if (p.distortAmplitude > 0.0f) {
                float fU = (p.distortFrequency > 0.0f) ? 1.0f / p.distortFrequency : 0.0f;
                float du = distortNoise(cellPx, cellPy, p.seed * 7 + 13, fU);
                float dv = distortNoise(cellPx, cellPy, p.seed * 7 + 31, fU);
                cellPx += du * p.distortAmplitude;
                cellPy += dv * p.distortAmplitude;
            }
            float2 r01 = boxValid ? cellRandom01(cellPx, cellPy) : float2(0.5f);
            float boxU = bxLo + r01.x * (bxHi - bxLo);
            float boxV = byLo + r01.y * (byHi - byLo);

            float pixU = float(qx) + 0.5f;
            float pixV = float(qy) + 0.5f;
            float sampU = boxU + (pixU - cellPx);
            float sampV = boxV + (pixV - cellPy);

            float4 plateS = sampleBilinear(plate, p.plateRowStride, W, H, sampU - 0.5f, sampV - 0.5f);
            float4 degS   = sampleBilinear(deg,   p.degRowStride,   W, H, sampU - 0.5f, sampV - 0.5f);
            float gR = plateS.x - degS.x;
            float gG = plateS.y - degS.y;
            float gB = plateS.z - degS.z;
            float cR = sampleCurve(curve, p.curveSize, p.curveMinX, p.curveMaxX, 0, degS.x);
            float cG = sampleCurve(curve, p.curveSize, p.curveMinX, p.curveMaxX, 1, degS.y);
            float cB = sampleCurve(curve, p.curveSize, p.curveMinX, p.curveMaxX, 2, degS.z);
            float eps = 0.01f;
            accum.x += gR * eps / max(cR, eps);
            accum.y += gG * eps / max(cG, eps);
            accum.z += gB * eps / max(cB, eps);
            accum.w += 1.0f;
            ++accumN;
        }
    }
    float invN = (accumN > 0) ? 1.0f / float(accumN) : 0.0f;
    device float* outPx = dst + int(gid.y) * p.dstRowStride + int(gid.x) * 4;
    if (p.overlayPattern != 0) {
        Cell cr = cellAt(sx_pix, sy_pix, W, H, aspect, frequency, p.seed);
        int ci = int(floor(cr.u));
        int cj = int(floor(cr.v));
        outPx[0] = (valueNoise3D(ci, cj, 0, p.seed * 11 + 1) + 1.0f) * 0.5f;
        outPx[1] = (valueNoise3D(ci, cj, 0, p.seed * 11 + 2) + 1.0f) * 0.5f;
        outPx[2] = (valueNoise3D(ci, cj, 0, p.seed * 11 + 3) + 1.0f) * 0.5f;
        outPx[3] = 1.0f;
    } else {
        outPx[0] = accum.x * invN;
        outPx[1] = accum.y * invN;
        outPx[2] = accum.z * invN;
        outPx[3] = accum.w * invN;
    }
}
)";

struct CachedPipeline {
    id<MTLDevice>                device   = nil;
    id<MTLComputePipelineState>  pipeline = nil;
};
static std::mutex                                              gPipelineMutex;
static std::unordered_map<id<MTLDevice>, CachedPipeline>       gPipelineCache;

CachedPipeline getPipeline(id<MTLDevice> device) {
    std::lock_guard<std::mutex> lock(gPipelineMutex);
    auto it = gPipelineCache.find(device);
    if (it != gPipelineCache.end()) return it->second;
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kKernelSrc];
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        return {};
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"scatter"];
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&err];
    CachedPipeline cp = { device, p };
    gPipelineCache[device] = cp;
    return cp;
}

}  // namespace

void runScatterMetal(void* metalCmdQueue,
                     const ScatterParams& p,
                     ImageViewConst plateSample,
                     ImageViewConst degSample,
                     ImageView      dst) {
    if (!metalCmdQueue) {
        runScatterCPU(p, plateSample, degSample, dst);
        return;
    }
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metalCmdQueue;
        id<MTLDevice> device = queue.device;
        CachedPipeline cp = getPipeline(device);
        if (!cp.pipeline) {
            runScatterCPU(p, plateSample, degSample, dst);
            return;
        }
        const int W  = plateSample.width;
        const int H  = plateSample.height;
        const int dw = dst.width;
        const int dh = dst.height;
        if (W <= 0 || H <= 0 || dw <= 0 || dh <= 0) return;

        const size_t plateBytes = size_t(W) * H * 4 * sizeof(float);
        const size_t dstBytes   = size_t(dw) * dh * 4 * sizeof(float);
        const int curveLen      = std::max(p.curve.size, 0) * 3;
        const size_t curveBytes = size_t(std::max(curveLen, 1)) * sizeof(float);

        id<MTLBuffer> plateBuf = [device newBufferWithBytes:plateSample.data length:plateBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> degBuf   = [device newBufferWithBytes:degSample.data   length:plateBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> dstBuf   = [device newBufferWithLength:dstBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> curveBuf = nil;
        if (p.curve.data && p.curve.size > 0) {
            curveBuf = [device newBufferWithBytes:p.curve.data length:curveBytes options:MTLResourceStorageModeShared];
        } else {
            float zero = 0.0f;
            curveBuf = [device newBufferWithBytes:&zero length:sizeof(float) options:MTLResourceStorageModeShared];
        }

        struct Params {
            float boxX, boxY, boxR, boxT;
            float cellSize;
            int   seed;
            int   overlayPattern;
            float distortAmplitude;
            float distortFrequency;
            int   edgeBlendSize;
            int   dstOffsetX, dstOffsetY;
            int   curveSize;
            float curveMinX, curveMaxX;
            int   plateW, plateH;
            int   plateRowStride;
            int   degRowStride;
            int   dstRowStride;
            int   dstW, dstH;
        } pp;
        pp.boxX             = p.boxX;
        pp.boxY             = p.boxY;
        pp.boxR             = p.boxR;
        pp.boxT             = p.boxT;
        pp.cellSize         = p.cellSize;
        pp.seed             = p.seed;
        pp.overlayPattern   = p.overlayPattern;
        pp.distortAmplitude = p.distortAmplitude;
        pp.distortFrequency = p.distortFrequency;
        pp.edgeBlendSize    = p.edgeBlendSize;
        pp.dstOffsetX       = p.dstOffsetX;
        pp.dstOffsetY       = p.dstOffsetY;
        pp.curveSize        = p.curve.size;
        pp.curveMinX        = p.curve.minX;
        pp.curveMaxX        = p.curve.maxX;
        pp.plateW           = W;
        pp.plateH           = H;
        pp.plateRowStride   = W * 4;
        pp.degRowStride     = W * 4;
        pp.dstRowStride     = dw * 4;
        pp.dstW             = dw;
        pp.dstH             = dh;

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:cp.pipeline];
        [ce setBuffer:plateBuf offset:0 atIndex:0];
        [ce setBuffer:degBuf   offset:0 atIndex:1];
        [ce setBuffer:dstBuf   offset:0 atIndex:2];
        [ce setBuffer:curveBuf offset:0 atIndex:3];
        [ce setBytes:&pp length:sizeof(pp) atIndex:4];
        MTLSize tgs = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake((dw + 15) / 16, (dh + 15) / 16, 1);
        [ce dispatchThreadgroups:grid threadsPerThreadgroup:tgs];
        [ce endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        const auto* outFlat = static_cast<const float*>([dstBuf contents]);
        for (int y = 0; y < dh; ++y) {
            std::memcpy(reinterpret_cast<uint8_t*>(dst.data) + y * dst.stride,
                        outFlat + size_t(y) * dw * 4,
                        size_t(dw) * 4 * sizeof(float));
        }
    }
}

}  // namespace kernels
}  // namespace dasgrain
