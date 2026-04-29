// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// OpenCL Voronoi scatter kernel. Mirrors the CPU reference in
// src/kernels/cpu/VoronoiCPU.cpp byte-for-byte.

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif

#include "../Kernels.h"

namespace dasgrain {
namespace kernels {

namespace {

constexpr const char* kSrc = R"(
typedef struct {
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
} Params;

static int intValueNoise3D(int x, int y, int z, int seed) {
    uint n = (uint)((1619 * x + 31337 * y + 6971 * z + 1013 * seed) & 0x7fffffff);
    n = (n >> 13) ^ n;
    return (int)((n * (n * n * 60493u + 19990303u) + 1376312589u) & 0x7fffffffu);
}
static float valueNoise3D(int x, int y, int z, int seed) {
    return 1.0f - ((float)intValueNoise3D(x, y, z, seed) / 1073741824.0f);
}
static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static void cellAt(int px, int py, int W, int H, float aspect, float frequency, int seed,
                   float* outU, float* outV) {
    float x = (float)px * aspect * frequency / (float)W;
    float y = (float)py * frequency / (float)W;
    int xInt = (x > 0.0f) ? (int)x : (int)x - 1;
    int yInt = (y > 0.0f) ? (int)y : (int)y - 1;
    float minDist = 2147483647.0f;
    float xCand = 0.0f, yCand = 0.0f;
    const float randomness = 0.5f;
    for (int yCur = yInt - 2; yCur <= yInt + 2; ++yCur) {
        for (int xCur = xInt - 2; xCur <= xInt + 2; ++xCur) {
            float xPos = (float)xCur + (valueNoise3D(xCur, yCur, 0, seed)     + 1.0f) * randomness + (1.0f - randomness) - 1.0f;
            float yPos = (float)yCur + (valueNoise3D(xCur, yCur, 0, seed + 1) + 1.0f) * randomness + (1.0f - randomness) - 1.0f;
            float dx = xPos - x, dy = yPos - y;
            float d = dx * dx + dy * dy;
            if (d < minDist) { minDist = d; xCand = xPos; yCand = yPos; }
        }
    }
    *outU = xCand / aspect / frequency;
    *outV = yCand / (float)H * (float)W / frequency;
}
static void fetchPx(__global const float* img, int rowFloats, int W, int H, int x, int y, float out[4]) {
    if (x < 0) x = 0; if (x > W - 1) x = W - 1;
    if (y < 0) y = 0; if (y > H - 1) y = H - 1;
    __global const float* p = img + y * rowFloats + x * 4;
    out[0] = p[0]; out[1] = p[1]; out[2] = p[2]; out[3] = p[3];
}
static void sampleBilinear(__global const float* img, int rowFloats, int W, int H, float u, float v, float out[4]) {
    float fx = clampf(u, 0.0f, (float)(W - 1));
    float fy = clampf(v, 0.0f, (float)(H - 1));
    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    int x1 = x0 + 1 < W ? x0 + 1 : W - 1;
    int y1 = y0 + 1 < H ? y0 + 1 : H - 1;
    float dx = fx - (float)x0;
    float dy = fy - (float)y0;
    float p00[4], p10[4], p01[4], p11[4];
    fetchPx(img, rowFloats, W, H, x0, y0, p00);
    fetchPx(img, rowFloats, W, H, x1, y0, p10);
    fetchPx(img, rowFloats, W, H, x0, y1, p01);
    fetchPx(img, rowFloats, W, H, x1, y1, p11);
    for (int c = 0; c < 4; ++c) {
        float a = p00[c] + (p10[c] - p00[c]) * dx;
        float b = p01[c] + (p11[c] - p01[c]) * dx;
        out[c] = a + (b - a) * dy;
    }
}
static float sampleCurve(__global const float* lut, int size, float minX, float maxX, int channel, float v) {
    if (size <= 0) return v > 0.01f ? v : 0.01f;
    float t  = (v - minX) / fmax(maxX - minX, 1e-12f);
    float ti = clampf(t, 0.0f, 1.0f) * (float)(size - 1);
    int   i0 = (int)floor(ti);
    int   i1 = i0 + 1 < size ? i0 + 1 : size - 1;
    float f  = ti - (float)i0;
    __global const float* row = lut + channel * size;
    return row[i0] + (row[i1] - row[i0]) * f;
}
static float distortNoise(float u, float v, int seed, float freq) {
    float fx = u * freq;
    float fy = v * freq;
    int xi = (int)floor(fx);
    int yi = (int)floor(fy);
    float tx = fx - (float)xi;
    float ty = fy - (float)yi;
    float v00 = valueNoise3D(xi,     yi,     0, seed);
    float v10 = valueNoise3D(xi + 1, yi,     0, seed);
    float v01 = valueNoise3D(xi,     yi + 1, 0, seed);
    float v11 = valueNoise3D(xi + 1, yi + 1, 0, seed);
    float a = v00 + (v10 - v00) * tx;
    float b = v01 + (v11 - v01) * tx;
    return a + (b - a) * ty;
}
static uint hashU32(uint x) {
    x = (x ^ 0x6c50b47cu) * 0xb82f1e52u;
    x = (x ^ (x >> 13))  * 0x7c4a02e9u;
    x ^= x >> 17;
    return x;
}
static void cellRandom01(float cu, float cv, float* rx, float* ry) {
    uint hx = hashU32((uint)((int)(cu * 1.0e6f)));
    uint hy = hashU32((uint)((int)(cv * 1.0e6f) ^ 0x9e3779b9u));
    *rx = (float)hx * (1.0f / 4294967296.0f);
    *ry = (float)hy * (1.0f / 4294967296.0f);
}

__kernel void scatterKernel(__global const float* plate,
                            __global const float* deg,
                            __global       float* dst,
                            __global const float* curve,
                            Params p) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= p.dstW || y >= p.dstH) return;
    int W = p.plateW;
    int H = p.plateH;
    float cellSize = p.cellSize > 1.0f ? p.cellSize : 1.0f;
    float frequency = (float)W / cellSize;
    float aspect = (float)W / (float)(H > 0 ? H : 1);

    float bxRaw0 = fmin(p.boxX, p.boxR);
    float byRaw0 = fmin(p.boxY, p.boxT);
    float bxRaw1 = fmax(p.boxX, p.boxR);
    float byRaw1 = fmax(p.boxY, p.boxT);
    int eb = p.edgeBlendSize > 0 ? p.edgeBlendSize : 0;
    float shrink = cellSize * 0.5f + ceil((float)eb * 0.5f) + p.distortAmplitude * 0.5f;
    float bxLo = bxRaw0 + shrink;
    float byLo = byRaw0 + shrink;
    float bxHi = bxRaw1 - shrink;
    float byHi = byRaw1 - shrink;
    int boxValid = (bxHi > bxLo) && (byHi > byLo) ? 1 : 0;

    int subN = (eb <= 0) ? 1 : (eb + 1);
    float subStep = 1.0f / (float)subN;
    int sx_pix = x + p.dstOffsetX;
    int sy_pix = y + p.dstOffsetY;

    float accum[4] = {0, 0, 0, 0};
    int accumN = 0;
    for (int by = 0; by < subN; ++by) {
        for (int bx = 0; bx < subN; ++bx) {
            float ox = (subN > 1) ? ((float)bx + 0.5f) * subStep - 0.5f : 0.0f;
            float oy = (subN > 1) ? ((float)by + 0.5f) * subStep - 0.5f : 0.0f;
            int qx = (int)round((float)sx_pix + ox);
            int qy = (int)round((float)sy_pix + oy);
            float cellPx, cellPy;
            cellAt(qx, qy, W, H, aspect, frequency, p.seed, &cellPx, &cellPy);
            if (p.distortAmplitude > 0.0f) {
                float fU = (p.distortFrequency > 0.0f) ? 1.0f / p.distortFrequency : 0.0f;
                float du = distortNoise(cellPx, cellPy, p.seed * 7 + 13, fU);
                float dv = distortNoise(cellPx, cellPy, p.seed * 7 + 31, fU);
                cellPx += du * p.distortAmplitude;
                cellPy += dv * p.distortAmplitude;
            }
            float rx = 0.5f, ry = 0.5f;
            if (boxValid != 0) cellRandom01(cellPx, cellPy, &rx, &ry);
            float boxU = bxLo + rx * (bxHi - bxLo);
            float boxV = byLo + ry * (byHi - byLo);
            float pixU = (float)qx + 0.5f;
            float pixV = (float)qy + 0.5f;
            float sampU = boxU + (pixU - cellPx);
            float sampV = boxV + (pixV - cellPy);

            float plateS[4], degS[4];
            sampleBilinear(plate, p.plateRowStride, W, H, sampU - 0.5f, sampV - 0.5f, plateS);
            sampleBilinear(deg,   p.degRowStride,   W, H, sampU - 0.5f, sampV - 0.5f, degS);
            float gR = plateS[0] - degS[0];
            float gG = plateS[1] - degS[1];
            float gB = plateS[2] - degS[2];
            float cR = sampleCurve(curve, p.curveSize, p.curveMinX, p.curveMaxX, 0, degS[0]);
            float cG = sampleCurve(curve, p.curveSize, p.curveMinX, p.curveMaxX, 1, degS[1]);
            float cB = sampleCurve(curve, p.curveSize, p.curveMinX, p.curveMaxX, 2, degS[2]);
            float eps = 0.01f;
            accum[0] += gR * eps / fmax(cR, eps);
            accum[1] += gG * eps / fmax(cG, eps);
            accum[2] += gB * eps / fmax(cB, eps);
            accum[3] += 1.0f;
            ++accumN;
        }
    }
    float invN = (accumN > 0) ? 1.0f / (float)accumN : 0.0f;
    __global float* outPx = dst + y * p.dstRowStride + x * 4;
    if (p.overlayPattern != 0) {
        float cu, cv;
        cellAt(sx_pix, sy_pix, W, H, aspect, frequency, p.seed, &cu, &cv);
        int ci = (int)floor(cu);
        int cj = (int)floor(cv);
        outPx[0] = (valueNoise3D(ci, cj, 0, p.seed * 11 + 1) + 1.0f) * 0.5f;
        outPx[1] = (valueNoise3D(ci, cj, 0, p.seed * 11 + 2) + 1.0f) * 0.5f;
        outPx[2] = (valueNoise3D(ci, cj, 0, p.seed * 11 + 3) + 1.0f) * 0.5f;
        outPx[3] = 1.0f;
    } else {
        outPx[0] = accum[0] * invN;
        outPx[1] = accum[1] * invN;
        outPx[2] = accum[2] * invN;
        outPx[3] = accum[3] * invN;
    }
}
)";

struct Cached {
    cl_context  context = nullptr;
    cl_program  program = nullptr;
    cl_kernel   kernel  = nullptr;
};

static std::mutex                                  gMutex;
static std::unordered_map<cl_context, Cached>      gCache;

Cached buildOrGet(cl_context ctx, cl_device_id dev) {
    std::lock_guard<std::mutex> lock(gMutex);
    auto it = gCache.find(ctx);
    if (it != gCache.end()) return it->second;
    cl_int err = CL_SUCCESS;
    const char* srcPtr = kSrc;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &srcPtr, nullptr, &err);
    if (err != CL_SUCCESS || !prog) return {};
    err = clBuildProgram(prog, 1, &dev, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseProgram(prog);
        return {};
    }
    cl_kernel kern = clCreateKernel(prog, "scatterKernel", &err);
    if (err != CL_SUCCESS || !kern) {
        clReleaseProgram(prog);
        return {};
    }
    Cached c{ ctx, prog, kern };
    gCache[ctx] = c;
    return c;
}

}  // namespace

void runScatterOpenCL(void* clCmdQueue,
                      const ScatterParams& p,
                      ImageViewConst plateSample,
                      ImageViewConst degSample,
                      ImageView      dst) {
    if (!clCmdQueue) {
        runScatterCPU(p, plateSample, degSample, dst);
        return;
    }
    cl_command_queue queue = static_cast<cl_command_queue>(clCmdQueue);
    cl_context ctx = nullptr;
    cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS ||
        clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) {
        runScatterCPU(p, plateSample, degSample, dst);
        return;
    }
    Cached cached = buildOrGet(ctx, dev);
    if (!cached.kernel) {
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
    const int curveLen      = (p.curve.size > 0 ? p.curve.size : 0) * 3;
    const size_t curveBytes = size_t(curveLen > 0 ? curveLen : 1) * sizeof(float);

    cl_int err = CL_SUCCESS;
    cl_mem mPlate = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                                   plateBytes, const_cast<float*>(plateSample.data), &err);
    cl_mem mDeg   = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                                   plateBytes, const_cast<float*>(degSample.data),   &err);
    cl_mem mDst   = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, dstBytes, nullptr, &err);
    cl_mem mCurve = (p.curve.data && p.curve.size > 0)
        ? clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         curveBytes, const_cast<float*>(p.curve.data), &err)
        : clCreateBuffer(ctx, CL_MEM_READ_ONLY, curveBytes, nullptr, &err);

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

    cl_kernel kern = cached.kernel;
    clSetKernelArg(kern, 0, sizeof(cl_mem),  &mPlate);
    clSetKernelArg(kern, 1, sizeof(cl_mem),  &mDeg);
    clSetKernelArg(kern, 2, sizeof(cl_mem),  &mDst);
    clSetKernelArg(kern, 3, sizeof(cl_mem),  &mCurve);
    clSetKernelArg(kern, 4, sizeof(pp),      &pp);

    size_t global[2] = { size_t(((dw + 15) / 16) * 16), size_t(((dh + 15) / 16) * 16) };
    size_t local[2]  = { 16, 16 };
    clEnqueueNDRangeKernel(queue, kern, 2, nullptr, global, local, 0, nullptr, nullptr);
    std::vector<float> tmp(size_t(dw) * dh * 4);
    clEnqueueReadBuffer(queue, mDst, CL_TRUE, 0, dstBytes, tmp.data(), 0, nullptr, nullptr);
    for (int y = 0; y < dh; ++y) {
        std::memcpy(reinterpret_cast<uint8_t*>(dst.data) + y * dst.stride,
                    tmp.data() + size_t(y) * dw * 4,
                    size_t(dw) * 4 * sizeof(float));
    }

    clReleaseMemObject(mPlate);
    clReleaseMemObject(mDeg);
    clReleaseMemObject(mDst);
    clReleaseMemObject(mCurve);
}

}  // namespace kernels
}  // namespace dasgrain
