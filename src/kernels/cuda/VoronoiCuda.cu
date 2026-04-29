// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// CUDA Voronoi scatter kernel. Mirrors the CPU reference in
// src/kernels/cpu/VoronoiCPU.cpp byte-for-byte: same Voronoi cell, same
// per-cell random-offset-within-shrunken-box, same UV distortion, same
// edge-blend trick.

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#include "../Kernels.h"

namespace dasgrain {
namespace kernels {

namespace {

struct ScatterParamsGPU {
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

__device__ inline int dgIntValueNoise3D(int x, int y, int z, int seed) {
    constexpr int kXNoise = 1619, kYNoise = 31337, kZNoise = 6971, kSeedNoise = 1013;
    unsigned int n = (unsigned int)((kXNoise * x + kYNoise * y + kZNoise * z + kSeedNoise * seed) & 0x7fffffff);
    n = (n >> 13) ^ n;
    return (int)((n * (n * n * 60493u + 19990303u) + 1376312589u) & 0x7fffffffu);
}
__device__ inline float dgValueNoise3D(int x, int y, int z, int seed) {
    return 1.0f - (float(dgIntValueNoise3D(x, y, z, seed)) / 1073741824.0f);
}
__device__ inline void dgCellAt(int px, int py, int W, int H, float aspect, float frequency, int seed,
                                float& outU, float& outV) {
    const float x = float(px) * aspect * frequency / float(W);
    const float y = float(py) * frequency / float(W);
    const int xInt = (x > 0.0f) ? int(x) : int(x) - 1;
    const int yInt = (y > 0.0f) ? int(y) : int(y) - 1;
    float minDist = 2147483647.0f;
    float xCand = 0.0f, yCand = 0.0f;
    const float randomness = 0.5f;
    for (int yCur = yInt - 2; yCur <= yInt + 2; ++yCur) {
        for (int xCur = xInt - 2; xCur <= xInt + 2; ++xCur) {
            const float xPos = float(xCur) + (dgValueNoise3D(xCur, yCur, 0, seed)     + 1.0f) * randomness + (1.0f - randomness) - 1.0f;
            const float yPos = float(yCur) + (dgValueNoise3D(xCur, yCur, 0, seed + 1) + 1.0f) * randomness + (1.0f - randomness) - 1.0f;
            const float dx = xPos - x, dy = yPos - y;
            const float d = dx * dx + dy * dy;
            if (d < minDist) { minDist = d; xCand = xPos; yCand = yPos; }
        }
    }
    outU = xCand / aspect / frequency;
    outV = yCand / float(H) * float(W) / frequency;
}
__device__ inline float dgClampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
__device__ inline void dgFetchPx(const float* img, int rowFloats, int W, int H,
                                 int x, int y, float out[4]) {
    if (x < 0) x = 0; if (x > W - 1) x = W - 1;
    if (y < 0) y = 0; if (y > H - 1) y = H - 1;
    const float* p = img + y * rowFloats + x * 4;
    out[0] = p[0]; out[1] = p[1]; out[2] = p[2]; out[3] = p[3];
}
__device__ inline void dgSampleBilinear(const float* img, int rowFloats, int W, int H,
                                        float u, float v, float out[4]) {
    const float fx = dgClampf(u, 0.0f, float(W - 1));
    const float fy = dgClampf(v, 0.0f, float(H - 1));
    const int   x0 = int(floorf(fx));
    const int   y0 = int(floorf(fy));
    const int   x1 = x0 + 1 < W ? x0 + 1 : W - 1;
    const int   y1 = y0 + 1 < H ? y0 + 1 : H - 1;
    const float dx = fx - float(x0);
    const float dy = fy - float(y0);
    float p00[4], p10[4], p01[4], p11[4];
    dgFetchPx(img, rowFloats, W, H, x0, y0, p00);
    dgFetchPx(img, rowFloats, W, H, x1, y0, p10);
    dgFetchPx(img, rowFloats, W, H, x0, y1, p01);
    dgFetchPx(img, rowFloats, W, H, x1, y1, p11);
    for (int c = 0; c < 4; ++c) {
        const float a = p00[c] + (p10[c] - p00[c]) * dx;
        const float b = p01[c] + (p11[c] - p01[c]) * dx;
        out[c] = a + (b - a) * dy;
    }
}
__device__ inline float dgSampleCurve(const float* lut, int size,
                                      float minX, float maxX,
                                      int channel, float v) {
    if (size <= 0) return v > 0.01f ? v : 0.01f;
    const float t  = (v - minX) / fmaxf(maxX - minX, 1e-12f);
    const float ti = dgClampf(t, 0.0f, 1.0f) * float(size - 1);
    const int   i0 = int(floorf(ti));
    const int   i1 = i0 + 1 < size ? i0 + 1 : size - 1;
    const float f  = ti - float(i0);
    const float* row = lut + channel * size;
    return row[i0] + (row[i1] - row[i0]) * f;
}
__device__ inline float dgDistortNoise(float u, float v, int seed, float freq) {
    const float fx = u * freq;
    const float fy = v * freq;
    const int xi = int(floorf(fx));
    const int yi = int(floorf(fy));
    const float tx = fx - float(xi);
    const float ty = fy - float(yi);
    const float v00 = dgValueNoise3D(xi,     yi,     0, seed);
    const float v10 = dgValueNoise3D(xi + 1, yi,     0, seed);
    const float v01 = dgValueNoise3D(xi,     yi + 1, 0, seed);
    const float v11 = dgValueNoise3D(xi + 1, yi + 1, 0, seed);
    const float a = v00 + (v10 - v00) * tx;
    const float b = v01 + (v11 - v01) * tx;
    return a + (b - a) * ty;
}
__device__ inline unsigned int dgHashU32(unsigned int x) {
    x = (x ^ 0x6c50b47cu) * 0xb82f1e52u;
    x = (x ^ (x >> 13))  * 0x7c4a02e9u;
    x ^= x >> 17;
    return x;
}
__device__ inline void dgCellRandom01(float cu, float cv, float& rx, float& ry) {
    unsigned int hx = dgHashU32((unsigned int)(int)(cu * 1.0e6f));
    unsigned int hy = dgHashU32((unsigned int)(int)(cv * 1.0e6f) ^ 0x9e3779b9u);
    rx = float(hx) * (1.0f / 4294967296.0f);
    ry = float(hy) * (1.0f / 4294967296.0f);
}

__global__ void scatterKernel(const float* __restrict__ plate,
                              const float* __restrict__ deg,
                              float*       __restrict__ dst,
                              const float* __restrict__ curve,
                              ScatterParamsGPU p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.dstW || y >= p.dstH) return;

    const int W = p.plateW;
    const int H = p.plateH;
    const float cellSize  = p.cellSize > 1.0f ? p.cellSize : 1.0f;
    const float frequency = float(W) / cellSize;
    const float aspect    = float(W) / float(H > 0 ? H : 1);

    const float bxRaw0 = fminf(p.boxX, p.boxR);
    const float byRaw0 = fminf(p.boxY, p.boxT);
    const float bxRaw1 = fmaxf(p.boxX, p.boxR);
    const float byRaw1 = fmaxf(p.boxY, p.boxT);
    const float shrink = cellSize * 0.5f
                       + ceilf(float(p.edgeBlendSize > 0 ? p.edgeBlendSize : 0) * 0.5f)
                       + p.distortAmplitude * 0.5f;
    const float bxLo = bxRaw0 + shrink;
    const float byLo = byRaw0 + shrink;
    const float bxHi = bxRaw1 - shrink;
    const float byHi = byRaw1 - shrink;
    const bool  boxValid = (bxHi > bxLo) && (byHi > byLo);

    const int blend = p.edgeBlendSize > 0 ? p.edgeBlendSize : 0;
    const int subN  = (blend <= 0) ? 1 : (blend + 1);
    const float subStep = 1.0f / float(subN);
    const int sx_pix = x + p.dstOffsetX;
    const int sy_pix = y + p.dstOffsetY;

    float accum[4] = {0, 0, 0, 0};
    int   accumN = 0;
    for (int by = 0; by < subN; ++by) {
        for (int bx = 0; bx < subN; ++bx) {
            const float ox = (subN > 1) ? (float(bx) + 0.5f) * subStep - 0.5f : 0.0f;
            const float oy = (subN > 1) ? (float(by) + 0.5f) * subStep - 0.5f : 0.0f;
            const int qx = int(roundf(float(sx_pix) + ox));
            const int qy = int(roundf(float(sy_pix) + oy));
            float cellPx, cellPy;
            dgCellAt(qx, qy, W, H, aspect, frequency, p.seed, cellPx, cellPy);
            if (p.distortAmplitude > 0.0f) {
                const float fU = (p.distortFrequency > 0.0f) ? 1.0f / p.distortFrequency : 0.0f;
                const float du = dgDistortNoise(cellPx, cellPy, p.seed * 7 + 13, fU);
                const float dv = dgDistortNoise(cellPx, cellPy, p.seed * 7 + 31, fU);
                cellPx += du * p.distortAmplitude;
                cellPy += dv * p.distortAmplitude;
            }
            float rx, ry;
            if (boxValid) {
                dgCellRandom01(cellPx, cellPy, rx, ry);
            } else {
                rx = 0.5f; ry = 0.5f;
            }
            const float boxU = bxLo + rx * (bxHi - bxLo);
            const float boxV = byLo + ry * (byHi - byLo);
            const float pixU = float(qx) + 0.5f;
            const float pixV = float(qy) + 0.5f;
            const float sampU = boxU + (pixU - cellPx);
            const float sampV = boxV + (pixV - cellPy);

            float plateS[4], degS[4];
            dgSampleBilinear(plate, p.plateRowStride, W, H, sampU - 0.5f, sampV - 0.5f, plateS);
            dgSampleBilinear(deg,   p.degRowStride,   W, H, sampU - 0.5f, sampV - 0.5f, degS);
            const float gR = plateS[0] - degS[0];
            const float gG = plateS[1] - degS[1];
            const float gB = plateS[2] - degS[2];
            const float cR = dgSampleCurve(curve, p.curveSize, p.curveMinX, p.curveMaxX, 0, degS[0]);
            const float cG = dgSampleCurve(curve, p.curveSize, p.curveMinX, p.curveMaxX, 1, degS[1]);
            const float cB = dgSampleCurve(curve, p.curveSize, p.curveMinX, p.curveMaxX, 2, degS[2]);
            const float eps = 0.01f;
            accum[0] += gR * eps / fmaxf(cR, eps);
            accum[1] += gG * eps / fmaxf(cG, eps);
            accum[2] += gB * eps / fmaxf(cB, eps);
            accum[3] += 1.0f;
            ++accumN;
        }
    }
    const float invN = (accumN > 0) ? 1.0f / float(accumN) : 0.0f;
    float* outPx = dst + y * p.dstRowStride + x * 4;
    if (p.overlayPattern != 0) {
        float cu, cv;
        dgCellAt(sx_pix, sy_pix, W, H, aspect, frequency, p.seed, cu, cv);
        const int ci = int(floorf(cu));
        const int cj = int(floorf(cv));
        outPx[0] = (dgValueNoise3D(ci, cj, 0, p.seed * 11 + 1) + 1.0f) * 0.5f;
        outPx[1] = (dgValueNoise3D(ci, cj, 0, p.seed * 11 + 2) + 1.0f) * 0.5f;
        outPx[2] = (dgValueNoise3D(ci, cj, 0, p.seed * 11 + 3) + 1.0f) * 0.5f;
        outPx[3] = 1.0f;
    } else {
        outPx[0] = accum[0] * invN;
        outPx[1] = accum[1] * invN;
        outPx[2] = accum[2] * invN;
        outPx[3] = accum[3] * invN;
    }
}

}  // namespace

void runScatterCuda(void* cudaStreamPtr,
                    const ScatterParams& p,
                    ImageViewConst plateSample,
                    ImageViewConst degSample,
                    ImageView      dst) {
    if (!plateSample.data || !degSample.data || !dst.data) return;
    const int W  = plateSample.width;
    const int H  = plateSample.height;
    const int dw = dst.width;
    const int dh = dst.height;
    if (W <= 0 || H <= 0 || dw <= 0 || dh <= 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cudaStreamPtr);
    const size_t plateBytes = size_t(W) * H * 4 * sizeof(float);
    const size_t dstBytes   = size_t(dw) * dh * 4 * sizeof(float);
    const int curveLen      = (p.curve.size > 0 ? p.curve.size : 0) * 3;
    const size_t curveBytes = size_t(curveLen > 0 ? curveLen : 1) * sizeof(float);

    float *dPlate = nullptr, *dDeg = nullptr, *dDst = nullptr, *dCurve = nullptr;
    cudaMalloc(&dPlate, plateBytes);
    cudaMalloc(&dDeg,   plateBytes);
    cudaMalloc(&dDst,   dstBytes);
    cudaMalloc(&dCurve, curveBytes);

    cudaMemcpyAsync(dPlate, plateSample.data, plateBytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dDeg,   degSample.data,   plateBytes, cudaMemcpyHostToDevice, stream);
    if (p.curve.data && p.curve.size > 0) {
        cudaMemcpyAsync(dCurve, p.curve.data, curveBytes, cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemsetAsync(dCurve, 0, curveBytes, stream);
    }

    ScatterParamsGPU pp;
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

    dim3 block(16, 16);
    dim3 grid((dw + 15) / 16, (dh + 15) / 16);
    scatterKernel<<<grid, block, 0, stream>>>(dPlate, dDeg, dDst, dCurve, pp);
    cudaMemcpyAsync(dst.data, dDst, dstBytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (dst.stride != int32_t(dw * 4 * sizeof(float))) {
        for (int y = dh - 1; y >= 0; --y) {
            const float* src = static_cast<const float*>(dst.data) + size_t(y) * dw * 4;
            float* dstRow = reinterpret_cast<float*>(
                reinterpret_cast<uint8_t*>(dst.data) + y * dst.stride);
            std::memmove(dstRow, src, size_t(dw) * 4 * sizeof(float));
        }
    }

    cudaFree(dPlate);
    cudaFree(dDeg);
    cudaFree(dDst);
    cudaFree(dCurve);
}

}  // namespace kernels
}  // namespace dasgrain
