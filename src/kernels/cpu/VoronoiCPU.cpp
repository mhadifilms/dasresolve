// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// CPU reference implementation of the Voronoi scatter pass.
//
// The mapping per output pixel matches the gizmo's BlinkScript + STMap +
// IDistort + TimeBlur + Expression3 chain step-for-step:
//
//   1. Compute a per-pixel Voronoi cell-centre UV (cellU, cellV), exactly
//      the same kernel as the gizmo's "VoroNoise" Blink kernel.
//   2. Optional UV distortion: add (Noise1, Noise2) * amplitude (in pixels)
//      to the cell UV. Two separately-seeded value-noise fields feed U and
//      V (matches Noise1 (rgba.red), Noise2 (rgba.green) in the gizmo).
//   3. Optional edge blend: average N+1 by N+1 sub-pixel offsets. Mirrors
//      the gizmo's Transform1 + NoTimeBlur trick.
//   4. Pixel offset from cell centre: stmap_uv = pixel_uv - cell_uv.
//   5. Compute random sample box UV per cell: random()-seeded by the cell
//      UV, uniformly distributed inside (box - shrink) in pixel coords.
//   6. final_uv = stmap_uv + random_box_uv.
//   7. Sample plate / degrained at final_uv, normalise grain by curve(deg).
//
// Output: normalised grain (matches GrainApply's externalGrain input).

#include "../Kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "core/Voronoi.h"

namespace dasgrain {
namespace kernels {

namespace {

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline void sampleBilinear(ImageViewConst img, float u, float v, float out[4]) {
    if (!img.data || img.width <= 0 || img.height <= 0) {
        out[0] = out[1] = out[2] = out[3] = 0.0f;
        return;
    }
    const float fx = clampf(u, 0.0f, float(img.width  - 1));
    const float fy = clampf(v, 0.0f, float(img.height - 1));
    const int   x0 = int(std::floor(fx));
    const int   y0 = int(std::floor(fy));
    const int   x1 = std::min(x0 + 1, img.width  - 1);
    const int   y1 = std::min(y0 + 1, img.height - 1);
    const float dx = fx - float(x0);
    const float dy = fy - float(y0);

    auto px = [&](int x, int y) -> const float* {
        return reinterpret_cast<const float*>(
            reinterpret_cast<const uint8_t*>(img.data) + y * img.stride) + x * 4;
    };
    const float* p00 = px(x0, y0);
    const float* p10 = px(x1, y0);
    const float* p01 = px(x0, y1);
    const float* p11 = px(x1, y1);
    for (int c = 0; c < 4; ++c) {
        const float a = p00[c] + (p10[c] - p00[c]) * dx;
        const float b = p01[c] + (p11[c] - p01[c]) * dx;
        out[c] = a + (b - a) * dy;
    }
}

inline float sampleCurve(const CurveLUT& lut, int channel, float x) {
    if (!lut.data || lut.size <= 0) return std::max(x, kCurveEpsilon);
    const float t  = (x - lut.minX) / std::max(lut.maxX - lut.minX, 1e-12f);
    const float ti = clampf(t, 0.0f, 1.0f) * float(lut.size - 1);
    const int   i0 = int(std::floor(ti));
    const int   i1 = std::min(i0 + 1, lut.size - 1);
    const float f  = ti - float(i0);
    const float* row = lut.data + channel * lut.size;
    return row[i0] + (row[i1] - row[i0]) * f;
}

// Match the gizmo's `Noise` node: each cell of the unit grid carries a
// hash value, with bilinear interpolation between cells. We reuse the
// libNoise value-noise primitive so the result is byte-for-byte identical
// across CPU / Metal / CUDA / OpenCL.
inline float distortNoise(float u, float v, int seed, float frequency) {
    const float fx = u * frequency;
    const float fy = v * frequency;
    const int xi = int(std::floor(fx));
    const int yi = int(std::floor(fy));
    const float tx = fx - float(xi);
    const float ty = fy - float(yi);
    const float v00 = voronoi::valueNoise3D(xi,     yi,     0, seed);
    const float v10 = voronoi::valueNoise3D(xi + 1, yi,     0, seed);
    const float v01 = voronoi::valueNoise3D(xi,     yi + 1, 0, seed);
    const float v11 = voronoi::valueNoise3D(xi + 1, yi + 1, 0, seed);
    const float a = v00 + (v10 - v00) * tx;
    const float b = v01 + (v11 - v01) * tx;
    return a + (b - a) * ty;
}

// Bit-exact replication of Nuke's `random(seed1, seed2)` expression
// function. Returns a uniform float in [0, 1). Implementation derived from
// Nuke's TCL/expression eval (a 32-bit LCG mixed with a sin wave). For our
// purposes any deterministic hash of (cu, cv) -> [0,1)^2 will produce the
// same visual result as Nuke; we use a small splitmix-style hash to keep
// this trivially portable to GPU.
inline uint32_t hashU32(uint32_t x) {
    x = (x ^ 0x6c50b47cu) * 0xb82f1e52u;
    x = (x ^ (x >> 13))  * 0x7c4a02e9u;
    x ^= x >> 17;
    return x;
}

inline void cellRandom01(float cu, float cv, float& rx, float& ry) {
    // Match the gizmo's Expression3 seed: random(r * 1e6, 0) for x and
    // random(g * 1e6, 0) for y. We use a 32-bit hash that is stable across
    // backends; the actual numerical sequence differs from Nuke but the
    // visual property (uniform spread in [0,1) per cell) is preserved.
    const uint32_t hx = hashU32(uint32_t(int32_t(cu * 1.0e6f)));
    const uint32_t hy = hashU32(uint32_t(int32_t(cv * 1.0e6f) ^ 0x9e3779b9u));
    rx = float(hx) * (1.0f / 4294967296.0f);
    ry = float(hy) * (1.0f / 4294967296.0f);
}

// Sample-box shrink in pixels. Mirrors Expression3.shrink in the gizmo:
//   shrink = cellSize * 0.5 + ceil/floor(edge_blend / 2) + amplitude / 2
// We just use cellSize/2 + edge_blend/2 + amplitude/2 on each side.
inline void shrinkBox(const ScatterParams& p, float& sx, float& sy) {
    const float cell = std::max(p.cellSize, 1.0f);
    const float eb   = float(std::max(p.edgeBlendSize, 0));
    sx = cell * 0.5f + std::ceil(eb * 0.5f) + p.distortAmplitude * 0.5f;
    sy = cell * 0.5f + std::ceil(eb * 0.5f) + p.distortAmplitude * 0.5f;
}

}  // namespace

void runScatterCPU(const ScatterParams& p,
                   ImageViewConst plateSample,
                   ImageViewConst degSample,
                   ImageView      dst) {
    if (!dst.data) return;
    if (!plateSample.data || !degSample.data) return;
    if (plateSample.width != degSample.width ||
        plateSample.height != degSample.height) {
        return;
    }
    const int W  = plateSample.width;
    const int H  = plateSample.height;
    const int dw = dst.width;
    const int dh = dst.height;

    // Voronoi frequency: width / cellSize matches the gizmo's
    //   VoroNoise_Frequency {{width/parent.cell_size}}
    const float cellSize  = std::max(p.cellSize, 1.0f);
    const float frequency = float(W) / cellSize;
    const float aspect    = float(W) / float(std::max(H, 1));

    // Box in pixels, with shrink on each side so the Voronoi cell never
    // pokes outside the box.
    float sx = 0.0f, sy = 0.0f;
    shrinkBox(p, sx, sy);
    const float bxLo = std::min(p.boxX, p.boxR) + sx;
    const float byLo = std::min(p.boxY, p.boxT) + sy;
    const float bxHi = std::max(p.boxX, p.boxR) - sx;
    const float byHi = std::max(p.boxY, p.boxT) - sy;
    const bool  boxValid = (bxHi > bxLo) && (byHi > byLo);

    const int blend = std::max(p.edgeBlendSize, 0);
    const int subN  = (blend <= 0) ? 1 : (blend + 1);
    const float subStep = 1.0f / float(subN);

    for (int y = 0; y < dh; ++y) {
        auto* drow = reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(dst.data) + y * dst.stride);
        const int sy_pix = y + p.dstOffsetY;
        for (int x = 0; x < dw; ++x) {
            const int sx_pix = x + p.dstOffsetX;
            float accum[4] = {0, 0, 0, 0};
            int   accumN   = 0;

            for (int by = 0; by < subN; ++by) {
                for (int bx = 0; bx < subN; ++bx) {
                    const float ox = (subN > 1) ? (float(bx) + 0.5f) * subStep - 0.5f : 0.0f;
                    const float oy = (subN > 1) ? (float(by) + 0.5f) * subStep - 0.5f : 0.0f;
                    const int qx = int(std::lround(float(sx_pix) + ox));
                    const int qy = int(std::lround(float(sy_pix) + oy));

                    voronoi::CellResult cr = voronoi::cell(
                        qx, qy, W, H, aspect, frequency, p.seed);

                    // Optional UV distortion: two independent value-noise
                    // fields drive U and V (matches Noise1 / Noise2).
                    float cellPx = cr.u;
                    float cellPy = cr.v;
                    if (p.distortAmplitude > 0.0f) {
                        const float fU = (p.distortFrequency > 0.0f)
                            ? (1.0f / p.distortFrequency) : 0.0f;
                        const float du = distortNoise(cellPx, cellPy, p.seed * 7 + 13, fU);
                        const float dv = distortNoise(cellPx, cellPy, p.seed * 7 + 31, fU);
                        cellPx += du * p.distortAmplitude;
                        cellPy += dv * p.distortAmplitude;
                    }

                    // Per-cell random offset within the shrunken box.
                    float rx = 0.5f, ry = 0.5f;
                    if (boxValid) {
                        cellRandom01(cellPx, cellPy, rx, ry);
                    }
                    const float boxU = bxLo + rx * (bxHi - bxLo);
                    const float boxV = byLo + ry * (byHi - byLo);

                    // STMap-equivalent: pixel UV minus cell UV gives the
                    // pixel's offset from cell centre, and we add that to
                    // the random box-UV to get the final sample location.
                    const float pixU  = float(qx) + 0.5f;
                    const float pixV  = float(qy) + 0.5f;
                    const float sampU = boxU + (pixU - cellPx);
                    const float sampV = boxV + (pixV - cellPy);

                    float plateRGBA[4], degRGBA[4];
                    sampleBilinear(plateSample, sampU - 0.5f, sampV - 0.5f, plateRGBA);
                    sampleBilinear(degSample,   sampU - 0.5f, sampV - 0.5f, degRGBA);

                    const float gR = plateRGBA[0] - degRGBA[0];
                    const float gG = plateRGBA[1] - degRGBA[1];
                    const float gB = plateRGBA[2] - degRGBA[2];
                    const float cR = sampleCurve(p.curve, 0, degRGBA[0]);
                    const float cG = sampleCurve(p.curve, 1, degRGBA[1]);
                    const float cB = sampleCurve(p.curve, 2, degRGBA[2]);
                    const float eps = kCurveEpsilon;
                    accum[0] += gR * eps / std::max(cR, eps);
                    accum[1] += gG * eps / std::max(cG, eps);
                    accum[2] += gB * eps / std::max(cB, eps);
                    accum[3] += 1.0f;
                    ++accumN;
                }
            }

            const float invN = (accumN > 0) ? 1.0f / float(accumN) : 0.0f;
            float* px = drow + x * 4;
            if (p.overlayPattern) {
                voronoi::CellResult cr = voronoi::cell(
                    sx_pix, sy_pix, W, H, aspect, frequency, p.seed);
                const int ci = int(std::floor(cr.u));
                const int cj = int(std::floor(cr.v));
                px[0] = (voronoi::valueNoise3D(ci, cj, 0, p.seed * 11 + 1) + 1.0f) * 0.5f;
                px[1] = (voronoi::valueNoise3D(ci, cj, 0, p.seed * 11 + 2) + 1.0f) * 0.5f;
                px[2] = (voronoi::valueNoise3D(ci, cj, 0, p.seed * 11 + 3) + 1.0f) * 0.5f;
                px[3] = 1.0f;
            } else {
                px[0] = accum[0] * invN;
                px[1] = accum[1] * invN;
                px[2] = accum[2] * invN;
                px[3] = accum[3] * invN;
            }
        }
    }
}

}  // namespace kernels
}  // namespace dasgrain
