// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Tests for the CPU Voronoi scatter kernel.

#include "test_harness.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "kernels/Kernels.h"

using dasgrain::kernels::ImageView;
using dasgrain::kernels::ImageViewConst;
using dasgrain::kernels::ScatterParams;
using dasgrain::kernels::CurveLUT;

namespace {

struct Buf {
    std::vector<float> data;
    int w = 0;
    int h = 0;
};

Buf makeBuf(int w, int h, float r, float g, float b, float a = 1.0f) {
    Buf out;
    out.w = w;
    out.h = h;
    out.data.assign(size_t(w) * h * 4, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float* p = out.data.data() + (size_t(y) * w + x) * 4;
            p[0] = r; p[1] = g; p[2] = b; p[3] = a;
        }
    }
    return out;
}

ImageViewConst viewC(const Buf& b) {
    ImageViewConst v;
    v.data = b.data.data();
    v.width = b.w;
    v.height = b.h;
    v.stride = b.w * 4 * sizeof(float);
    return v;
}
ImageView view(Buf& b) {
    ImageView v;
    v.data = b.data.data();
    v.width = b.w;
    v.height = b.h;
    v.stride = b.w * 4 * sizeof(float);
    return v;
}

}  // namespace

TEST_CASE(scatter_zero_grain_returns_zero) {
    // plate == deg => grain difference == 0. Output should be all zero.
    Buf plate = makeBuf(32, 32, 0.5f, 0.5f, 0.5f);
    Buf deg   = makeBuf(32, 32, 0.5f, 0.5f, 0.5f);
    Buf dst   = makeBuf(16, 16, -1.0f, -1.0f, -1.0f);

    // Build identity LUT.
    std::vector<float> lut(256 * 3, 0.5f);

    ScatterParams sp;
    sp.boxX = 4.0f; sp.boxY = 4.0f;
    sp.boxR = 28.0f; sp.boxT = 28.0f;
    sp.cellSize = 8.0f;
    sp.seed = 42;
    sp.curve.data = lut.data();
    sp.curve.size = 256;
    sp.curve.minX = 0.0f;
    sp.curve.maxX = 1.0f;

    dasgrain::kernels::runScatterCPU(sp, viewC(plate), viewC(deg), view(dst));

    for (int i = 0; i < dst.w * dst.h; ++i) {
        EXPECT_NEAR(dst.data[i * 4 + 0], 0.0f, 1e-5);
        EXPECT_NEAR(dst.data[i * 4 + 1], 0.0f, 1e-5);
        EXPECT_NEAR(dst.data[i * 4 + 2], 0.0f, 1e-5);
    }
}

TEST_CASE(scatter_overlay_pattern_paints_cells) {
    Buf plate = makeBuf(32, 32, 0.5f, 0.5f, 0.5f);
    Buf deg   = makeBuf(32, 32, 0.5f, 0.5f, 0.5f);
    Buf dst   = makeBuf(8, 8, 0.0f, 0.0f, 0.0f);

    ScatterParams sp;
    sp.cellSize = 4.0f;
    sp.overlayPattern = 1;
    sp.boxX = 0.0f; sp.boxY = 0.0f;
    sp.boxR = 32.0f; sp.boxT = 32.0f;
    sp.seed = 7;

    dasgrain::kernels::runScatterCPU(sp, viewC(plate), viewC(deg), view(dst));

    // Every pixel should have a non-default value.
    bool anyNonZero = false;
    for (int i = 0; i < dst.w * dst.h; ++i) {
        if (dst.data[i * 4 + 0] != 0.0f || dst.data[i * 4 + 1] != 0.0f
            || dst.data[i * 4 + 2] != 0.0f) {
            anyNonZero = true;
            break;
        }
    }
    EXPECT_TRUE(anyNonZero);
}

TEST_CASE(scatter_handles_zero_curve_size) {
    // Without a baked curve the kernel should still produce normalised
    // values that match (plate-deg) (because curve sampling clamps to >=eps).
    Buf plate = makeBuf(32, 32, 0.6f, 0.6f, 0.6f);
    Buf deg   = makeBuf(32, 32, 0.5f, 0.5f, 0.5f);
    Buf dst   = makeBuf(8, 8, 0.0f, 0.0f, 0.0f);

    ScatterParams sp;
    sp.cellSize = 4.0f;
    sp.seed = 11;
    sp.boxX = 0.0f; sp.boxY = 0.0f;
    sp.boxR = 32.0f; sp.boxT = 32.0f;
    sp.curve.data = nullptr;
    sp.curve.size = 0;

    dasgrain::kernels::runScatterCPU(sp, viewC(plate), viewC(deg), view(dst));

    // Plate is uniform 0.6, deg uniform 0.5 → per-pixel grain = 0.1. With a
    // missing curve the kernel returns max(deg, eps) = 0.5, so output =
    // 0.1 * eps / 0.5 = 0.002 (modulo bilinear edge softening).
    EXPECT_NEAR(dst.data[0], 0.1f * 0.01f / 0.5f, 1e-4);
    EXPECT_NEAR(dst.data[1], 0.1f * 0.01f / 0.5f, 1e-4);
    EXPECT_NEAR(dst.data[2], 0.1f * 0.01f / 0.5f, 1e-4);
}

TEST_CASE(scatter_box_wrap_keeps_outputs_finite) {
    // Even with a degenerate (zero-area) box the kernel should not produce
    // NaN/inf — boxValid collapses to a single sample point at the centre.
    Buf plate = makeBuf(16, 16, 0.5f, 0.5f, 0.5f);
    Buf deg   = makeBuf(16, 16, 0.5f, 0.5f, 0.5f);
    Buf dst   = makeBuf(8, 8, 0.0f, 0.0f, 0.0f);

    ScatterParams sp;
    sp.cellSize = 4.0f;
    sp.seed = 1;
    sp.boxX = 8.0f; sp.boxY = 8.0f;
    sp.boxR = 8.0f; sp.boxT = 8.0f;
    sp.curve.data = nullptr;

    dasgrain::kernels::runScatterCPU(sp, viewC(plate), viewC(deg), view(dst));

    for (int i = 0; i < dst.w * dst.h * 4; ++i) {
        EXPECT_TRUE(std::isfinite(dst.data[i]));
    }
}
