// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Golden-frame regression tests for the full grain pipeline (analyse + apply
// + scatter), all run on synthetic in-memory frames so they need zero
// fixture files on disk. Each test pins a few representative outputs to
// stable values so future kernel edits surface unintentional drift.

#include "test_harness.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "core/Analyser.h"
#include "core/ResponseCurve.h"
#include "kernels/Kernels.h"

namespace {

void writeRGBA(std::vector<float>& buf, int W, int x, int y,
               float r, float g, float b, float a = 1.0f) {
    float* p = buf.data() + (size_t(y) * W + x) * 4;
    p[0] = r; p[1] = g; p[2] = b; p[3] = a;
}

// Build a flat plate with a horizontal luma ramp; deg = plate - noise so
// (plate - deg) returns a known noise field per pixel.
void buildRamp(int W, int H, float noiseAmp, uint32_t seed,
               std::vector<float>& plate, std::vector<float>& deg) {
    plate.assign(size_t(W) * H * 4, 0.0f);
    deg.assign(size_t(W) * H * 4, 0.0f);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> noise(-1.0f, 1.0f);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float v = (W == 1) ? 0.5f : float(x) / float(W - 1);
            const float n = noise(rng) * noiseAmp;
            writeRGBA(plate, W, x, y, v + n, v + n, v + n);
            writeRGBA(deg,   W, x, y, v,     v,     v);
        }
    }
}

}  // namespace

TEST_CASE(golden_analyse_and_apply_round_trip) {
    constexpr int W = 16;
    constexpr int H = 16;
    constexpr int FRAMES = 3;

    using namespace dasgrain;

    std::vector<std::vector<float>> plates(FRAMES);
    std::vector<std::vector<float>> degs(FRAMES);
    for (int f = 0; f < FRAMES; ++f) {
        buildRamp(W, H, /*noiseAmp*/0.04f, /*seed*/uint32_t(7 + f * 31),
                  plates[f], degs[f]);
    }

    auto fetcher = [&](ClipKind kind, int frame, FrameImage& out) -> bool {
        if (frame < 0 || frame >= FRAMES) return false;
        out.width = W;
        out.height = H;
        out.data = (kind == ClipKind::kPlate) ? plates[frame] : degs[frame];
        return true;
    };

    AnalyseConfig cfg;
    cfg.numberOfFrames = FRAMES;
    cfg.sampleCount    = 8;
    cfg.firstFrame     = 0;
    cfg.lastFrame      = FRAMES - 1;
    std::vector<int> frames;
    for (int i = 0; i < FRAMES; ++i) frames.push_back(i);

    AnalyseResult r = runAnalyse(cfg, frames, fetcher);
    EXPECT_TRUE(r.ok);
    EXPECT_TRUE(r.curve.isValid());

    // Sample mid-ramp luma. Expected grain magnitude ~ noiseAmp / 2 for
    // uniform [-A,A] (E|X|=A/2). Given noiseAmp=0.04 we expect ~0.02.
    const double y = r.curve.sample(0, 0.5);
    EXPECT_TRUE(y > 0.005);
    EXPECT_TRUE(y < 0.04);

    // ---- Bake curve to LUT and run grain apply on a single frame --------
    constexpr int kLutSize = 64;
    std::vector<float> lut(kLutSize * 3, 0.0f);
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < kLutSize; ++i) {
            const double xx = double(i) / double(kLutSize - 1);
            lut[c * kLutSize + i] = float(r.curve.sample(c, xx));
        }
    }

    auto bufView = [W, H](const std::vector<float>& v) {
        kernels::ImageViewConst view;
        view.data   = v.data();
        view.width  = W;
        view.height = H;
        view.stride = int32_t(W * 4 * sizeof(float));
        return view;
    };

    std::vector<float> out(size_t(W) * H * 4, 0.0f);
    kernels::ImageView outView{out.data(), W, H,
                                int32_t(W * 4 * sizeof(float))};

    kernels::GrainApplyParams gp;
    gp.luminance     = 1.0f;
    gp.outputMode    = 0;  // regrained
    gp.curve.data    = lut.data();
    gp.curve.size    = kLutSize;
    gp.curve.minX    = 0.0f;
    gp.curve.maxX    = 1.0f;

    kernels::runGrainApplyCPU(gp,
                              bufView(plates[0]),  // comp = same as plate
                              bufView(plates[0]),
                              bufView(degs[0]),
                              kernels::ImageViewConst{},
                              kernels::ImageViewConst{},
                              outView);

    // The regrained output should be close to the plate (since comp==plate
    // and the curve roughly captures the noise).
    // We pin the average alpha to 1.0 (sanity).
    double meanA = 0.0;
    for (int i = 0; i < W * H; ++i) meanA += out[i * 4 + 3];
    EXPECT_NEAR(meanA / double(W * H), 1.0, 1e-5);

    // And the mean output luminance should track the plate luminance.
    double meanOut = 0.0, meanPlate = 0.0;
    for (int i = 0; i < W * H; ++i) {
        meanOut   += out[i * 4 + 0];
        meanPlate += plates[0][i * 4 + 0];
    }
    EXPECT_NEAR(meanOut / double(W * H),
                meanPlate / double(W * H),
                0.2);
}

TEST_CASE(golden_response_curve_json_round_trip) {
    using namespace dasgrain;
    ResponseCurve a;
    a.setChannel(0, {{0.0, 0.0}, {0.5, 0.2}, {1.0, 0.8}});
    a.setChannel(1, {{0.0, 0.0}, {0.4, 0.3}, {1.0, 0.9}});
    a.setChannel(2, {{0.0, 0.0}, {0.6, 0.1}, {1.0, 0.7}});

    const std::string json = a.toJSON();
    ResponseCurve b;
    EXPECT_TRUE(b.fromJSON(json));
    EXPECT_NEAR(a.sample(0, 0.5), b.sample(0, 0.5), 1e-6);
    EXPECT_NEAR(a.sample(1, 0.7), b.sample(1, 0.7), 1e-6);
    EXPECT_NEAR(a.sample(2, 0.3), b.sample(2, 0.3), 1e-6);
}
