// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Tests for the multi-frame Analyse pipeline. We synthesise tiny RGBA images
// in memory and feed them through the public Analyser surface so we can
// validate the curve construction without touching OFX.

#include "test_harness.h"

#include <cmath>
#include <random>

#include "core/Analyser.h"
#include "core/ColorSpace.h"
#include "core/ResponseCurve.h"

using dasgrain::AnalyseConfig;
using dasgrain::AnalyseResult;
using dasgrain::ClipKind;
using dasgrain::FrameImage;
using dasgrain::ResponseCurve;

namespace {

// Build a flat plate with a horizontal luma ramp and a "noise" signal added.
// The same ramp is used as the degrained image (so plate - degrained == noise).
void makeFrame(int w, int h, FrameImage& plate, FrameImage& deg,
               float noiseAmplitude, uint32_t seed) {
    plate.width = deg.width = w;
    plate.height = deg.height = h;
    plate.data.assign(size_t(w) * h * 4, 0.0f);
    deg.data.assign(size_t(w) * h * 4, 0.0f);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> noise(-1.0f, 1.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float t = (w == 1) ? 0.5f : float(x) / float(w - 1);
            const float v = t;  // linear ramp 0..1
            float* P = plate.data.data() + (size_t(y) * w + x) * 4;
            float* D = deg.data.data()   + (size_t(y) * w + x) * 4;
            const float n = noise(rng) * noiseAmplitude;
            P[0] = v + n;
            P[1] = v + n;
            P[2] = v + n;
            P[3] = 1.0f;
            D[0] = v;
            D[1] = v;
            D[2] = v;
            D[3] = 1.0f;
        }
    }
}

}  // namespace

TEST_CASE(analyser_compute_sample_frames_uses_frame_range) {
    AnalyseConfig cfg;
    cfg.numberOfFrames = 5;
    cfg.firstFrame     = 1000;
    cfg.lastFrame      = 1010;
    auto frames = dasgrain::computeSampleFrames(cfg);
    EXPECT_TRUE(frames.size() >= 5);
    EXPECT_TRUE(frames.front() >= 1000);
    EXPECT_TRUE(frames.back() <= 1010);
}

TEST_CASE(analyser_reduce_frame_stats_finds_min_max) {
    FrameImage plate, deg;
    makeFrame(/*w*/16, /*h*/16, plate, deg, /*noise*/0.0f, /*seed*/1);
    auto s = dasgrain::reduceFrameStats(deg);
    EXPECT_TRUE(s.any);
    // The degrained ramp goes 0..1 in linear; LogC of 0 is ~0.0928, LogC of
    // 1 is ~0.5709. Ranges should be roughly that, identical across channels.
    EXPECT_TRUE(s.minR < 0.2f);
    EXPECT_TRUE(s.maxR > 0.5f);
    EXPECT_NEAR(s.minR, s.minG, 1e-6);
    EXPECT_NEAR(s.maxB, s.maxG, 1e-6);
}

TEST_CASE(analyser_run_analyse_produces_valid_curve) {
    // Two synthetic frames with constant noise amplitude => the curve y
    // values should be roughly equal to the noise amplitude across buckets.
    FrameImage p1, d1, p2, d2;
    makeFrame(32, 32, p1, d1, /*noise*/0.05f, /*seed*/11);
    makeFrame(32, 32, p2, d2, /*noise*/0.05f, /*seed*/22);

    AnalyseConfig cfg;
    cfg.numberOfFrames = 2;
    cfg.sampleCount    = 8;
    cfg.firstFrame     = 1000;
    cfg.lastFrame      = 1001;
    std::vector<int> frames = {1000, 1001};

    auto fetch = [&](ClipKind kind, int frame, FrameImage& out) -> bool {
        const FrameImage& src = (frame == 1000)
            ? (kind == ClipKind::kPlate ? p1 : d1)
            : (kind == ClipKind::kPlate ? p2 : d2);
        out = src;
        return true;
    };

    AnalyseResult r = dasgrain::runAnalyse(cfg, frames, fetch);
    EXPECT_TRUE(r.ok);
    EXPECT_TRUE(r.curve.isValid());
    // Sample mid-range; expect the mean |grain| to be on the order of
    // noiseAmplitude / 2 (uniform [-A,A] => E|X| = A/2).
    const double y = r.curve.sample(0, 0.5);
    EXPECT_TRUE(y > 0.005);
    EXPECT_TRUE(y < 0.10);
}

TEST_CASE(analyser_aborts_when_progress_returns_false) {
    FrameImage p1, d1;
    makeFrame(8, 8, p1, d1, /*noise*/0.02f, /*seed*/3);

    AnalyseConfig cfg;
    cfg.numberOfFrames = 1;
    cfg.sampleCount    = 4;
    std::vector<int> frames = {1000};
    auto fetch = [&](ClipKind kind, int /*frame*/, FrameImage& out) -> bool {
        out = (kind == ClipKind::kPlate) ? p1 : d1;
        return true;
    };

    auto progress = [](double) { return false; };
    AnalyseResult r = dasgrain::runAnalyse(cfg, frames, fetch, progress);
    EXPECT_TRUE(!r.ok);
    EXPECT_TRUE(r.error == "aborted");
}

TEST_CASE(analyser_propagates_fetch_failures) {
    AnalyseConfig cfg;
    cfg.numberOfFrames = 1;
    cfg.sampleCount    = 4;
    std::vector<int> frames = {1000};
    auto fetch = [](ClipKind, int, FrameImage&) { return false; };
    AnalyseResult r = dasgrain::runAnalyse(cfg, frames, fetch);
    EXPECT_TRUE(!r.ok);
    EXPECT_TRUE(!r.error.empty());
}
