// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Analyser: drives the multi-frame Analyse render pass. The work happens in
// two CPU reductions; on Apple Silicon and modern x86 this fits comfortably
// in the Render call's time budget. The plan reserves a GPU-accelerated
// reduction for future work — the surface here is GPU-friendly so we can
// drop in a Metal / CUDA / OpenCL implementation later without touching the
// plugin layer.
//
// The reductions operate on AlexaV3LogC luma values:
//   1. min/max of clean (= degrained) luma per channel across all frames
//   2. per-luminance-bucket mean of |plate - degrained| per channel

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "ResponseCurve.h"

namespace dasgrain {

struct AnalyseConfig {
    int    numberOfFrames   = 10;
    int    sampleCount      = 20;
    double luminance        = 1.0;   // luminance compensation amount
    int    firstFrame       = 0;
    int    lastFrame        = 0;
    std::string additionalFrames;

    // Optional analyse-mask routing: when `useMask` is true, only pixels
    // where the mask alpha is > 0 contribute to the bucket accumulation.
    // `invertMask` flips the test (matches the gizmo's "invert" toggle).
    bool   useMask          = false;
    bool   invertMask       = false;
};

struct AnalyseResult {
    ResponseCurve curve;
    bool          ok    = false;
    std::string   error;
};

// Compute the union sampling-frame list for the analysis pass.
std::vector<int> computeSampleFrames(const AnalyseConfig& cfg);

// ---------------------------------------------------------------------------
// Per-frame image accessor. The OFX layer wraps fetchImage / pixel data in
// this. `out` is filled with interleaved RGBA float32 (row-major, no
// padding) of size `width * height * 4`. Returns false on fetch failure.
struct FrameImage {
    std::vector<float> data;     // RGBA float32
    int width  = 0;
    int height = 0;
};
enum class ClipKind { kPlate, kDegrained, kMask };
using FrameFetcher = std::function<bool(ClipKind kind, int frame, FrameImage& out)>;

// Stage 1: per-channel (min, max) of degrained LogC luma across all frames.
struct FrameStats {
    float minR = 1e30f, maxR = -1e30f;
    float minG = 1e30f, maxG = -1e30f;
    float minB = 1e30f, maxB = -1e30f;
    bool  any  = false;
};
FrameStats reduceFrameStats(FrameImage const& degrained);

// Combine `b` into `a` in-place.
void mergeStats(FrameStats& a, const FrameStats& b);

// Stage 2: per-bucket sum-of-|grain| and pixel-count, for ONE frame, per
// channel. Output buckets are [bucketCount * 3] = [R0..R_{n-1}, G0.., B0..]
// for both `sums` and `counts`.
struct Histogram {
    std::vector<double> sums;
    std::vector<double> counts;
    int bucketCount = 0;
};

void accumulateHistogram(FrameImage const& plate,
                         FrameImage const& degrained,
                         const FrameStats& stats,
                         Histogram& hist,
                         FrameImage const* mask = nullptr,
                         bool invertMask = false);

// Stage 3: build the final ResponseCurve from accumulated histograms. We
// emit `bucketCount` control points per channel, each placed at the linear
// luminance corresponding to the bucket centre in LogC space.
ResponseCurve buildCurve(const Histogram& hist,
                         const FrameStats& stats);

// Convenience top-level driver. Pulls each frame via `fetcher`, runs the
// reductions, and produces an AnalyseResult. Reports progress (0..1) via
// the optional `progress` callback; if it returns false the analysis is
// aborted and result.ok stays false.
AnalyseResult runAnalyse(const AnalyseConfig& cfg,
                         const std::vector<int>& sampleFrames,
                         FrameFetcher fetcher,
                         std::function<bool(double)> progress = {});

}  // namespace dasgrain
