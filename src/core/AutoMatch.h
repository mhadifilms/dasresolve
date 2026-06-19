// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// AutoMatch: optional calibration pass that suggests the same artist-facing
// knobs artists would normally tune by eye: grain_amount and RGB trims. It
// deliberately leaves the analysed ResponseCurve and render kernels unchanged.

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "Analyser.h"
#include "ResponseCurve.h"

namespace dasgrain {

struct AutoMatchFrame {
    FrameImage source;     // Comp / Source input
    FrameImage plate;      // Original grainy plate
    FrameImage degrained;  // Denoised plate
    FrameImage mask;       // Optional mask alpha
};

struct AutoMatchConfig {
    double luminance = 1.0;
    double shadowGrain = 1.0;
    double midtoneGrain = 1.0;
    double highlightGrain = 1.0;
    double curveContrast = 1.0;
    double curvePivot = 0.5;

    bool useMask = false;
    bool invertMask = false;

    // Low-frequency source-vs-plate differences are treated as changed regions
    // (mouth replacements, edits, large lighting changes) and ignored.
    int blockSize = 16;
    int rejectPaddingBlocks = 2;
    double changedRegionThreshold = 0.006;

    // Conservative knob bounds. The plugin descriptor exposes 0..4; this keeps
    // bad inputs from writing extreme values while still allowing real fixes.
    double minGain = 0.0;
    double maxGain = 4.0;
    double minChannelGain = 0.0;
    double maxChannelGain = 4.0;
};

struct AutoMatchResult {
    bool ok = false;
    std::string error;

    double grainAmount = 1.0;
    double shadowGrain = 1.0;
    double midtoneGrain = 1.0;
    double highlightGrain = 1.0;
    double redGrain = 1.0;
    double greenGrain = 1.0;
    double blueGrain = 1.0;

    std::array<double, 3> channelProducts{{1.0, 1.0, 1.0}};
    std::array<double, 3> targetEnergy{{0.0, 0.0, 0.0}};
    std::array<double, 3> fittedEnergy{{0.0, 0.0, 0.0}};
    double energyCoverageBoost = 1.0;

    std::uint64_t pixelsSampled = 0;
    int framesSampled = 0;
};

enum class AutoMatchClipKind { kSource, kPlate, kDegrained, kMask };
using AutoMatchFrameFetcher = std::function<bool(AutoMatchClipKind kind,
                                                 int frame,
                                                 FrameImage& out)>;

AutoMatchResult runAutoMatch(const AutoMatchConfig& cfg,
                             const ResponseCurve& curve,
                             const std::vector<AutoMatchFrame>& frames);

AutoMatchResult runAutoMatch(const AutoMatchConfig& cfg,
                             const ResponseCurve& curve,
                             const std::vector<int>& sampleFrames,
                             AutoMatchFrameFetcher fetcher,
                             std::function<bool(double)> progress = {});

}  // namespace dasgrain
