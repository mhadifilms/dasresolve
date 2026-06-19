// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "core/AutoMatch.h"

using dasgrain::AutoMatchConfig;
using dasgrain::AutoMatchClipKind;
using dasgrain::AutoMatchFrame;
using dasgrain::ResponseCurve;

namespace {

dasgrain::FrameImage makeImage(int w, int h, float v) {
    dasgrain::FrameImage img;
    img.width = w;
    img.height = h;
    img.data.assign(static_cast<size_t>(w) * h * 4, 1.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float* px = img.data.data() + (static_cast<size_t>(y) * w + x) * 4;
            px[0] = v;
            px[1] = v;
            px[2] = v;
            px[3] = 1.0f;
        }
    }
    return img;
}

float noiseAt(int x, int y, int channel) {
    const int sign = ((x * 17 + y * 31 + channel * 7) & 1) ? 1 : -1;
    return sign * (0.01f + 0.002f * float((x + y + channel) % 3));
}

void addNoise(dasgrain::FrameImage& img, float scaleR, float scaleG, float scaleB) {
    const float scale[3] = {scaleR, scaleG, scaleB};
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            float* px = img.data.data() + (static_cast<size_t>(y) * img.width + x) * 4;
            for (int c = 0; c < 3; ++c) {
                px[c] += noiseAt(x, y, c) * scale[c];
            }
        }
    }
}

ResponseCurve flatCurve(double v) {
    ResponseCurve curve;
    for (int c = 0; c < 3; ++c) {
        curve.setChannel(c, {{0.0, v}, {1.0, v}});
    }
    return curve;
}

void assertNear(double got, double want, double eps) {
    if (std::fabs(got - want) > eps) {
        std::cerr << "assertNear failed: got=" << got << " want=" << want
                  << " eps=" << eps << "\n";
        assert(false);
    }
}

void neutralFitFindsOneThirdProduct() {
    AutoMatchFrame frame;
    frame.degrained = makeImage(32, 32, 0.4f);
    frame.source = frame.degrained;
    frame.plate = frame.degrained;
    addNoise(frame.plate, 1.0f, 1.0f, 1.0f);

    AutoMatchConfig cfg;
    cfg.luminance = 0.0;
    cfg.changedRegionThreshold = 1.0;
    auto result = runAutoMatch(cfg, flatCurve(0.1), std::vector<AutoMatchFrame>{frame});
    assert(result.ok);
    assertNear(result.channelProducts[0], 1.0 / 3.0, 0.02);
    assertNear(result.channelProducts[1], 1.0 / 3.0, 0.02);
    assertNear(result.channelProducts[2], 1.0 / 3.0, 0.02);
    assertNear(result.redGrain, 1.0, 0.03);
    assertNear(result.greenGrain, 1.0, 0.03);
    assertNear(result.blueGrain, 1.0, 0.03);
}

void underGrainedSourceGetsEnergyTopUp() {
    AutoMatchFrame frame;
    frame.degrained = makeImage(32, 32, 0.4f);
    frame.plate = frame.degrained;
    frame.source = frame.degrained;
    addNoise(frame.plate, 1.0f, 1.0f, 1.0f);
    addNoise(frame.source, 0.88f, 0.88f, 0.88f);

    AutoMatchConfig cfg;
    cfg.luminance = 0.0;
    cfg.changedRegionThreshold = 1.0;
    auto result = runAutoMatch(cfg, flatCurve(0.1), std::vector<AutoMatchFrame>{frame});
    assert(result.ok);
    assertNear(result.grainAmount, 0.158, 0.02);
    assertNear(result.redGrain, 1.0, 0.05);
    assertNear(result.greenGrain, 1.0, 0.05);
    assertNear(result.blueGrain, 1.0, 0.05);
}

void channelBiasProducesChannelTrim() {
    AutoMatchFrame frame;
    frame.degrained = makeImage(32, 32, 0.4f);
    frame.plate = frame.degrained;
    frame.source = frame.degrained;
    addNoise(frame.plate, 1.0f, 1.4f, 0.7f);
    addNoise(frame.source, 0.9f, 1.1f, 0.6f);

    AutoMatchConfig cfg;
    cfg.luminance = 0.0;
    cfg.changedRegionThreshold = 1.0;
    auto result = runAutoMatch(cfg, flatCurve(0.1), std::vector<AutoMatchFrame>{frame});
    assert(result.ok);
    assert(result.greenGrain > result.redGrain);
    assert(result.blueGrain > 0.0);
}

void changedRegionIsRejected() {
    AutoMatchFrame frame;
    frame.degrained = makeImage(64, 64, 0.4f);
    frame.plate = frame.degrained;
    frame.source = frame.degrained;
    addNoise(frame.plate, 1.0f, 1.0f, 1.0f);
    for (int y = 20; y < 44; ++y) {
        for (int x = 20; x < 44; ++x) {
            float* px = frame.source.data.data() + (static_cast<size_t>(y) * frame.source.width + x) * 4;
            px[0] += 0.1f;
            px[1] += 0.1f;
            px[2] += 0.1f;
        }
    }

    AutoMatchConfig cfg;
    cfg.luminance = 0.0;
    cfg.changedRegionThreshold = 0.006;
    cfg.rejectPaddingBlocks = 0;
    auto result = runAutoMatch(cfg, flatCurve(0.1), std::vector<AutoMatchFrame>{frame});
    assert(result.ok);
    assert(result.pixelsSampled < static_cast<std::uint64_t>((64 - 2) * (64 - 2)));
}

void streamingFetcherMatchesVectorPath() {
    AutoMatchFrame frameA;
    frameA.degrained = makeImage(32, 32, 0.35f);
    frameA.plate = frameA.degrained;
    frameA.source = frameA.degrained;
    addNoise(frameA.plate, 1.0f, 1.2f, 0.8f);
    addNoise(frameA.source, 0.8f, 0.9f, 0.7f);

    AutoMatchFrame frameB;
    frameB.degrained = makeImage(32, 32, 0.55f);
    frameB.plate = frameB.degrained;
    frameB.source = frameB.degrained;
    addNoise(frameB.plate, 1.1f, 1.1f, 0.9f);
    addNoise(frameB.source, 0.9f, 0.85f, 0.75f);

    const std::vector<AutoMatchFrame> frames{frameA, frameB};
    AutoMatchConfig cfg;
    cfg.luminance = 0.0;
    cfg.changedRegionThreshold = 1.0;
    const ResponseCurve curve = flatCurve(0.1);
    const auto vectorResult = runAutoMatch(cfg, curve, frames);

    const auto fetcher = [&](AutoMatchClipKind kind, int frame, dasgrain::FrameImage& out) {
        const AutoMatchFrame& src = frames.at(static_cast<size_t>(frame));
        switch (kind) {
            case AutoMatchClipKind::kSource: out = src.source; return true;
            case AutoMatchClipKind::kPlate: out = src.plate; return true;
            case AutoMatchClipKind::kDegrained: out = src.degrained; return true;
            case AutoMatchClipKind::kMask: out = src.mask; return true;
        }
        return false;
    };
    const auto streamResult = runAutoMatch(cfg, curve, std::vector<int>{0, 1}, fetcher);

    assert(vectorResult.ok);
    assert(streamResult.ok);
    assertNear(streamResult.grainAmount, vectorResult.grainAmount, 1e-12);
    assertNear(streamResult.redGrain, vectorResult.redGrain, 1e-12);
    assertNear(streamResult.greenGrain, vectorResult.greenGrain, 1e-12);
    assertNear(streamResult.blueGrain, vectorResult.blueGrain, 1e-12);
}

}  // namespace

int main() {
    neutralFitFindsOneThirdProduct();
    underGrainedSourceGetsEnergyTopUp();
    channelBiasProducesChannelTrim();
    changedRegionIsRejected();
    streamingFetcherMatchesVectorPath();
    std::cout << "auto_match_test: ok\n";
    return 0;
}
