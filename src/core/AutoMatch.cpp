// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0

#include "AutoMatch.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "ColorSpace.h"

namespace dasgrain {
namespace {

constexpr double kEps = 1e-6;

struct Accumulator {
    std::array<double, 3> denominator{{0.0, 0.0, 0.0}};
    std::array<double, 3> targetSq{{0.0, 0.0, 0.0}};
    std::array<double, 3> sourceSq{{0.0, 0.0, 0.0}};
    std::uint64_t pixelsSampled = 0;
    int framesSampled = 0;
};

double clampDouble(double v, double lo, double hi) {
    if (!std::isfinite(v)) return lo;
    return std::clamp(v, lo, hi);
}

double median3(double a, double b, double c) {
    std::array<double, 3> values{{a, b, c}};
    std::sort(values.begin(), values.end());
    return values[1];
}

bool sameFormat(const FrameImage& a, const FrameImage& b) {
    return !a.data.empty() && !b.data.empty()
        && a.width == b.width
        && a.height == b.height
        && a.width > 2
        && a.height > 2;
}

const float* pixel(const FrameImage& img, int x, int y) {
    return img.data.data() + (static_cast<size_t>(y) * img.width + x) * 4;
}

double luma709(const float* p) {
    return colorspace::luma709(p[0], p[1], p[2]);
}

double toneWeight(const AutoMatchConfig& cfg, const float* source) {
    const double y = clampDouble(luma709(source), 0.0, 1.0);
    const double pivot = clampDouble(cfg.curvePivot, 0.0, 1.0);
    const double loDen = std::max(pivot, kEps);
    const double hiDen = std::max(1.0 - pivot, kEps);

    double shadowW = 0.0;
    double midW = 0.0;
    double highW = 0.0;
    if (y <= pivot) {
        const double t = y / loDen;
        shadowW = 1.0 - t;
        midW = t;
    } else {
        const double t = (y - pivot) / hiDen;
        midW = 1.0 - t;
        highW = t;
    }
    const double shaped = shadowW * cfg.shadowGrain
                        + midW * cfg.midtoneGrain
                        + highW * cfg.highlightGrain;
    return 1.0 + (shaped - 1.0) * std::max(cfg.curveContrast, 0.0);
}

std::vector<unsigned char> changedBlockMask(const AutoMatchConfig& cfg,
                                            const FrameImage& source,
                                            const FrameImage& plate,
                                            int& blocksX,
                                            int& blocksY) {
    const int block = std::max(1, cfg.blockSize);
    blocksX = (source.width + block - 1) / block;
    blocksY = (source.height + block - 1) / block;
    std::vector<unsigned char> raw(static_cast<size_t>(blocksX) * blocksY, 0);
    std::vector<unsigned char> out(static_cast<size_t>(blocksX) * blocksY, 0);

    for (int by = 0; by < blocksY; ++by) {
        for (int bx = 0; bx < blocksX; ++bx) {
            const int x0 = bx * block;
            const int y0 = by * block;
            const int x1 = std::min(source.width, x0 + block);
            const int y1 = std::min(source.height, y0 + block);
            double sumS = 0.0;
            double sumP = 0.0;
            double count = 0.0;
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    sumS += luma709(pixel(source, x, y));
                    sumP += luma709(pixel(plate, x, y));
                    count += 1.0;
                }
            }
            if (count > 0.0) {
                const double diff = std::fabs(sumS / count - sumP / count);
                raw[static_cast<size_t>(by) * blocksX + bx] =
                    diff > cfg.changedRegionThreshold ? 1 : 0;
            }
        }
    }

    const int pad = std::max(0, cfg.rejectPaddingBlocks);
    for (int by = 0; by < blocksY; ++by) {
        for (int bx = 0; bx < blocksX; ++bx) {
            bool reject = false;
            for (int yy = std::max(0, by - pad); yy <= std::min(blocksY - 1, by + pad); ++yy) {
                for (int xx = std::max(0, bx - pad); xx <= std::min(blocksX - 1, bx + pad); ++xx) {
                    reject = reject || raw[static_cast<size_t>(yy) * blocksX + xx] != 0;
                }
            }
            out[static_cast<size_t>(by) * blocksX + bx] = reject ? 1 : 0;
        }
    }
    return out;
}

bool maskAllows(const AutoMatchConfig& cfg, const AutoMatchFrame& frame, int x, int y) {
    if (!cfg.useMask || frame.mask.data.empty()) return true;
    if (frame.mask.width != frame.source.width || frame.mask.height != frame.source.height) return true;
    const float a = pixel(frame.mask, x, y)[3];
    const bool inside = cfg.invertMask ? (a < 0.5f) : (a > 0.5f);
    return inside;
}

double baseGrain(const AutoMatchConfig& cfg,
                 const ResponseCurve& curve,
                 const AutoMatchFrame& frame,
                 int x,
                 int y,
                 int channel) {
    const float* source = pixel(frame.source, x, y);
    const float* plate = pixel(frame.plate, x, y);
    const float* deg = pixel(frame.degrained, x, y);

    const double gOrig[3] = {
        double(plate[0]) - double(deg[0]),
        double(plate[1]) - double(deg[1]),
        double(plate[2]) - double(deg[2]),
    };
    const double yG = colorspace::luma709(gOrig[0], gOrig[1], gOrig[2]);
    const double lyG = yG * cfg.luminance;
    const double gLc = gOrig[channel] + lyG;
    const double clean = double(deg[channel]) - lyG;
    const double cClean = std::max(curve.sample(channel, clean), kEps);
    const double normalised = gLc * kEps / cClean;
    const double cSource = curve.sample(channel, double(source[channel]));
    const double tone = toneWeight(cfg, source);

    // Same scalar energy shape as the render kernels, but without grain_amount
    // or per-channel trim. AutoMatch solves those products.
    return normalised * cSource / kEps * tone * 3.0;
}

double hpPlate(const AutoMatchFrame& frame, int x, int y, int channel) {
    return double(pixel(frame.plate, x, y)[channel])
        - 0.25 * (double(pixel(frame.plate, x - 1, y)[channel])
                + double(pixel(frame.plate, x + 1, y)[channel])
                + double(pixel(frame.plate, x, y - 1)[channel])
                + double(pixel(frame.plate, x, y + 1)[channel]));
}

double hpSource(const AutoMatchFrame& frame, int x, int y, int channel) {
    return double(pixel(frame.source, x, y)[channel])
        - 0.25 * (double(pixel(frame.source, x - 1, y)[channel])
                + double(pixel(frame.source, x + 1, y)[channel])
                + double(pixel(frame.source, x, y - 1)[channel])
                + double(pixel(frame.source, x, y + 1)[channel]));
}

double hpBase(const AutoMatchConfig& cfg,
              const ResponseCurve& curve,
              const AutoMatchFrame& frame,
              int x,
              int y,
              int channel) {
    return baseGrain(cfg, curve, frame, x, y, channel)
        - 0.25 * (baseGrain(cfg, curve, frame, x - 1, y, channel)
                + baseGrain(cfg, curve, frame, x + 1, y, channel)
                + baseGrain(cfg, curve, frame, x, y - 1, channel)
                + baseGrain(cfg, curve, frame, x, y + 1, channel));
}

bool accumulateFrame(const AutoMatchConfig& cfg,
                     const ResponseCurve& curve,
                     const AutoMatchFrame& frame,
                     Accumulator& accum,
                     std::string& error) {
    if (!sameFormat(frame.source, frame.plate) || !sameFormat(frame.source, frame.degrained)) {
        error = "frame format mismatch";
        return false;
    }

    int blocksX = 0;
    int blocksY = 0;
    const auto rejectBlocks = changedBlockMask(cfg, frame.source, frame.plate, blocksX, blocksY);
    const int block = std::max(1, cfg.blockSize);

    for (int y = 1; y < frame.source.height - 1; ++y) {
        for (int x = 1; x < frame.source.width - 1; ++x) {
            const int bx = std::clamp(x / block, 0, blocksX - 1);
            const int by = std::clamp(y / block, 0, blocksY - 1);
            if (rejectBlocks[static_cast<size_t>(by) * blocksX + bx]) continue;
            if (!maskAllows(cfg, frame, x, y)
                || !maskAllows(cfg, frame, x - 1, y)
                || !maskAllows(cfg, frame, x + 1, y)
                || !maskAllows(cfg, frame, x, y - 1)
                || !maskAllows(cfg, frame, x, y + 1)) {
                continue;
            }

            for (int c = 0; c < 3; ++c) {
                const double target = hpPlate(frame, x, y, c);
                const double source = hpSource(frame, x, y, c);
                const double base = hpBase(cfg, curve, frame, x, y, c);
                if (!std::isfinite(target) || !std::isfinite(source) || !std::isfinite(base)) {
                    continue;
                }
                // Grain matching is an energy problem, not a pixel-correlation
                // problem: the source may already contain uncorrelated grain.
                // Fit the missing energy needed to bring source variance up to
                // the original plate.
                accum.denominator[c] += base * base;
                accum.targetSq[c] += target * target;
                accum.sourceSq[c] += source * source;
            }
            accum.pixelsSampled += 1;
        }
    }
    accum.framesSampled += 1;
    return true;
}

AutoMatchResult finishResult(const AutoMatchConfig& cfg, const Accumulator& accum) {
    AutoMatchResult result;
    result.pixelsSampled = accum.pixelsSampled;
    result.framesSampled = accum.framesSampled;

    if (result.pixelsSampled == 0) {
        result.error = "no valid pixels after masks/rejection";
        return result;
    }

    for (int c = 0; c < 3; ++c) {
        if (accum.denominator[c] <= kEps) {
            result.error = "generated grain energy is zero";
            return result;
        }
        const double missingSq = std::max(accum.targetSq[c] - accum.sourceSq[c], 0.0);
        result.channelProducts[c] = clampDouble(
            std::sqrt(missingSq / accum.denominator[c]),
            cfg.minGain,
            cfg.maxGain);
        result.targetEnergy[c] = std::sqrt(accum.targetSq[c] / double(result.pixelsSampled));
    }

    const double global = clampDouble(
        median3(result.channelProducts[0], result.channelProducts[1], result.channelProducts[2]),
        cfg.minGain,
        cfg.maxGain);
    result.grainAmount = global;
    const double safeGlobal = std::max(global, kEps);
    result.redGrain = clampDouble(result.channelProducts[0] / safeGlobal,
                                  cfg.minChannelGain, cfg.maxChannelGain);
    result.greenGrain = clampDouble(result.channelProducts[1] / safeGlobal,
                                    cfg.minChannelGain, cfg.maxChannelGain);
    result.blueGrain = clampDouble(result.channelProducts[2] / safeGlobal,
                                   cfg.minChannelGain, cfg.maxChannelGain);

    const double products[3] = {
        result.grainAmount * result.redGrain,
        result.grainAmount * result.greenGrain,
        result.grainAmount * result.blueGrain,
    };
    for (int c = 0; c < 3; ++c) {
        const double sourceEnergySq = accum.sourceSq[c] / double(result.pixelsSampled);
        const double addedEnergySq = products[c] * products[c]
            * accum.denominator[c] / double(result.pixelsSampled);
        result.fittedEnergy[c] = std::sqrt(sourceEnergySq + addedEnergySq);
    }

    result.ok = true;
    return result;
}

}  // namespace

AutoMatchResult runAutoMatch(const AutoMatchConfig& cfg,
                             const ResponseCurve& curve,
                             const std::vector<AutoMatchFrame>& frames) {
    if (frames.empty()) {
        AutoMatchResult result;
        result.error = "no frames";
        return result;
    }
    if (!curve.isValid()) {
        AutoMatchResult result;
        result.error = "response curve is not valid";
        return result;
    }

    Accumulator accum;
    std::string error;
    for (const AutoMatchFrame& frame : frames) {
        if (!accumulateFrame(cfg, curve, frame, accum, error)) {
            AutoMatchResult result;
            result.error = error;
            return result;
        }
    }
    return finishResult(cfg, accum);
}

AutoMatchResult runAutoMatch(const AutoMatchConfig& cfg,
                             const ResponseCurve& curve,
                             const std::vector<int>& sampleFrames,
                             AutoMatchFrameFetcher fetcher,
                             std::function<bool(double)> progress) {
    if (sampleFrames.empty() || !fetcher) {
        AutoMatchResult result;
        result.error = "no frames or no fetcher";
        return result;
    }
    if (!curve.isValid()) {
        AutoMatchResult result;
        result.error = "response curve is not valid";
        return result;
    }

    auto reportProgress = [&](double t) {
        return progress ? progress(t) : true;
    };

    Accumulator accum;
    std::string error;
    for (size_t i = 0; i < sampleFrames.size(); ++i) {
        const int frameNumber = sampleFrames[i];
        AutoMatchFrame frame;
        if (!fetcher(AutoMatchClipKind::kSource, frameNumber, frame.source)
            || !fetcher(AutoMatchClipKind::kPlate, frameNumber, frame.plate)
            || !fetcher(AutoMatchClipKind::kDegrained, frameNumber, frame.degrained)) {
            AutoMatchResult result;
            result.error = "fetch frame failed";
            return result;
        }
        if (cfg.useMask) {
            fetcher(AutoMatchClipKind::kMask, frameNumber, frame.mask);
        }
        if (!accumulateFrame(cfg, curve, frame, accum, error)) {
            AutoMatchResult result;
            result.error = error;
            return result;
        }
        if (!reportProgress(double(i + 1) / double(sampleFrames.size()))) {
            AutoMatchResult result;
            result.error = "aborted";
            return result;
        }
    }
    return finishResult(cfg, accum);
}

}  // namespace dasgrain
