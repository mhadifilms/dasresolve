// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0

#include "Analyser.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "ColorSpace.h"
#include "FrameRange.h"

namespace dasgrain {

std::vector<int> computeSampleFrames(const AnalyseConfig& cfg) {
    return FrameRange::generate(cfg.numberOfFrames,
                                cfg.additionalFrames,
                                cfg.firstFrame,
                                cfg.lastFrame);
}

// ---------------------------------------------------------------------------
// Stage 1: min / max LogC luma per channel.
// ---------------------------------------------------------------------------
FrameStats reduceFrameStats(FrameImage const& deg) {
    FrameStats s;
    if (deg.data.empty() || deg.width <= 0 || deg.height <= 0) return s;
    const int n = deg.width * deg.height;
    const float* px = deg.data.data();
    for (int i = 0; i < n; ++i) {
        const float r = float(colorspace::linearToLogC(double(px[0])));
        const float g = float(colorspace::linearToLogC(double(px[1])));
        const float b = float(colorspace::linearToLogC(double(px[2])));
        s.minR = std::min(s.minR, r); s.maxR = std::max(s.maxR, r);
        s.minG = std::min(s.minG, g); s.maxG = std::max(s.maxG, g);
        s.minB = std::min(s.minB, b); s.maxB = std::max(s.maxB, b);
        px += 4;
    }
    s.any = true;
    return s;
}

void mergeStats(FrameStats& a, const FrameStats& b) {
    if (!b.any) return;
    if (!a.any) { a = b; return; }
    a.minR = std::min(a.minR, b.minR); a.maxR = std::max(a.maxR, b.maxR);
    a.minG = std::min(a.minG, b.minG); a.maxG = std::max(a.maxG, b.maxG);
    a.minB = std::min(a.minB, b.minB); a.maxB = std::max(a.maxB, b.maxB);
}

// ---------------------------------------------------------------------------
// Stage 2: per-bucket sum-of-|grain|, one frame at a time.
// ---------------------------------------------------------------------------
namespace {

inline int bucketIndex(float v, float lo, float hi, int n) {
    if (n <= 1 || hi <= lo) return 0;
    const float t = (v - lo) / (hi - lo);
    if (!(t == t)) return -1;  // NaN guard
    if (t < 0.0f || t > 1.0f) return -1;
    int idx = int(std::floor(t * float(n - 1) + 0.5f));
    return std::clamp(idx, 0, n - 1);
}

}  // namespace

void accumulateHistogram(FrameImage const& plate,
                         FrameImage const& deg,
                         const FrameStats& stats,
                         Histogram& hist,
                         FrameImage const* mask,
                         bool invertMask) {
    if (plate.data.empty() || deg.data.empty()) return;
    if (plate.width != deg.width || plate.height != deg.height) return;
    if (hist.bucketCount <= 0) return;
    const int n   = plate.width * plate.height;
    const int B   = hist.bucketCount;
    if (hist.sums.size()   != size_t(B * 3)) hist.sums.assign(size_t(B * 3), 0.0);
    if (hist.counts.size() != size_t(B * 3)) hist.counts.assign(size_t(B * 3), 0.0);

    const bool useMask = (mask != nullptr) && !mask->data.empty()
                       && mask->width == plate.width
                       && mask->height == plate.height;

    const float* pp = plate.data.data();
    const float* dp = deg.data.data();
    const float* mp = useMask ? mask->data.data() : nullptr;
    for (int i = 0; i < n; ++i) {
        // Match the gizmo's ChannelMerge1 (multiply by mask alpha): pixels
        // where alpha == 0 contribute zero count to the bucket, so they do
        // not influence the curve's mean.
        if (useMask) {
            const float a = mp[3];
            const bool inside = invertMask ? (a < 0.5f) : (a > 0.5f);
            if (!inside) {
                pp += 4; dp += 4; mp += 4;
                continue;
            }
        }
        const float dr = float(colorspace::linearToLogC(double(dp[0])));
        const float dg = float(colorspace::linearToLogC(double(dp[1])));
        const float db = float(colorspace::linearToLogC(double(dp[2])));
        const int bR = bucketIndex(dr, stats.minR, stats.maxR, B);
        const int bG = bucketIndex(dg, stats.minG, stats.maxG, B);
        const int bB = bucketIndex(db, stats.minB, stats.maxB, B);
        const float gr = std::fabs(pp[0] - dp[0]);
        const float gg = std::fabs(pp[1] - dp[1]);
        const float gb = std::fabs(pp[2] - dp[2]);
        if (bR >= 0) { hist.sums[0 * B + bR] += gr; hist.counts[0 * B + bR] += 1.0; }
        if (bG >= 0) { hist.sums[1 * B + bG] += gg; hist.counts[1 * B + bG] += 1.0; }
        if (bB >= 0) { hist.sums[2 * B + bB] += gb; hist.counts[2 * B + bB] += 1.0; }
        pp += 4;
        dp += 4;
        if (mp) mp += 4;
    }
}

// ---------------------------------------------------------------------------
// Stage 3: build ResponseCurve.
// ---------------------------------------------------------------------------
ResponseCurve buildCurve(const Histogram& hist, const FrameStats& stats) {
    ResponseCurve curve;
    if (hist.bucketCount <= 1 || hist.sums.empty()) return curve;
    const int B = hist.bucketCount;
    const float lo[3] = {stats.minR, stats.minG, stats.minB};
    const float hi[3] = {stats.maxR, stats.maxG, stats.maxB};

    for (int c = 0; c < 3; ++c) {
        std::vector<CurvePoint> points;
        points.reserve(B);
        for (int i = 0; i < B; ++i) {
            const float t = (B == 1) ? 0.0f : float(i) / float(B - 1);
            const float bucketLogC = lo[c] + t * (hi[c] - lo[c]);
            const float xLinear    = float(colorspace::logCToLinear(double(bucketLogC)));
            const double cnt = hist.counts[c * B + i];
            const double yMean = (cnt > 0) ? (hist.sums[c * B + i] / cnt) : 0.0;
            points.push_back({double(xLinear), yMean});
        }
        // Make sure the curve is non-degenerate. If every bucket was empty
        // we leave an identity (0,0)..(1,1) at the channel level.
        if (!points.empty() &&
            std::any_of(points.begin(), points.end(),
                        [](const CurvePoint& p){ return p.y > 0.0; })) {
            curve.setChannel(c, std::move(points));
        }
    }
    return curve;
}

// ---------------------------------------------------------------------------
// Top-level driver: two passes over `sampleFrames`.
// ---------------------------------------------------------------------------
AnalyseResult runAnalyse(const AnalyseConfig& cfg,
                         const std::vector<int>& sampleFrames,
                         FrameFetcher fetcher,
                         std::function<bool(double)> progress) {
    AnalyseResult res;
    if (sampleFrames.empty() || !fetcher) {
        res.error = "no analysis frames or no fetcher";
        return res;
    }
    if (cfg.sampleCount < 2) {
        res.error = "sample_count must be >= 2";
        return res;
    }

    auto reportProgress = [&](double t) {
        return progress ? progress(t) : true;
    };

    // ---- Pass 1: min / max ------------------------------------------------
    FrameStats stats;
    const size_t total = sampleFrames.size() * 2;
    size_t step = 0;
    for (int frame : sampleFrames) {
        FrameImage degImg;
        if (!fetcher(ClipKind::kDegrained, frame, degImg)) {
            res.error = "fetch degrained failed during stats pass";
            return res;
        }
        mergeStats(stats, reduceFrameStats(degImg));
        ++step;
        if (!reportProgress(double(step) / double(total))) {
            res.error = "aborted";
            return res;
        }
    }
    if (!stats.any) {
        res.error = "no pixels";
        return res;
    }

    // ---- Pass 2: histogram ------------------------------------------------
    Histogram hist;
    hist.bucketCount = cfg.sampleCount;
    hist.sums.assign(size_t(cfg.sampleCount) * 3, 0.0);
    hist.counts.assign(size_t(cfg.sampleCount) * 3, 0.0);

    for (int frame : sampleFrames) {
        FrameImage plateImg, degImg, maskImg;
        if (!fetcher(ClipKind::kPlate, frame, plateImg)) {
            res.error = "fetch plate failed during histogram pass";
            return res;
        }
        if (!fetcher(ClipKind::kDegrained, frame, degImg)) {
            res.error = "fetch degrained failed during histogram pass";
            return res;
        }
        FrameImage* maskPtr = nullptr;
        if (cfg.useMask) {
            // Mask fetch is allowed to silently fail (clip might not be
            // connected); we only honour the mask when we got pixels back.
            if (fetcher(ClipKind::kMask, frame, maskImg) && !maskImg.data.empty()) {
                maskPtr = &maskImg;
            }
        }
        accumulateHistogram(plateImg, degImg, stats, hist, maskPtr, cfg.invertMask);
        ++step;
        if (!reportProgress(double(step) / double(total))) {
            res.error = "aborted";
            return res;
        }
    }

    // ---- Stage 3: build curve --------------------------------------------
    res.curve = buildCurve(hist, stats);
    res.ok = true;
    return res;
}

}  // namespace dasgrain
