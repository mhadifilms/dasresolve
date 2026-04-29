// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// ResponseCurve: a 3-channel piecewise-linear LUT mapping linear luminance
// to per-channel grain intensity. Mirrors Nuke's `ColorLookup.lut` with red,
// green, blue curves and is the artefact produced by the Analyse pass.

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace dasgrain {

struct CurvePoint {
    double x;  // input value (linear luminance, 0..1+)
    double y;  // output value (grain intensity)
};

// One curve per RGB channel. Points are kept sorted by x.
class ResponseCurve {
public:
    static constexpr int kChannelCount = 3;

    ResponseCurve();

    // Reset to identity (y = x, two endpoints (0,0) and (1,1)).
    void reset();

    // Replace a channel's curve wholesale. `points` must be sorted by x.
    void setChannel(int channel, std::vector<CurvePoint> points);

    // Append a control point to a channel. The point list will be re-sorted
    // by x after the addition.
    void addPoint(int channel, double x, double y);

    // Sample channel at x, with endpoint extrapolation matching Nuke's
    // ColorLookup behaviour: clamped to first/last y outside the domain.
    double sample(int channel, double x) const;

    const std::vector<CurvePoint>& channel(int channel) const {
        return points_[channel];
    }
    std::vector<CurvePoint>& channel(int channel) {
        return points_[channel];
    }

    // True if every channel has at least 2 points and is non-degenerate.
    bool isValid() const;

    // ---- (de)serialisation ---------------------------------------------------
    // We use a tiny hand-rolled JSON writer/reader so the plugin stays free of
    // external runtime deps. Format:
    //
    //   {"version":1,"curves":[[[x,y],...], [[x,y],...], [[x,y],...]]}
    //
    std::string toJSON() const;
    bool fromJSON(const std::string& json);

private:
    std::array<std::vector<CurvePoint>, kChannelCount> points_;
};

}  // namespace dasgrain
