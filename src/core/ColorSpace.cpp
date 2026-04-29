// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ColorSpace.h"

#include <cmath>

namespace dasgrain {
namespace colorspace {

double linearToLogC(double v) {
    if (v > kLogC_cut) {
        return kLogC_c * std::log10(kLogC_a * v + kLogC_b) + kLogC_d;
    }
    return kLogC_e * v + kLogC_f;
}

double logCToLinear(double v) {
    // Mirrors Nuke's `_log2lin` helper inside the analyse script.
    if (v > kLogC_e * kLogC_cut + kLogC_f) {
        return (std::pow(10.0, (v - kLogC_d) / kLogC_c) - kLogC_b) / kLogC_a;
    }
    return (v - kLogC_f) / kLogC_e;
}

double luma709(double r, double g, double b) {
    return kRec709Yr * r + kRec709Yg * g + kRec709Yb * b;
}

}  // namespace colorspace
}  // namespace dasgrain
