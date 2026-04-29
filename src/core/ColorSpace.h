// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Color-space helpers used by DasGrain.
//
// The Nuke gizmo runs the grain analysis in **AlexaV3LogC** so that grain
// samples are spread evenly across the perceptual luminance range; it also
// uses **YCbCr (Rec.709)** for the luminance-only mix used by the
// DegrainHelper / luminance-compensation paths.
//
// We only need scalar versions for the analysis state machine and constants
// shared with the GPU kernels. The kernels themselves embed the same
// constants for performance.

#pragma once

namespace dasgrain {
namespace colorspace {

// AlexaV3LogC parameters (matching Nuke's `AlexaV3LogC` colour-space).
inline constexpr double kLogC_a    = 5.555556;
inline constexpr double kLogC_b    = 0.052272;
inline constexpr double kLogC_c    = 0.247190;
inline constexpr double kLogC_d    = 0.385537;
inline constexpr double kLogC_e    = 5.367655;
inline constexpr double kLogC_f    = 0.092809;
inline constexpr double kLogC_cut  = 0.010591;  // linear cut

// Convert a single channel between linear scene and AlexaV3LogC encoded.
double linearToLogC(double v);
double logCToLinear(double v);

// Rec.709 Y' from non-linear R'G'B' (note: the gizmo applies this to *log*
// values via a Colorspace node set to YCbCr, which is mathematically the
// luma matrix on whatever signal it gets). The matrix is identical whether
// the input is linear or non-linear; we only need the Y component.
inline constexpr double kRec709Yr = 0.2126;
inline constexpr double kRec709Yg = 0.7152;
inline constexpr double kRec709Yb = 0.0722;

double luma709(double r, double g, double b);

}  // namespace colorspace
}  // namespace dasgrain
