// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// CPU reference implementation of the fused grain-apply kernel.
//
// Per-pixel algorithm (linear RGBA float32 throughout). This mirrors the
// Nuke gizmo's data flow exactly:
//
//   g_orig  = plate - degrained                              # original grain
//   yG      = Y(g_orig) using Rec.709                        # luma of grain
//   lf      = (fix_ghosting || L>=1) ? 0 : (1/L - 1)         # luma-comp factor
//   g_lc    = g_orig    + (yG,yG,yG) * lf                    # luma-compensated grain
//   clean   = degrained - (yG,yG,yG) * lf                    # estimate of clean signal
//
//   cClean  = (curve_R(clean.r),   curve_G(clean.g),   curve_B(clean.b))
//   cComp   = (curve_R(comp.r),    curve_G(comp.g),    curve_B(comp.b))
//
//   g_norm    = g_lc * eps / max(cClean, eps)
//   g_source  = externalGrain ? ExternalGrain : g_norm
//   toneGain  = shadow/midtone/highlight shaping at comp luminance
//   rgbGain   = per-channel user trims
//   g_adapted = g_source * cComp / eps * toneGain * rgbGain * grainAmount
//   m         = mask alpha (or 1)  [optionally inverted]
//   regrain   = comp + g_adapted * m
//
// Output mode picks one of: regrain / g_orig / g_norm / g_adapted / QC.
// Phase 6 swaps the QC mode for the |x - blur(x)| * 50 form.

#include "../Kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "../../core/ColorSpace.h"
#include "plugin/Params.h"

#if DASGRAIN_HAS_OPENMP
#include <omp.h>
#endif

namespace dasgrain {
namespace kernels {

namespace {

// Sample a per-channel CurveLUT at `x`, clamping at endpoints (matches Nuke
// ColorLookup behaviour for values outside the keyed domain).
inline float sampleCurve(const CurveLUT& lut, int channel, float x) {
    if (lut.size <= 0 || lut.data == nullptr) return x;
    const float t = std::clamp((x - lut.minX) / (lut.maxX - lut.minX), 0.0f, 1.0f);
    const float fpos = t * float(lut.size - 1);
    const int   i0   = int(std::floor(fpos));
    const int   i1   = std::min(i0 + 1, lut.size - 1);
    const float fr   = fpos - float(i0);
    const float* ch  = lut.data + channel * lut.size;
    return ch[i0] * (1.0f - fr) + ch[i1] * fr;
}

// Resolve cannot expose Nuke's original editable curve widget, so the UI
// presents live tone-region controls. This helper turns Source luma into
// shadow/midtone/highlight weights around curvePivot, then blends the three
// gains and applies curveContrast as the shaping strength.
inline float toneGain(const GrainApplyParams& p, const float C[3]) {
    const float pivot = std::clamp(p.curvePivot, 0.001f, 0.999f);
    const float y = std::clamp(float(colorspace::luma709(C[0], C[1], C[2])), 0.0f, 1.0f);
    const float shadow = std::clamp((pivot - y) / pivot, 0.0f, 1.0f);
    const float highlight = std::clamp((y - pivot) / (1.0f - pivot), 0.0f, 1.0f);
    const float midtone = std::max(0.0f, 1.0f - std::max(shadow, highlight));
    const float shaped =
        p.shadowGrain * shadow + p.midtoneGrain * midtone + p.highlightGrain * highlight;
    return std::max(0.0f, 1.0f + (shaped - 1.0f) * std::max(p.curveContrast, 0.0f));
}

inline const float* readPixel(const ImageViewConst& v, int x, int y) {
    if (!v.data) return nullptr;
    if (x < 0 || y < 0 || x >= v.width || y >= v.height) return nullptr;
    const auto* row = reinterpret_cast<const uint8_t*>(v.data) + y * v.stride;
    return reinterpret_cast<const float*>(row) + x * 4;
}

inline float* writePixel(const ImageView& v, int x, int y) {
    if (!v.data) return nullptr;
    if (x < 0 || y < 0 || x >= v.width || y >= v.height) return nullptr;
    auto* row = reinterpret_cast<uint8_t*>(v.data) + y * v.stride;
    return reinterpret_cast<float*>(row) + x * 4;
}

}  // namespace

void runGrainApplyCPU(const GrainApplyParams& p,
                      ImageViewConst comp,
                      ImageViewConst plate,
                      ImageViewConst degrained,
                      ImageViewConst mask,
                      ImageViewConst externalGrain,
                      ImageView dst) {
    if (!dst.data) return;

    // Luma-compensation factor. Disabled when L>=1 (no compensation needed)
    // OR when the user has flipped fix_ghosting on (because the luma comp
    // path bakes plate detail into the comp on repo'd / retimed shots).
    const float lumaFactor =
        (p.fixGhosting != 0 || p.luminance >= 1.0f || p.luminance <= 0.0f)
            ? 0.0f
            : (1.0f / p.luminance - 1.0f);
    const float eps     = kCurveEpsilon;
    const float invEps  = 1.0f / eps;
    const bool  useExt  = (p.externalGrain != 0) && externalGrain.data != nullptr;

#if DASGRAIN_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < dst.height; ++y) {
        for (int x = 0; x < dst.width; ++x) {
            float* dpx = writePixel(dst, x, y);
            if (!dpx) continue;

            const float* cpx = readPixel(comp,      x, y);
            const float* ppx = readPixel(plate,     x, y);
            const float* gpx = readPixel(degrained, x, y);
            const float* mpx = readPixel(mask,      x, y);
            const float* epx = readPixel(externalGrain, x, y);

            // Sensible defaults if any clip is missing pixels at this coord.
            float C[3] = {0, 0, 0};
            float P[3] = {0, 0, 0};
            float D[3] = {0, 0, 0};
            float A    = 1.0f;  // alpha out
            if (cpx) { C[0] = cpx[0]; C[1] = cpx[1]; C[2] = cpx[2]; A = cpx[3]; }
            if (ppx) { P[0] = ppx[0]; P[1] = ppx[1]; P[2] = ppx[2]; }
            if (gpx) { D[0] = gpx[0]; D[1] = gpx[1]; D[2] = gpx[2]; }

            // 1. Original grain (raw difference).
            float g_orig[3] = {P[0] - D[0], P[1] - D[1], P[2] - D[2]};

            // 2. Luma-comp factor (per-pixel constant).
            const float yG  = float(colorspace::luma709(g_orig[0], g_orig[1], g_orig[2]));
            const float lyG = yG * lumaFactor;
            float g_lc[3]    = {g_orig[0] + lyG, g_orig[1] + lyG, g_orig[2] + lyG};
            float clean[3]   = {D[0]      - lyG, D[1]      - lyG, D[2]      - lyG};

            // 3. Per-channel curve at the clean estimate. Matches the
            //    gizmo's ColorLookup -> Expression chain exactly.
            const float cClean[3] = {
                sampleCurve(p.curve, 0, clean[0]),
                sampleCurve(p.curve, 1, clean[1]),
                sampleCurve(p.curve, 2, clean[2]),
            };
            float g_norm[3] = {
                g_lc[0] * eps / std::max(cClean[0], eps),
                g_lc[1] * eps / std::max(cClean[1], eps),
                g_lc[2] * eps / std::max(cClean[2], eps),
            };

            // 4. Pick grain source (external pre-rendered grain replaces
            //    the in-line normalised grain when requested).
            float g_source[3];
            if (useExt && epx) {
                g_source[0] = epx[0]; g_source[1] = epx[1]; g_source[2] = epx[2];
            } else {
                g_source[0] = g_norm[0]; g_source[1] = g_norm[1]; g_source[2] = g_norm[2];
            }

            // 5. Adapt by curve(comp).
            const float cC[3] = {
                sampleCurve(p.curve, 0, C[0]),
                sampleCurve(p.curve, 1, C[1]),
                sampleCurve(p.curve, 2, C[2]),
            };
            const float tone = toneGain(p, C);
            const float channelGain[3] = {
                std::max(p.redGrain, 0.0f),
                std::max(p.greenGrain, 0.0f),
                std::max(p.blueGrain, 0.0f),
            };
            const float liveGain = std::max(p.grainAmount, 0.0f);
            const float g_adapted[3] = {
                g_source[0] * cC[0] * invEps * tone * channelGain[0] * liveGain,
                g_source[1] * cC[1] * invEps * tone * channelGain[1] * liveGain,
                g_source[2] * cC[2] * invEps * tone * channelGain[2] * liveGain,
            };

            // 6. Mask. When connected we use the alpha channel of `mask`;
            //    when not, the implicit mask is fully on.
            float m = 1.0f;
            if (p.hasMask && mpx) {
                // For RGBA mask clips the host hands us alpha at index 3;
                // for Alpha-only clips OFX still maps to RGBA where alpha
                // mirrors the alpha plane.
                m = mpx[3];
                m = std::clamp(m, 0.0f, 1.0f);
                if (p.invertMask) m = 1.0f - m;
            }

            // 7. Compose final regrained pixel.
            const float regrain[3] = {
                C[0] + g_adapted[0] * m,
                C[1] + g_adapted[1] * m,
                C[2] + g_adapted[2] * m,
            };

            // 8. Output mode select.
            float out[3];
            switch (p.outputMode) {
                default:
                case params::kOutputRegrained:
                    out[0] = regrain[0]; out[1] = regrain[1]; out[2] = regrain[2];
                    break;
                case params::kOutputOriginalGrain:
                    out[0] = g_orig[0]; out[1] = g_orig[1]; out[2] = g_orig[2];
                    break;
                case params::kOutputNormalisedGrain:
                    out[0] = g_norm[0]; out[1] = g_norm[1]; out[2] = g_norm[2];
                    break;
                case params::kOutputAdaptedGrain:
                    out[0] = g_adapted[0]; out[1] = g_adapted[1]; out[2] = g_adapted[2];
                    break;
                case params::kOutputGrainQC:
                    // Phase 6 turns this into the |x - blur(x)| * 50 form.
                    // For now we output the regrained image, which still
                    // visualises the result without the blur dependency.
                    out[0] = regrain[0]; out[1] = regrain[1]; out[2] = regrain[2];
                    break;
            }

            dpx[0] = out[0];
            dpx[1] = out[1];
            dpx[2] = out[2];
            dpx[3] = A;
        }
    }
}

}  // namespace kernels
}  // namespace dasgrain
