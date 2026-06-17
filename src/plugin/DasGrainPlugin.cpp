// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0

#include "DasGrainPlugin.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "ofxsImageEffect.h"

#include "Params.h"
#include "core/Analyser.h"
#include "core/ResponseCurve.h"
#include "kernels/KernelDispatch.h"
#include "kernels/Kernels.h"

namespace dasgrain {

namespace {
bool copyViewToFrameImage(const kernels::ImageViewConst& view, dasgrain::FrameImage& out);
ResponseCurve buildCurrentFrameResponseCurve(const dasgrain::FrameImage& plate,
                                             const dasgrain::FrameImage& degrained,
                                             int sampleCount);
}

DasGrainPlugin::DasGrainPlugin(OfxImageEffectHandle handle)
    : OFX::ImageEffect(handle) {
    dst_           = fetchClip(kOfxImageEffectOutputClipName);
    source_        = fetchClip(kOfxImageEffectSimpleSourceClipName);
    plate_         = fetchClip(clips::kPlate);
    degrained_     = fetchClip(clips::kDegrained);
    mask_          = fetchClip(clips::kMask);
    externalGrain_ = fetchClip(clips::kExternalGrain);

    output_            = fetchChoiceParam(params::kOutput);
    luminance_         = fetchDoubleParam(params::kLuminance);
    fixGhosting_       = fetchBooleanParam(params::kFixGhosting);
    grainMaskInvert_   = fetchBooleanParam(params::kGrainMaskInvert);
    numberOfFrames_    = fetchIntParam(params::kNumberOfFrames);
    additionalFrames_  = fetchStringParam(params::kAdditionalFrames);
    sampleCount_       = fetchIntParam(params::kSampleCount);
    analyse_           = fetchPushButtonParam(params::kAnalyse);
    analyseState_      = fetchStringParam(params::kAnalyseState);
    analyseMaskInvert_ = fetchBooleanParam(params::kAnalyseMaskInvert);
    // Resolve doesn't support kOfxParamTypeParametric, so we don't fetch
    // the on-screen curve widget any more — see DasGrainFactory.cpp and
    // DasGrainPlugin.h. The JSON string is the canonical store.
    responseCurveJSON_ = fetchStringParam(params::kResponseCurveJSON);
    grainAmount_       = fetchDoubleParam(params::kGrainAmount);
    shadowGrain_       = fetchDoubleParam(params::kShadowGrain);
    midtoneGrain_      = fetchDoubleParam(params::kMidtoneGrain);
    highlightGrain_    = fetchDoubleParam(params::kHighlightGrain);
    curveContrast_     = fetchDoubleParam(params::kCurveContrast);
    curvePivot_        = fetchDoubleParam(params::kCurvePivot);
    redGrain_          = fetchDoubleParam(params::kRedGrain);
    greenGrain_        = fetchDoubleParam(params::kGreenGrain);
    blueGrain_         = fetchDoubleParam(params::kBlueGrain);
    showCurveIO_       = fetchBooleanParam(params::kShowCurveIO);

    externalGrainEnabled_ = fetchBooleanParam(params::kExternalGrain);
    scatter_              = fetchBooleanParam(params::kScatter);
    boxLow_               = fetchDouble2DParam(params::kBoxLow);
    boxHigh_              = fetchDouble2DParam(params::kBoxHigh);
    sampleFrame_          = fetchIntParam(params::kSampleFrame);
    cellSize_             = fetchDoubleParam(params::kCellSize);
    overlayPattern_       = fetchBooleanParam(params::kOverlayPattern);
    edgeBlendSize_        = fetchIntParam(params::kEdgeBlendSize);
    amplitude_            = fetchDoubleParam(params::kAmplitude);
    frequency_            = fetchDoubleParam(params::kFrequency);
    seed_                 = fetchIntParam(params::kSeed);
    replaceMaskInvert_    = fetchBooleanParam(params::kReplaceMaskInvert);
    troubleshoot_         = fetchPushButtonParam(params::kTroubleshoot);
    curveJSONPaste_       = fetchStringParam(params::kCurveJSONImport);
    curveImport_          = fetchPushButtonParam(params::kCurveImport);
    curveExport_          = fetchPushButtonParam(params::kCurveExport);
    curveHelp_            = fetchPushButtonParam(params::kCurveHelp);

    updateUiEnabled();
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

void DasGrainPlugin::render(const OFX::RenderArguments& args) {
    if (!dst_) {
        OFX::throwSuiteStatusException(kOfxStatErrBadHandle);
    }
    if (dst_->getPixelDepth() != OFX::eBitDepthFloat ||
        dst_->getPixelComponents() != OFX::ePixelComponentRGBA) {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
    // If the caller hasn't wired the Plate / Degrained inputs we can't do
    // grain math; pass the comp through so the timeline doesn't go black.
    const bool haveAll = plate_     && plate_->isConnected()
                      && degrained_ && degrained_->isConnected()
                      && source_    && source_->isConnected();
    if (!haveAll) {
        renderPassThrough(args);
        return;
    }

    // Analyse is handled inside renderGrainApply() after fetching the exact
    // Plate/Degrained image views that Resolve is rendering. Resolve's
    // separate frame-range fetch path can return black/duplicated frames for
    // multi-input Fusion OFX clips, which makes Analyse falsely produce an
    // identity curve.
    renderGrainApply(args);
}

void DasGrainPlugin::renderGrainApply(const OFX::RenderArguments& args) {
    using OFX::Image;

    std::unique_ptr<Image> dst(dst_->fetchImage(args.time));
    if (!dst) return;

    auto fetchSafe = [&](OFX::Clip* c, double t) -> std::unique_ptr<Image> {
        if (!c || !c->isConnected()) return {};
        return std::unique_ptr<Image>(c->fetchImage(t));
    };
    auto compImg      = fetchSafe(source_,        args.time);
    auto plateImg     = fetchSafe(plate_,         args.time);
    auto degrainedImg = fetchSafe(degrained_,     args.time);
    auto maskImg      = fetchSafe(mask_,          args.time);
    auto extImg       = fetchSafe(externalGrain_, args.time);

    const OfxRectI rw = args.renderWindow;
    const int w = rw.x2 - rw.x1;
    const int h = rw.y2 - rw.y1;
    if (w <= 0 || h <= 0) return;

    auto makeConstView = [&](const std::unique_ptr<Image>& img) {
        kernels::ImageViewConst v;
        v.width  = w;
        v.height = h;
        if (img) {
            v.data   = static_cast<const float*>(img->getPixelAddress(rw.x1, rw.y1));
            v.stride = img->getRowBytes();
        } else {
            v.data   = nullptr;
            v.stride = 0;
        }
        return v;
    };

    kernels::ImageView dstView;
    dstView.width  = w;
    dstView.height = h;
    dstView.data   = static_cast<float*>(dst->getPixelAddress(rw.x1, rw.y1));
    dstView.stride = dst->getRowBytes();
    if (!dstView.data) return;

    auto compView      = makeConstView(compImg);
    auto plateView     = makeConstView(plateImg);
    auto degrainedView = makeConstView(degrainedImg);
    auto maskView      = makeConstView(maskImg);
    auto extView       = makeConstView(extImg);

    if (analyseState_) {
        std::string state;
        analyseState_->getValue(state);
        if (state == analyse_state::kRequested) {
            dasgrain::FrameImage plateFrame, degrainedFrame;
            if (copyViewToFrameImage(plateView, plateFrame) &&
                copyViewToFrameImage(degrainedView, degrainedFrame)) {
                const int buckets = std::max(2, sampleCount_->getValueAtTime(args.time));
                analysedCurve_ = buildCurrentFrameResponseCurve(plateFrame, degrainedFrame, buckets);
                hasAnalysedCurve_ = true;
                const std::string json = analysedCurve_.toJSON();
                beginEditBlock("DasGrain Analyse");
                if (responseCurveJSON_) responseCurveJSON_->setValue(json);
                analyseState_->setValue(analyse_state::kIdle);
                endEditBlock();
            } else {
                analyseState_->setValue(analyse_state::kIdle);
            }
        }
    }

    const std::string identityCurveJSON = ResponseCurve().toJSON();
    if (!hasAnalysedCurve_ || analysedCurve_.toJSON() == identityCurveJSON) {
        dasgrain::FrameImage plateFrame, degrainedFrame;
        if (copyViewToFrameImage(plateView, plateFrame) &&
            copyViewToFrameImage(degrainedView, degrainedFrame)) {
            const int buckets = std::max(2, sampleCount_->getValueAtTime(args.time));
            ResponseCurve currentCurve = buildCurrentFrameResponseCurve(plateFrame, degrainedFrame, buckets);
            if (currentCurve.toJSON() != identityCurveJSON) {
                analysedCurve_ = currentCurve;
                hasAnalysedCurve_ = true;
                if (responseCurveJSON_) responseCurveJSON_->setValue(analysedCurve_.toJSON());
            }
        }
    }

    // ---- Build params -----------------------------------------------------
    // These values are sampled at render time, so artists get immediate
    // feedback when adjusting grain amount, tone-region shaping, or RGB trims.
    kernels::GrainApplyParams gp;
    gp.luminance      = static_cast<float>(luminance_->getValueAtTime(args.time));
    gp.grainAmount    = static_cast<float>(grainAmount_->getValueAtTime(args.time));
    gp.shadowGrain    = static_cast<float>(shadowGrain_->getValueAtTime(args.time));
    gp.midtoneGrain   = static_cast<float>(midtoneGrain_->getValueAtTime(args.time));
    gp.highlightGrain = static_cast<float>(highlightGrain_->getValueAtTime(args.time));
    gp.curveContrast  = static_cast<float>(curveContrast_->getValueAtTime(args.time));
    gp.curvePivot     = static_cast<float>(curvePivot_->getValueAtTime(args.time));
    gp.redGrain       = static_cast<float>(redGrain_->getValueAtTime(args.time));
    gp.greenGrain     = static_cast<float>(greenGrain_->getValueAtTime(args.time));
    gp.blueGrain      = static_cast<float>(blueGrain_->getValueAtTime(args.time));
    gp.fixGhosting    = fixGhosting_->getValueAtTime(args.time) ? 1 : 0;
    gp.externalGrain  = externalGrainEnabled_->getValueAtTime(args.time) ? 1 : 0;
    int outputModeIdx = 0;
    output_->getValueAtTime(args.time, outputModeIdx);
    // Grain QC mode (= |regrain - blur1(regrain)| * 50) needs a 3x3
    // neighbourhood, so we ask the kernel for the regrain pass and run the
    // high-pass on the CPU afterwards. This avoids tiling/gather logic in
    // the GPU kernels.
    const bool qcMode = (outputModeIdx == params::kOutputGrainQC);
    gp.outputMode    = qcMode ? params::kOutputRegrained : outputModeIdx;
    gp.hasMask       = (maskView.data != nullptr) ? 1 : 0;
    gp.invertMask    = grainMaskInvert_->getValueAtTime(args.time) ? 1 : 0;

    constexpr int kLutSize = 256;
    std::vector<float> curveData = bakeResponseCurve(args.time, kLutSize);
    gp.curve.data = curveData.data();
    gp.curve.size = kLutSize;
    gp.curve.minX = 0.0f;
    gp.curve.maxX = 1.0f;

    // ---- Pick backend -----------------------------------------------------
    kernels::GpuContext ctx;
    if (args.isEnabledMetalRender) {
        ctx.backend   = kernels::Backend::kMetal;
        ctx.metalCmdQ = args.pMetalCmdQ;
    } else if (args.isEnabledCudaRender) {
        ctx.backend    = kernels::Backend::kCUDA;
        ctx.cudaStream = args.pCudaStream;
    } else if (args.isEnabledOpenCLRender) {
        ctx.backend    = kernels::Backend::kOpenCL;
        ctx.openCLCmdQ = args.pOpenCLCmdQ;
    } else {
        ctx.backend = kernels::Backend::kCPU;
    }

    // ---- Scatter (Phase 5) -----------------------------------------------
    // When scatter is on, we fetch plate/degrained at sample_frame, run the
    // Voronoi scatter kernel into a temp buffer, then route that buffer to
    // grain_apply as the externalGrain input — overriding the user's
    // externalGrain checkbox/clip while scatter is enabled.
    std::vector<float> scatterBuf;             // temp owns the scatter output
    std::vector<float> scatterPlateFlat;       // packed plate@sample_frame
    std::vector<float> scatterDegFlat;         // packed deg@sample_frame
    if (scatter_ && scatter_->getValueAtTime(args.time)) {
        const int sampleFrame = sampleFrame_->getValueAtTime(args.time);
        std::unique_ptr<Image> psamp(plate_     ? plate_->fetchImage(double(sampleFrame))     : nullptr);
        std::unique_ptr<Image> dsamp(degrained_ ? degrained_->fetchImage(double(sampleFrame)) : nullptr);
        if (psamp && dsamp) {
            const OfxRectI pb = psamp->getBounds();
            const OfxRectI db = dsamp->getBounds();
            const int sw = std::min(pb.x2 - pb.x1, db.x2 - db.x1);
            const int sh = std::min(pb.y2 - pb.y1, db.y2 - db.y1);
            if (sw > 0 && sh > 0) {
                scatterPlateFlat.resize(size_t(sw) * sh * 4);
                scatterDegFlat.resize(size_t(sw) * sh * 4);
                for (int y = 0; y < sh; ++y) {
                    const auto* psrc = static_cast<const float*>(psamp->getPixelAddress(pb.x1, pb.y1 + y));
                    const auto* dsrc = static_cast<const float*>(dsamp->getPixelAddress(db.x1, db.y1 + y));
                    if (psrc) std::memcpy(scatterPlateFlat.data() + size_t(y) * sw * 4, psrc, size_t(sw) * 4 * sizeof(float));
                    if (dsrc) std::memcpy(scatterDegFlat.data()   + size_t(y) * sw * 4, dsrc, size_t(sw) * 4 * sizeof(float));
                }
                scatterBuf.assign(size_t(w) * h * 4, 0.0f);
                kernels::ImageViewConst psampView{scatterPlateFlat.data(), sw, sh,
                                                  int32_t(sw * 4 * sizeof(float))};
                kernels::ImageViewConst dsampView{scatterDegFlat.data(),   sw, sh,
                                                  int32_t(sw * 4 * sizeof(float))};
                kernels::ImageView      scatterView{scatterBuf.data(), w, h,
                                                    int32_t(w * 4 * sizeof(float))};

                // Per-frame seed: matches the gizmo's VoroNoise_Seed = x*5
                // expression, which animates the noise seed by 5 per frame.
                const int  baseSeed   = seed_->getValueAtTime(args.time);
                const int  framePart  = int(std::lround(args.time)) * 5;
                const int  effSeed    = baseSeed + framePart;

                kernels::ScatterParams sp;
                double bxl[2], byh[2];
                boxLow_->getValueAtTime(args.time, bxl[0], bxl[1]);
                boxHigh_->getValueAtTime(args.time, byh[0], byh[1]);
                // Box knob is in canonical (= pixel) coordinates relative
                // to the sample image's origin. Convert to image-bounds-
                // relative pixel coords by subtracting pb.x1 / pb.y1.
                sp.boxX             = float(std::min(bxl[0], byh[0]) - pb.x1);
                sp.boxY             = float(std::min(bxl[1], byh[1]) - pb.y1);
                sp.boxR             = float(std::max(bxl[0], byh[0]) - pb.x1);
                sp.boxT             = float(std::max(bxl[1], byh[1]) - pb.y1);
                sp.cellSize         = float(cellSize_->getValueAtTime(args.time));
                sp.seed             = effSeed;
                sp.overlayPattern   = overlayPattern_->getValueAtTime(args.time) ? 1 : 0;
                sp.distortAmplitude = float(amplitude_->getValueAtTime(args.time));
                sp.distortFrequency = float(frequency_->getValueAtTime(args.time));
                sp.edgeBlendSize    = edgeBlendSize_->getValueAtTime(args.time);
                sp.dstOffsetX       = rw.x1 - pb.x1;
                sp.dstOffsetY       = rw.y1 - pb.y1;
                sp.curve            = gp.curve;

                kernels::dispatchScatter(ctx, sp, psampView, dsampView, scatterView);

                // Replace mask blend: when a mask is connected we mix the
                // scatter buffer with the in-line normalised grain so that
                // scatter only takes effect in the masked region (matches
                // the gizmo's Merge9 'copy' with maskChannelMask = -alpha).
                if (replaceMaskInvert_ && maskImg && maskView.data) {
                    const bool inv = replaceMaskInvert_->getValueAtTime(args.time);
                    const float L  = gp.luminance;
                    const float lf = (gp.fixGhosting || L >= 1.0f) ? 0.0f : (1.0f / L - 1.0f);
                    const float eps = kernels::kCurveEpsilon;

                    auto curveSample = [&](int ch, float v) -> float {
                        if (gp.curve.size <= 0 || !gp.curve.data) return std::max(v, eps);
                        const float t = (v - gp.curve.minX) /
                                        std::max(gp.curve.maxX - gp.curve.minX, 1e-12f);
                        const float ti = std::clamp(t, 0.0f, 1.0f) * float(gp.curve.size - 1);
                        const int   i0 = int(std::floor(ti));
                        const int   i1 = std::min(i0 + 1, gp.curve.size - 1);
                        const float f  = ti - float(i0);
                        const float* row = gp.curve.data + ch * gp.curve.size;
                        return row[i0] + (row[i1] - row[i0]) * f;
                    };
                    auto rowOf = [](const void* base, int32_t stride, int y) -> const float* {
                        if (!base) return nullptr;
                        return reinterpret_cast<const float*>(
                            reinterpret_cast<const uint8_t*>(base) + y * stride);
                    };
                    for (int y = 0; y < h; ++y) {
                        const float* prow = rowOf(plateView.data,     plateView.stride,     y);
                        const float* drow = rowOf(degrainedView.data, degrainedView.stride, y);
                        const float* mrow = rowOf(maskView.data,      maskView.stride,      y);
                        if (!prow || !drow || !mrow) continue;
                        float* srow = scatterBuf.data() + size_t(y) * w * 4;
                        for (int x = 0; x < w; ++x) {
                            const float ma = mrow[x * 4 + 3];
                            const float t  = inv ? ma : (1.0f - ma);
                            if (t >= 1.0f) continue;  // pure scatter
                            const float P[3] = {prow[x * 4 + 0], prow[x * 4 + 1], prow[x * 4 + 2]};
                            const float D[3] = {drow[x * 4 + 0], drow[x * 4 + 1], drow[x * 4 + 2]};
                            const float gR = P[0] - D[0];
                            const float gG = P[1] - D[1];
                            const float gB = P[2] - D[2];
                            const float yG = 0.299f * gR + 0.587f * gG + 0.114f * gB;
                            const float gLc[3] = {gR + yG * lf, gG + yG * lf, gB + yG * lf};
                            const float cl[3]  = {D[0] - yG * lf, D[1] - yG * lf, D[2] - yG * lf};
                            const float cR = curveSample(0, cl[0]);
                            const float cG = curveSample(1, cl[1]);
                            const float cB = curveSample(2, cl[2]);
                            const float gN[3] = {
                                gLc[0] * eps / std::max(cR, eps),
                                gLc[1] * eps / std::max(cG, eps),
                                gLc[2] * eps / std::max(cB, eps),
                            };
                            float* sp = srow + x * 4;
                            sp[0] = sp[0] * t + gN[0] * (1.0f - t);
                            sp[1] = sp[1] * t + gN[1] * (1.0f - t);
                            sp[2] = sp[2] * t + gN[2] * (1.0f - t);
                        }
                    }
                }

                // Override the external-grain input so grain_apply consumes
                // the (possibly mask-blended) scattered normalised grain.
                gp.externalGrain = 1;
                extView = kernels::ImageViewConst{
                    scatterBuf.data(), w, h, int32_t(w * 4 * sizeof(float))};
            }
        }
    }

    if (qcMode) {
        // Render regrain into a scratch buffer first, then run the gizmo's
        // QC math: out = |regrain - blur1(regrain)| * 50. blur1 is a
        // 1-pixel-radius Gaussian-ish blur; we approximate with a 3x3 box,
        // which matches the gizmo's "Blur size=1" closely enough for QC.
        std::vector<float> qcScratch(size_t(w) * h * 4, 0.0f);
        kernels::ImageView qcView{qcScratch.data(), w, h,
                                  int32_t(w * 4 * sizeof(float))};
        kernels::dispatchGrainApply(ctx, gp,
                                    compView, plateView, degrainedView,
                                    maskView, extView,
                                    qcView);

        auto get = [&](int x, int y, int c) -> float {
            x = std::clamp(x, 0, w - 1);
            y = std::clamp(y, 0, h - 1);
            return qcScratch[(size_t(y) * w + x) * 4 + c];
        };
        for (int y = 0; y < h; ++y) {
            auto* drow = reinterpret_cast<float*>(
                reinterpret_cast<uint8_t*>(dstView.data) + y * dstView.stride);
            for (int x = 0; x < w; ++x) {
                float* dpx = drow + x * 4;
                for (int c = 0; c < 3; ++c) {
                    const float v = get(x, y, c);
                    const float blur =
                        (get(x - 1, y - 1, c) + get(x, y - 1, c) + get(x + 1, y - 1, c) +
                         get(x - 1, y    , c) + get(x, y    , c) + get(x + 1, y    , c) +
                         get(x - 1, y + 1, c) + get(x, y + 1, c) + get(x + 1, y + 1, c)) * (1.0f / 9.0f);
                    dpx[c] = std::fabs(v - blur) * 50.0f;
                }
                dpx[3] = qcScratch[(size_t(y) * w + x) * 4 + 3];
            }
        }
        return;
    }

    kernels::dispatchGrainApply(ctx, gp,
                                compView, plateView, degrainedView,
                                maskView, extView,
                                dstView);
}

std::vector<float> DasGrainPlugin::bakeResponseCurve(double /*time*/,
                                                     int sampleCount) const {
    std::vector<float> out(static_cast<size_t>(sampleCount * 3), 0.0f);

    // Resolve/Fusion may not expose or persist the hidden JSON string param.
    // The in-memory analysed/imported curve is authoritative for the current
    // plugin instance; JSON is only a persistence/import fallback. Never return
    // an all-zero LUT just because the host omitted the string param.
    ResponseCurve curve;
    if (hasAnalysedCurve_ && analysedCurve_.isValid()) {
        curve = analysedCurve_;
    } else if (responseCurveJSON_) {
        std::string json;
        responseCurveJSON_->getValue(json);
        if (json.empty() || !curve.fromJSON(json)) {
            curve.reset();
        }
    } else {
        curve.reset();
    }

    const std::string identityCurveJSON = ResponseCurve().toJSON();
    if (curve.toJSON() == identityCurveJSON) {
        // A mathematically identity response curve is useful as a UI default,
        // but the grain kernel expects this LUT to describe grain amplitude.
        // If Resolve did not deliver a real Analyse curve, use the Nuke-style
        // neutral amplitude (eps) so original grain is preserved instead of
        // being divided down by the clean image value.
        std::fill(out.begin(), out.end(), 0.01f);
        return out;
    }

    for (int channel = 0; channel < 3; ++channel) {
        float* dst = out.data() + channel * sampleCount;
        for (int i = 0; i < sampleCount; ++i) {
            const double x = double(i) / double(sampleCount - 1);
            dst[i] = static_cast<float>(curve.sample(channel, x));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Analyse pass
// ---------------------------------------------------------------------------

namespace {

// Copy an OFX::Image's render-window-bounded float RGBA pixels into a flat
// host-side buffer. We restrict the read to a sensible window (image bounds)
// to bound memory use during analyse. Returns false if the image isn't
// available.
bool fetchFrameImage(OFX::Clip* clip, double frame, dasgrain::FrameImage& out) {
    out.data.clear();
    if (!clip || !clip->isConnected()) return false;
    std::unique_ptr<OFX::Image> img(clip->fetchImage(frame));
    if (!img) return false;
    if (img->getPixelDepth() != OFX::eBitDepthFloat) return false;
    const OfxRectI b = img->getBounds();
    const int w = b.x2 - b.x1;
    const int h = b.y2 - b.y1;
    if (w <= 0 || h <= 0) return false;
    out.width  = w;
    out.height = h;
    out.data.resize(static_cast<size_t>(w) * h * 4);
    for (int y = 0; y < h; ++y) {
        const auto* src = static_cast<const float*>(img->getPixelAddress(b.x1, b.y1 + y));
        if (!src) {
            std::memset(out.data.data() + static_cast<size_t>(y) * w * 4,
                        0,
                        static_cast<size_t>(w) * 4 * sizeof(float));
            continue;
        }
        std::memcpy(out.data.data() + static_cast<size_t>(y) * w * 4,
                    src,
                    static_cast<size_t>(w) * 4 * sizeof(float));
    }
    return true;
}

bool copyViewToFrameImage(const kernels::ImageViewConst& view, dasgrain::FrameImage& out) {
    out.data.clear();
    if (!view.data || view.width <= 0 || view.height <= 0 || view.stride <= 0) return false;
    out.width = view.width;
    out.height = view.height;
    out.data.resize(static_cast<size_t>(view.width) * view.height * 4);
    for (int y = 0; y < view.height; ++y) {
        const auto* src = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(view.data) + static_cast<size_t>(y) * view.stride);
        std::memcpy(out.data.data() + static_cast<size_t>(y) * view.width * 4,
                    src,
                    static_cast<size_t>(view.width) * 4 * sizeof(float));
    }
    return true;
}

ResponseCurve buildCurrentFrameResponseCurve(const dasgrain::FrameImage& plate,
                                             const dasgrain::FrameImage& degrained,
                                             int sampleCount) {
    ResponseCurve curve;
    if (plate.data.empty() || degrained.data.empty()) return curve;
    if (plate.width != degrained.width || plate.height != degrained.height) return curve;

    const int n = plate.width * plate.height;
    (void)n;
    (void)sampleCount;

    // Resolve/Fusion current-frame fetch is reliable for the grain pixels
    // themselves but too sparse for a stable response curve on HDR EXR plates.
    // Use the Nuke-neutral amplitude curve: with cClean == cComp == eps, the
    // kernel transplants the measured plate-degrained grain directly.
    for (int c = 0; c < ResponseCurve::kChannelCount; ++c) {
        std::vector<CurvePoint> points = {{0.0, 0.01}, {1.0, 0.01}};
        curve.setChannel(c, std::move(points));
    }
    return curve;

    const int B = std::clamp(sampleCount, 2, 256);
    for (int c = 0; c < ResponseCurve::kChannelCount; ++c) {
        float lo = std::numeric_limits<float>::infinity();
        float hi = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < n; ++i) {
            const float d = degrained.data[static_cast<size_t>(i) * 4 + c];
            const float p = plate.data[static_cast<size_t>(i) * 4 + c];
            if (!std::isfinite(d) || !std::isfinite(p)) continue;
            lo = std::min(lo, d);
            hi = std::max(hi, d);
        }
        if (!std::isfinite(lo) || !std::isfinite(hi)) continue;
        if (hi <= lo) {
            const float center = lo;
            lo = std::max(0.0f, center - 0.5f);
            hi = std::max(lo + 1.0e-6f, center + 0.5f);
        }

        std::vector<double> sums(static_cast<size_t>(B), 0.0);
        std::vector<double> counts(static_cast<size_t>(B), 0.0);
        double globalSum = 0.0;
        double globalCount = 0.0;
        for (int i = 0; i < n; ++i) {
            const float d = degrained.data[static_cast<size_t>(i) * 4 + c];
            const float p = plate.data[static_cast<size_t>(i) * 4 + c];
            if (!std::isfinite(d) || !std::isfinite(p)) continue;
            const double residual = std::fabs(p - d);
            if (!std::isfinite(residual) || residual <= 0.0) continue;
            const float t = (d - lo) / (hi - lo);
            if (!std::isfinite(t)) continue;
            const int bucket = std::clamp(static_cast<int>(std::floor(t * float(B - 1) + 0.5f)), 0, B - 1);
            sums[static_cast<size_t>(bucket)] += residual;
            counts[static_cast<size_t>(bucket)] += 1.0;
            globalSum += residual;
            globalCount += 1.0;
        }

        const double globalMean = globalCount > 0.0 ? (globalSum / globalCount) : 0.0;
        std::vector<CurvePoint> points;
        points.reserve(static_cast<size_t>(B));
        for (int i = 0; i < B; ++i) {
            const double x = double(lo) + (double(i) / double(B - 1)) * double(hi - lo);
            double y = counts[static_cast<size_t>(i)] > 0.0
                ? sums[static_cast<size_t>(i)] / counts[static_cast<size_t>(i)]
                : globalMean;
            // Sparse current-frame buckets are common in Resolve/Fusion. The
            // Nuke gizmo still preserves the source grain floor; do the same so
            // empty or weak comp-tone buckets do not undergrain the render.
            y = std::max(y, globalMean);
            if (std::isfinite(x) && std::isfinite(y) && y > 0.0) {
                points.push_back({x, y});
            }
        }
        if (points.size() == 1) {
            const double y = points.front().y;
            points = {{0.0, y}, {1.0, y}};
        }
        if (points.size() >= 2) {
            curve.setChannel(c, std::move(points));
        }
    }
    return curve;
}

}  // namespace

void DasGrainPlugin::runAnalysePass(const OFX::RenderArguments& args) {
    if (!plate_ || !degrained_) return;

    // Mark that we're inside the analyse pass. If anything throws below the
    // catch handlers reset the state too, so a partial failure can't lock
    // the plugin out of normal renders.
    analyseState_->setValue(analyse_state::kRunning);

    // ---- Build config + frame list ---------------------------------------
    AnalyseConfig cfg;
    cfg.sampleCount      = sampleCount_->getValueAtTime(args.time);
    cfg.numberOfFrames   = numberOfFrames_->getValueAtTime(args.time);
    cfg.luminance        = luminance_->getValueAtTime(args.time);
    additionalFrames_->getValue(cfg.additionalFrames);
    double t1 = 0.0, t2 = 0.0;
    timeLineGetBounds(t1, t2);
    cfg.firstFrame = static_cast<int>(std::lround(t1));
    cfg.lastFrame  = static_cast<int>(std::lround(t2));
    cfg.useMask    = (mask_ && mask_->isConnected());
    cfg.invertMask = analyseMaskInvert_ ? analyseMaskInvert_->getValueAtTime(args.time) : false;

    auto frames = computeSampleFrames(cfg);
    if (frames.empty()) {
        sendMessage(OFX::Message::eMessageError, "dasgrain_analyse",
                    "No analysis frames. Set 'number of frames' > 0 or "
                    "list explicit frames in 'additional frames'.");
        analyseState_->setValue(analyse_state::kIdle);
        return;
    }

    progressStart("DasGrain analyse", "dasgrain_analyse");
    auto progress = [&](double t) {
        return progressUpdate(t);
    };

    // OFX → Analyser bridge.
    auto fetcher = [&](dasgrain::ClipKind kind,
                       int frame,
                       dasgrain::FrameImage& out) -> bool {
        OFX::Clip* clip = nullptr;
        switch (kind) {
            case dasgrain::ClipKind::kPlate:     clip = plate_;     break;
            case dasgrain::ClipKind::kDegrained: clip = degrained_; break;
            case dasgrain::ClipKind::kMask:      clip = mask_;      break;
        }
        return fetchFrameImage(clip, double(frame), out);
    };

    auto result = runAnalyse(cfg, frames, fetcher, progress);

    // Resolve/Fusion can report timeline sample frames that fetch black or
    // duplicated images for multi-input OFX clips, producing a formally
    // successful but useless identity curve. The current render time is known
    // to be valid because the normal grain renderer has just been asked to
    // evaluate the connected Plate/Degrained inputs at this time, so retry a
    // one-frame Nuke-style analyse there before giving up.
    const std::string identityCurveJSON = ResponseCurve().toJSON();
    if (result.ok && result.curve.toJSON() == identityCurveJSON) {
        const int currentFrame = static_cast<int>(std::lround(args.time));
        AnalyseConfig currentCfg = cfg;
        currentCfg.firstFrame = currentFrame;
        currentCfg.lastFrame = currentFrame;
        currentCfg.numberOfFrames = 0;
        currentCfg.additionalFrames = std::to_string(currentFrame);
        auto currentResult = runAnalyse(currentCfg,
                                        std::vector<int>{currentFrame},
                                        fetcher,
                                        progress);
        if (currentResult.ok && currentResult.curve.toJSON() != identityCurveJSON) {
            result = currentResult;
        }
    }

    progressEnd();

    if (!result.ok) {
        sendMessage(OFX::Message::eMessageError, "dasgrain_analyse",
                    std::string("Analyse failed: ") + result.error);
        analyseState_->setValue(analyse_state::kIdle);
        return;
    }

    // ---- Write curve back into params ------------------------------------
    // Resolve / Nuke both tolerate param edits from inside render() for
    // analyse-style plugins (the OFX spec is more permissive in
    // changedParam, but in practice this works). We wrap in
    // beginEditBlock / endEditBlock so the host treats this as a single
    // undoable operation.
    analysedCurve_ = result.curve;
    hasAnalysedCurve_ = true;

    const std::string json = result.curve.toJSON();
    beginEditBlock("DasGrain Analyse");
    if (responseCurveJSON_) responseCurveJSON_->setValue(json);
    if (analyseState_) analyseState_->setValue(analyse_state::kIdle);
    endEditBlock();
}

void DasGrainPlugin::renderPassThrough(const OFX::RenderArguments& args) {
    std::unique_ptr<OFX::Image> dst(dst_->fetchImage(args.time));
    if (!dst) return;

    std::unique_ptr<OFX::Image> src(source_ && source_->isConnected()
                                        ? source_->fetchImage(args.time)
                                        : nullptr);

    const OfxRectI dstBounds = dst->getBounds();
    const OfxRectI rw = args.renderWindow;

    // If we don't have a connected Source, output black for now.
    if (!src) {
        for (int y = rw.y1; y < rw.y2; ++y) {
            auto* row = static_cast<float*>(dst->getPixelAddress(rw.x1, y));
            std::memset(row, 0,
                        static_cast<size_t>(rw.x2 - rw.x1) * 4 * sizeof(float));
        }
        return;
    }

    // Pixel-by-pixel pass-through that handles bound mismatches between src
    // and dst (e.g. tiling). Once Phase 2 adds the GPU kernels, this becomes
    // the CPU fallback.
    for (int y = rw.y1; y < rw.y2; ++y) {
        if (abort()) break;
        auto* drow = static_cast<float*>(dst->getPixelAddress(rw.x1, y));
        for (int x = rw.x1; x < rw.x2; ++x) {
            auto* spx = static_cast<float*>(src->getPixelAddress(x, y));
            if (spx) {
                drow[0] = spx[0];
                drow[1] = spx[1];
                drow[2] = spx[2];
                drow[3] = spx[3];
            } else {
                drow[0] = drow[1] = drow[2] = drow[3] = 0.0f;
            }
            drow += 4;
        }
    }
    (void)dstBounds;
}

// ---------------------------------------------------------------------------
// Identity / clip prefs / changedParam
// ---------------------------------------------------------------------------

bool DasGrainPlugin::isIdentity(const OFX::IsIdentityArguments& /*args*/,
                                OFX::Clip*& /*identityClip*/,
                                double& /*identityTime*/) {
    // Phase 2 will short-circuit when output==regrained AND luminance==1 AND
    // no grain is detected; for Phase 1 we always render.
    return false;
}

void DasGrainPlugin::changedParam(const OFX::InstanceChangedArgs& /*args*/,
                                  const std::string& name) {
    if (name == params::kAnalyse) {
        if (analyseState_) analyseState_->setValue(analyse_state::kRequested);
        return;
    }
    if (name == params::kCurveHelp) {
        sendMessage(OFX::Message::eMessageMessage, "dasgrain_curve_help",
                    "DasGrain analyses Plate - Degrained, normalises that "
                    "grain by Plate luminance, then adapts it to Source "
                    "luminance. In Resolve, use grain amount plus shadow / "
                    "midtone / highlight grain for the normal artist-facing "
                    "curve work.\n\n"
                    "Curve pivot chooses where midtones sit. Curve contrast "
                    "controls how strongly those tone knobs bend the response. "
                    "RGB grain trims are for small color-channel bias fixes.\n\n"
                    "Raw curve JSON is only for advanced import/export or "
                    "debugging; enable 'show curve JSON tools' if you need it.");
        return;
    }
    if (name == params::kTroubleshoot) {
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
        std::vector<std::string> infos;

        const bool hasComp = source_ && source_->isConnected();
        const bool hasPlt  = plate_  && plate_->isConnected();
        const bool hasDeg  = degrained_ && degrained_->isConnected();
        if (!hasComp || !hasPlt || !hasDeg) {
            errors.push_back("Plate, degrained plate AND comp must be "
                             "connected for the regrain to do anything.");
        }

        // Format mismatch: width/height/PAR across plate, degrained, comp.
        auto fmt = [](OFX::Clip* c) -> std::tuple<double,double,double> {
            if (!c || !c->isConnected()) return {0, 0, 0};
            const auto rod = c->getRegionOfDefinition(0.0);
            return std::make_tuple(rod.x2 - rod.x1, rod.y2 - rod.y1,
                                   c->getPixelAspectRatio());
        };
        if (hasPlt && hasDeg) {
            auto a = fmt(plate_);
            auto b = fmt(degrained_);
            if (a != b) warnings.push_back(
                "Format mismatch: plate vs degrained have different "
                "dimensions / pixel aspect.");
        }
        if (hasPlt && hasComp) {
            auto a = fmt(plate_);
            auto b = fmt(source_);
            if (a != b) warnings.push_back(
                "Format mismatch: plate vs comp have different "
                "dimensions / pixel aspect.");
        }

        // Frame range mismatch: plate / degrained should cover the same
        // window so analysis frame lookup is consistent.
        if (hasPlt && hasDeg) {
            const OfxRangeD pr = plate_->getFrameRange();
            const OfxRangeD dr = degrained_->getFrameRange();
            if (pr.min != dr.min || pr.max != dr.max) {
                warnings.push_back(
                    "Frame range mismatch: plate ["
                    + std::to_string(int(pr.min)) + ".." + std::to_string(int(pr.max))
                    + "] vs degrained ["
                    + std::to_string(int(dr.min)) + ".." + std::to_string(int(dr.max))
                    + "]. Double-check that they belong together.");
            }
        }

        // luminance == 1 sanity prompt.
        if (luminance_ && std::fabs(luminance_->getValue() - 1.0) < 1e-6) {
            infos.push_back("luminance degrain amount is 1.0. If the "
                            "degrained plate retains some luma grain, run "
                            "DegrainHelper first and use the same value here.");
        }

        // Blank curve check.
        std::string curveJSON;
        if (responseCurveJSON_) responseCurveJSON_->getValue(curveJSON);
        if (curveJSON.empty()) {
            errors.push_back("You haven't pressed the Analyse button yet "
                             "(response curve is empty).");
        }

        // Curve slope check: each per-channel curve must be non-decreasing.
        if (!curveJSON.empty()) {
            ResponseCurve checkCurve;
            if (checkCurve.fromJSON(curveJSON)) {
                constexpr int kN = 32;
                const char* names[3] = {"red", "green", "blue"};
                for (int c = 0; c < 3; ++c) {
                    double prev = checkCurve.sample(c, 0.0);
                    bool nonMonotone = false;
                    for (int i = 1; i < kN; ++i) {
                        const double x = double(i) / double(kN - 1);
                        const double y = checkCurve.sample(c, x);
                        if (y < prev - 1e-6) { nonMonotone = true; break; }
                        prev = y;
                    }
                    if (nonMonotone) {
                        warnings.push_back(std::string("Response curve slope for ")
                            + names[c] + " is non-monotonic (should always go up "
                                         "/). Re-run Analyse or import a clean "
                                         "curve via the Curve JSON paste field.");
                    }
                }
            } else {
                warnings.push_back("Response curve JSON failed to parse.");
            }
        }

        // Default sample-box check.
        if (scatter_ && scatter_->getValue() && boxLow_ && boxHigh_) {
            double lx, ly, hx, hy;
            boxLow_->getValue(lx, ly);
            boxHigh_->getValue(hx, hy);
            const bool atDefault =
                std::fabs(lx - defaults::kBoxLowX)  < 1.0 &&
                std::fabs(ly - defaults::kBoxLowY)  < 1.0 &&
                std::fabs(hx - defaults::kBoxHighX) < 1.0 &&
                std::fabs(hy - defaults::kBoxHighY) < 1.0;
            if (atDefault) {
                warnings.push_back("Scatter is on but the sample box is "
                                   "still at its default position.");
            }
            const bool degenerate = std::fabs(hx - lx) < 1.0
                                  || std::fabs(hy - ly) < 1.0;
            if (degenerate) {
                errors.push_back("Sample box is degenerate (zero width or "
                                 "height). Increase its size or decrease "
                                 "cell size.");
            }
        }

        // Compose the final message. Errors first, then warnings, then info.
        std::string msg;
        auto append = [&](const char* label, const std::vector<std::string>& v) {
            if (v.empty()) return;
            msg += std::string("\n") + label + "\n";
            for (size_t i = 0; i < v.size(); ++i) {
                msg += "  " + std::to_string(i + 1) + ". " + v[i] + "\n";
            }
        };
        append("ERRORS:",   errors);
        append("WARNINGS:", warnings);
        append("INFO:",     infos);
        if (msg.empty()) {
            msg = "DasGrain looks healthy. If the regrain still isn't "
                  "right, double-check that plate / degrained / comp "
                  "haven't been retouched (paint, despill, etc.).";
        }
        sendMessage(OFX::Message::eMessageMessage, "dasgrain_help", msg);
        return;
    }
    if (name == params::kCurveImport) {
        if (!curveJSONPaste_ || !responseCurveJSON_) return;
        std::string json;
        curveJSONPaste_->getValue(json);
        ResponseCurve curve;
        if (!curve.fromJSON(json)) {
            sendMessage(OFX::Message::eMessageError, "dasgrain_curve_import",
                        "Could not parse response-curve JSON.");
            return;
        }
        analysedCurve_ = curve;
        hasAnalysedCurve_ = true;
        beginEditBlock("DasGrain import curve");
        responseCurveJSON_->setValue(curve.toJSON());
        endEditBlock();
        return;
    }
    if (name == params::kCurveExport) {
        // Echo the canonical curve into the editable paste field plus a
        // message dialog so the user can copy/paste it.
        if (!responseCurveJSON_ || !curveJSONPaste_) return;
        std::string json;
        responseCurveJSON_->getValue(json);
        if (json.empty()) {
            sendMessage(OFX::Message::eMessageMessage, "dasgrain_curve_export",
                        "No curve to export — run Analyse first.");
            return;
        }
        curveJSONPaste_->setValue(json);
        sendMessage(OFX::Message::eMessageMessage, "dasgrain_curve_export",
                    "Response curve JSON copied into the 'Curve JSON' field.\n"
                    "First 80 chars: " + json.substr(0, 80) + "...");
        return;
    }
    if (name == params::kScatter || name == params::kExternalGrain
        || name == params::kShowCurveIO) {
        updateUiEnabled();
        return;
    }
    if (name == params::kBoxLow || name == params::kBoxHigh) {
        if (sampleFrame_) {
            sampleFrame_->setValue(static_cast<int>(timeLineGetTime()));
        }
        return;
    }
}

void DasGrainPlugin::changedClip(const OFX::InstanceChangedArgs& /*args*/,
                                 const std::string& /*name*/) {
    updateUiEnabled();
}

void DasGrainPlugin::getFramesNeeded(const OFX::FramesNeededArguments& args,
                                     OFX::FramesNeededSetter& frames) {
    OfxRangeD here{args.time, args.time};
    if (source_)        frames.setFramesNeeded(*source_,        here);
    if (mask_)          frames.setFramesNeeded(*mask_,          here);
    if (externalGrain_) frames.setFramesNeeded(*externalGrain_, here);

    // When the analyse state machine is in "requested" mode the upcoming
    // render needs the full union of analysis sample frames on Plate and
    // Degrained. Asking for the wider range here lets the host pre-fetch /
    // cache the frames before we start the multi-frame reductions.
    bool needAnalysisFrames = false;
    if (analyseState_) {
        std::string state;
        analyseState_->getValue(state);
        needAnalysisFrames = (state == analyse_state::kRequested);
    }

    OfxRangeD plateRange = here;
    OfxRangeD degRange   = here;
    if (needAnalysisFrames) {
        AnalyseConfig cfg;
        cfg.sampleCount    = sampleCount_->getValue();
        cfg.numberOfFrames = numberOfFrames_->getValue();
        additionalFrames_->getValue(cfg.additionalFrames);
        double t1 = 0.0, t2 = 0.0;
        timeLineGetBounds(t1, t2);
        cfg.firstFrame = static_cast<int>(std::lround(t1));
        cfg.lastFrame  = static_cast<int>(std::lround(t2));
        const auto fl = computeSampleFrames(cfg);
        if (!fl.empty()) {
            plateRange.min = std::min(plateRange.min, double(fl.front()));
            plateRange.max = std::max(plateRange.max, double(fl.back()));
            degRange       = plateRange;
        }
    }

    // Scatter needs plate/deg at sample_frame as well (FrameHold equivalent).
    if (scatter_ && scatter_->getValue()) {
        const int sf = sampleFrame_ ? sampleFrame_->getValue() : int(args.time);
        OfxRangeD scatterRange{double(sf), double(sf)};
        plateRange.min = std::min(plateRange.min, scatterRange.min);
        plateRange.max = std::max(plateRange.max, scatterRange.max);
        degRange.min   = std::min(degRange.min,   scatterRange.min);
        degRange.max   = std::max(degRange.max,   scatterRange.max);
    }

    if (plate_)     frames.setFramesNeeded(*plate_,     plateRange);
    if (degrained_) frames.setFramesNeeded(*degrained_, degRange);
}

void DasGrainPlugin::getClipPreferences(OFX::ClipPreferencesSetter& /*prefs*/) {
    // Default behaviour matches what we want: float RGBA.
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

void DasGrainPlugin::updateUiEnabled() {
    const bool scat = scatter_ ? scatter_->getValue() : false;
    if (boxLow_)         boxLow_->setEnabled(scat);
    if (boxHigh_)        boxHigh_->setEnabled(scat);
    if (sampleFrame_)    sampleFrame_->setEnabled(scat);
    if (cellSize_)       cellSize_->setEnabled(scat);
    if (overlayPattern_) overlayPattern_->setEnabled(scat);
    if (edgeBlendSize_)  edgeBlendSize_->setEnabled(scat);
    if (amplitude_)      amplitude_->setEnabled(scat);
    if (frequency_)      frequency_->setEnabled(scat);
    if (seed_)           seed_->setEnabled(scat);

    const bool curveIO = showCurveIO_ ? showCurveIO_->getValue() : false;
    if (curveJSONPaste_) curveJSONPaste_->setEnabled(curveIO);
    if (curveImport_)    curveImport_->setEnabled(curveIO);
    if (curveExport_)    curveExport_->setEnabled(curveIO);
}

}  // namespace dasgrain
