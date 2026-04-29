// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0

#include "DegrainHelperPlugin.h"

#include <algorithm>
#include <cstring>
#include <memory>

#include "ofxsImageEffect.h"

namespace dasgrain {

namespace {

constexpr const char* kPluginName        = "DegrainHelper";
constexpr const char* kPluginGrouping    = "OFX/DasGrain";
constexpr const char* kPluginDescription =
    "Mixes a fraction of the plate's luminance grain back onto a fully "
    "degrained plate. Use the same value here and in DasGrain's "
    "'luminance degrain amount' knob.";
constexpr const char* kPluginIdentifier  = "com.dasgrain.ofx.DegrainHelper";
constexpr unsigned int kPluginMajor = 1;
constexpr unsigned int kPluginMinor = 0;

constexpr const char* kPlateClip     = "Plate";
constexpr const char* kDegrainedClip = "Degrained";

constexpr const char* kLumaMixParam  = "luma_mix";

// Rec.601 / Rec.709 BT.709-ish RGB→Y mix. Matches Nuke's YCbCr default.
inline float lumaY(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

}  // namespace

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

DegrainHelperPlugin::DegrainHelperPlugin(OfxImageEffectHandle handle)
    : OFX::ImageEffect(handle) {
    dst_       = fetchClip(kOfxImageEffectOutputClipName);
    plate_     = fetchClip(kPlateClip);
    degrained_ = fetchClip(kDegrainedClip);
    lumaMix_   = fetchDoubleParam(kLumaMixParam);
}

void DegrainHelperPlugin::render(const OFX::RenderArguments& args) {
    if (!dst_) OFX::throwSuiteStatusException(kOfxStatErrBadHandle);
    if (dst_->getPixelDepth() != OFX::eBitDepthFloat ||
        dst_->getPixelComponents() != OFX::ePixelComponentRGBA) {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }

    std::unique_ptr<OFX::Image> dst(dst_->fetchImage(args.time));
    std::unique_ptr<OFX::Image> plateImg(plate_     ? plate_->fetchImage(args.time)     : nullptr);
    std::unique_ptr<OFX::Image> degImg  (degrained_ ? degrained_->fetchImage(args.time) : nullptr);
    if (!dst) return;

    const OfxRectI rw = args.renderWindow;
    const float mix   = float(lumaMix_->getValueAtTime(args.time));

    for (int y = rw.y1; y < rw.y2; ++y) {
        if (abort()) break;
        auto* drow = static_cast<float*>(dst->getPixelAddress(rw.x1, y));
        for (int x = rw.x1; x < rw.x2; ++x) {
            const auto* prow = plateImg ? static_cast<const float*>(plateImg->getPixelAddress(x, y)) : nullptr;
            const auto* gpx  = degImg   ? static_cast<const float*>(degImg  ->getPixelAddress(x, y)) : nullptr;

            // No degrained → output black so the user notices the missing input.
            if (!gpx) {
                drow[0] = drow[1] = drow[2] = drow[3] = 0.0f;
                drow += 4;
                continue;
            }
            // Without a plate input we just pass the degrained through.
            if (!prow) {
                drow[0] = gpx[0]; drow[1] = gpx[1]; drow[2] = gpx[2]; drow[3] = gpx[3];
                drow += 4;
                continue;
            }

            // The gizmo's DegrainHelper:
            //   plate(YCbCr).Y mixed onto degrained(YCbCr).Y by (1 - luma_mix)
            // (the Copy node uses `mix {{1-parent.luma_mix}}` and copies
            // rgba.red — i.e. the Y channel after Colorspace YCbCr).
            //   out.Y = lerp(deg.Y, plate.Y, 1 - luma_mix)
            const float yP = lumaY(prow[0], prow[1], prow[2]);
            const float yD = lumaY(gpx[0],  gpx[1],  gpx[2]);
            const float t  = std::clamp(1.0f - mix, 0.0f, 1.0f);
            const float yOut = yD + (yP - yD) * t;
            // Add the Y-delta to all RGB channels (matches Nuke YCbCr where
            // Y maps additively to R/G/B).
            const float dY = yOut - yD;
            drow[0] = gpx[0] + dY;
            drow[1] = gpx[1] + dY;
            drow[2] = gpx[2] + dY;
            drow[3] = gpx[3];
            drow += 4;
        }
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

DegrainHelperFactory::DegrainHelperFactory()
    : OFX::PluginFactoryHelper<DegrainHelperFactory>(kPluginIdentifier,
                                                     kPluginMajor,
                                                     kPluginMinor) {}

void DegrainHelperFactory::describe(OFX::ImageEffectDescriptor& desc) {
    desc.setLabels(kPluginName, kPluginName, kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(kPluginDescription);

    desc.addSupportedContext(OFX::eContextGeneral);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);

    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(false);
    desc.setSupportsTiles(false);
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);
}

void DegrainHelperFactory::describeInContext(OFX::ImageEffectDescriptor& desc,
                                             OFX::ContextEnum /*context*/) {
    using namespace OFX;

    auto* sourceClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    sourceClip->addSupportedComponent(ePixelComponentRGBA);
    sourceClip->setSupportsTiles(false);
    sourceClip->setIsMask(false);
    sourceClip->setOptional(true);
    sourceClip->setLabel("(unused)");

    auto* plateClip = desc.defineClip(kPlateClip);
    plateClip->addSupportedComponent(ePixelComponentRGBA);
    plateClip->setSupportsTiles(false);
    plateClip->setIsMask(false);
    plateClip->setOptional(false);
    plateClip->setLabel("Plate");

    auto* degrainedClip = desc.defineClip(kDegrainedClip);
    degrainedClip->addSupportedComponent(ePixelComponentRGBA);
    degrainedClip->setSupportsTiles(false);
    degrainedClip->setIsMask(false);
    degrainedClip->setOptional(false);
    degrainedClip->setLabel("Degrained");

    auto* dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->setSupportsTiles(false);

    auto* page = desc.definePageParam("page_main");

    auto* p = desc.defineDoubleParam(kLumaMixParam);
    p->setLabels("luminance degrain amount",
                 "luminance degrain amount",
                 "luminance degrain amount");
    p->setScriptName(kLumaMixParam);
    p->setHint("0 = use plate luma, 1 = use degrained luma. Use the same "
               "value as DasGrain's 'luminance degrain amount' knob.");
    p->setRange(0.0, 1.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.8);
    p->setAnimates(false);
    page->addChild(*p);
}

OFX::ImageEffect* DegrainHelperFactory::createInstance(OfxImageEffectHandle handle,
                                                        OFX::ContextEnum /*context*/) {
    return new DegrainHelperPlugin(handle);
}

}  // namespace dasgrain
