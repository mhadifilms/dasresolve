// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0

#include "DasGrainFactory.h"

#include "ofxsImageEffect.h"
#include "ofxsParam.h"

#include "DasGrainInteract.h"
#include "DasGrainPlugin.h"
#include "Params.h"

namespace dasgrain {

namespace {

constexpr const char* kPluginName        = "DasGrain";
constexpr const char* kPluginGrouping    = "OFX/DasGrain";
constexpr const char* kPluginDescription =
    "Re-grain a comp by analysing the original plate against its degrained "
    "version, then adapting the extracted grain back onto the comp using a "
    "luminance-driven response curve. Optional Voronoi scatter lets you "
    "reuse a small clean grain area to cover the whole frame.";
constexpr const char* kPluginIdentifier = "com.dasgrain.ofx.DasGrain";
constexpr unsigned int kPluginMajor = 1;
constexpr unsigned int kPluginMinor = 0;

// Tiles default for now; we can flip this on once the kernels respect
// tile-bounded inputs (Phase 2 implements that).
constexpr bool kSupportsTiles            = false;
constexpr bool kSupportsMultiResolution  = false;
constexpr bool kSupportsMultipleClipPARs = false;

// ---------------------------------------------------------------------------
// Param-construction helpers
// ---------------------------------------------------------------------------

OFX::DoubleParamDescriptor* defineDouble(OFX::ImageEffectDescriptor& desc,
                                         const char* name,
                                         const char* label,
                                         const char* hint,
                                         double minVal, double maxVal,
                                         double defaultVal,
                                         OFX::PageParamDescriptor* page,
                                         OFX::GroupParamDescriptor* parent = nullptr) {
    auto* p = desc.defineDoubleParam(name);
    p->setLabels(label, label, label);
    p->setScriptName(name);
    p->setHint(hint);
    p->setRange(minVal, maxVal);
    p->setDisplayRange(minVal, maxVal);
    p->setDefault(defaultVal);
    p->setAnimates(false);
    if (parent) p->setParent(*parent);
    if (page)   page->addChild(*p);
    return p;
}

OFX::IntParamDescriptor* defineInt(OFX::ImageEffectDescriptor& desc,
                                   const char* name,
                                   const char* label,
                                   const char* hint,
                                   int minVal, int maxVal,
                                   int defaultVal,
                                   OFX::PageParamDescriptor* page,
                                   OFX::GroupParamDescriptor* parent = nullptr) {
    auto* p = desc.defineIntParam(name);
    p->setLabels(label, label, label);
    p->setScriptName(name);
    p->setHint(hint);
    p->setRange(minVal, maxVal);
    p->setDisplayRange(minVal, maxVal);
    p->setDefault(defaultVal);
    p->setAnimates(false);
    if (parent) p->setParent(*parent);
    if (page)   page->addChild(*p);
    return p;
}

OFX::BooleanParamDescriptor* defineBool(OFX::ImageEffectDescriptor& desc,
                                        const char* name,
                                        const char* label,
                                        const char* hint,
                                        bool defaultVal,
                                        OFX::PageParamDescriptor* page,
                                        OFX::GroupParamDescriptor* parent = nullptr) {
    auto* p = desc.defineBooleanParam(name);
    p->setLabels(label, label, label);
    p->setScriptName(name);
    p->setHint(hint);
    p->setDefault(defaultVal);
    p->setAnimates(false);
    if (parent) p->setParent(*parent);
    if (page)   page->addChild(*p);
    return p;
}

OFX::StringParamDescriptor* defineString(OFX::ImageEffectDescriptor& desc,
                                         const char* name,
                                         const char* label,
                                         const char* hint,
                                         const char* defaultVal,
                                         OFX::PageParamDescriptor* page,
                                         OFX::GroupParamDescriptor* parent = nullptr) {
    auto* p = desc.defineStringParam(name);
    p->setLabels(label, label, label);
    p->setScriptName(name);
    p->setHint(hint);
    if (defaultVal) p->setDefault(defaultVal);
    p->setAnimates(false);
    if (parent) p->setParent(*parent);
    if (page)   page->addChild(*p);
    return p;
}

}  // namespace

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

DasGrainFactory::DasGrainFactory()
    : OFX::PluginFactoryHelper<DasGrainFactory>(kPluginIdentifier, kPluginMajor, kPluginMinor) {}

void DasGrainFactory::describe(OFX::ImageEffectDescriptor& desc) {
    desc.setLabels(kPluginName, kPluginName, kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(kPluginDescription);

    // We declare both eContextFilter (single Source — required by Resolve's
    // Color/Edit page OFX host and by Resolve's embedded Fusion adapter)
    // and eContextGeneral (multi-input — Source + Plate + Degrained + Mask
    // + ExternalGrain — which Fusion Studio standalone honours and which
    // is the original DasGrain workflow). When instantiated in Filter
    // context, the non-Source clips simply stay unconnected and the plugin
    // falls back to a passthrough render.
    desc.addSupportedContext(OFX::eContextFilter);
    desc.addSupportedContext(OFX::eContextGeneral);

    desc.addSupportedBitDepth(OFX::eBitDepthFloat);

    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(kSupportsMultiResolution);
    desc.setSupportsTiles(kSupportsTiles);
    // Analyse pulls multiple frames per render; the scatter pass uses a
    // FrameHold-equivalent for sample_frame. Both require temporal access.
    desc.setTemporalClipAccess(true);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

    // GPU support flags.
    desc.setSupportsOpenCLBuffersRender(true);
#ifndef __APPLE__
    desc.setSupportsCudaRender(true);
    desc.setSupportsCudaStream(true);
#endif
#ifdef __APPLE__
    desc.setSupportsMetalRender(true);
#endif

    // Sample-box overlay. The descriptor is owned by InteractMainEntry so
    // we just hand a pointer in — Support deletes it.
    desc.setOverlayInteractDescriptor(new DasGrainOverlayDescriptor());
}

void DasGrainFactory::describeInContext(OFX::ImageEffectDescriptor& desc,
                                        OFX::ContextEnum /*context*/) {
    using namespace OFX;

    // ---- Clips ------------------------------------------------------------
    auto setupImageClip = [](ClipDescriptor* c, bool optional) {
        c->addSupportedComponent(ePixelComponentRGBA);
        c->setTemporalClipAccess(true);
        c->setSupportsTiles(kSupportsTiles);
        c->setIsMask(false);
        c->setOptional(optional);
    };

    // OFX requires a Source clip; we use it as the COMP input.
    auto* sourceClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    setupImageClip(sourceClip, /*optional=*/false);
    sourceClip->setLabel("Comp");

    // Plate / Degrained MUST be optional in OFX terms so the plugin can
    // be instantiated in single-input contexts (Resolve Color page, single
    // -track Fusion comp). The render path still requires both to be
    // connected for the full grain pipeline; if they're missing it falls
    // back to passthrough on the comp.
    auto* plateClip = desc.defineClip(clips::kPlate);
    setupImageClip(plateClip, /*optional=*/true);
    plateClip->setLabel("Plate");

    auto* degrainedClip = desc.defineClip(clips::kDegrained);
    setupImageClip(degrainedClip, /*optional=*/true);
    degrainedClip->setLabel("Degrained");

    auto* maskClip = desc.defineClip(clips::kMask);
    maskClip->addSupportedComponent(ePixelComponentRGBA);
    maskClip->addSupportedComponent(ePixelComponentAlpha);
    maskClip->setTemporalClipAccess(true);
    maskClip->setSupportsTiles(kSupportsTiles);
    maskClip->setIsMask(true);
    maskClip->setOptional(true);
    maskClip->setLabel("Mask");

    auto* externalGrainClip = desc.defineClip(clips::kExternalGrain);
    setupImageClip(externalGrainClip, /*optional=*/true);
    externalGrainClip->setLabel("External Grain");

    auto* dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->setSupportsTiles(kSupportsTiles);

    // ---- Pages (one per gizmo tab) ---------------------------------------
    auto* pageAnalyse = desc.definePageParam(params::kPageAnalyse);
    auto* pageAdjust  = desc.definePageParam(params::kPageAdjust);
    auto* pageReplace = desc.definePageParam(params::kPageReplace);
    auto* pageHelp    = desc.definePageParam(params::kPageHelp);

    // ---- Output mode (visible on every tab in the gizmo; we put it on
    //      Analyse to avoid duplication) -----------------------------------
    {
        auto* p = desc.defineChoiceParam(params::kOutput);
        p->setLabels("Output", "Output", "Output");
        p->setScriptName(params::kOutput);
        p->setHint("regrained comp / original grain / normalised grain / "
                   "adapted grain / grain QC.");
        p->appendOption("regrained comp");
        p->appendOption("original grain");
        p->appendOption("normalised grain");
        p->appendOption("adapted grain");
        p->appendOption("grain QC");
        p->setDefault(params::kOutputRegrained);
        p->setAnimates(false);
        pageAnalyse->addChild(*p);
        pageAdjust->addChild(*p);
        pageReplace->addChild(*p);
    }

    // ---- Analyse tab ------------------------------------------------------
    auto* groupLevel = desc.defineGroupParam("group_level");
    groupLevel->setLabels("Degrain amount", "Degrain amount", "Degrain amount");
    pageAnalyse->addChild(*groupLevel);

    defineDouble(desc, params::kLuminance, "luminance degrain amount",
                 "Leave at 1 for a fully degrained plate. If the degrained "
                 "plate retains some luminance grain (DegrainHelper), set "
                 "this to the same factor used there.",
                 0.0, 1.0, defaults::kLuminance, pageAnalyse, groupLevel);

    defineBool(desc, params::kFixGhosting, "fix ghosting",
               "Fixes detail ghosting introduced by luminance compensation "
               "on repo'd / retimed plates.",
               false, pageAnalyse, groupLevel);

    defineBool(desc, params::kGrainMaskInvert, "invert grain level mask",
               "Invert the grain-level mask channel.",
               false, pageAnalyse, groupLevel);

    auto* groupAnalyse = desc.defineGroupParam("group_analyse");
    groupAnalyse->setLabels("Analyse", "Analyse", "Analyse");
    pageAnalyse->addChild(*groupAnalyse);

    defineInt(desc, params::kNumberOfFrames, "number of frames",
              "Sample frames spread evenly over the input range. More "
              "frames = more accuracy. Use 0 to define every frame manually "
              "via 'additional frames'.",
              0, 200, defaults::kNumberOfFrames, pageAnalyse, groupAnalyse);

    defineString(desc, params::kAdditionalFrames, "additional frames",
                 "Comma-separated extra frames or ranges, e.g. "
                 "'1001,1020-1040,1100-1120x4'.",
                 "", pageAnalyse, groupAnalyse);

    defineInt(desc, params::kSampleCount, "sample count",
              "Luminance buckets used when fitting the grain response curve. "
              "More buckets = more curve detail but slower analysis.",
              4, 64, defaults::kSampleCount, pageAnalyse, groupAnalyse);

    {
        auto* p = desc.definePushButtonParam(params::kAnalyse);
        p->setLabels("Analyse", "Analyse", "Analyse");
        p->setScriptName(params::kAnalyse);
        p->setHint("Run the multi-frame analysis. Sample frames are pulled, "
                   "the grain response curve is fit and stored.");
        p->setParent(*groupAnalyse);
        pageAnalyse->addChild(*p);
    }

    {
        auto* p = defineString(desc, params::kAnalyseState,
                               "(internal) analyse state",
                               "Internal state machine. Don't touch.",
                               analyse_state::kIdle, nullptr, groupAnalyse);
        p->setIsSecret(true);
        p->setEvaluateOnChange(false);
        p->setEnabled(false);
    }

    defineBool(desc, params::kAnalyseMaskInvert, "invert analysis mask",
               "Invert the analysis mask.",
               false, pageAnalyse, groupAnalyse);

    // ---- Adjust tab -------------------------------------------------------
    {
        auto* p = desc.definePushButtonParam(params::kCurveHelp);
        p->setLabels("What am I looking at?",
                     "What am I looking at?",
                     "What am I looking at?");
        p->setHint("Pop up a quick explanation of the response curves.");
        pageAdjust->addChild(*p);
    }
    // NOTE: the original gizmo had an interactive parametric curve widget
    // here. OFX's `kOfxParamTypeParametric` is a host-optional feature that
    // DaVinci Resolve does NOT implement — calling defineParametricParam
    // makes Resolve's host fail OfxActionCreateInstance with
    // kOfxStatErrUnknown, which is exactly what showed up in
    // davinci_resolve.log when we tried to drop the plugin. So in this
    // port the curve is JSON-only: the Analyse pass writes the canonical
    // form into `response_curve_json`, and the user edits it via the
    // Curve JSON paste / Import / Export controls below.

    {
        auto* p = defineString(desc, params::kResponseCurveJSON,
                               "(internal) response curve JSON",
                               "Canonical persisted form of the response curve.",
                               "", pageAdjust);
        p->setIsSecret(true);
        p->setEvaluateOnChange(false);
        p->setEnabled(false);
    }

    // Curve JSON import/export. The user pastes the JSON for a previously
    // analysed curve here and clicks Import; Export echoes the current
    // curve back into this field plus a message dialog. The OFX
    // `eStringTypeMultiLine` mode is host-optional and Resolve doesn't
    // honour it (the curve JSON fits on one line in practice anyway).
    {
        auto* p = desc.defineStringParam(params::kCurveJSONImport);
        p->setLabels("Curve JSON",
                     "Curve JSON",
                     "Curve JSON");
        p->setHint("Paste / read response-curve JSON here. Use the Import "
                   "button to apply, or Export to populate it from the "
                   "current curve.");
        p->setStringType(eStringTypeSingleLine);
        p->setAnimates(false);
        p->setEvaluateOnChange(false);
        pageAdjust->addChild(*p);
    }
    {
        auto* p = desc.definePushButtonParam(params::kCurveImport);
        p->setLabels("Import curve", "Import curve", "Import curve");
        p->setHint("Apply the JSON in the field above to the response curve.");
        pageAdjust->addChild(*p);
    }
    {
        auto* p = desc.definePushButtonParam(params::kCurveExport);
        p->setLabels("Export curve", "Export curve", "Export curve");
        p->setHint("Write the current response curve into the JSON field "
                   "above (and into the diagnostic message dialog).");
        pageAdjust->addChild(*p);
    }

    // ---- Replace tab ------------------------------------------------------
    auto* groupExternal = desc.defineGroupParam("group_external");
    groupExternal->setLabels("External Grain", "External Grain", "External Grain");
    pageReplace->addChild(*groupExternal);

    defineBool(desc, params::kExternalGrain, "use external grain",
               "Use the ExternalGrain input (a normalised grain plate from "
               "another DasGrain) instead of the in-line normalised grain.",
               false, pageReplace, groupExternal);

    auto* groupScatter = desc.defineGroupParam("group_scatter");
    groupScatter->setLabels("Scatter", "Scatter", "Scatter");
    pageReplace->addChild(*groupScatter);

    defineBool(desc, params::kScatter, "activate scatter",
               "Voronoi-tile the sample-box across the frame to fabricate "
               "grain everywhere.",
               false, pageReplace, groupScatter);

    {
        auto* p = desc.defineDouble2DParam(params::kBoxLow);
        p->setLabels("sample box (low)", "sample box (low)", "sample box (low)");
        p->setHint("Lower-left corner of the sample box, in image-relative "
                   "coordinates 0..1.");
        p->setRange(0.0, 0.0, 1.0, 1.0);
        p->setDisplayRange(0.0, 0.0, 1.0, 1.0);
        p->setDefault(defaults::kBoxLowX, defaults::kBoxLowY);
        p->setParent(*groupScatter);
        pageReplace->addChild(*p);
    }
    {
        auto* p = desc.defineDouble2DParam(params::kBoxHigh);
        p->setLabels("sample box (high)", "sample box (high)", "sample box (high)");
        p->setHint("Upper-right corner of the sample box, in image-relative "
                   "coordinates 0..1.");
        p->setRange(0.0, 0.0, 1.0, 1.0);
        p->setDisplayRange(0.0, 0.0, 1.0, 1.0);
        p->setDefault(defaults::kBoxHighX, defaults::kBoxHighY);
        p->setParent(*groupScatter);
        pageReplace->addChild(*p);
    }

    defineInt(desc, params::kSampleFrame, "sample frame",
              "Frame at which the grain is sampled for the scatter source.",
              0, 100000, defaults::kSampleFrame, pageReplace, groupScatter);

    defineBool(desc, params::kOverlayPattern, "overlay cell pattern",
               "Overlay the Voronoi cell pattern (debug).",
               false, pageReplace, groupScatter);

    defineDouble(desc, params::kCellSize, "cell size",
                 "Size of the Voronoi cells (pixels).",
                 5.0, 100.0, defaults::kCellSize, pageReplace, groupScatter);

    defineInt(desc, params::kEdgeBlendSize, "edge blend size",
              "Conceal cell seams. Slow — keep below 3 if possible.",
              0, 8, defaults::kEdgeBlendSize, pageReplace, groupScatter);

    defineDouble(desc, params::kAmplitude, "distortion amplitude",
                 "Distortion amplitude in pixels.",
                 0.0, 50.0, defaults::kAmplitude, pageReplace, groupScatter);

    defineDouble(desc, params::kFrequency, "distortion frequency",
                 "Distortion frequency.",
                 0.0, 50.0, defaults::kFrequency, pageReplace, groupScatter);

    defineInt(desc, params::kSeed, "voronoi seed",
              "Seed for the Voronoi pattern.",
              0, 1000000, defaults::kSeed, pageReplace, groupScatter);

    defineBool(desc, params::kReplaceMaskInvert, "invert replace mask",
               "Invert the replace mask channel.",
               false, pageReplace);

    // ---- Help tab ---------------------------------------------------------
    {
        auto* p = desc.definePushButtonParam(params::kTroubleshoot);
        p->setLabels("Troubleshoot", "Troubleshoot", "Troubleshoot");
        p->setScriptName(params::kTroubleshoot);
        p->setHint("Sanity-check connections, formats, response curve and the "
                   "sample box. Sends a diagnostic message.");
        pageHelp->addChild(*p);
    }
}

OFX::ImageEffect* DasGrainFactory::createInstance(OfxImageEffectHandle handle,
                                                  OFX::ContextEnum /*context*/) {
    return new DasGrainPlugin(handle);
}

}  // namespace dasgrain
