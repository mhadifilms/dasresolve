// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ofxsImageEffect.h"

#include "core/ResponseCurve.h"

namespace dasgrain {

// Per-instance plugin state. Owns nothing it doesn't allocate; clip and
// param pointers are non-owning views into the OFX support library state.
class DasGrainPlugin : public OFX::ImageEffect {
public:
    explicit DasGrainPlugin(OfxImageEffectHandle handle);

    // OFX overrides ----------------------------------------------------------
    // Entry point for every rendered tile/frame. Dispatches analyse requests
    // first, then routes the current frame to the grain-apply backend.
    void render(const OFX::RenderArguments& args) override;

    // DasGrain is only identity when the host asks before clips are connected;
    // otherwise render() decides whether to pass through or apply grain.
    bool isIdentity(const OFX::IsIdentityArguments& args,
                    OFX::Clip*& identityClip,
                    double& identityTime) override;

    // Handles button clicks and UI toggles: Analyse state, help dialogs,
    // Troubleshoot diagnostics, JSON import/export, and dependent UI enabling.
    void changedParam(const OFX::InstanceChangedArgs& args,
                      const std::string& name) override;

    // Clip connection changes can affect which controls are meaningful.
    void changedClip(const OFX::InstanceChangedArgs& args,
                     const std::string& name) override;

    // Tells the host which extra frames are needed for analyse and scatter
    // so Resolve/Fusion can prefetch Plate and Degrained correctly.
    void getFramesNeeded(const OFX::FramesNeededArguments& args,
                         OFX::FramesNeededSetter& frames) override;

    // Keeps Resolve/Fusion on float RGBA, matching the kernel assumptions.
    void getClipPreferences(OFX::ClipPreferencesSetter& prefs) override;

private:
    // Helpers ---------------------------------------------------------------
    // Copy Source to dst when required inputs are missing. This keeps a node
    // dropped on a clip from rendering black before Plate/Degrained are wired.
    void renderPassThrough(const OFX::RenderArguments& args);

    // Fetches OFX images/params for one render window, handles optional scatter
    // and QC modes, and dispatches CPU/Metal/CUDA/OpenCL grain application.
    void renderGrainApply(const OFX::RenderArguments& args);

    // Enables controls that depend on feature toggles, e.g. scatter-only
    // settings and advanced curve JSON tools.
    void updateUiEnabled();

    // Bake the persisted response-curve JSON into a flat RGB LUT for kernels.
    // Returns a vector owning the data; CurveLUT.data points into it.
    std::vector<float> bakeResponseCurve(double time, int sampleCount) const;

    // Run the multi-frame analysis pass. Called from render() when
    // analyse_state is "requested". Updates the hidden JSON curve store via
    // beginEditBlock/endEditBlock and resets the state.
    void runAnalysePass(const OFX::RenderArguments& args);

    // Clips ----------------------------------------------------------------
    OFX::Clip* dst_           = nullptr;
    OFX::Clip* source_        = nullptr;     // COMP
    OFX::Clip* plate_         = nullptr;
    OFX::Clip* degrained_     = nullptr;
    OFX::Clip* mask_          = nullptr;
    OFX::Clip* externalGrain_ = nullptr;

    // Params (subset materialised in Phase 1; the rest are fetched in Phase 2) -
    OFX::ChoiceParam*     output_           = nullptr;
    OFX::DoubleParam*     luminance_        = nullptr;
    OFX::BooleanParam*    fixGhosting_      = nullptr;
    OFX::BooleanParam*    grainMaskInvert_  = nullptr;
    OFX::IntParam*        numberOfFrames_   = nullptr;
    OFX::StringParam*     additionalFrames_ = nullptr;
    OFX::IntParam*        sampleCount_      = nullptr;
    OFX::PushButtonParam* analyse_          = nullptr;
    OFX::StringParam*     analyseState_     = nullptr;
    OFX::BooleanParam*    analyseMaskInvert_ = nullptr;
    // The on-screen parametric curve widget is intentionally absent: OFX's
    // kOfxParamTypeParametric is host-optional and DaVinci Resolve does
    // not implement it. The canonical curve lives in `responseCurveJSON_`.
    OFX::StringParam*     responseCurveJSON_ = nullptr;
    OFX::DoubleParam*     grainAmount_       = nullptr;
    OFX::DoubleParam*     shadowGrain_       = nullptr;
    OFX::DoubleParam*     midtoneGrain_      = nullptr;
    OFX::DoubleParam*     highlightGrain_    = nullptr;
    OFX::DoubleParam*     curveContrast_     = nullptr;
    OFX::DoubleParam*     curvePivot_        = nullptr;
    OFX::DoubleParam*     redGrain_          = nullptr;
    OFX::DoubleParam*     greenGrain_        = nullptr;
    OFX::DoubleParam*     blueGrain_         = nullptr;
    OFX::BooleanParam*    showCurveIO_       = nullptr;
    OFX::BooleanParam*    externalGrainEnabled_ = nullptr;
    OFX::BooleanParam*    scatter_          = nullptr;
    OFX::Double2DParam*   boxLow_           = nullptr;
    OFX::Double2DParam*   boxHigh_          = nullptr;
    OFX::IntParam*        sampleFrame_      = nullptr;
    OFX::DoubleParam*     cellSize_         = nullptr;
    OFX::BooleanParam*    overlayPattern_   = nullptr;
    OFX::IntParam*        edgeBlendSize_    = nullptr;
    OFX::DoubleParam*     amplitude_        = nullptr;
    OFX::DoubleParam*     frequency_        = nullptr;
    OFX::IntParam*        seed_             = nullptr;
    OFX::BooleanParam*    replaceMaskInvert_ = nullptr;
    OFX::PushButtonParam* troubleshoot_     = nullptr;
    OFX::StringParam*     curveJSONPaste_   = nullptr;
    OFX::PushButtonParam* curveImport_      = nullptr;
    OFX::PushButtonParam* curveExport_      = nullptr;
    OFX::PushButtonParam* curveHelp_        = nullptr;

    // Resolve/Fusion does not reliably persist hidden String params on OFX
    // plugins, so keep the most recent analysed/imported curve in instance
    // memory and use the JSON param only as a persistence fallback.
    ResponseCurve analysedCurve_;
    bool hasAnalysedCurve_ = false;
};

}  // namespace dasgrain
