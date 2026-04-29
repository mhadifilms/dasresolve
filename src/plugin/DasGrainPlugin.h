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
    void render(const OFX::RenderArguments& args) override;
    bool isIdentity(const OFX::IsIdentityArguments& args,
                    OFX::Clip*& identityClip,
                    double& identityTime) override;
    void changedParam(const OFX::InstanceChangedArgs& args,
                      const std::string& name) override;
    void changedClip(const OFX::InstanceChangedArgs& args,
                     const std::string& name) override;
    void getFramesNeeded(const OFX::FramesNeededArguments& args,
                         OFX::FramesNeededSetter& frames) override;
    void getClipPreferences(OFX::ClipPreferencesSetter& prefs) override;

private:
    // Helpers ---------------------------------------------------------------
    void renderPassThrough(const OFX::RenderArguments& args);
    void renderGrainApply(const OFX::RenderArguments& args);
    void updateUiEnabled();

    // Bake the parametric response curve into a flat LUT for the kernel.
    // Returns a vector owning the data; CurveLUT.data points into it.
    std::vector<float> bakeResponseCurve(double time, int sampleCount) const;

    // Run the multi-frame analysis pass. Called from render() when
    // analyse_state is "requested". Updates the parametric param + JSON
    // string param via paramEditBegin/End and resets the state.
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
};

}  // namespace dasgrain
