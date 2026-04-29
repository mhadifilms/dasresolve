// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// DegrainHelper: companion to DasGrain. Mixes the original plate's luma
// grain back onto a fully-degrained plate by `1 - luma_mix` in YCbCr's Y
// channel. The same `luma_mix` value is then plugged into DasGrain's
// `luminance` knob so the analyse pipeline can compensate.
//
// Mirrors the DegrainHelper Group inside the DasGrain.nk gizmo.

#pragma once

#include "ofxsImageEffect.h"
#include "ofxsParam.h"

namespace dasgrain {

class DegrainHelperPlugin : public OFX::ImageEffect {
public:
    explicit DegrainHelperPlugin(OfxImageEffectHandle handle);

    void render(const OFX::RenderArguments& args) override;

private:
    OFX::Clip* dst_         = nullptr;
    OFX::Clip* plate_       = nullptr;
    OFX::Clip* degrained_   = nullptr;

    OFX::DoubleParam* lumaMix_ = nullptr;
};

class DegrainHelperFactory : public OFX::PluginFactoryHelper<DegrainHelperFactory> {
public:
    DegrainHelperFactory();
    void load() override {}
    void unload() override {}
    void describe(OFX::ImageEffectDescriptor& desc) override;
    void describeInContext(OFX::ImageEffectDescriptor& desc,
                           OFX::ContextEnum context) override;
    OFX::ImageEffect* createInstance(OfxImageEffectHandle handle,
                                     OFX::ContextEnum context) override;
};

}  // namespace dasgrain
