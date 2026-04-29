// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ofxsImageEffect.h"

namespace dasgrain {

class DasGrainFactory : public OFX::PluginFactoryHelper<DasGrainFactory> {
public:
    DasGrainFactory();
    void load() override {}
    void unload() override {}
    void describe(OFX::ImageEffectDescriptor& desc) override;
    void describeInContext(OFX::ImageEffectDescriptor& desc,
                           OFX::ContextEnum context) override;
    OFX::ImageEffect* createInstance(OfxImageEffectHandle handle,
                                     OFX::ContextEnum context) override;
};

}  // namespace dasgrain
