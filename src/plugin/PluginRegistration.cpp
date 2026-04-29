// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ofxsImageEffect.h"

#include "DasGrainFactory.h"
#include "DegrainHelperPlugin.h"

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& factories) {
    static dasgrain::DasGrainFactory       dasGrainFactory;
    static dasgrain::DegrainHelperFactory  degrainHelperFactory;
    factories.push_back(&dasGrainFactory);
    factories.push_back(&degrainHelperFactory);
}
