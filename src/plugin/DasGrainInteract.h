// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Sample-box overlay. Draws a draggable rectangle on the viewer that defines
// the source area for the Voronoi scatter pass. Box coordinates are stored
// in two Double2D params (`box_low`, `box_high`) in normalised UV space
// [0..1], where 1 maps to the source image's longest dimension; the interact
// converts to/from canonical (image-pixel) coordinates for drawing and pen
// events.

#pragma once

#include "ofxsInteract.h"

namespace dasgrain {

class DasGrainInteract : public OFX::OverlayInteract {
public:
    DasGrainInteract(OfxInteractHandle handle, OFX::ImageEffect* effect);

    bool draw(const OFX::DrawArgs& args) override;
    bool penMotion(const OFX::PenArgs& args) override;
    bool penDown(const OFX::PenArgs& args) override;
    bool penUp(const OFX::PenArgs& args) override;

private:
    enum class Handle {
        kNone,
        kBottomLeft,
        kBottomRight,
        kTopLeft,
        kTopRight,
        kInside,
    };

    // Returns which handle (if any) is under (canonical) point `p` given the
    // current box and screen pixel scale.
    Handle hitTest(double px, double py,
                   double imgW, double imgH,
                   const OfxPointD& pixelScale) const;

    // Convert between canonical (image) coords and normalised UV coords.
    // `imgW` is the image width in pixels (we use it as the unit for both
    // axes so the box stays aspect-correct).
    void toUV(double px, double py, double imgW, double& u, double& v) const;
    void fromUV(double u, double v, double imgW, double& px, double& py) const;

    Handle    activeHandle_  = Handle::kNone;
    OfxPointD dragOriginCanonical_ {0.0, 0.0};
    OfxPointD dragOriginLow_       {0.0, 0.0};
    OfxPointD dragOriginHigh_      {0.0, 0.0};
};

class DasGrainOverlayDescriptor : public OFX::DefaultEffectOverlayDescriptor<
        DasGrainOverlayDescriptor, DasGrainInteract> {};

}  // namespace dasgrain
