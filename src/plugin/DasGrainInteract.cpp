// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0

#include "DasGrainInteract.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

// macOS deprecated the legacy GL immediate-mode API in 10.14, but Resolve
// still wraps interacts in a legacy GL context. Silence the warnings here.
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION 1
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <algorithm>
#include <cmath>

#include "ofxsImageEffect.h"

#include "Params.h"

namespace dasgrain {

namespace {

// Visual size of corner handles, in pixels (multiplied by pixelScale to stay
// constant on screen across zoom levels).
constexpr double kHandlePxRadius = 6.0;

OfxPointD getImageSize(OFX::ImageEffect* effect) {
    OfxPointD s{1920.0, 1080.0};
    if (!effect) return s;
    if (auto* clip = effect->fetchClip(kOfxImageEffectSimpleSourceClipName)) {
        if (clip->isConnected()) {
            const OfxRectD r = clip->getRegionOfDefinition(0.0);
            s.x = std::max(r.x2 - r.x1, 1.0);
            s.y = std::max(r.y2 - r.y1, 1.0);
            return s;
        }
    }
    if (auto* clip = effect->fetchClip(clips::kPlate)) {
        if (clip->isConnected()) {
            const OfxRectD r = clip->getRegionOfDefinition(0.0);
            s.x = std::max(r.x2 - r.x1, 1.0);
            s.y = std::max(r.y2 - r.y1, 1.0);
            return s;
        }
    }
    return s;
}

}  // namespace

DasGrainInteract::DasGrainInteract(OfxInteractHandle handle, OFX::ImageEffect* /*effect*/)
    : OFX::OverlayInteract(handle) {}

void DasGrainInteract::toUV(double px, double py, double imgW,
                            double& u, double& v) const {
    u = px / std::max(imgW, 1.0);
    v = py / std::max(imgW, 1.0);
}

void DasGrainInteract::fromUV(double u, double v, double imgW,
                              double& px, double& py) const {
    px = u * imgW;
    py = v * imgW;
}

DasGrainInteract::Handle DasGrainInteract::hitTest(double px, double py,
                                                   double imgW, double /*imgH*/,
                                                   const OfxPointD& pixelScale) const {
    auto* low  = _effect->fetchDouble2DParam(params::kBoxLow);
    auto* high = _effect->fetchDouble2DParam(params::kBoxHigh);
    if (!low || !high) return Handle::kNone;
    double lx, ly, hx, hy;
    low->getValue(lx, ly);
    high->getValue(hx, hy);
    double bxLow, byLow, bxHigh, byHigh;
    fromUV(std::min(lx, hx), std::min(ly, hy), imgW, bxLow,  byLow);
    fromUV(std::max(lx, hx), std::max(ly, hy), imgW, bxHigh, byHigh);
    const double rx = kHandlePxRadius * pixelScale.x;
    const double ry = kHandlePxRadius * pixelScale.y;

    auto near = [&](double ax, double ay) {
        return std::fabs(px - ax) <= rx && std::fabs(py - ay) <= ry;
    };
    if (near(bxLow,  byLow))  return Handle::kBottomLeft;
    if (near(bxHigh, byLow))  return Handle::kBottomRight;
    if (near(bxLow,  byHigh)) return Handle::kTopLeft;
    if (near(bxHigh, byHigh)) return Handle::kTopRight;
    if (px >= bxLow && px <= bxHigh && py >= byLow && py <= byHigh) {
        return Handle::kInside;
    }
    return Handle::kNone;
}

bool DasGrainInteract::draw(const OFX::DrawArgs& args) {
    if (!_effect) return false;
    auto* scatter = _effect->fetchBooleanParam(params::kScatter);
    if (!scatter || !scatter->getValue()) return false;
    auto* low  = _effect->fetchDouble2DParam(params::kBoxLow);
    auto* high = _effect->fetchDouble2DParam(params::kBoxHigh);
    if (!low || !high) return false;
    double lx, ly, hx, hy;
    low->getValue(lx, ly);
    high->getValue(hx, hy);

    const OfxPointD imgSize = getImageSize(_effect);
    double bxLow, byLow, bxHigh, byHigh;
    fromUV(std::min(lx, hx), std::min(ly, hy), imgSize.x, bxLow,  byLow);
    fromUV(std::max(lx, hx), std::max(ly, hy), imgSize.x, bxHigh, byHigh);
    const double hrx = kHandlePxRadius * args.pixelScale.x;
    const double hry = kHandlePxRadius * args.pixelScale.y;

    glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT);
    glLineWidth(2.0f);

    glColor4f(1.0f, 0.85f, 0.2f, 0.9f);
    glBegin(GL_LINE_LOOP);
    glVertex2d(bxLow,  byLow);
    glVertex2d(bxHigh, byLow);
    glVertex2d(bxHigh, byHigh);
    glVertex2d(bxLow,  byHigh);
    glEnd();

    auto drawHandle = [&](double cx, double cy, bool active) {
        if (active) glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        else        glColor4f(1.0f, 0.85f, 0.2f, 1.0f);
        glBegin(GL_LINE_LOOP);
        glVertex2d(cx - hrx, cy - hry);
        glVertex2d(cx + hrx, cy - hry);
        glVertex2d(cx + hrx, cy + hry);
        glVertex2d(cx - hrx, cy + hry);
        glEnd();
    };
    drawHandle(bxLow,  byLow,  activeHandle_ == Handle::kBottomLeft);
    drawHandle(bxHigh, byLow,  activeHandle_ == Handle::kBottomRight);
    drawHandle(bxLow,  byHigh, activeHandle_ == Handle::kTopLeft);
    drawHandle(bxHigh, byHigh, activeHandle_ == Handle::kTopRight);

    glPopAttrib();
    return true;
}

bool DasGrainInteract::penDown(const OFX::PenArgs& args) {
    if (!_effect) return false;
    auto* scatter = _effect->fetchBooleanParam(params::kScatter);
    if (!scatter || !scatter->getValue()) return false;

    const OfxPointD imgSize = getImageSize(_effect);
    Handle h = hitTest(args.penPosition.x, args.penPosition.y,
                       imgSize.x, imgSize.y, args.pixelScale);
    if (h == Handle::kNone) return false;
    activeHandle_ = h;
    dragOriginCanonical_ = args.penPosition;

    auto* low  = _effect->fetchDouble2DParam(params::kBoxLow);
    auto* high = _effect->fetchDouble2DParam(params::kBoxHigh);
    low->getValue(dragOriginLow_.x,  dragOriginLow_.y);
    high->getValue(dragOriginHigh_.x, dragOriginHigh_.y);
    _effect->redrawOverlays();
    return true;
}

bool DasGrainInteract::penMotion(const OFX::PenArgs& args) {
    if (!_effect || activeHandle_ == Handle::kNone) return false;
    auto* low  = _effect->fetchDouble2DParam(params::kBoxLow);
    auto* high = _effect->fetchDouble2DParam(params::kBoxHigh);
    if (!low || !high) return false;

    const OfxPointD imgSize = getImageSize(_effect);
    double pu, pv;
    toUV(args.penPosition.x, args.penPosition.y, imgSize.x, pu, pv);
    double ou, ov;
    toUV(dragOriginCanonical_.x, dragOriginCanonical_.y, imgSize.x, ou, ov);
    const double du = pu - ou;
    const double dv = pv - ov;

    double newLowU  = dragOriginLow_.x;
    double newLowV  = dragOriginLow_.y;
    double newHighU = dragOriginHigh_.x;
    double newHighV = dragOriginHigh_.y;
    switch (activeHandle_) {
        case Handle::kBottomLeft:  newLowU  += du; newLowV  += dv; break;
        case Handle::kBottomRight: newHighU += du; newLowV  += dv; break;
        case Handle::kTopLeft:     newLowU  += du; newHighV += dv; break;
        case Handle::kTopRight:    newHighU += du; newHighV += dv; break;
        case Handle::kInside:
            newLowU  += du; newHighU += du;
            newLowV  += dv; newHighV += dv;
            break;
        default: break;
    }
    _effect->beginEditBlock("DasGrain box drag");
    low ->setValue(newLowU,  newLowV);
    high->setValue(newHighU, newHighV);
    _effect->endEditBlock();
    _effect->redrawOverlays();
    return true;
}

bool DasGrainInteract::penUp(const OFX::PenArgs& /*args*/) {
    if (activeHandle_ == Handle::kNone) return false;
    activeHandle_ = Handle::kNone;
    if (_effect) _effect->redrawOverlays();
    return true;
}

}  // namespace dasgrain
