// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Centralised place for every clip name and parameter name used by the
// plugin. Keeping these in one header avoids string-typo bugs and makes the
// plugin's UI surface easy to scan at a glance.

#pragma once

#include <string>

namespace dasgrain {

// ---------------------------------------------------------------------------
// Clip names. We expose five named inputs in Fusion, mirroring the original
// DasGrain Nuke gizmo. "Source" is OFX's mandatory output-driving input; in
// our case it carries the COMP that we re-grain.
// ---------------------------------------------------------------------------
namespace clips {
constexpr const char* kSource        = "Source";        // == COMP
constexpr const char* kPlate         = "Plate";         // grainy plate
constexpr const char* kDegrained     = "Degrained";     // denoised plate
constexpr const char* kMask          = "Mask";          // optional mask
constexpr const char* kExternalGrain = "ExternalGrain"; // optional pre-rendered grain
}  // namespace clips

// ---------------------------------------------------------------------------
// Parameter names.
// ---------------------------------------------------------------------------
namespace params {

// Output mode. Matches the Switch/Output node in the gizmo.
constexpr const char* kOutput = "output";
enum OutputMode : int {
    kOutputRegrained        = 0,  // regrained comp
    kOutputOriginalGrain    = 1,  // plate - degrained
    kOutputNormalisedGrain  = 2,  // grain / curve(luma)
    kOutputAdaptedGrain     = 3,  // normGrain * curve(compLuma)
    kOutputGrainQC          = 4,  // |regrained - blur(regrained)| * 50
};

// Analyse tab.
constexpr const char* kLuminance        = "luminance";          // 0..1
constexpr const char* kFixGhosting      = "fix_ghosting";
constexpr const char* kGrainMaskInvert  = "grain_mask_invert";
constexpr const char* kNumberOfFrames   = "number_of_frames";
constexpr const char* kAdditionalFrames = "additional_frames";
constexpr const char* kSampleCount      = "sample_count";
constexpr const char* kAnalyse          = "analyse";            // PushButton
constexpr const char* kAnalyseState     = "analyse_state";      // hidden String
constexpr const char* kAnalyseMaskInvert = "analyse_mask_invert";

// Adjust tab.
constexpr const char* kResponseCurve     = "response_curve";    // legacy id (unused; see kResponseCurveJSON)
constexpr const char* kResponseCurveJSON = "response_curve_json"; // hidden String

// Replace tab.
constexpr const char* kExternalGrain     = "external_grain";    // bool
constexpr const char* kScatter           = "scatter";           // bool
constexpr const char* kBoxLow            = "box_low";           // Double2D (x1, y1)
constexpr const char* kBoxHigh           = "box_high";          // Double2D (x2, y2)
constexpr const char* kSampleFrame       = "sample_frame";
constexpr const char* kCellSize          = "cell_size";
constexpr const char* kOverlayPattern    = "overlay_pattern";
constexpr const char* kEdgeBlendSize     = "edge_blend_size";
constexpr const char* kAmplitude         = "amplitude";
constexpr const char* kFrequency         = "frequency";
constexpr const char* kSeed              = "seed";
constexpr const char* kReplaceMaskInvert = "replace_mask_invert";

// Help tab.
constexpr const char* kTroubleshoot   = "troubleshoot";  // PushButton

// Adjust tab — "What am I looking at?" help button. Mirrors the gizmo's
// `whatsthis` knob; opens a message dialog explaining the response curve.
constexpr const char* kCurveHelp      = "curve_help";    // PushButton

// Adjust tab — extra: response curve import/export. The user pastes JSON
// into `kCurveJSONImport` and clicks `kCurveImport` to apply; clicking
// `kCurveExport` writes the current curve into the (read-only) hidden
// `kResponseCurveJSON` and shows it via a message dialog.
constexpr const char* kCurveJSONImport = "response_curve_json_paste";
constexpr const char* kCurveImport     = "response_curve_import";   // PushButton
constexpr const char* kCurveExport     = "response_curve_export";   // PushButton

// Pages.
constexpr const char* kPageAnalyse  = "Analyse";
constexpr const char* kPageAdjust   = "Adjust";
constexpr const char* kPageReplace  = "Replace";
constexpr const char* kPageHelp     = "Help";

}  // namespace params

// State machine values for params::kAnalyseState.
namespace analyse_state {
constexpr const char* kIdle      = "idle";
constexpr const char* kRequested = "requested";
constexpr const char* kRunning   = "running";
}  // namespace analyse_state

// Defaults.
namespace defaults {
constexpr int    kSampleCount      = 20;
constexpr int    kNumberOfFrames   = 10;
constexpr int    kCurvePoints      = 32;     // points along response curve
constexpr double kLuminance        = 1.0;
constexpr double kCellSize         = 40.0;
constexpr int    kEdgeBlendSize    = 0;
constexpr double kAmplitude        = 0.0;
constexpr double kFrequency        = 15.0;
constexpr int    kSeed             = 0;
constexpr int    kSampleFrame      = 1001;
// Default sample-box (in normalised image coords, like Nuke).
constexpr double kBoxLowX  = 0.10;
constexpr double kBoxLowY  = 0.10;
constexpr double kBoxHighX = 0.40;
constexpr double kBoxHighY = 0.30;
}  // namespace defaults

}  // namespace dasgrain
