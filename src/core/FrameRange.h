// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// FrameRange: parses Nuke-style frame-range strings like
//
//     "1001,1020,1053"          (single frames)
//     "1020-1040"               (inclusive range)
//     "1020-1040x4"             (range with step)
//     "1001,1100-1120x2,1200"   (combinations)

#pragma once

#include <string>
#include <vector>

namespace dasgrain {

class FrameRange {
public:
    // Parse the user input plus a sliding count of N evenly-spread frames in
    // [firstFrame, lastFrame], identical to the gizmo's `_generate_frame_list`.
    // Frames are returned sorted, deduplicated, and clamped to the inclusive
    // range [firstFrame, lastFrame].
    static std::vector<int> generate(int numberOfFrames,
                                     const std::string& additional,
                                     int firstFrame,
                                     int lastFrame);

    // Parse just the additional-frames string. Returns the deduplicated,
    // unsorted frames.
    static std::vector<int> parseAdditional(const std::string& s);
};

}  // namespace dasgrain
