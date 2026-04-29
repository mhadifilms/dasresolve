// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0

#include "FrameRange.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>
#include <sstream>

namespace dasgrain {
namespace {

// Trim leading/trailing whitespace (in place).
void trim(std::string& s) {
    auto issp = [](unsigned char c) { return std::isspace(c); };
    while (!s.empty() && issp(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && issp(static_cast<unsigned char>(s.back())))  s.pop_back();
}

// Try to parse "<start>-<end>[xstep]" or "<frame>". Returns the list of
// frames produced by that single token.
std::vector<int> parseToken(const std::string& tokIn) {
    std::vector<int> out;
    std::string tok = tokIn;
    trim(tok);
    if (tok.empty()) return out;

    const auto dash = tok.find('-');
    if (dash == std::string::npos) {
        // Single frame.
        char* end = nullptr;
        long v = std::strtol(tok.c_str(), &end, 10);
        if (end != tok.c_str()) out.push_back(static_cast<int>(v));
        return out;
    }
    // Range with optional step.
    std::string startStr = tok.substr(0, dash);
    std::string rest     = tok.substr(dash + 1);
    int step = 1;
    const auto x = rest.find_first_of("xX");
    std::string endStr = (x == std::string::npos) ? rest : rest.substr(0, x);
    if (x != std::string::npos) {
        char* e = nullptr;
        long sv = std::strtol(rest.c_str() + x + 1, &e, 10);
        if (e != rest.c_str() + x + 1 && sv > 0) step = static_cast<int>(sv);
    }
    char* es = nullptr;
    char* ee = nullptr;
    long s0 = std::strtol(startStr.c_str(), &es, 10);
    long s1 = std::strtol(endStr.c_str(),   &ee, 10);
    if (es == startStr.c_str() || ee == endStr.c_str()) return out;
    int start = static_cast<int>(s0);
    int end   = static_cast<int>(s1);
    if (start > end) std::swap(start, end);
    for (int f = start; f <= end; f += step) out.push_back(f);
    return out;
}

}  // namespace

std::vector<int> FrameRange::parseAdditional(const std::string& s) {
    std::set<int> uniq;
    std::vector<int> out;
    std::string cur;
    for (size_t i = 0; i <= s.size(); ++i) {
        char c = (i < s.size()) ? s[i] : ',';
        if (c == ',' || c == ';') {
            if (!cur.empty()) {
                for (int f : parseToken(cur)) {
                    if (uniq.insert(f).second) out.push_back(f);
                }
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    return out;
}

std::vector<int> FrameRange::generate(int numberOfFrames,
                                      const std::string& additional,
                                      int firstFrame,
                                      int lastFrame) {
    if (lastFrame < firstFrame) std::swap(firstFrame, lastFrame);

    std::set<int> uniq;
    std::vector<int> out;

    if (numberOfFrames > 0) {
        const double distance = double(lastFrame - firstFrame) / numberOfFrames;
        double frame = double(firstFrame) + distance / 2.0;
        for (int n = 0; n < numberOfFrames; ++n) {
            const int rounded = int(std::lround(frame));
            const int clamped = std::max(firstFrame, std::min(lastFrame, rounded));
            if (uniq.insert(clamped).second) out.push_back(clamped);
            frame += distance;
        }
    }

    for (int f : parseAdditional(additional)) {
        if (f >= firstFrame && f <= lastFrame && uniq.insert(f).second) {
            out.push_back(f);
        }
    }

    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace dasgrain
