// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ResponseCurve.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace dasgrain {

ResponseCurve::ResponseCurve() {
    reset();
}

void ResponseCurve::reset() {
    for (int c = 0; c < kChannelCount; ++c) {
        points_[c].clear();
        points_[c].push_back({0.0, 0.0});
        points_[c].push_back({1.0, 1.0});
    }
}

void ResponseCurve::setChannel(int channel, std::vector<CurvePoint> points) {
    if (channel < 0 || channel >= kChannelCount) return;
    std::sort(points.begin(), points.end(),
              [](const CurvePoint& a, const CurvePoint& b) { return a.x < b.x; });
    points_[channel] = std::move(points);
}

void ResponseCurve::addPoint(int channel, double x, double y) {
    if (channel < 0 || channel >= kChannelCount) return;
    auto& pts = points_[channel];
    pts.push_back({x, y});
    std::sort(pts.begin(), pts.end(),
              [](const CurvePoint& a, const CurvePoint& b) { return a.x < b.x; });
}

double ResponseCurve::sample(int channel, double x) const {
    if (channel < 0 || channel >= kChannelCount) return x;
    const auto& pts = points_[channel];
    if (pts.empty()) return x;
    if (x <= pts.front().x) return pts.front().y;
    if (x >= pts.back().x)  return pts.back().y;

    // Binary search for the segment containing x.
    auto it = std::upper_bound(pts.begin(), pts.end(), x,
        [](double xv, const CurvePoint& p) { return xv < p.x; });
    auto hi = it;
    auto lo = it - 1;
    const double t = (lo->x == hi->x) ? 0.0 : (x - lo->x) / (hi->x - lo->x);
    return lo->y + t * (hi->y - lo->y);
}

bool ResponseCurve::isValid() const {
    for (int c = 0; c < kChannelCount; ++c) {
        if (points_[c].size() < 2) return false;
        // monotonic in x is required for sample() to behave; we already
        // enforce that on insertion.
    }
    return true;
}

// ---------------------------------------------------------------------------
// JSON I/O (minimal, no external deps).
// ---------------------------------------------------------------------------

namespace {

void writeNum(std::ostringstream& os, double v) {
    char buf[64];
    // %.10g keeps round-trip accuracy without being noisy.
    std::snprintf(buf, sizeof(buf), "%.10g", v);
    os << buf;
}

// Trivial JSON parser: skips whitespace, parses arrays of numbers. Strict
// enough for the shape we emit; tolerant enough for hand edits.
struct Parser {
    const std::string& s;
    size_t i = 0;
    bool ok = true;

    explicit Parser(const std::string& str) : s(str) {}

    void skip() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }
    bool consume(char c) {
        skip();
        if (i < s.size() && s[i] == c) { ++i; return true; }
        ok = false; return false;
    }
    bool peek(char c) {
        skip();
        return i < s.size() && s[i] == c;
    }
    bool parseNumber(double& out) {
        skip();
        char* end = nullptr;
        const char* start = s.c_str() + i;
        out = std::strtod(start, &end);
        if (end == start) { ok = false; return false; }
        i += static_cast<size_t>(end - start);
        return true;
    }
    // Returns the unquoted string between the next pair of '"'.
    bool parseString(std::string& out) {
        skip();
        if (i >= s.size() || s[i] != '"') { ok = false; return false; }
        ++i;
        out.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                out.push_back(s[i + 1]);
                i += 2;
            } else {
                out.push_back(s[i]);
                ++i;
            }
        }
        if (i >= s.size()) { ok = false; return false; }
        ++i;  // closing quote
        return true;
    }
    // Parse a [x, y] pair.
    bool parsePair(CurvePoint& p) {
        if (!consume('[')) return false;
        if (!parseNumber(p.x)) return false;
        if (!consume(',')) return false;
        if (!parseNumber(p.y)) return false;
        if (!consume(']')) return false;
        return true;
    }
};

}  // namespace

std::string ResponseCurve::toJSON() const {
    std::ostringstream os;
    os << "{\"version\":1,\"curves\":[";
    for (int c = 0; c < kChannelCount; ++c) {
        if (c) os << ",";
        os << "[";
        const auto& pts = points_[c];
        for (size_t k = 0; k < pts.size(); ++k) {
            if (k) os << ",";
            os << "[";
            writeNum(os, pts[k].x);
            os << ",";
            writeNum(os, pts[k].y);
            os << "]";
        }
        os << "]";
    }
    os << "]}";
    return os.str();
}

bool ResponseCurve::fromJSON(const std::string& json) {
    if (json.empty()) return false;
    Parser p(json);
    if (!p.consume('{')) return false;

    std::array<std::vector<CurvePoint>, kChannelCount> staged;
    bool gotCurves = false;

    while (p.ok) {
        p.skip();
        if (p.peek('}')) break;
        std::string key;
        if (!p.parseString(key)) return false;
        if (!p.consume(':')) return false;

        if (key == "curves") {
            if (!p.consume('[')) return false;
            for (int c = 0; c < kChannelCount; ++c) {
                if (!p.consume('[')) return false;
                while (p.ok && !p.peek(']')) {
                    CurvePoint cp{};
                    if (!p.parsePair(cp)) return false;
                    staged[c].push_back(cp);
                    p.skip();
                    if (p.peek(',')) ++p.i;
                }
                if (!p.consume(']')) return false;
                if (c + 1 < kChannelCount) {
                    if (!p.consume(',')) return false;
                }
            }
            if (!p.consume(']')) return false;
            gotCurves = true;
        } else {
            // Skip an unknown value: a number, a string, or a brace-balanced
            // structure. Cheap heuristic: scan until top-level ',' or '}'.
            int depth = 0;
            while (p.i < p.s.size()) {
                char c = p.s[p.i];
                if (c == '[' || c == '{') ++depth;
                else if (c == ']' || c == '}') {
                    if (depth == 0) break;
                    --depth;
                } else if (c == ',' && depth == 0) {
                    break;
                }
                ++p.i;
            }
        }

        p.skip();
        if (p.peek(',')) ++p.i;
    }
    if (!p.consume('}')) return false;
    if (!gotCurves) return false;

    for (int c = 0; c < kChannelCount; ++c) {
        if (staged[c].size() < 2) return false;
        std::sort(staged[c].begin(), staged[c].end(),
                  [](const CurvePoint& a, const CurvePoint& b) { return a.x < b.x; });
        points_[c] = std::move(staged[c]);
    }
    return true;
}

}  // namespace dasgrain
