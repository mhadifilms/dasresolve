// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Voronoi: CPU reference port of the BlinkScript kernel from the original
// DasGrain gizmo. Same constants and ValueNoise3D() so the Mac CPU path
// produces frame-identical results to the GPU path.
//
// Mostly used for unit tests and the CPU fallback; the GPU kernels embed an
// equivalent translation in the host shading language.

#pragma once

#include <cstdint>

namespace dasgrain {
namespace voronoi {

// libNoise prime constants used by the original blink kernel.
inline constexpr int32_t kXNoise    = 1619;
inline constexpr int32_t kYNoise    = 31337;
inline constexpr int32_t kZNoise    = 6971;
inline constexpr int32_t kSeedNoise = 1013;

// 3D integer value noise from libNoise. Returns a 31-bit positive integer.
int32_t intValueNoise3D(int x, int y, int z, int seed);

// Same, normalised to [-1, 1].
float valueNoise3D(int x, int y, int z, int seed);

// Output (xCandidate, yCandidate) for the Voronoi cell centre at (x, y) in
// frequency-space, exactly matching the BlinkScript kernel. Coordinates are
// pixel-domain, transformed inside the function.
struct CellResult {
    float u;  // sample-x in pixel-domain (after divide-by-frequency)
    float v;  // sample-y in pixel-domain
};

CellResult cell(int px, int py,
                int width, int height,
                float aspectRatio,
                float frequency,
                int seed,
                float randomness = 0.5f);

}  // namespace voronoi
}  // namespace dasgrain
