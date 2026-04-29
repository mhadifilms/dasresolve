// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0

#include "Voronoi.h"

#include <cstdint>

namespace dasgrain {
namespace voronoi {

int32_t intValueNoise3D(int x, int y, int z, int seed) {
    // Match the BlinkScript kernel byte-for-byte. We do the arithmetic in
    // uint32_t to avoid signed-overflow UB; the trailing mask reproduces the
    // reference behaviour.
    using u32 = uint32_t;
    u32 n = (u32(kXNoise)    * u32(x)
           + u32(kYNoise)    * u32(y)
           + u32(kZNoise)    * u32(z)
           + u32(kSeedNoise) * u32(seed)) & u32(0x7fffffff);
    n = (n >> 13) ^ n;
    const u32 result = (n * (n * n * u32(60493) + u32(19990303)) + u32(1376312589))
                       & u32(0x7fffffff);
    return int32_t(result);
}

float valueNoise3D(int x, int y, int z, int seed) {
    return 1.0f - (float(intValueNoise3D(x, y, z, seed)) / 1073741824.0f);
}

CellResult cell(int px, int py,
                int width, int height,
                float aspectRatio,
                float frequency,
                int seed,
                float randomness) {
    const float x = float(px) * aspectRatio * frequency / float(width);
    const float y = float(py) * frequency / float(width);
    const int xInt = (x > 0.0f) ? int(x) : int(x) - 1;
    const int yInt = (y > 0.0f) ? int(y) : int(y) - 1;

    float minDist = 2147483647.0f;
    float xCandidate = 0.0f;
    float yCandidate = 0.0f;

    for (int yCur = yInt - 2; yCur <= yInt + 2; ++yCur) {
        for (int xCur = xInt - 2; xCur <= xInt + 2; ++xCur) {
            const float xPos = xCur + (valueNoise3D(xCur, yCur, 0, seed)     + 1.0f) * randomness + (1.0f - randomness) - 1.0f;
            const float yPos = yCur + (valueNoise3D(xCur, yCur, 0, seed + 1) + 1.0f) * randomness + (1.0f - randomness) - 1.0f;
            const float dx = xPos - x;
            const float dy = yPos - y;
            const float d  = dx * dx + dy * dy;
            if (d < minDist) {
                minDist = d;
                xCandidate = xPos;
                yCandidate = yPos;
            }
        }
    }

    CellResult r;
    r.u = xCandidate / aspectRatio / frequency;
    r.v = yCandidate / float(height) * float(width) / frequency;
    return r;
}

}  // namespace voronoi
}  // namespace dasgrain
