#include "test_harness.h"

#include "core/Voronoi.h"

using dasgrain::voronoi::cell;
using dasgrain::voronoi::CellResult;
using dasgrain::voronoi::valueNoise3D;

TEST_CASE(voronoi_value_noise_is_in_range) {
    for (int z = 0; z < 4; ++z)
    for (int y = -3; y < 3; ++y)
    for (int x = -3; x < 3; ++x) {
        const float v = valueNoise3D(x, y, z, /*seed=*/42);
        EXPECT_TRUE(v >= -1.0f && v <= 1.0f);
    }
}

TEST_CASE(voronoi_cell_returns_finite_values) {
    // Pixel-domain test: 1920x1080, freq=10, aspect=1.
    const int w = 1920, h = 1080;
    const float freq = 10.0f;
    const float aspect = 1.0f;
    for (int py = 0; py < h; py += 137) {
        for (int px = 0; px < w; px += 191) {
            const CellResult r = cell(px, py, w, h, aspect, freq, /*seed=*/0);
            EXPECT_TRUE(std::isfinite(r.u));
            EXPECT_TRUE(std::isfinite(r.v));
        }
    }
}

TEST_CASE(voronoi_cell_seed_changes_output) {
    const CellResult a = cell(960, 540, 1920, 1080, 1.0f, 10.0f, /*seed=*/0);
    const CellResult b = cell(960, 540, 1920, 1080, 1.0f, 10.0f, /*seed=*/100);
    // Extremely unlikely to be identical.
    EXPECT_TRUE(a.u != b.u || a.v != b.v);
}
