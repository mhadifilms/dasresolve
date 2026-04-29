#include "test_harness.h"

#include "core/ColorSpace.h"

using namespace dasgrain::colorspace;

TEST_CASE(logc_round_trip) {
    for (double v : {0.0, 0.005, 0.01, 0.05, 0.18, 0.5, 1.0, 4.0, 16.0}) {
        const double encoded = linearToLogC(v);
        const double back    = logCToLinear(encoded);
        EXPECT_NEAR(back, v, 1e-6);
    }
}

TEST_CASE(luma709_known_values) {
    EXPECT_NEAR(luma709(1.0, 0.0, 0.0), 0.2126, 1e-9);
    EXPECT_NEAR(luma709(0.0, 1.0, 0.0), 0.7152, 1e-9);
    EXPECT_NEAR(luma709(0.0, 0.0, 1.0), 0.0722, 1e-9);
    EXPECT_NEAR(luma709(1.0, 1.0, 1.0), 1.0,    1e-9);
}
