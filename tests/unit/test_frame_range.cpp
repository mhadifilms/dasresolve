#include "test_harness.h"

#include "core/FrameRange.h"

#include <vector>

using dasgrain::FrameRange;

TEST_CASE(frame_range_single_frames) {
    auto frames = FrameRange::parseAdditional("1001,1020,1053");
    std::vector<int> expected{1001, 1020, 1053};
    EXPECT_EQ(frames.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(frames[i], expected[i]);
}

TEST_CASE(frame_range_simple_range) {
    auto frames = FrameRange::parseAdditional("1020-1023");
    std::vector<int> expected{1020, 1021, 1022, 1023};
    EXPECT_EQ(frames.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(frames[i], expected[i]);
}

TEST_CASE(frame_range_with_step) {
    auto frames = FrameRange::parseAdditional("1020-1040x4");
    std::vector<int> expected{1020, 1024, 1028, 1032, 1036, 1040};
    EXPECT_EQ(frames.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(frames[i], expected[i]);
}

TEST_CASE(frame_range_combined_input) {
    auto frames = FrameRange::parseAdditional("1001,1020-1024,1100x2");
    EXPECT_EQ(frames.size(), 7u);
    EXPECT_EQ(frames[0], 1001);
    EXPECT_EQ(frames[1], 1020);
    EXPECT_EQ(frames[2], 1021);
    EXPECT_EQ(frames[3], 1022);
    EXPECT_EQ(frames[4], 1023);
    EXPECT_EQ(frames[5], 1024);
    EXPECT_EQ(frames[6], 1100);
}

TEST_CASE(frame_range_generate_spreads_evenly) {
    // 10 frames spread across [1001, 1100] should mirror Nuke's algo.
    auto frames = FrameRange::generate(10, "", 1001, 1100);
    EXPECT_EQ(frames.size(), 10u);
    EXPECT_TRUE(frames.front() >= 1001);
    EXPECT_TRUE(frames.back()  <= 1100);
    // Strictly increasing and unique.
    for (size_t i = 1; i < frames.size(); ++i) EXPECT_TRUE(frames[i] > frames[i - 1]);
}

TEST_CASE(frame_range_generate_includes_additional) {
    auto frames = FrameRange::generate(0, "1010,1020,1030", 1000, 1100);
    EXPECT_EQ(frames.size(), 3u);
    EXPECT_EQ(frames[0], 1010);
    EXPECT_EQ(frames[1], 1020);
    EXPECT_EQ(frames[2], 1030);
}

TEST_CASE(frame_range_generate_clamps_oob) {
    auto frames = FrameRange::generate(0, "999,1010,1500", 1000, 1100);
    EXPECT_EQ(frames.size(), 1u);
    EXPECT_EQ(frames[0], 1010);
}
