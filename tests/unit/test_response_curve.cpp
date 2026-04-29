#include "test_harness.h"

#include "core/ResponseCurve.h"

using dasgrain::ResponseCurve;

TEST_CASE(response_curve_default_is_identity) {
    ResponseCurve c;
    for (int ch = 0; ch < ResponseCurve::kChannelCount; ++ch) {
        EXPECT_NEAR(c.sample(ch, 0.0),  0.0, 1e-9);
        EXPECT_NEAR(c.sample(ch, 0.5),  0.5, 1e-9);
        EXPECT_NEAR(c.sample(ch, 1.0),  1.0, 1e-9);
        // Endpoint clamping outside the domain.
        EXPECT_NEAR(c.sample(ch, -1.0), 0.0, 1e-9);
        EXPECT_NEAR(c.sample(ch, 2.0),  1.0, 1e-9);
    }
}

TEST_CASE(response_curve_set_channel_and_sample) {
    ResponseCurve c;
    c.setChannel(0, {{0.0, 0.0}, {0.25, 0.5}, {1.0, 1.0}});
    EXPECT_NEAR(c.sample(0, 0.0),    0.0,  1e-9);
    EXPECT_NEAR(c.sample(0, 0.125),  0.25, 1e-9);  // mid of (0,0)-(0.25,0.5)
    EXPECT_NEAR(c.sample(0, 0.25),   0.5,  1e-9);
    EXPECT_NEAR(c.sample(0, 0.625),  0.75, 1e-9);  // mid of (0.25,0.5)-(1,1)
}

TEST_CASE(response_curve_json_roundtrip) {
    ResponseCurve c;
    c.setChannel(0, {{0.0, 0.0}, {0.5, 0.3}, {1.0, 1.2}});
    c.setChannel(1, {{0.0, 0.1}, {1.0, 0.9}});
    c.setChannel(2, {{0.0, 0.0}, {0.4, 0.5}, {0.8, 0.8}, {1.0, 1.1}});

    const std::string json = c.toJSON();
    EXPECT_TRUE(json.find("\"version\":1") != std::string::npos);
    EXPECT_TRUE(json.find("\"curves\"")    != std::string::npos);

    ResponseCurve c2;
    EXPECT_TRUE(c2.fromJSON(json));
    for (int ch = 0; ch < ResponseCurve::kChannelCount; ++ch) {
        EXPECT_EQ(c.channel(ch).size(), c2.channel(ch).size());
        for (size_t i = 0; i < c.channel(ch).size(); ++i) {
            EXPECT_NEAR(c.channel(ch)[i].x, c2.channel(ch)[i].x, 1e-9);
            EXPECT_NEAR(c.channel(ch)[i].y, c2.channel(ch)[i].y, 1e-9);
        }
    }
}

TEST_CASE(response_curve_rejects_garbage) {
    ResponseCurve c;
    EXPECT_TRUE(!c.fromJSON(""));
    EXPECT_TRUE(!c.fromJSON("not json"));
    EXPECT_TRUE(!c.fromJSON("{\"curves\":[]}"));
}
