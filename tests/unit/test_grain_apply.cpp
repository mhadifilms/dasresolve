#include "test_harness.h"

#include <vector>

#include "kernels/Kernels.h"
#include "plugin/Params.h"

using namespace dasgrain;
using namespace dasgrain::kernels;

namespace {

// Build a 1x1 RGBA float buffer with a given colour.
std::vector<float> pixel(float r, float g, float b, float a = 1.0f) {
    return {r, g, b, a};
}

ImageViewConst toConst(std::vector<float>& v, int w = 1, int h = 1) {
    ImageViewConst out;
    out.data   = v.data();
    out.width  = w;
    out.height = h;
    out.stride = static_cast<int>(w * 4 * sizeof(float));
    return out;
}

ImageView toView(std::vector<float>& v, int w = 1, int h = 1) {
    ImageView out;
    out.data   = v.data();
    out.width  = w;
    out.height = h;
    out.stride = static_cast<int>(w * 4 * sizeof(float));
    return out;
}

// Identity 1-D LUT (3 channels, n points): y = x.
std::vector<float> identityLUT(int size) {
    std::vector<float> data(static_cast<size_t>(size * 3), 0.0f);
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < size; ++i) {
            data[c * size + i] = float(i) / float(size - 1);
        }
    }
    return data;
}

}  // namespace

TEST_CASE(grain_apply_zero_grain_returns_comp) {
    auto comp  = pixel(0.5f, 0.5f, 0.5f);
    auto plate = pixel(0.5f, 0.5f, 0.5f);
    auto degr  = pixel(0.5f, 0.5f, 0.5f);   // grain == 0
    std::vector<float> emptyMask, emptyExt;
    std::vector<float> out(4, -1.0f);

    auto lut = identityLUT(64);
    GrainApplyParams gp;
    gp.outputMode = params::kOutputRegrained;
    gp.luminance  = 1.0f;
    gp.curve.data = lut.data();
    gp.curve.size = 64;
    gp.curve.minX = 0.0f;
    gp.curve.maxX = 1.0f;

    runGrainApplyCPU(gp,
                     toConst(comp),
                     toConst(plate),
                     toConst(degr),
                     ImageViewConst{},
                     ImageViewConst{},
                     toView(out));
    EXPECT_NEAR(out[0], 0.5f, 1e-6);
    EXPECT_NEAR(out[1], 0.5f, 1e-6);
    EXPECT_NEAR(out[2], 0.5f, 1e-6);
    EXPECT_NEAR(out[3], 1.0f, 1e-6);
}

TEST_CASE(grain_apply_identity_curve_grain_passes_through) {
    // With identity curve, grain on plate at level 0.5:
    //   curve(plate=0.5) = 0.5
    //   normalised = grain * eps / 0.5
    //   adapted    = normalised * curve(comp=0.5) / eps = grain
    auto comp  = pixel(0.5f, 0.5f, 0.5f);
    auto plate = pixel(0.51f, 0.52f, 0.49f);  // grain = (+0.01, +0.02, -0.01)
    auto degr  = pixel(0.5f, 0.5f, 0.5f);
    std::vector<float> out(4, -1.0f);

    auto lut = identityLUT(64);
    GrainApplyParams gp;
    gp.outputMode = params::kOutputRegrained;
    gp.luminance  = 1.0f;
    gp.curve.data = lut.data();
    gp.curve.size = 64;
    gp.curve.minX = 0.0f;
    gp.curve.maxX = 1.0f;

    runGrainApplyCPU(gp, toConst(comp), toConst(plate), toConst(degr),
                     ImageViewConst{}, ImageViewConst{}, toView(out));

    // Comp + grain (because curve(plate) ≈ curve(comp) ≈ 0.5).
    EXPECT_NEAR(out[0], 0.51f, 1e-3);
    EXPECT_NEAR(out[1], 0.52f, 1e-3);
    EXPECT_NEAR(out[2], 0.49f, 1e-3);
}

TEST_CASE(grain_apply_output_modes) {
    auto comp  = pixel(0.5f, 0.5f, 0.5f);
    auto plate = pixel(0.6f, 0.5f, 0.5f);     // grain = (+0.1, 0, 0)
    auto degr  = pixel(0.5f, 0.5f, 0.5f);

    auto lut = identityLUT(64);
    GrainApplyParams gp;
    gp.luminance  = 1.0f;
    gp.curve.data = lut.data();
    gp.curve.size = 64;
    gp.curve.minX = 0.0f;
    gp.curve.maxX = 1.0f;

    // mode 1 = original grain
    {
        std::vector<float> out(4);
        gp.outputMode = params::kOutputOriginalGrain;
        runGrainApplyCPU(gp, toConst(comp), toConst(plate), toConst(degr),
                         ImageViewConst{}, ImageViewConst{}, toView(out));
        EXPECT_NEAR(out[0],  0.10f, 1e-6);
        EXPECT_NEAR(out[1],  0.00f, 1e-6);
        EXPECT_NEAR(out[2],  0.00f, 1e-6);
    }

    // mode 2 = normalised grain. With identity curve, L=1 (lumaFactor=0),
    // clean = degrained = 0.5, so:
    //   norm.r = grain * eps / curve(degrained) = 0.1 * 0.01 / 0.5
    {
        std::vector<float> out(4);
        gp.outputMode = params::kOutputNormalisedGrain;
        runGrainApplyCPU(gp, toConst(comp), toConst(plate), toConst(degr),
                         ImageViewConst{}, ImageViewConst{}, toView(out));
        EXPECT_NEAR(out[0], 0.1f * 0.01f / 0.5f, 1e-4);
    }

    // mode 3 = adapted grain. norm * curve(comp) / eps = norm * 0.5 / 0.01
    {
        std::vector<float> out(4);
        gp.outputMode = params::kOutputAdaptedGrain;
        runGrainApplyCPU(gp, toConst(comp), toConst(plate), toConst(degr),
                         ImageViewConst{}, ImageViewConst{}, toView(out));
        const double expectedNorm  = 0.1 * 0.01 / 0.5;
        const double expectedAdapt = expectedNorm * 0.5 / 0.01;
        EXPECT_NEAR(out[0], expectedAdapt, 1e-4);
    }
}

TEST_CASE(grain_apply_external_grain_replaces_normalised) {
    auto comp  = pixel(0.5f, 0.5f, 0.5f);
    auto plate = pixel(0.7f, 0.7f, 0.7f);
    auto degr  = pixel(0.7f, 0.7f, 0.7f);  // local grain == 0
    auto ext   = pixel(0.05f, 0.05f, 0.05f);

    std::vector<float> out(4);
    auto lut = identityLUT(64);
    GrainApplyParams gp;
    gp.outputMode    = params::kOutputAdaptedGrain;
    gp.externalGrain = 1;
    gp.luminance     = 1.0f;
    gp.curve.data    = lut.data();
    gp.curve.size    = 64;
    gp.curve.minX    = 0.0f;
    gp.curve.maxX    = 1.0f;

    runGrainApplyCPU(gp, toConst(comp), toConst(plate), toConst(degr),
                     ImageViewConst{}, toConst(ext), toView(out));
    // adapted = ext * curve(comp=0.5) / eps = 0.05 * 0.5 / 0.01 = 2.5
    EXPECT_NEAR(out[0], 2.5f, 1e-4);
}

TEST_CASE(grain_apply_luma_compensation_scales_grain) {
    // Grain is luma-only here. With L=0.5 (50% degrain), the luma-comp
    // factor is (1/0.5 - 1) = 1.0, so g_lc = g + Y(g)*1 = 2*g for pure luma
    // grain. clean = degrained - Y(g) = degrained - g (for pure luma grain).
    auto comp  = pixel(0.5f, 0.5f, 0.5f);
    auto plate = pixel(0.55f, 0.55f, 0.55f);   // luma-only grain
    auto degr  = pixel(0.5f, 0.5f, 0.5f);
    std::vector<float> out(4);

    auto lut = identityLUT(64);
    GrainApplyParams gp;
    gp.outputMode  = params::kOutputAdaptedGrain;
    gp.luminance   = 0.5f;
    gp.fixGhosting = 0;
    gp.curve.data  = lut.data();
    gp.curve.size  = 64;
    gp.curve.minX  = 0.0f;
    gp.curve.maxX  = 1.0f;

    runGrainApplyCPU(gp, toConst(comp), toConst(plate), toConst(degr),
                     ImageViewConst{}, ImageViewConst{}, toView(out));
    // Grain.r = 0.05; yG = 0.05 (luma-only); lumaFactor = 1.0 -> lyG = 0.05.
    // g_lc = grain + lyG = 0.05 + 0.05 = 0.10
    // clean = 0.5 - 0.05 = 0.45 (identity curve -> 0.45)
    // norm  = 0.10 * 0.01 / 0.45
    // adapt = norm * curve(comp=0.5) / 0.01 = norm * 50 = 0.10 * 0.5 / 0.45
    EXPECT_NEAR(out[0], 0.10 * 0.5 / 0.45, 1e-3);
}

TEST_CASE(grain_apply_fix_ghosting_disables_luma_comp) {
    // Same input as above, but with fix_ghosting=1, the luma compensation
    // should be skipped (lumaFactor=0). Result mirrors the L=1 path.
    auto comp  = pixel(0.5f, 0.5f, 0.5f);
    auto plate = pixel(0.55f, 0.55f, 0.55f);
    auto degr  = pixel(0.5f, 0.5f, 0.5f);
    std::vector<float> out(4);

    auto lut = identityLUT(64);
    GrainApplyParams gp;
    gp.outputMode  = params::kOutputAdaptedGrain;
    gp.luminance   = 0.5f;     // would normally engage luma comp
    gp.fixGhosting = 1;        // ... but fix_ghosting overrides
    gp.curve.data  = lut.data();
    gp.curve.size  = 64;
    gp.curve.minX  = 0.0f;
    gp.curve.maxX  = 1.0f;

    runGrainApplyCPU(gp, toConst(comp), toConst(plate), toConst(degr),
                     ImageViewConst{}, ImageViewConst{}, toView(out));
    // With lumaFactor=0:
    // clean = degr = 0.5; curve(clean)=0.5;
    // norm = grain * 0.01/0.5 = 0.05 * 0.01/0.5 = 0.001
    // adapt = 0.001 * curve(comp=0.5) / 0.01 = 0.05
    EXPECT_NEAR(out[0], 0.05, 1e-4);
}

TEST_CASE(grain_apply_mask_zero_disables_grain) {
    auto comp  = pixel(0.5f, 0.5f, 0.5f);
    auto plate = pixel(0.6f, 0.5f, 0.5f);
    auto degr  = pixel(0.5f, 0.5f, 0.5f);
    auto mask  = pixel(0.0f, 0.0f, 0.0f, /*alpha*/0.0f);

    std::vector<float> out(4);
    auto lut = identityLUT(64);
    GrainApplyParams gp;
    gp.outputMode = params::kOutputRegrained;
    gp.luminance  = 1.0f;
    gp.hasMask    = 1;
    gp.curve.data = lut.data();
    gp.curve.size = 64;
    gp.curve.minX = 0.0f;
    gp.curve.maxX = 1.0f;

    runGrainApplyCPU(gp, toConst(comp), toConst(plate), toConst(degr),
                     toConst(mask), ImageViewConst{}, toView(out));
    // mask alpha is 0, so result should equal comp.
    EXPECT_NEAR(out[0], 0.5f, 1e-6);
    EXPECT_NEAR(out[1], 0.5f, 1e-6);
    EXPECT_NEAR(out[2], 0.5f, 1e-6);
}
