// Tiny zero-dependency test helper. We only need a few asserts and a
// per-file registration mechanism. No other test infrastructure is pulled
// into the build.

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace dasgrain_test {

struct Case { const char* name; void (*fn)(); };
inline std::vector<Case>& registry() { static std::vector<Case> v; return v; }

struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

inline int& failureCount() { static int n = 0; return n; }

#define TEST_CASE(name) \
    static void name(); \
    static ::dasgrain_test::Registrar reg_##name(#name, name); \
    static void name()

#define EXPECT_TRUE(cond) do { \
    if (!(cond)) { \
        std::printf("  FAIL %s:%d: EXPECT_TRUE(%s)\n", __FILE__, __LINE__, #cond); \
        ::dasgrain_test::failureCount()++; \
    } \
} while (0)

#define EXPECT_EQ(a, b) do { \
    if (!((a) == (b))) { \
        std::printf("  FAIL %s:%d: EXPECT_EQ(%s, %s)\n", __FILE__, __LINE__, #a, #b); \
        ::dasgrain_test::failureCount()++; \
    } \
} while (0)

#define EXPECT_NEAR(a, b, tol) do { \
    const double _da = double(a); \
    const double _db = double(b); \
    if (std::fabs(_da - _db) > (tol)) { \
        std::printf("  FAIL %s:%d: EXPECT_NEAR(%s=%.10g, %s=%.10g, tol=%.3g)\n", \
                    __FILE__, __LINE__, #a, _da, #b, _db, double(tol)); \
        ::dasgrain_test::failureCount()++; \
    } \
} while (0)

}  // namespace dasgrain_test
