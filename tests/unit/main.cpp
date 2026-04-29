#include "test_harness.h"

#include <cstdio>

int main() {
    auto& cases = dasgrain_test::registry();
    std::printf("Running %zu tests\n", cases.size());
    for (auto& c : cases) {
        std::printf("[ RUN      ] %s\n", c.name);
        const int beforeFails = dasgrain_test::failureCount();
        c.fn();
        const int afterFails  = dasgrain_test::failureCount();
        std::printf("[ %s ] %s\n",
                    (afterFails == beforeFails) ? "      OK" : "FAILED  ",
                    c.name);
    }
    const int n = dasgrain_test::failureCount();
    std::printf("%s. Failed asserts: %d\n", n == 0 ? "All passed" : "FAILURES", n);
    return n == 0 ? 0 : 1;
}
