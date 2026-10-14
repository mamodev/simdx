#include "ops_kernel.h"

#include <cmath>
#include <cstdio>
#include <random>

namespace {

int failures = 0;

void check(bool ok, const char* isa, const char* what)
{
    if (!ok) {
        failures++;
        std::printf("FAIL %-7s %s\n", isa, what);
    }
}

bool same(float a, float b)
{
    return a == b || (std::isnan(a) && std::isnan(b));
}

bool close(float a, float b, float relative)
{
    return std::fabs(a - b) <= relative * std::fmax(1.0f, std::fabs(b));
}

simdx::Mask expectedMask(const float* a, const float* b, int first, int lanes, bool (*predicate)(float, float))
{
    simdx::Mask mask = 0;
    for (int lane = 0; lane < lanes; lane++)
        if (predicate(a[first + lane], b[first + lane]))
            mask |= 1u << lane;
    return mask;
}

void verify(const char* isa, ops::Apply* apply, const float* a, const float* b, const float* c)
{
    ops::Results r{};
    apply(a, b, c, r);
    const int n = ops::kValues;

    bool exact = true, fused = true, sqrtOk = true;
    double total = 0;
    for (int i = 0; i < n; i++) {
        exact &= same(r.add[i], a[i] + b[i]) && same(r.sub[i], a[i] - b[i]) && same(r.mul[i], a[i] * b[i]) &&
                 same(r.div[i], a[i] / b[i]) && same(r.neg[i], -a[i]) && same(r.abs[i], std::fabs(a[i])) &&
                 same(r.min[i], a[i] < b[i] ? a[i] : b[i]) && same(r.max[i], a[i] > b[i] ? a[i] : b[i]);
        fused &= std::isnan(b[i]) ? std::isnan(r.fma[i]) : close(r.fma[i], std::fma(a[i], b[i], c[i]), 1e-6f);
        sqrtOk &= r.sqrt[i] == std::sqrt(std::fabs(a[i]));
        total += a[i];
    }
    check(exact, isa, "add sub mul div neg abs min max are exact, NaN included");
    check(fused, isa, "fma");
    check(sqrtOk, isa, "sqrt is exact");
    check(close(r.sum, static_cast<float>(total), 1e-5f), isa, "sum");

    bool masks = true;
    for (int i = 0; i < n; i += r.lanes) {
        masks &= r.lt[i] == expectedMask(a, b, i, r.lanes, [](float x, float y) { return x < y; });
        masks &= r.le[i] == expectedMask(a, b, i, r.lanes, [](float x, float y) { return x <= y; });
        masks &= r.gt[i] == expectedMask(a, b, i, r.lanes, [](float x, float y) { return x > y; });
        masks &= r.ge[i] == expectedMask(a, b, i, r.lanes, [](float x, float y) { return x >= y; });
        masks &= r.eq[i] == expectedMask(a, b, i, r.lanes, [](float x, float y) { return x == y; });
        masks &= r.ne[i] == expectedMask(a, b, i, r.lanes, [](float x, float y) { return x != y; });
    }
    check(masks, isa, "compares match scalar, NaN included");
}

}  // namespace

int main()
{
    std::mt19937 random(42);
    std::uniform_real_distribution<float> value(-100.0f, 100.0f);
    float a[ops::kValues], b[ops::kValues], c[ops::kValues];
    for (int i = 0; i < ops::kValues; i++) {
        a[i] = value(random);
        b[i] = i % 5 == 0 ? a[i] : value(random);    // some equal lanes for == / <=
        c[i] = value(random);
    }
    b[7] = NAN;
    b[0] = b[0] == 0.0f ? 1.0f : b[0];

    const char* names[] = {"sse2", "avx2", "avx512"};
    ops::Apply* variants[] = {SIMDX_VARIANTS(ops, apply)};
    for (int isa = 0; isa < 3; isa++) {
        if (!simdx::supported(static_cast<simdx::Isa>(isa))) {
            std::printf("skip %s (not supported by this CPU)\n", names[isa]);
            continue;
        }
        verify(names[isa], variants[isa], a, b, c);
        std::printf("ran  %s\n", names[isa]);
    }
    check(simdx::dispatch(SIMDX_VARIANTS(ops, apply)) == variants[static_cast<int>(simdx::best())], "dispatch",
          "picks the best supported variant");

    std::printf(failures ? "%d FAILURES\n" : "ALL PASSED\n", failures);
    return failures != 0;
}
