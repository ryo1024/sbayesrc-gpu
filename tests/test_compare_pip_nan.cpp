// Standalone test for the NaN-safe PIP comparator. The GCTB-side codebase is
// compiled with -O3 -ffast-math (-ffinite-math-only), which makes the compiler
// assume no FP operand is ever NaN and silently optimize away "a != a" and
// std::isnan(a). The fix relies on bit-pattern detection via __builtin_memcpy +
// __attribute__((optimize("no-finite-math-only"))) to survive that.
//
// This file mirrors the comparator implementation byte-for-byte and runs the
// same suite under -ffast-math as the production build, so a regression where
// the attribute is dropped or the bit-pattern check is rewritten back to isnan()
// would fail here without needing a GPU.
//
// Build:
//   g++ -O3 -ffast-math -std=c++17 -o test_compare_pip_nan test_compare_pip_nan.cpp && ./test_compare_pip_nan
//
// Exits 0 on success, non-zero with a description on failure.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <limits>

struct SnpInfo {
    char _pad[0x154];     // mirror the field offset GCTB sees
    float pip;
};

__attribute__((optimize("no-finite-math-only")))
static bool comparePIP(const SnpInfo* snpi, const SnpInfo* snpj) {
    float a = snpi->pip, b = snpj->pip;
    unsigned ai, bi;
    __builtin_memcpy(&ai, &a, sizeof(ai));
    __builtin_memcpy(&bi, &b, sizeof(bi));
    bool an = ((ai & 0x7f800000u) == 0x7f800000u) && (ai & 0x7fffffu);
    bool bn = ((bi & 0x7f800000u) == 0x7f800000u) && (bi & 0x7fffffu);
    if (an && bn) return false;
    if (an) return false;
    if (bn) return true;
    return a > b;
}

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s (line %d): %s\n", msg, __LINE__, #cond); ++failures; } \
} while (0)

static SnpInfo make_snp(float pip) {
    SnpInfo s{};
    s.pip = pip;
    return s;
}

int main() {
    const float nan_v = std::numeric_limits<float>::quiet_NaN();
    const float inf_v = std::numeric_limits<float>::infinity();

    SnpInfo s_low   = make_snp(0.1f);
    SnpInfo s_mid   = make_snp(0.5f);
    SnpInfo s_high  = make_snp(0.9f);
    SnpInfo s_zero  = make_snp(0.0f);
    SnpInfo s_one   = make_snp(1.0f);
    SnpInfo s_nan_a = make_snp(nan_v);
    SnpInfo s_nan_b = make_snp(nan_v);

    // 1) Basic ordering: higher PIP comes first.
    CHECK( comparePIP(&s_high, &s_low),  "high > low");
    CHECK(!comparePIP(&s_low,  &s_high), "low not > high");
    CHECK(!comparePIP(&s_mid,  &s_mid),  "equal not >");
    CHECK( comparePIP(&s_one,  &s_zero), "1.0 > 0.0");

    // 2) NaN is treated as smallest — anything finite sorts before NaN.
    CHECK( comparePIP(&s_low,   &s_nan_a), "any finite > NaN");
    CHECK( comparePIP(&s_zero,  &s_nan_a), "0.0 > NaN");
    CHECK( comparePIP(&s_high,  &s_nan_a), "0.9 > NaN");
    CHECK(!comparePIP(&s_nan_a, &s_low),   "NaN not > finite");
    CHECK(!comparePIP(&s_nan_a, &s_zero),  "NaN not > 0.0");

    // 3) NaN vs NaN — not strictly greater either way (equivalent).
    CHECK(!comparePIP(&s_nan_a, &s_nan_b), "NaN not > NaN");
    CHECK(!comparePIP(&s_nan_b, &s_nan_a), "NaN not > NaN (sym)");

    // 4) +Inf is NOT NaN (exp=0xff, mantissa=0). It should compare normally.
    SnpInfo s_pinf = make_snp(inf_v);
    CHECK( comparePIP(&s_pinf, &s_high), "+inf > 0.9");
    CHECK(!comparePIP(&s_high, &s_pinf), "0.9 not > +inf");

    // 5) Strict weak ordering: !(a<b) && !(b<a)  ⇒  a equivalent to b.
    //    Two NaNs are equivalent; a NaN and a finite are NOT equivalent (NaN < finite).
    auto sw_eq = [](const SnpInfo& a, const SnpInfo& b) {
        return !comparePIP(&a, &b) && !comparePIP(&b, &a);
    };
    CHECK( sw_eq(s_nan_a, s_nan_b),     "SWO: NaN ~ NaN");
    CHECK(!sw_eq(s_nan_a, s_low),       "SWO: NaN ≢ finite");
    CHECK( sw_eq(s_mid, s_mid),         "SWO: x ~ x");

    // 6) Sort a vector containing many NaNs — std::sort with a comparator that
    //    violates strict weak ordering can write OOB on libstdc++ introsort.
    //    Run a larger sort to stress that path.
    {
        std::vector<SnpInfo> v;
        for (int i = 0; i < 1000; ++i) {
            float p;
            if (i % 7 == 0)      p = nan_v;
            else if (i % 11 == 0) p = inf_v;
            else                  p = (i * 0.001237f);
            v.push_back(make_snp(p));
        }
        std::vector<SnpInfo*> ptrs;
        ptrs.reserve(v.size());
        for (auto& s : v) ptrs.push_back(&s);
        std::sort(ptrs.begin(), ptrs.end(), comparePIP);
        // After sort, NaNs must all be at the end. Find first NaN and verify
        // every subsequent element is also NaN.
        size_t first_nan = ptrs.size();
        for (size_t i = 0; i < ptrs.size(); ++i) {
            unsigned bits;
            __builtin_memcpy(&bits, &ptrs[i]->pip, sizeof(bits));
            bool is_nan = ((bits & 0x7f800000u) == 0x7f800000u) && (bits & 0x7fffffu);
            if (is_nan) { first_nan = i; break; }
        }
        for (size_t i = first_nan; i < ptrs.size(); ++i) {
            unsigned bits;
            __builtin_memcpy(&bits, &ptrs[i]->pip, sizeof(bits));
            bool is_nan = ((bits & 0x7f800000u) == 0x7f800000u) && (bits & 0x7fffffu);
            CHECK(is_nan, "NaNs contiguous at tail after sort");
        }
        // Finite head should be in descending order.
        for (size_t i = 1; i < first_nan; ++i) {
            CHECK(ptrs[i-1]->pip >= ptrs[i]->pip, "finite portion descending");
        }
    }

    if (failures) {
        std::fprintf(stderr, "\n%d test(s) failed.\n", failures);
        return 1;
    }
    std::printf("All comparePIP NaN tests passed.\n");
    return 0;
}
