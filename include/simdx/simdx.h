#pragma once

// simdx: portable SIMD for kernels written once and compiled once per ISA.
//
// The ISA is chosen by the flags of the file that includes this header:
//     -mavx512f -mavx2 -mfma   avx512   16 lanes
//     -mavx2 -mfma             avx2      8 lanes
//     (none)                   sse2      4 lanes
// make/simdx.mk builds a kernel file three times with these flags.
//
// Everything here sits in namespace simdx::SIMDX_ISA, so the copies built for different ISAs never
// collide at link time: the linker can never hand an AVX-512 copy to a baseline caller.
//
// Kernel file, built once per ISA:
//     #include <simdx/simdx.h>
//     namespace dsp::SIMDX_ISA {
//     void scale(float* x, int n, float k)
//     {
//         using simdx::Float;
//         for (int i = 0; i < n; i += Float::kLanes)
//             (Float::load(x + i) * Float::broadcast(k)).store(x + i);
//     }
//     }
//
// Caller, built normally:
//     namespace dsp { SIMDX_DECLARE(void scale(float* x, int n, float k)) }
//     static const auto scale = simdx::dispatch(SIMDX_VARIANTS(dsp, scale));

#include <cstdint>
#include <immintrin.h>

#if defined(__AVX512F__) && defined(__AVX2__) && defined(__FMA__)
#define SIMDX_ISA avx512
#define SIMDX_AVX512 1
#elif defined(__AVX2__) && defined(__FMA__)
#define SIMDX_ISA avx2
#define SIMDX_AVX2 1
#else
#define SIMDX_ISA sse2
#define SIMDX_SSE2 1
#endif

#define SIMDX_DECLARE(...)                 \
    namespace sse2 { __VA_ARGS__; }        \
    namespace avx2 { __VA_ARGS__; }        \
    namespace avx512 { __VA_ARGS__; }

#define SIMDX_VARIANTS(ns, name) ns::sse2::name, ns::avx2::name, ns::avx512::name

namespace simdx {
inline namespace SIMDX_ISA {

// Ordered: each ISA can run everything below it.
enum class Isa { Sse2, Avx2, Avx512 };

// One bit per lane, lane 0 in bit 0, whatever the ISA.
using Mask = uint32_t;

namespace native {

#if SIMDX_AVX512

using Vector = __m512;
constexpr int kLanes = 16;

inline Vector load(const float* p) noexcept { return _mm512_loadu_ps(p); }
inline void store(float* p, Vector v) noexcept { _mm512_storeu_ps(p, v); }
inline Vector broadcast(float x) noexcept { return _mm512_set1_ps(x); }
inline Vector add(Vector a, Vector b) noexcept { return _mm512_add_ps(a, b); }
inline Vector sub(Vector a, Vector b) noexcept { return _mm512_sub_ps(a, b); }
inline Vector mul(Vector a, Vector b) noexcept { return _mm512_mul_ps(a, b); }
inline Vector div(Vector a, Vector b) noexcept { return _mm512_div_ps(a, b); }
inline Vector fma(Vector a, Vector b, Vector c) noexcept { return _mm512_fmadd_ps(a, b, c); }
inline Vector abs(Vector a) noexcept { return _mm512_abs_ps(a); }
inline Vector min(Vector a, Vector b) noexcept { return _mm512_min_ps(a, b); }
inline Vector max(Vector a, Vector b) noexcept { return _mm512_max_ps(a, b); }
inline Vector sqrt(Vector a) noexcept { return _mm512_sqrt_ps(a); }
inline float sum(Vector a) noexcept { return _mm512_reduce_add_ps(a); }

template <int Predicate>
inline Mask compare(Vector a, Vector b) noexcept { return _mm512_cmp_ps_mask(a, b, Predicate); }

#elif SIMDX_AVX2

using Vector = __m256;
constexpr int kLanes = 8;

inline Vector load(const float* p) noexcept { return _mm256_loadu_ps(p); }
inline void store(float* p, Vector v) noexcept { _mm256_storeu_ps(p, v); }
inline Vector broadcast(float x) noexcept { return _mm256_set1_ps(x); }
inline Vector add(Vector a, Vector b) noexcept { return _mm256_add_ps(a, b); }
inline Vector sub(Vector a, Vector b) noexcept { return _mm256_sub_ps(a, b); }
inline Vector mul(Vector a, Vector b) noexcept { return _mm256_mul_ps(a, b); }
inline Vector div(Vector a, Vector b) noexcept { return _mm256_div_ps(a, b); }
inline Vector fma(Vector a, Vector b, Vector c) noexcept { return _mm256_fmadd_ps(a, b, c); }
inline Vector abs(Vector a) noexcept { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a); }
inline Vector min(Vector a, Vector b) noexcept { return _mm256_min_ps(a, b); }
inline Vector max(Vector a, Vector b) noexcept { return _mm256_max_ps(a, b); }
inline Vector sqrt(Vector a) noexcept { return _mm256_sqrt_ps(a); }

inline float sum(Vector a) noexcept
{
    __m128 v = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    v = _mm_add_ps(v, _mm_movehl_ps(v, v));
    return _mm_cvtss_f32(_mm_add_ss(v, _mm_shuffle_ps(v, v, 1)));
}

template <int Predicate>
inline Mask compare(Vector a, Vector b) noexcept
{
    return static_cast<Mask>(_mm256_movemask_ps(_mm256_cmp_ps(a, b, Predicate)));
}

#else

using Vector = __m128;
constexpr int kLanes = 4;

inline Vector load(const float* p) noexcept { return _mm_loadu_ps(p); }
inline void store(float* p, Vector v) noexcept { _mm_storeu_ps(p, v); }
inline Vector broadcast(float x) noexcept { return _mm_set1_ps(x); }
inline Vector add(Vector a, Vector b) noexcept { return _mm_add_ps(a, b); }
inline Vector sub(Vector a, Vector b) noexcept { return _mm_sub_ps(a, b); }
inline Vector mul(Vector a, Vector b) noexcept { return _mm_mul_ps(a, b); }
inline Vector div(Vector a, Vector b) noexcept { return _mm_div_ps(a, b); }
inline Vector fma(Vector a, Vector b, Vector c) noexcept { return _mm_add_ps(_mm_mul_ps(a, b), c); }
inline Vector abs(Vector a) noexcept { return _mm_andnot_ps(_mm_set1_ps(-0.0f), a); }
inline Vector min(Vector a, Vector b) noexcept { return _mm_min_ps(a, b); }
inline Vector max(Vector a, Vector b) noexcept { return _mm_max_ps(a, b); }
inline Vector sqrt(Vector a) noexcept { return _mm_sqrt_ps(a); }

inline float sum(Vector a) noexcept
{
    const __m128 v = _mm_add_ps(a, _mm_movehl_ps(a, a));
    return _mm_cvtss_f32(_mm_add_ss(v, _mm_shuffle_ps(v, v, 1)));
}

// SSE2 has no predicate operand: map the AVX predicates used by Float onto its compares.
template <int Predicate>
inline Mask compare(Vector a, Vector b) noexcept
{
    Vector r;
    if constexpr (Predicate == _CMP_LT_OQ)
        r = _mm_cmplt_ps(a, b);
    else if constexpr (Predicate == _CMP_LE_OQ)
        r = _mm_cmple_ps(a, b);
    else if constexpr (Predicate == _CMP_GT_OQ)
        r = _mm_cmpgt_ps(a, b);
    else if constexpr (Predicate == _CMP_GE_OQ)
        r = _mm_cmpge_ps(a, b);
    else if constexpr (Predicate == _CMP_EQ_OQ)
        r = _mm_cmpeq_ps(a, b);
    else
        r = _mm_cmpneq_ps(a, b);
    return static_cast<Mask>(_mm_movemask_ps(r));
}

#endif

}  // namespace native

class Float {
public:
    static constexpr int kLanes = native::kLanes;

    Float() noexcept = default;

    static Float load(const float* p) noexcept { return Float(native::load(p)); }
    static Float broadcast(float x) noexcept { return Float(native::broadcast(x)); }
    static Float zero() noexcept { return broadcast(0.0f); }
    void store(float* p) const noexcept { native::store(p, v_); }

    friend Float operator+(Float a, Float b) noexcept { return Float(native::add(a.v_, b.v_)); }
    friend Float operator-(Float a, Float b) noexcept { return Float(native::sub(a.v_, b.v_)); }
    friend Float operator*(Float a, Float b) noexcept { return Float(native::mul(a.v_, b.v_)); }
    friend Float operator/(Float a, Float b) noexcept { return Float(native::div(a.v_, b.v_)); }
    friend Float operator-(Float a) noexcept { return zero() - a; }
    Float& operator+=(Float b) noexcept { return *this = *this + b; }
    Float& operator-=(Float b) noexcept { return *this = *this - b; }
    Float& operator*=(Float b) noexcept { return *this = *this * b; }
    Float& operator/=(Float b) noexcept { return *this = *this / b; }

    // a * b + c; a single rounding on AVX2 / AVX-512, two on SSE2.
    friend Float fma(Float a, Float b, Float c) noexcept { return Float(native::fma(a.v_, b.v_, c.v_)); }
    friend Float abs(Float a) noexcept { return Float(native::abs(a.v_)); }
    // x86 semantics: b when either lane is NaN.
    friend Float min(Float a, Float b) noexcept { return Float(native::min(a.v_, b.v_)); }
    friend Float max(Float a, Float b) noexcept { return Float(native::max(a.v_, b.v_)); }
    friend Float sqrt(Float a) noexcept { return Float(native::sqrt(a.v_)); }
    friend float sum(Float a) noexcept { return native::sum(a.v_); }

    // Ordered compares: a NaN lane is false, except for !=.
    friend Mask operator<(Float a, Float b) noexcept { return native::compare<_CMP_LT_OQ>(a.v_, b.v_); }
    friend Mask operator<=(Float a, Float b) noexcept { return native::compare<_CMP_LE_OQ>(a.v_, b.v_); }
    friend Mask operator>(Float a, Float b) noexcept { return native::compare<_CMP_GT_OQ>(a.v_, b.v_); }
    friend Mask operator>=(Float a, Float b) noexcept { return native::compare<_CMP_GE_OQ>(a.v_, b.v_); }
    friend Mask operator==(Float a, Float b) noexcept { return native::compare<_CMP_EQ_OQ>(a.v_, b.v_); }
    friend Mask operator!=(Float a, Float b) noexcept { return native::compare<_CMP_NEQ_UQ>(a.v_, b.v_); }

private:
    explicit Float(native::Vector v) noexcept : v_(v) {}

    native::Vector v_;
};

template <class Visit>
void forEachSet(Mask mask, Visit&& visit)
{
    for (; mask != 0; mask &= mask - 1)
        visit(__builtin_ctz(mask));
}

// AVX-512 requires AVX2 and FMA too, matching the flags simdx.mk builds it with.
inline Isa best() noexcept
{
    static const Isa isa = [] {
        __builtin_cpu_init();
        const bool avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
        if (avx2 && __builtin_cpu_supports("avx512f"))
            return Isa::Avx512;
        return avx2 ? Isa::Avx2 : Isa::Sse2;
    }();
    return isa;
}

inline bool supported(Isa isa) noexcept { return isa <= best(); }

template <class Function>
Function* dispatch(Function* sse2, Function* avx2, Function* avx512) noexcept
{
    switch (best()) {
    case Isa::Avx512:
        return avx512;
    case Isa::Avx2:
        return avx2;
    default:
        return sse2;
    }
}

}  // inline namespace SIMDX_ISA
}  // namespace simdx
