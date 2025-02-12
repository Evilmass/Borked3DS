#ifndef COMMON_VECTOR_MATH_H
#define COMMON_VECTOR_MATH_H

// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// Copyright 2014 Tony Wasserka
// Copyright 2025 Borked3DS Emulator Project
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the owner nor the names of its contributors may
//       be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <boost/serialization/access.hpp>

// SIMD includes
#if defined(__x86_64__) || defined(_M_X64)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#if defined(__SSE2__)
#define HAVE_SSE2
#endif
#if defined(__SSE4_1__)
#define HAVE_SSE4_1
#endif
#if defined(__AVX__)
#define HAVE_AVX
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#ifdef _MSC_VER
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#define HAVE_NEON
#endif

namespace Common {

template <typename T>
class Vec2;
template <typename T>
class Vec3;
template <typename T>
class Vec4;

namespace detail {
#if defined(HAVE_SSE2) || defined(HAVE_NEON)
constexpr bool has_simd_support = true;
#else
constexpr bool has_simd_support = false;
#endif

template <typename T>
struct is_vectorizable {
    static constexpr bool value = std::is_same_v<T, float>;
};

// Helper functions for NEON operations
#ifdef __ARM_NEON
inline float32x4_t vdivq_f32(float32x4_t num, float32x4_t den) {
    // Initial estimate
    float32x4_t recip = vrecpeq_f32(den);

    // Newton-Raphson iterations for improved precision
    recip = vmulq_f32(recip, vrecpsq_f32(den, recip));
    recip = vmulq_f32(recip, vrecpsq_f32(den, recip));

    return vmulq_f32(num, recip);
}

inline float32x2_t vdiv_f32(float32x2_t num, float32x2_t den) {
    // Initial estimate
    float32x2_t recip = vrecpe_f32(den);

    // Newton-Raphson iterations for improved precision
    recip = vmul_f32(recip, vrecps_f32(den, recip));
    recip = vmul_f32(recip, vrecps_f32(den, recip));

    return vmul_f32(num, recip);
}
#endif

} // namespace detail

template <typename T>
class alignas(8) Vec2 {
    static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                  "Vector type must be floating point or integral");

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        ar & x;
        ar & y;
    }

public:
    T x{};
    T y{};

    // Ensure proper alignment for SIMD operations
    static constexpr std::size_t alignment = 8;
    static_assert(alignof(Vec2) >= alignment, "Vec2 must be properly aligned for SIMD operations");

    constexpr Vec2() = default;
    constexpr Vec2(const T& x_, const T& y_) : x(x_), y(y_) {}

    template <typename T2>
    [[nodiscard]] constexpr Vec2<T2> Cast() const {
        return Vec2<T2>(static_cast<T2>(x), static_cast<T2>(y));
    }

    [[nodiscard]] static constexpr Vec2 AssignToAll(const T& f) {
        return Vec2(f, f);
    }

    [[nodiscard]] constexpr Vec2<decltype(T{} + T{})> operator+(const Vec2& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
            Vec2<T> result;
#if defined(HAVE_SSE2)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 b = _mm_setr_ps(other.x, other.y, 0.0f, 0.0f);
            __m128 sum = _mm_add_ps(a, b);
            _mm_store_ss(&result.x, sum);
            _mm_store_ss(&result.y, _mm_movehl_ps(sum, sum));
#elif defined(HAVE_NEON)
            float32x2_t a =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&x))));
            float32x2_t b =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&other.x))));
            float32x2_t sum = vadd_f32(a, b);
            vst1_f32(&result.x, sum);
#endif
            return result;
        } else {
            return {x + other.x, y + other.y};
        }
    }

    constexpr Vec2& operator+=(const Vec2& other) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 b = _mm_setr_ps(other.x, other.y, 0.0f, 0.0f);
            __m128 sum = _mm_add_ps(a, b);
            _mm_store_ss(&x, sum);
            _mm_store_ss(&y, _mm_movehl_ps(sum, sum));
#elif defined(HAVE_NEON)
            float32x2_t a = vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<uint64_t*>(&x))));
            float32x2_t b =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&other.x))));
            float32x2_t sum = vadd_f32(a, b);
            vst1_f32(&x, sum);
#endif
        } else {
            x += other.x;
            y += other.y;
        }
        return *this;
    }

    [[nodiscard]] constexpr Vec2<decltype(T{} - T{})> operator-(const Vec2& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
            Vec2<T> result;
#if defined(HAVE_SSE2)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 b = _mm_setr_ps(other.x, other.y, 0.0f, 0.0f);
            __m128 diff = _mm_sub_ps(a, b);
            _mm_store_ss(&result.x, diff);
            _mm_store_ss(&result.y, _mm_movehl_ps(diff, diff));
#elif defined(HAVE_NEON)
            float32x2_t a =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&x))));
            float32x2_t b =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&other.x))));
            float32x2_t diff = vsub_f32(a, b);
            vst1_f32(&result.x, diff);
#endif
            return result;
        } else {
            return {x - other.x, y - other.y};
        }
    }

    constexpr Vec2& operator-=(const Vec2& other) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 b = _mm_setr_ps(other.x, other.y, 0.0f, 0.0f);
            __m128 diff = _mm_sub_ps(a, b);
            _mm_store_ss(&x, diff);
            _mm_store_ss(&y, _mm_movehl_ps(diff, diff));
#elif defined(HAVE_NEON)
            float32x2_t a = vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<uint64_t*>(&x))));
            float32x2_t b =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&other.x))));
            float32x2_t diff = vsub_f32(a, b);
            vst1_f32(&x, diff);
#endif
        } else {
            x -= other.x;
            y -= other.y;
        }
        return *this;
    }

    template <typename U = T>
    [[nodiscard]] constexpr Vec2<std::enable_if_t<std::is_signed_v<U>, U>> operator-() const {
        if constexpr (detail::is_vectorizable<T>::value) {
            Vec2<T> result;
#if defined(HAVE_SSE2)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 neg = _mm_sub_ps(_mm_setzero_ps(), a);
            _mm_store_ss(&result.x, neg);
            _mm_store_ss(&result.y, _mm_movehl_ps(neg, neg));
#elif defined(HAVE_NEON)
            float32x2_t a =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&x))));
            float32x2_t neg = vneg_f32(a);
            vst1_f32(&result.x, neg);
#endif
            return result;
        } else {
            return {-x, -y};
        }
    }

    [[nodiscard]] constexpr Vec2<decltype(T{} * T{})> operator*(const Vec2& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
            Vec2<T> result;
#if defined(HAVE_SSE2)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 b = _mm_setr_ps(other.x, other.y, 0.0f, 0.0f);
            __m128 prod = _mm_mul_ps(a, b);
            _mm_store_ss(&result.x, prod);
            _mm_store_ss(&result.y, _mm_movehl_ps(prod, prod));
#elif defined(HAVE_NEON)
            float32x2_t a =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&x))));
            float32x2_t b =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&other.x))));
            float32x2_t prod = vmul_f32(a, b);
            vst1_f32(&result.x, prod);
#endif
            return result;
        } else {
            return {x * other.x, y * other.y};
        }
    }

    template <typename V>
    [[nodiscard]] constexpr Vec2<decltype(T{} * V{})> operator*(const V& f) const {
        if constexpr (detail::is_vectorizable<T>::value && std::is_same_v<V, float>) {
            Vec2<decltype(T{} * V{})> result;
#if defined(HAVE_SSE2)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 scalar = _mm_set1_ps(f);
            __m128 prod = _mm_mul_ps(a, scalar);
            _mm_store_ss(&result.x, prod);
            _mm_store_ss(&result.y, _mm_movehl_ps(prod, prod));
#elif defined(HAVE_NEON)
            float32x2_t a =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&x))));
            float32x2_t scalar = vdup_n_f32(f);
            float32x2_t prod = vmul_f32(a, scalar);
            vst1_f32(&result.x, prod);
#endif
            return result;
        } else {
            return {x * f, y * f};
        }
    }

    template <typename V>
    constexpr Vec2& operator*=(const V& f) {
        if constexpr (detail::is_vectorizable<T>::value && std::is_same_v<V, float>) {
#if defined(HAVE_SSE2)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 scalar = _mm_set1_ps(f);
            __m128 prod = _mm_mul_ps(a, scalar);
            _mm_store_ss(&x, prod);
            _mm_store_ss(&y, _mm_movehl_ps(prod, prod));
#elif defined(HAVE_NEON)
            float32x2_t a = vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<uint64_t*>(&x))));
            float32x2_t scalar = vdup_n_f32(f);
            float32x2_t prod = vmul_f32(a, scalar);
            vst1_f32(&x, prod);
#endif
        } else {
            x *= f;
            y *= f;
        }
        return *this;
    }

    template <typename V>
    [[nodiscard]] constexpr Vec2<decltype(T{} / V{})> operator/(const V& f) const {
        if constexpr (detail::is_vectorizable<T>::value && std::is_same_v<V, float>) {
            Vec2<decltype(T{} / V{})> result;
#if defined(HAVE_SSE2)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 scalar = _mm_set1_ps(f);
            __m128 div = _mm_div_ps(a, scalar);
            _mm_store_ss(&result.x, div);
            _mm_store_ss(&result.y, _mm_movehl_ps(div, div));
#elif defined(HAVE_NEON)
            float32x2_t a =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&x))));
            float32x2_t scalar = vdup_n_f32(f);
            float32x2_t div = detail::vdiv_f32(a, scalar);
            vst1_f32(&result.x, div);
#endif
            return result;
        } else {
            return {x / f, y / f};
        }
    }

    template <typename V>
    constexpr Vec2& operator/=(const V& f) {
        if constexpr (detail::is_vectorizable<T>::value && std::is_same_v<V, float>) {
#if defined(HAVE_SSE2)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 scalar = _mm_set1_ps(f);
            __m128 div = _mm_div_ps(a, scalar);
            _mm_store_ss(&x, div);
            _mm_store_ss(&y, _mm_movehl_ps(div, div));
#elif defined(HAVE_NEON)
            float32x2_t a = vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<uint64_t*>(&x))));
            float32x2_t scalar = vdup_n_f32(f);
            float32x2_t div = detail::vdiv_f32(a, scalar);
            vst1_f32(&x, div);
#endif
        } else {
            x /= f;
            y /= f;
        }
        return *this;
    }

    [[nodiscard]] constexpr bool operator==(const Vec2& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 b = _mm_setr_ps(other.x, other.y, 0.0f, 0.0f);
            __m128 cmp = _mm_cmpeq_ps(a, b);
            return (_mm_movemask_ps(cmp) & 0x3) == 0x3;
#elif defined(HAVE_NEON)
            float32x2_t a =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&x))));
            float32x2_t b =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&other.x))));
            uint32x2_t cmp = vceq_f32(a, b);
            return (vget_lane_u32(cmp, 0) & vget_lane_u32(cmp, 1)) == 0xFFFFFFFF;
#endif
        } else {
            return x == other.x && y == other.y;
        }
    }

    [[nodiscard]] constexpr bool operator!=(const Vec2& other) const {
        return !(*this == other);
    }

    [[nodiscard]] T Length2() const {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE4_1)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            return _mm_cvtss_f32(_mm_dp_ps(a, a, 0x31));
#elif defined(HAVE_SSE2)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 sq = _mm_mul_ps(a, a);
            __m128 sum = _mm_add_ss(sq, _mm_movehl_ps(sq, sq));
            return _mm_cvtss_f32(sum);
#elif defined(HAVE_NEON)
            float32x2_t a =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&x))));
            float32x2_t sq = vmul_f32(a, a);
            return vget_lane_f32(vpadd_f32(sq, sq), 0);
#endif
        } else {
            return x * x + y * y;
        }
    }

    [[nodiscard]] float Length() const {
        const float length2 = Length2();
        if (length2 < std::numeric_limits<float>::epsilon()) {
            return 0.0f;
        }
#if defined(HAVE_SSE2)
        __m128 sq = _mm_set_ss(length2);
        return _mm_cvtss_f32(_mm_sqrt_ss(sq));
#elif defined(HAVE_NEON)
        float32x2_t sq = vdup_n_f32(length2);
        return vget_lane_f32(vsqrt_f32(sq), 0);
#else
        return std::sqrt(length2);
#endif
    }

    [[nodiscard]] Vec2 Normalized() const {
        const float length = Length();
        if (length < std::numeric_limits<float>::epsilon()) {
            return Vec2{0, 0};
        }
        return *this / length;
    }

    float Normalize() {
        const float length = Length();
        if (length < std::numeric_limits<float>::epsilon()) {
            x = 0;
            y = 0;
            return 0;
        }
        *this /= length;
        return length;
    }

    [[nodiscard]] constexpr T& operator[](std::size_t i) {
        return *((&x) + i);
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t i) const {
        return *((&x) + i);
    }

    void SetZero() {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            __m128 zero = _mm_setzero_ps();
            _mm_store_ss(&x, zero);
            _mm_store_ss(&y, zero);
#elif defined(HAVE_NEON)
            float32x2_t zero = vdup_n_f32(0.0f);
            vst1_f32(&x, zero);
#endif
        } else {
            x = 0;
            y = 0;
        }
    }

    // Swizzle operations
    [[nodiscard]] constexpr Vec2 yx() const {
        return Vec2(y, x);
    }
};

template <typename T, typename V>
[[nodiscard]] constexpr Vec2<T> operator*(const V& f, const Vec2<T>& vec) {
    return vec * f;
}

using Vec2f = Vec2<float>;
using Vec2i = Vec2<int>;
using Vec2u = Vec2<unsigned int>;

template <typename T>
class alignas(16) Vec3 {
    static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                  "Vector type must be floating point or integral");

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        ar & x;
        ar & y;
        ar & z;
    }

public:
    union {
        struct {
            T x, y, z;
            T pad; // For SIMD alignment
        };
#if defined(HAVE_SSE2)
        struct {
            volatile __m128 simd;
        };
#elif defined(HAVE_NEON)
        struct {
            volatile float32x4_t simd;
        };
#endif
    };

    static constexpr std::size_t alignment = 16;
    static_assert(alignof(Vec3) >= alignment, "Vec3 must be properly aligned for SIMD operations");

    constexpr Vec3() : x(0), y(0), z(0), pad(0) {}
    constexpr Vec3(const T& x_, const T& y_, const T& z_) : x(x_), y(y_), z(z_), pad(0) {}

    template <typename T2>
    [[nodiscard]] constexpr Vec3<T2> Cast() const {
        if constexpr (detail::is_vectorizable<T>::value && detail::is_vectorizable<T2>::value) {
            Vec3<T2> result;
#if defined(HAVE_SSE2)
            result.simd = simd;
#elif defined(HAVE_NEON)
            result.simd = simd;
#endif
            return result;
        } else {
            return Vec3<T2>(static_cast<T2>(x), static_cast<T2>(y), static_cast<T2>(z));
        }
    }

    [[nodiscard]] static constexpr Vec3 AssignToAll(const T& f) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            Vec3 result;
            result.simd = _mm_set1_ps(f);
            return result;
#elif defined(HAVE_NEON)
            Vec3 result;
            result.simd = vdupq_n_f32(f);
            return result;
#endif
        }
        return Vec3(f, f, f);
    }

    [[nodiscard]] constexpr Vec3<decltype(T{} + T{})> operator+(const Vec3& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
            Vec3<T> result;
#if defined(HAVE_SSE2)
            result.simd = _mm_add_ps(simd, other.simd);
#elif defined(HAVE_NEON)
            result.simd = vaddq_f32(simd, other.simd);
#endif
            return result;
        } else {
            return {x + other.x, y + other.y, z + other.z};
        }
    }

    constexpr Vec3& operator+=(const Vec3& other) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            simd = _mm_add_ps(simd, other.simd);
#elif defined(HAVE_NEON)
            simd = vaddq_f32(simd, other.simd);
#endif
        } else {
            x += other.x;
            y += other.y;
            z += other.z;
        }
        return *this;
    }

    [[nodiscard]] constexpr Vec3<decltype(T{} - T{})> operator-(const Vec3& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
            Vec3<T> result;
#if defined(HAVE_SSE2)
            result.simd = _mm_sub_ps(simd, other.simd);
#elif defined(HAVE_NEON)
            result.simd = vsubq_f32(simd, other.simd);
#endif
            return result;
        } else {
            return {x - other.x, y - other.y, z - other.z};
        }
    }

    constexpr Vec3& operator-=(const Vec3& other) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            simd = _mm_sub_ps(simd, other.simd);
#elif defined(HAVE_NEON)
            simd = vsubq_f32(simd, other.simd);
#endif
        } else {
            x -= other.x;
            y -= other.y;
            z -= other.z;
        }
        return *this;
    }

    template <typename U = T>
    [[nodiscard]] constexpr Vec3<std::enable_if_t<std::is_signed_v<U>, U>> operator-() const {
        if constexpr (detail::is_vectorizable<T>::value) {
            Vec3<T> result;
#if defined(HAVE_SSE2)
            result.simd = _mm_sub_ps(_mm_setzero_ps(), simd);
#elif defined(HAVE_NEON)
            result.simd = vnegq_f32(simd);
#endif
            return result;
        } else {
            return {-x, -y, -z};
        }
    }

    [[nodiscard]] constexpr Vec3<decltype(T{} * T{})> operator*(const Vec3& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
            Vec3<T> result;
#if defined(HAVE_SSE2)
            result.simd = _mm_mul_ps(simd, other.simd);
#elif defined(HAVE_NEON)
            result.simd = vmulq_f32(simd, other.simd);
#endif
            return result;
        } else {
            return {x * other.x, y * other.y, z * other.z};
        }
    }

    template <typename V>
    [[nodiscard]] constexpr Vec3<decltype(T{} * V{})> operator*(const V& f) const {
        if constexpr (detail::is_vectorizable<T>::value && std::is_same_v<V, float>) {
            Vec3<decltype(T{} * V{})> result;
#if defined(HAVE_SSE2)
            result.simd = _mm_mul_ps(simd, _mm_set1_ps(f));
#elif defined(HAVE_NEON)
            result.simd = vmulq_f32(simd, vdupq_n_f32(f));
#endif
            return result;
        } else {
            return {x * f, y * f, z * f};
        }
    }

    template <typename V>
    constexpr Vec3& operator*=(const V& f) {
        if constexpr (detail::is_vectorizable<T>::value && std::is_same_v<V, float>) {
#if defined(HAVE_SSE2)
            simd = _mm_mul_ps(simd, _mm_set1_ps(f));
#elif defined(HAVE_NEON)
            simd = vmulq_f32(simd, vdupq_n_f32(f));
#endif
        } else {
            x *= f;
            y *= f;
            z *= f;
        }
        return *this;
    }

    template <typename V>
    [[nodiscard]] constexpr Vec3<decltype(T{} / V{})> operator/(const V& f) const {
        if constexpr (detail::is_vectorizable<T>::value && std::is_same_v<V, float>) {
            Vec3<decltype(T{} / V{})> result;
            if (f < std::numeric_limits<float>::epsilon()) {
                result.SetZero();
                return result;
            }
#if defined(HAVE_SSE2)
            result.simd = _mm_div_ps(simd, _mm_set1_ps(f));
#elif defined(HAVE_NEON)
            // Use helper function for improved precision division
            result.simd = detail::vdivq_f32(simd, vdupq_n_f32(f));
#endif
            return result;
        } else {
            return {x / f, y / f, z / f};
        }
    }

    template <typename V>
    constexpr Vec3& operator/=(const V& f) {
        if constexpr (detail::is_vectorizable<T>::value && std::is_same_v<V, float>) {
            if (f < std::numeric_limits<float>::epsilon()) {
                SetZero();
                return *this;
            }
#if defined(HAVE_SSE2)
            simd = _mm_div_ps(simd, _mm_set1_ps(f));
#elif defined(HAVE_NEON)
            // Use helper function for improved precision division
            simd = detail::vdivq_f32(simd, vdupq_n_f32(f));
#endif
        } else {
            x /= f;
            y /= f;
            z /= f;
        }
        return *this;
    }

    [[nodiscard]] constexpr bool operator==(const Vec3& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            __m128 cmp = _mm_cmpeq_ps(simd, other.simd);
            return (_mm_movemask_ps(cmp) & 0x7) == 0x7;
#elif defined(HAVE_NEON)
            uint32x4_t cmp = vceqq_f32(simd, other.simd);
            uint32x2_t hi_lo = vand_u32(vget_high_u32(cmp), vget_low_u32(cmp));
            uint32x2_t folded = vand_u32(hi_lo, vrev64_u32(hi_lo));
            return (vget_lane_u32(folded, 0) & 0xFFFFFFFF) == 0xFFFFFFFF;
#endif
        } else {
            return x == other.x && y == other.y && z == other.z;
        }
    }

    [[nodiscard]] constexpr bool operator!=(const Vec3& other) const {
        return !(*this == other);
    }

    [[nodiscard]] T Length2() const {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE4_1)
            return _mm_cvtss_f32(_mm_dp_ps(simd, simd, 0x71));
#elif defined(HAVE_SSE2)
            __m128 sq = _mm_mul_ps(simd, simd);
            __m128 sum = _mm_add_ps(_mm_add_ps(sq, _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(2, 3, 0, 1))),
                                    _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtss_f32(sum);
#elif defined(HAVE_NEON)
            float32x4_t sq = vmulq_f32(simd, simd);
            float32x2_t sum = vadd_f32(vget_low_f32(sq), vget_high_f32(sq));
            return vget_lane_f32(vpadd_f32(sum, sum), 0);
#endif
        } else {
            return x * x + y * y + z * z;
        }
    }

    [[nodiscard]] float Length() const {
        const float length2 = Length2();
        if (length2 < std::numeric_limits<float>::epsilon()) {
            return 0.0f;
        }
#if defined(HAVE_SSE2)
        return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(length2)));
#elif defined(HAVE_NEON)
        float32x2_t len = vsqrt_f32(vdup_n_f32(length2));
        return vget_lane_f32(len, 0);
#else
        return std::sqrt(length2);
#endif
    }

    [[nodiscard]] Vec3 Normalized() const {
        const float length = Length();
        if (length < std::numeric_limits<float>::epsilon()) {
            return Vec3{0, 0, 0};
        }
        return *this / length;
    }

    float Normalize() {
        const float length = Length();
        if (length < std::numeric_limits<float>::epsilon()) {
            SetZero();
            return 0.0f;
        }
        *this /= length;
        return length;
    }

    [[nodiscard]] constexpr T& operator[](std::size_t i) {
        return *((&x) + i);
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t i) const {
        return *((&x) + i);
    }

    void SetZero() {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            simd = _mm_setzero_ps();
#elif defined(HAVE_NEON)
            simd = vdupq_n_f32(0.0f);
#endif
        } else {
            x = 0;
            y = 0;
            z = 0;
        }
        pad = 0;
    }

    [[nodiscard]] static Vec3 Cross(const Vec3& a, const Vec3& b) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            // Implementation using SSE shuffles:
            // x = a.y * b.z - a.z * b.y
            // y = a.z * b.x - a.x * b.z
            // z = a.x * b.y - a.y * b.x
            __m128 a1 = _mm_shuffle_ps(a.simd, a.simd, _MM_SHUFFLE(3, 0, 2, 1));
            __m128 b1 = _mm_shuffle_ps(b.simd, b.simd, _MM_SHUFFLE(3, 1, 0, 2));
            __m128 a2 = _mm_shuffle_ps(a.simd, a.simd, _MM_SHUFFLE(3, 1, 0, 2));
            __m128 b2 = _mm_shuffle_ps(b.simd, b.simd, _MM_SHUFFLE(3, 0, 2, 1));
            Vec3 result;
            result.simd = _mm_sub_ps(_mm_mul_ps(a1, b1), _mm_mul_ps(a2, b2));
            return result;
#elif defined(HAVE_NEON)
            // Implementation using NEON:
            float32x4x2_t ab = vtrnq_f32(a.simd, b.simd);
            float32x4_t a1 = vextq_f32(a.simd, a.simd, 1);
            float32x4_t b1 = vextq_f32(b.simd, b.simd, 1);
            float32x4_t a2 = vextq_f32(a.simd, a.simd, 2);
            float32x4_t b2 = vextq_f32(b.simd, b.simd, 2);
            Vec3 result;
            result.simd = vsubq_f32(vmulq_f32(a1, b2), vmulq_f32(a2, b1));
            return result;
#endif
        }
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }

    [[nodiscard]] static T Dot(const Vec3& a, const Vec3& b) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE4_1)
            return _mm_cvtss_f32(_mm_dp_ps(a.simd, b.simd, 0x71));
#elif defined(HAVE_SSE2)
            __m128 prod = _mm_mul_ps(a.simd, b.simd);
            __m128 sum =
                _mm_add_ps(_mm_add_ps(prod, _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(2, 3, 0, 1))),
                           _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtss_f32(sum);
#elif defined(HAVE_NEON)
            float32x4_t prod = vmulq_f32(a.simd, b.simd);
            float32x2_t sum = vadd_f32(vget_low_f32(prod), vget_high_f32(prod));
            return vget_lane_f32(vpadd_f32(sum, sum), 0);
#endif
        }
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    // Swizzle operations
    [[nodiscard]] constexpr Vec3 yzx() const {
        return Vec3(y, z, x);
    }
    [[nodiscard]] constexpr Vec3 zxy() const {
        return Vec3(z, x, y);
    }
    [[nodiscard]] constexpr Vec2<T> xy() const {
        return Vec2<T>(x, y);
    }
    [[nodiscard]] constexpr Vec2<T> xz() const {
        return Vec2<T>(x, z);
    }
    [[nodiscard]] constexpr Vec2<T> yz() const {
        return Vec2<T>(y, z);
    }
    [[nodiscard]] constexpr Vec2<T> yx() const {
        return Vec2<T>(y, x);
    }
    [[nodiscard]] constexpr Vec2<T> zy() const {
        return Vec2<T>(z, y);
    }
    [[nodiscard]] constexpr Vec2<T> zx() const {
        return Vec2<T>(z, x);
    }
};

template <typename T, typename V>
[[nodiscard]] constexpr Vec3<T> operator*(const V& f, const Vec3<T>& vec) {
    return vec * f;
}

using Vec3f = Vec3<float>;
using Vec3i = Vec3<int>;
using Vec3u = Vec3<unsigned int>;

template <typename T>
class alignas(16) Vec4 {
    static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                  "Vector type must be floating point or integral");

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        ar & x;
        ar & y;
        ar & z;
        ar & w;
    }

public:
    union {
        struct {
            T x, y, z, w;
        };
#if defined(HAVE_SSE2)
        struct {
            volatile __m128 simd;
        };
#elif defined(HAVE_NEON)
        struct {
            volatile float32x4_t simd;
        };
#endif
    };

    static constexpr std::size_t alignment = 16;
    static_assert(alignof(Vec4) >= alignment, "Vec4 must be properly aligned for SIMD operations");

    constexpr Vec4() : x(0), y(0), z(0), w(0) {}
    constexpr Vec4(const T& x_, const T& y_, const T& z_, const T& w_)
        : x(x_), y(y_), z(z_), w(w_) {}

    // Construct from Vec3 + w component
    constexpr Vec4(const Vec3<T>& xyz, const T& w_) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            simd = _mm_set_ps(w_, xyz.z, xyz.y, xyz.x);
#elif defined(HAVE_NEON)
            float32x2_t lo =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&xyz.x))));
            float32x2_t hi =
                vcreate_f32(static_cast<uint64_t>(*(reinterpret_cast<const uint64_t*>(&xyz.z))));
            simd = vcombine_f32(lo, vset_lane_f32(w_, hi, 1));
#endif
        } else {
            x = xyz.x;
            y = xyz.y;
            z = xyz.z;
            w = w_;
        }
    }

    template <typename T2>
    [[nodiscard]] constexpr Vec4<T2> Cast() const {
        if constexpr (detail::is_vectorizable<T>::value && detail::is_vectorizable<T2>::value) {
            Vec4<T2> result;
#if defined(HAVE_SSE2)
            result.simd = simd;
#elif defined(HAVE_NEON)
            result.simd = simd;
#endif
            return result;
        } else {
            return Vec4<T2>(static_cast<T2>(x), static_cast<T2>(y), static_cast<T2>(z),
                            static_cast<T2>(w));
        }
    }

    [[nodiscard]] static constexpr Vec4 AssignToAll(const T& f) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            Vec4 result;
            result.simd = _mm_set1_ps(f);
            return result;
#elif defined(HAVE_NEON)
            Vec4 result;
            result.simd = vdupq_n_f32(f);
            return result;
#endif
        }
        return Vec4(f, f, f, f);
    }

    [[nodiscard]] constexpr Vec4<decltype(T{} + T{})> operator+(const Vec4& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
            Vec4<T> result;
#if defined(HAVE_SSE2)
            result.simd = _mm_add_ps(simd, other.simd);
#elif defined(HAVE_NEON)
            result.simd = vaddq_f32(simd, other.simd);
#endif
            return result;
        } else {
            return {x + other.x, y + other.y, z + other.z, w + other.w};
        }
    }

    constexpr Vec4& operator+=(const Vec4& other) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            simd = _mm_add_ps(simd, other.simd);
#elif defined(HAVE_NEON)
            simd = vaddq_f32(simd, other.simd);
#endif
        } else {
            x += other.x;
            y += other.y;
            z += other.z;
            w += other.w;
        }
        return *this;
    }

    [[nodiscard]] constexpr Vec4<decltype(T{} - T{})> operator-(const Vec4& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
            Vec4<T> result;
#if defined(HAVE_SSE2)
            result.simd = _mm_sub_ps(simd, other.simd);
#elif defined(HAVE_NEON)
            result.simd = vsubq_f32(simd, other.simd);
#endif
            return result;
        } else {
            return {x - other.x, y - other.y, z - other.z, w - other.w};
        }
    }

    constexpr Vec4& operator-=(const Vec4& other) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            simd = _mm_sub_ps(simd, other.simd);
#elif defined(HAVE_NEON)
            simd = vsubq_f32(simd, other.simd);
#endif
        } else {
            x -= other.x;
            y -= other.y;
            z -= other.z;
            w -= other.w;
        }
        return *this;
    }

    template <typename U = T>
    [[nodiscard]] constexpr Vec4<std::enable_if_t<std::is_signed_v<U>, U>> operator-() const {
        if constexpr (detail::is_vectorizable<T>::value) {
            Vec4<T> result;
#if defined(HAVE_SSE2)
            result.simd = _mm_sub_ps(_mm_setzero_ps(), simd);
#elif defined(HAVE_NEON)
            result.simd = vnegq_f32(simd);
#endif
            return result;
        } else {
            return {-x, -y, -z, -w};
        }
    }

    [[nodiscard]] constexpr Vec4<decltype(T{} * T{})> operator*(const Vec4& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
            Vec4<T> result;
#if defined(HAVE_SSE2)
            result.simd = _mm_mul_ps(simd, other.simd);
#elif defined(HAVE_NEON)
            result.simd = vmulq_f32(simd, other.simd);
#endif
            return result;
        } else {
            return {x * other.x, y * other.y, z * other.z, w * other.w};
        }
    }

    template <typename V>
    [[nodiscard]] constexpr Vec4<decltype(T{} * V{})> operator*(const V& f) const {
        if constexpr (detail::is_vectorizable<T>::value && std::is_same_v<V, float>) {
            Vec4<decltype(T{} * V{})> result;
#if defined(HAVE_SSE2)
            result.simd = _mm_mul_ps(simd, _mm_set1_ps(f));
#elif defined(HAVE_NEON)
            result.simd = vmulq_f32(simd, vdupq_n_f32(f));
#endif
            return result;
        } else {
            return {x * f, y * f, z * f, w * f};
        }
    }

    template <typename V>
    constexpr Vec4& operator*=(const V& f) {
        if constexpr (detail::is_vectorizable<T>::value && std::is_same_v<V, float>) {
#if defined(HAVE_SSE2)
            simd = _mm_mul_ps(simd, _mm_set1_ps(f));
#elif defined(HAVE_NEON)
            simd = vmulq_f32(simd, vdupq_n_f32(f));
#endif
        } else {
            x *= f;
            y *= f;
            z *= f;
            w *= f;
        }
        return *this;
    }

    template <typename V>
    [[nodiscard]] constexpr Vec4<decltype(T{} / V{})> operator/(const V& f) const {
        if constexpr (detail::is_vectorizable<T>::value && std::is_same_v<V, float>) {
            Vec4<decltype(T{} / V{})> result;
            if (f < std::numeric_limits<float>::epsilon()) {
                result.SetZero();
                return result;
            }
#if defined(HAVE_SSE2)
            result.simd = _mm_div_ps(simd, _mm_set1_ps(f));
#elif defined(HAVE_NEON)
            result.simd = detail::vdivq_f32(simd, vdupq_n_f32(f));
#endif
            return result;
        } else {
            return {x / f, y / f, z / f, w / f};
        }
    }

    template <typename V>
    constexpr Vec4& operator/=(const V& f) {
        if constexpr (detail::is_vectorizable<T>::value && std::is_same_v<V, float>) {
            if (f < std::numeric_limits<float>::epsilon()) {
                SetZero();
                return *this;
            }
#if defined(HAVE_SSE2)
            simd = _mm_div_ps(simd, _mm_set1_ps(f));
#elif defined(HAVE_NEON)
            simd = detail::vdivq_f32(simd, vdupq_n_f32(f));
#endif
        } else {
            x /= f;
            y /= f;
            z /= f;
            w /= f;
        }
        return *this;
    }

    [[nodiscard]] constexpr bool operator==(const Vec4& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            __m128 cmp = _mm_cmpeq_ps(simd, other.simd);
            return (_mm_movemask_ps(cmp) & 0xF) == 0xF;
#elif defined(HAVE_NEON)
            uint32x4_t cmp = vceqq_f32(simd, other.simd);
            uint32x2_t hi_lo = vand_u32(vget_high_u32(cmp), vget_low_u32(cmp));
            uint32x2_t folded = vand_u32(hi_lo, vrev64_u32(hi_lo));
            return (vget_lane_u32(folded, 0) & 0xFFFFFFFF) == 0xFFFFFFFF;
#endif
        } else {
            return x == other.x && y == other.y && z == other.z && w == other.w;
        }
    }

    [[nodiscard]] constexpr bool operator!=(const Vec4& other) const {
        return !(*this == other);
    }

    [[nodiscard]] T Length2() const {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE4_1)
            return _mm_cvtss_f32(_mm_dp_ps(simd, simd, 0xF1));
#elif defined(HAVE_SSE2)
            __m128 sq = _mm_mul_ps(simd, simd);
            __m128 sum = _mm_add_ps(_mm_add_ps(sq, _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(2, 3, 0, 1))),
                                    _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtss_f32(sum);
#elif defined(HAVE_NEON)
            float32x4_t sq = vmulq_f32(simd, simd);
            float32x2_t sum = vadd_f32(vget_low_f32(sq), vget_high_f32(sq));
            return vget_lane_f32(vpadd_f32(sum, sum), 0);
#endif
        } else {
            return x * x + y * y + z * z + w * w;
        }
    }

    [[nodiscard]] float Length() const {
        const float length2 = Length2();
        if (length2 < std::numeric_limits<float>::epsilon()) {
            return 0.0f;
        }
#if defined(HAVE_SSE2)
        return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(length2)));
#elif defined(HAVE_NEON)
        float32x2_t len = vsqrt_f32(vdup_n_f32(length2));
        return vget_lane_f32(len, 0);
#else
        return std::sqrt(length2);
#endif
    }

    [[nodiscard]] Vec4 Normalized() const {
        const float length = Length();
        if (length < std::numeric_limits<float>::epsilon()) {
            return Vec4{0, 0, 0, 0};
        }
        return *this / length;
    }

    float Normalize() {
        const float length = Length();
        if (length < std::numeric_limits<float>::epsilon()) {
            SetZero();
            return 0.0f;
        }
        *this /= length;
        return length;
    }

    [[nodiscard]] constexpr T& operator[](std::size_t i) {
        return *((&x) + i);
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t i) const {
        return *((&x) + i);
    }

    void SetZero() {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            simd = _mm_setzero_ps();
#elif defined(HAVE_NEON)
            simd = vdupq_n_f32(0.0f);
#endif
        } else {
            x = 0;
            y = 0;
            z = 0;
            w = 0;
        }
    }

    [[nodiscard]] static T Dot(const Vec4& a, const Vec4& b) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE4_1)
            return _mm_cvtss_f32(_mm_dp_ps(a.simd, b.simd, 0xF1));
#elif defined(HAVE_SSE2)
            __m128 prod = _mm_mul_ps(a.simd, b.simd);
            __m128 sum =
                _mm_add_ps(_mm_add_ps(prod, _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(2, 3, 0, 1))),
                           _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtss_f32(sum);
#elif defined(HAVE_NEON)
            float32x4_t prod = vmulq_f32(a.simd, b.simd);
            float32x2_t sum = vadd_f32(vget_low_f32(prod), vget_high_f32(prod));
            return vget_lane_f32(vpadd_f32(sum, sum), 0);
#endif
        }
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    // Swizzle operations
    [[nodiscard]] constexpr Vec3<T> xyz() const {
        return Vec3<T>(x, y, z);
    }
    [[nodiscard]] constexpr Vec3<T> xyw() const {
        return Vec3<T>(x, y, w);
    }
    [[nodiscard]] constexpr Vec3<T> xzw() const {
        return Vec3<T>(x, z, w);
    }
    [[nodiscard]] constexpr Vec3<T> yzw() const {
        return Vec3<T>(y, z, w);
    }
    [[nodiscard]] constexpr Vec2<T> xy() const {
        return Vec2<T>(x, y);
    }
    [[nodiscard]] constexpr Vec2<T> xz() const {
        return Vec2<T>(x, z);
    }
    [[nodiscard]] constexpr Vec2<T> xw() const {
        return Vec2<T>(x, w);
    }
    [[nodiscard]] constexpr Vec2<T> yz() const {
        return Vec2<T>(y, z);
    }
    [[nodiscard]] constexpr Vec2<T> yw() const {
        return Vec2<T>(y, w);
    }
    [[nodiscard]] constexpr Vec2<T> zw() const {
        return Vec2<T>(z, w);
    }
};

template <typename T, typename V>
[[nodiscard]] constexpr Vec4<T> operator*(const V& f, const Vec4<T>& vec) {
    return vec * f;
}

using Vec4f = Vec4<float>;
using Vec4i = Vec4<int>;
using Vec4u = Vec4<unsigned int>;

} // namespace Common

#endif
