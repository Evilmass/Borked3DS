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
struct is_vectorizable : std::bool_constant<std::is_same_v<T, float> && has_simd_support> {};
} // namespace detail

// Set up ARM NEON FP control if available
#if defined(HAVE_NEON) && defined(__aarch64__)
namespace {
inline void ConfigureNEONFP() {
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1 << 24); // Set flush-to-zero
    __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
}
} // namespace
#endif

template <typename T>
class Vec2 {
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        ar & x;
        ar & y;
    }

public:
    T x;
    T y;

    constexpr Vec2() = default;

    constexpr Vec2(const T& x_, const T& y_) : x(x_), y(y_) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_NEON)
            float values[2] = {x_, y_};
            float32x2_t temp = vld1_f32(values);
            x = vget_lane_f32(temp, 0);
            y = vget_lane_f32(temp, 1);
#endif
        }
    }

    [[nodiscard]] T* AsArray() {
        return &x;
    }

    [[nodiscard]] const T* AsArray() const {
        return &x;
    }

    template <typename T2>
    [[nodiscard]] constexpr Vec2<T2> Cast() const {
        return Vec2<T2>(static_cast<T2>(x), static_cast<T2>(y));
    }

    [[nodiscard]] static constexpr Vec2 AssignToAll(const T& f) {
        return Vec2{f, f};
    }

    [[nodiscard]] constexpr Vec2<decltype(T{} + T{})> operator+(const Vec2& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
            Vec2<T> result;
#if defined(HAVE_SSE2)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 b = _mm_setr_ps(other.x, other.y, 0.0f, 0.0f);
            __m128 sum = _mm_add_ps(a, b);
            result.x = _mm_cvtss_f32(sum);
            result.y = _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 1, 1, 1)));
#elif defined(HAVE_NEON)
            float32x2_t a = vld1_f32(&x);
            float32x2_t b = vld1_f32(&other.x);
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
            x = _mm_cvtss_f32(sum);
            y = _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 1, 1, 1)));
#elif defined(HAVE_NEON)
            float32x2_t a = vld1_f32(&x);
            float32x2_t b = vld1_f32(&other.x);
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
            result.x = _mm_cvtss_f32(diff);
            result.y = _mm_cvtss_f32(_mm_shuffle_ps(diff, diff, _MM_SHUFFLE(1, 1, 1, 1)));
#elif defined(HAVE_NEON)
            float32x2_t a = vld1_f32(&x);
            float32x2_t b = vld1_f32(&other.x);
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
            x = _mm_cvtss_f32(diff);
            y = _mm_cvtss_f32(_mm_shuffle_ps(diff, diff, _MM_SHUFFLE(1, 1, 1, 1)));
#elif defined(HAVE_NEON)
            float32x2_t a = vld1_f32(&x);
            float32x2_t b = vld1_f32(&other.x);
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
            result.x = _mm_cvtss_f32(neg);
            result.y = _mm_cvtss_f32(_mm_shuffle_ps(neg, neg, _MM_SHUFFLE(1, 1, 1, 1)));
#elif defined(HAVE_NEON)
            float32x2_t a = vld1_f32(&x);
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
            result.x = _mm_cvtss_f32(prod);
            result.y = _mm_cvtss_f32(_mm_shuffle_ps(prod, prod, _MM_SHUFFLE(1, 1, 1, 1)));
#elif defined(HAVE_NEON)
            float32x2_t a = vld1_f32(&x);
            float32x2_t b = vld1_f32(&other.x);
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
            __m128 b = _mm_set1_ps(f);
            __m128 prod = _mm_mul_ps(a, b);
            result.x = _mm_cvtss_f32(prod);
            result.y = _mm_cvtss_f32(_mm_shuffle_ps(prod, prod, _MM_SHUFFLE(1, 1, 1, 1)));
#elif defined(HAVE_NEON)
            float32x2_t a = vld1_f32(&x);
            float32x2_t b = vdup_n_f32(f);
            float32x2_t prod = vmul_f32(a, b);
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
            __m128 b = _mm_set1_ps(f);
            __m128 prod = _mm_mul_ps(a, b);
            x = _mm_cvtss_f32(prod);
            y = _mm_cvtss_f32(_mm_shuffle_ps(prod, prod, _MM_SHUFFLE(1, 1, 1, 1)));
#elif defined(HAVE_NEON)
            float32x2_t a = vld1_f32(&x);
            float32x2_t b = vdup_n_f32(f);
            float32x2_t prod = vmul_f32(a, b);
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
            __m128 b = _mm_set1_ps(f);
            __m128 quot = _mm_div_ps(a, b);
            result.x = _mm_cvtss_f32(quot);
            result.y = _mm_cvtss_f32(_mm_shuffle_ps(quot, quot, _MM_SHUFFLE(1, 1, 1, 1)));
#elif defined(HAVE_NEON)
            float32x2_t a = vld1_f32(&x);
            // Improved precision with two Newton-Raphson iterations
            float32x2_t recip = vdup_n_f32(1.0f / f);
            recip = vmul_f32(recip, vrsqrts_f32(vdup_n_f32(f), vmul_f32(recip, recip)));
            recip = vmul_f32(recip, vrsqrts_f32(vdup_n_f32(f), vmul_f32(recip, recip)));
            float32x2_t quot = vmul_f32(a, recip);
            vst1_f32(&result.x, quot);
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
            __m128 b = _mm_set1_ps(f);
            __m128 quot = _mm_div_ps(a, b);
            x = _mm_cvtss_f32(quot);
            y = _mm_cvtss_f32(_mm_shuffle_ps(quot, quot, _MM_SHUFFLE(1, 1, 1, 1)));
#elif defined(HAVE_NEON)
            float32x2_t a = vld1_f32(&x);
            // Improved precision with two Newton-Raphson iterations
            float32x2_t recip = vdup_n_f32(1.0f / f);
            recip = vmul_f32(recip, vrsqrts_f32(vdup_n_f32(f), vmul_f32(recip, recip)));
            recip = vmul_f32(recip, vrsqrts_f32(vdup_n_f32(f), vmul_f32(recip, recip)));
            float32x2_t quot = vmul_f32(a, recip);
            vst1_f32(&x, quot);
#endif
        } else {
            x /= f;
            y /= f;
        }
        return *this;
    }

    [[nodiscard]] constexpr T Length2() const {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            __m128 v = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 sq = _mm_mul_ps(v, v);
            return _mm_cvtss_f32(_mm_add_ss(sq, _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(1, 1, 1, 1))));
#elif defined(HAVE_NEON)
            float32x2_t v = vld1_f32(&x);
            float32x2_t sq = vmul_f32(v, v);
            return vget_lane_f32(vpadd_f32(sq, sq), 0);
#endif
        } else {
            return x * x + y * y;
        }
    }

    [[nodiscard]] constexpr bool operator!=(const Vec2& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 b = _mm_setr_ps(other.x, other.y, 0.0f, 0.0f);
            __m128 cmp = _mm_cmpeq_ps(a, b);
            return (_mm_movemask_ps(cmp) & 0x3) != 0x3;
#elif defined(HAVE_NEON)
            float32x2_t a = vld1_f32(&x);
            float32x2_t b = vld1_f32(&other.x);
            uint32x2_t cmp = vceq_f32(a, b);
            return (vget_lane_u32(cmp, 0) & vget_lane_u32(cmp, 1)) != 0xFFFFFFFF;
#endif
        } else {
            return std::memcmp(AsArray(), other.AsArray(), sizeof(Vec2)) != 0;
        }
    }

    [[nodiscard]] constexpr bool operator==(const Vec2& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            __m128 a = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 b = _mm_setr_ps(other.x, other.y, 0.0f, 0.0f);
            __m128 cmp = _mm_cmpeq_ps(a, b);
            return (_mm_movemask_ps(cmp) & 0x3) == 0x3;
#elif defined(HAVE_NEON)
            float32x2_t a = vld1_f32(&x);
            float32x2_t b = vld1_f32(&other.x);
            uint32x2_t cmp = vceq_f32(a, b);
            return (vget_lane_u32(cmp, 0) & vget_lane_u32(cmp, 1)) == 0xFFFFFFFF;
#endif
        } else {
            return std::memcmp(AsArray(), other.AsArray(), sizeof(Vec2)) == 0;
        }
    }

    // Only implemented for T=float
    [[nodiscard]] float Length() const;
    float Normalize(); // returns the previous length, which is often useful

    [[nodiscard]] constexpr T& operator[](std::size_t i) {
        return *((&x) + i);
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t i) const {
        return *((&x) + i);
    }

    constexpr void SetZero() {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            _mm_store_ps(&x, _mm_setzero_ps());
#elif defined(HAVE_NEON)
            vst1_f32(&x, vdup_n_f32(0.0f));
#endif
        } else {
            x = 0;
            y = 0;
        }
    }

    // Common aliases: UV (texel coordinates), ST (texture coordinates)
    [[nodiscard]] constexpr T& u() {
        return x;
    }
    [[nodiscard]] constexpr T& v() {
        return y;
    }
    [[nodiscard]] constexpr T& s() {
        return x;
    }
    [[nodiscard]] constexpr T& t() {
        return y;
    }

    [[nodiscard]] constexpr const T& u() const {
        return x;
    }
    [[nodiscard]] constexpr const T& v() const {
        return y;
    }
    [[nodiscard]] constexpr const T& s() const {
        return x;
    }
    [[nodiscard]] constexpr const T& t() const {
        return y;
    }

    // swizzlers - create a subvector of specific components
    [[nodiscard]] constexpr Vec2 yx() const {
        return Vec2(y, x);
    }
    [[nodiscard]] constexpr Vec2 vu() const {
        return Vec2(y, x);
    }
    [[nodiscard]] constexpr Vec2 ts() const {
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

template <>
inline float Vec2<float>::Length() const {
#if defined(HAVE_SSE4_1)
    // Use SSE4.1's dedicated dot product instruction
    __m128 v = _mm_setr_ps(x, y, 0.0f, 0.0f);
    return _mm_cvtss_f32(_mm_sqrt_ss(
        _mm_dp_ps(v, v, 0x31))); // 0x31: only multiply xy (0x3) and store in lowest (0x1)
#elif defined(HAVE_SSE2)
    __m128 v = _mm_setr_ps(x, y, 0.0f, 0.0f);
    __m128 sq = _mm_mul_ps(v, v);
    __m128 sum = _mm_add_ss(sq, _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(_mm_sqrt_ss(sum));
#elif defined(HAVE_NEON)
    float32x2_t v = vld1_f32(&x);
    float32x2_t sq = vmul_f32(v, v);
    float32x2_t sum = vpadd_f32(sq, sq);
    return sqrtf(vget_lane_f32(sum, 0));
#else
    return std::sqrt(x * x + y * y);
#endif
}

template <>
inline float Vec2<float>::Normalize() {
    float length = Length();
    if constexpr (detail::is_vectorizable<float>::value) {
#if defined(HAVE_SSE2)
        if (length != 0) {
            __m128 vec = _mm_setr_ps(x, y, 0.0f, 0.0f);
            __m128 len = _mm_set1_ps(length);
            __m128 normalized = _mm_div_ps(vec, len);
            x = _mm_cvtss_f32(normalized);
            y = _mm_cvtss_f32(_mm_shuffle_ps(normalized, normalized, _MM_SHUFFLE(1, 1, 1, 1)));
        }
#elif defined(HAVE_NEON)
        if (length != 0) {
            float32x2_t vec = vld1_f32(&x);
            float32x2_t recip = vdup_n_f32(1.0f / length);
            float32x2_t normalized = vmul_f32(vec, recip);
            vst1_f32(&x, normalized);
        }
#endif
    } else {
        *this /= length;
    }
    return length;
}

template <typename T>
class alignas(16) Vec3 {
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        ar & x;
        ar & y;
        ar & z;
    }

public:
    T x;
    T y;
    T z;
    T pad; // For SIMD alignment

    constexpr Vec3() = default;

    constexpr Vec3(const T& x_, const T& y_, const T& z_) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            __m128 vec = _mm_setr_ps(x_, y_, z_, 0.0f);
            _mm_store_ps(&x, vec);
#elif defined(HAVE_NEON)
            float values[4] = {x_, y_, z_, 0.0f};
            vst1q_f32(&x, vld1q_f32(values));
#endif
        } else {
            x = x_;
            y = y_;
            z = z_;
        }
    }

    [[nodiscard]] T* AsArray() {
        return &x;
    }

    [[nodiscard]] const T* AsArray() const {
        return &x;
    }

    template <typename T2>
    [[nodiscard]] constexpr Vec3<T2> Cast() const {
        if constexpr (detail::is_vectorizable<T>::value && detail::is_vectorizable<T2>::value) {
            Vec3<T2> result;
#if defined(HAVE_SSE2)
            if constexpr (std::is_same_v<T, float> && std::is_same_v<T2, float>) {
                _mm_store_ps(&result.x, _mm_load_ps(&x));
            } else {
                result.x = static_cast<T2>(x);
                result.y = static_cast<T2>(y);
                result.z = static_cast<T2>(z);
            }
#elif defined(HAVE_NEON)
            if constexpr (std::is_same_v<T, float> && std::is_same_v<T2, float>) {
                vst1q_f32(&result.x, vld1q_f32(&x));
            } else {
                result.x = static_cast<T2>(x);
                result.y = static_cast<T2>(y);
                result.z = static_cast<T2>(z);
            }
#endif
            return result;
        } else {
            return Vec3<T2>(static_cast<T2>(x), static_cast<T2>(y), static_cast<T2>(z));
        }
    }

    [[nodiscard]] static constexpr Vec3 AssignToAll(const T& f) {
        if constexpr (detail::is_vectorizable<T>::value) {
            Vec3<T> result;
#if defined(HAVE_SSE2)
            _mm_store_ps(&result.x, _mm_set1_ps(f));
#elif defined(HAVE_NEON)
            vst1q_f32(&result.x, vdupq_n_f32(f));
#endif
            return result;
        } else {
            return Vec3(f, f, f);
        }
    }

    [[nodiscard]] constexpr Vec3<decltype(T{} + T{})> operator+(const Vec3& other) const {
        if constexpr (detail::is_vectorizable<T>::value) {
            Vec3<T> result;
#if defined(HAVE_SSE2)
            _mm_store_ps(&result.x, _mm_add_ps(_mm_load_ps(&x), _mm_load_ps(&other.x)));
#elif defined(HAVE_NEON)
            vst1q_f32(&result.x, vaddq_f32(vld1q_f32(&x), vld1q_f32(&other.x)));
#endif
            return result;
        } else {
            return {x + other.x, y + other.y, z + other.z};
        }
    }

    constexpr Vec3& operator+=(const Vec3& other) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            _mm_store_ps(&x, _mm_add_ps(_mm_load_ps(&x), _mm_load_ps(&other.x)));
#elif defined(HAVE_NEON)
            vst1q_f32(&x, vaddq_f32(vld1q_f32(&x), vld1q_f32(&other.x)));
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
            _mm_store_ps(&result.x, _mm_sub_ps(_mm_load_ps(&x), _mm_load_ps(&other.x)));
#elif defined(HAVE_NEON)
            vst1q_f32(&result.x, vsubq_f32(vld1q_f32(&x), vld1q_f32(&other.x)));
#endif
            return result;
        } else {
            return {x - other.x, y - other.y, z - other.z};
        }
    }

    constexpr Vec3& operator-=(const Vec3& other) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            _mm_store_ps(&x, _mm_sub_ps(_mm_load_ps(&x), _mm_load_ps(&other.x)));
#elif defined(HAVE_NEON)
            vst1q_f32(&x, vsubq_f32(vld1q_f32(&x), vld1q_f32(&other.x)));
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
            _mm_store_ps(&result.x, _mm_sub_ps(_mm_setzero_ps(), _mm_load_ps(&x)));
#elif defined(HAVE_NEON)
            vst1q_f32(&result.x, vnegq_f32(vld1q_f32(&x)));
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
            _mm_store_ps(&result.x, _mm_mul_ps(_mm_load_ps(&x), _mm_load_ps(&other.x)));
#elif defined(HAVE_NEON)
            vst1q_f32(&result.x, vmulq_f32(vld1q_f32(&x), vld1q_f32(&other.x)));
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
            _mm_store_ps(&result.x, _mm_mul_ps(_mm_load_ps(&x), _mm_set1_ps(f)));
#elif defined(HAVE_NEON)
            vst1q_f32(&result.x, vmulq_f32(vld1q_f32(&x), vdupq_n_f32(f)));
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
            _mm_store_ps(&x, _mm_mul_ps(_mm_load_ps(&x), _mm_set1_ps(f)));
#elif defined(HAVE_NEON)
            vst1q_f32(&x, vmulq_f32(vld1q_f32(&x), vdupq_n_f32(f)));
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
#if defined(HAVE_SSE2)
            _mm_store_ps(&result.x, _mm_div_ps(_mm_load_ps(&x), _mm_set1_ps(f)));
#elif defined(HAVE_NEON)
            // Improved precision with two Newton-Raphson iterations
            float32x4_t recip = vrecpeq_f32(vdupq_n_f32(f));
            recip = vmulq_f32(recip, vrecpsq_f32(vdupq_n_f32(f), recip));
            recip = vmulq_f32(recip, vrecpsq_f32(vdupq_n_f32(f), recip));
            vst1q_f32(&result.x, vmulq_f32(vld1q_f32(&x), recip));
#endif
            return result;
        } else {
            return {x / f, y / f, z / f};
        }
    }

    template <typename V>
    constexpr Vec3& operator/=(const V& f) {
        if constexpr (detail::is_vectorizable<T>::value && std::is_same_v<V, float>) {
#if defined(HAVE_SSE2)
            _mm_store_ps(&x, _mm_div_ps(_mm_load_ps(&x), _mm_set1_ps(f)));
#elif defined(HAVE_NEON)
            // Improved precision with two Newton-Raphson iterations
            float32x4_t recip = vrecpeq_f32(vdupq_n_f32(f));
            recip = vmulq_f32(recip, vrecpsq_f32(vdupq_n_f32(f), recip));
            recip = vmulq_f32(recip, vrecpsq_f32(vdupq_n_f32(f), recip));
            vst1q_f32(&x, vmulq_f32(vld1q_f32(&x), recip));
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
            __m128 cmp = _mm_cmpeq_ps(_mm_load_ps(&x), _mm_load_ps(&other.x));
            return (_mm_movemask_ps(cmp) & 0x7) == 0x7; // Check only x,y,z (mask 0x7)
#elif defined(HAVE_NEON)
            uint32x4_t cmp = vceqq_f32(vld1q_f32(&x), vld1q_f32(&other.x));
            uint32x2_t hi_lo = vand_u32(vget_high_u32(cmp), vget_low_u32(cmp));
            return (vget_lane_u32(hi_lo, 0) & vget_lane_u32(hi_lo, 1)) == 0xFFFFFFFF;
#endif
        } else {
            return x == other.x && y == other.y && z == other.z;
        }
    }

    [[nodiscard]] constexpr bool operator!=(const Vec3& other) const {
        return !(*this == other);
    }

    [[nodiscard]] constexpr T Length2() const {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            __m128 v = _mm_load_ps(&x);
            __m128 sq = _mm_mul_ps(v, v);
            __m128 sum = _mm_add_ss(_mm_add_ss(sq, _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(1, 1, 1, 1))),
                                    _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(2, 2, 2, 2)));
            return _mm_cvtss_f32(sum);
#elif defined(HAVE_NEON)
            float32x4_t v = vld1q_f32(&x);
            float32x4_t sq = vmulq_f32(v, v);
            float32x2_t sum = vpadd_f32(vget_low_f32(sq), vget_high_f32(sq));
            return vget_lane_f32(vpadd_f32(sum, vdup_n_f32(0.0f)), 0);
#endif
        } else {
            return x * x + y * y + z * z;
        }
    }

    [[nodiscard]] float Length() const;
    [[nodiscard]] Vec3 Normalized() const;
    float Normalize(); // returns the previous length, which is often useful

    [[nodiscard]] constexpr T& operator[](std::size_t i) {
        return *((&x) + i);
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t i) const {
        return *((&x) + i);
    }

    constexpr void SetZero() {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            _mm_store_ps(&x, _mm_setzero_ps());
#elif defined(HAVE_NEON)
            vst1q_f32(&x, vdupq_n_f32(0.0f));
#endif
        } else {
            x = 0;
            y = 0;
            z = 0;
        }
    }

    // Common aliases: UVW (texel coordinates), RGB (colors), STQ (texture coordinates)
    [[nodiscard]] constexpr T& u() {
        return x;
    }
    [[nodiscard]] constexpr T& v() {
        return y;
    }
    [[nodiscard]] constexpr T& w() {
        return z;
    }

    [[nodiscard]] constexpr T& r() {
        return x;
    }
    [[nodiscard]] constexpr T& g() {
        return y;
    }
    [[nodiscard]] constexpr T& b() {
        return z;
    }

    [[nodiscard]] constexpr T& s() {
        return x;
    }
    [[nodiscard]] constexpr T& t() {
        return y;
    }
    [[nodiscard]] constexpr T& q() {
        return z;
    }

    [[nodiscard]] constexpr const T& u() const {
        return x;
    }
    [[nodiscard]] constexpr const T& v() const {
        return y;
    }
    [[nodiscard]] constexpr const T& w() const {
        return z;
    }

    [[nodiscard]] constexpr const T& r() const {
        return x;
    }
    [[nodiscard]] constexpr const T& g() const {
        return y;
    }
    [[nodiscard]] constexpr const T& b() const {
        return z;
    }

    [[nodiscard]] constexpr const T& s() const {
        return x;
    }
    [[nodiscard]] constexpr const T& t() const {
        return y;
    }
    [[nodiscard]] constexpr const T& q() const {
        return z;
    }

// swizzlers - create a subvector of specific components
#define _DEFINE_SWIZZLER2(a, b, name)                                                              \
    [[nodiscard]] constexpr Vec2<T> name() const {                                                 \
        return Vec2<T>(a, b);                                                                      \
    }

#define DEFINE_SWIZZLER2(a, b, a2, b2, a3, b3, a4, b4)                                             \
    _DEFINE_SWIZZLER2(a, b, a##b);                                                                 \
    _DEFINE_SWIZZLER2(a, b, a2##b2);                                                               \
    _DEFINE_SWIZZLER2(a, b, a3##b3);                                                               \
    _DEFINE_SWIZZLER2(a, b, a4##b4);                                                               \
    _DEFINE_SWIZZLER2(b, a, b##a);                                                                 \
    _DEFINE_SWIZZLER2(b, a, b2##a2);                                                               \
    _DEFINE_SWIZZLER2(b, a, b3##a3);                                                               \
    _DEFINE_SWIZZLER2(b, a, b4##a4)

    DEFINE_SWIZZLER2(x, y, r, g, u, v, s, t);
    DEFINE_SWIZZLER2(x, z, r, b, u, w, s, q);
    DEFINE_SWIZZLER2(y, z, g, b, v, w, t, q);

#undef DEFINE_SWIZZLER2
#undef _DEFINE_SWIZZLER2
};

template <typename T, typename V>
[[nodiscard]] constexpr Vec3<T> operator*(const V& f, const Vec3<T>& vec) {
    return vec * f;
}

using Vec3f = Vec3<float>;
using Vec3i = Vec3<int>;
using Vec3u = Vec3<unsigned int>;

template <>
inline float Vec3<float>::Length() const {
#if defined(HAVE_SSE4_1)
    // Use SSE4.1's dedicated dot product instruction
    __m128 v = _mm_load_ps(&x);
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(v, v, 0x71)));
#elif defined(HAVE_SSE2)
    __m128 v = _mm_load_ps(&x);
    __m128 sq = _mm_mul_ps(v, v);
    __m128 sum = _mm_add_ss(_mm_add_ss(sq, _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(1, 1, 1, 1))),
                            _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(2, 2, 2, 2)));
    return _mm_cvtss_f32(_mm_sqrt_ss(sum));
#elif defined(HAVE_NEON)
    float32x4_t v = vld1q_f32(&x);
    float32x4_t sq = vmulq_f32(v, v);
    float32x2_t sum = vpadd_f32(vget_low_f32(sq), vget_high_f32(sq));
    float32x2_t total = vpadd_f32(sum, vdup_n_f32(0.0f));
    return sqrtf(vget_lane_f32(total, 0));
#else
    return std::sqrt(x * x + y * y + z * z);
#endif
}

template <>
inline Vec3<float> Vec3<float>::Normalized() const {
#if defined(HAVE_SSE2)
    __m128 v = _mm_load_ps(&x);
    __m128 sq = _mm_mul_ps(v, v);
    __m128 sum = _mm_add_ss(_mm_add_ss(sq, _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(1, 1, 1, 1))),
                            _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(2, 2, 2, 2)));
    __m128 length = _mm_sqrt_ss(sum);
    __m128 scale =
        _mm_div_ps(_mm_set1_ps(1.0f), _mm_shuffle_ps(length, length, _MM_SHUFFLE(0, 0, 0, 0)));
    Vec3<float> result;
    _mm_store_ps(&result.x, _mm_mul_ps(v, scale));
    return result;
#elif defined(HAVE_NEON)
    float32x4_t v = vld1q_f32(&x);
    float32x4_t sq = vmulq_f32(v, v);
    float32x2_t sum = vpadd_f32(vget_low_f32(sq), vget_high_f32(sq));
    float32x2_t len = vsqrt_f32(vpadd_f32(sum, vdup_n_f32(0.0f)));
    float32x4_t scale = vdupq_n_f32(1.0f / vget_lane_f32(len, 0));
    Vec3<float> result;
    vst1q_f32(&result.x, vmulq_f32(v, scale));
    return result;
#else
    return *this / Length();
#endif
}

template <>
inline float Vec3<float>::Normalize() {
    float length = Length();
    if constexpr (detail::is_vectorizable<float>::value) {
#if defined(HAVE_SSE2)
        if (length != 0) {
            __m128 vec = _mm_load_ps(&x);
            __m128 len = _mm_set1_ps(length);
            __m128 normalized = _mm_div_ps(vec, len);
            _mm_store_ps(&x, normalized);
        }
#elif defined(HAVE_NEON)
        if (length != 0) {
            float32x4_t vec = vld1q_f32(&x);
            float32x4_t scale = vdupq_n_f32(1.0f / length);
            vst1q_f32(&x, vmulq_f32(vec, scale));
        }
#endif
    } else {
        *this /= length;
    }
    return length;
}

template <typename T>
class alignas(16) Vec4 {
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
        __m128 simd;
#elif defined(HAVE_NEON)
        float32x4_t simd;
#endif
    };

    [[nodiscard]] T* AsArray() {
        return &x;
    }

    [[nodiscard]] const T* AsArray() const {
        return &x;
    }

    constexpr Vec4() = default;

    constexpr Vec4(const T& x_, const T& y_, const T& z_, const T& w_) {
        if constexpr (detail::is_vectorizable<T>::value) {
#if defined(HAVE_SSE2)
            simd = _mm_set_ps(w_, z_, y_, x_);
#elif defined(HAVE_NEON)
            float values[4] = {x_, y_, z_, w_};
            simd = vld1q_f32(values);
#endif
        } else {
            x = x_;
            y = y_;
            z = z_;
            w = w_;
        }
    }

    template <typename T2>
    [[nodiscard]] constexpr Vec4<T2> Cast() const {
        if constexpr (detail::is_vectorizable<T>::value && detail::is_vectorizable<T2>::value) {
            Vec4<T2> result;
#if defined(HAVE_SSE2)
            if constexpr (std::is_same_v<T, float> && std::is_same_v<T2, float>) {
                result.simd = simd;
            } else {
                result.x = static_cast<T2>(x);
                result.y = static_cast<T2>(y);
                result.z = static_cast<T2>(z);
                result.w = static_cast<T2>(w);
            }
#elif defined(HAVE_NEON)
            if constexpr (std::is_same_v<T, float> && std::is_same_v<T2, float>) {
                result.simd = simd;
            } else {
                result.x = static_cast<T2>(x);
                result.y = static_cast<T2>(y);
                result.z = static_cast<T2>(z);
                result.w = static_cast<T2>(w);
            }
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
#if defined(HAVE_SSE2)
            result.simd = _mm_div_ps(simd, _mm_set1_ps(f));
#elif defined(HAVE_NEON)
            // Improved precision with two Newton-Raphson iterations
            float32x4_t recip = vrecpeq_f32(vdupq_n_f32(f));
            recip = vmulq_f32(recip, vrecpsq_f32(vdupq_n_f32(f), recip));
            recip = vmulq_f32(recip, vrecpsq_f32(vdupq_n_f32(f), recip));
            result.simd = vmulq_f32(simd, recip);
#endif
            return result;
        } else {
            return {x / f, y / f, z / f, w / f};
        }
    }

    template <typename V>
    constexpr Vec4& operator/=(const V& f) {
        if constexpr (detail::is_vectorizable<T>::value && std::is_same_v<V, float>) {
#if defined(HAVE_SSE2)
            simd = _mm_div_ps(simd, _mm_set1_ps(f));
#elif defined(HAVE_NEON)
            // Improved precision with two Newton-Raphson iterations
            float32x4_t recip = vrecpeq_f32(vdupq_n_f32(f));
            recip = vmulq_f32(recip, vrecpsq_f32(vdupq_n_f32(f), recip));
            recip = vmulq_f32(recip, vrecpsq_f32(vdupq_n_f32(f), recip));
            simd = vmulq_f32(simd, recip);
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
            return (vget_lane_u32(vand_u32(hi_lo, vrev64_u32(hi_lo)), 0)) == 0xFFFFFFFF;
#endif
        } else {
            return x == other.x && y == other.y && z == other.z && w == other.w;
        }
    }

    [[nodiscard]] constexpr bool operator!=(const Vec4& other) const {
        return !(*this == other);
    }

    [[nodiscard]] constexpr T Length2() const {
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
            float32x2_t sum = vpadd_f32(vget_low_f32(sq), vget_high_f32(sq));
            return vget_lane_f32(vpadd_f32(sum, sum), 0);
#endif
        } else {
            return x * x + y * y + z * z + w * w;
        }
    }

    [[nodiscard]] float Length() const;
    [[nodiscard]] Vec4 Normalized() const;
    float Normalize(); // returns the previous length, which is often useful

    [[nodiscard]] constexpr T& operator[](std::size_t i) {
        return *((&x) + i);
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t i) const {
        return *((&x) + i);
    }

    constexpr void SetZero() {
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

    // Common aliases: RGBA (colors), STPQ (texture coordinates)
    [[nodiscard]] constexpr T& r() {
        return x;
    }
    [[nodiscard]] constexpr T& g() {
        return y;
    }
    [[nodiscard]] constexpr T& b() {
        return z;
    }
    [[nodiscard]] constexpr T& a() {
        return w;
    }

    [[nodiscard]] constexpr T& s() {
        return x;
    }
    [[nodiscard]] constexpr T& t() {
        return y;
    }
    [[nodiscard]] constexpr T& p() {
        return z;
    }
    [[nodiscard]] constexpr T& q() {
        return w;
    }

    [[nodiscard]] constexpr const T& r() const {
        return x;
    }
    [[nodiscard]] constexpr const T& g() const {
        return y;
    }
    [[nodiscard]] constexpr const T& b() const {
        return z;
    }
    [[nodiscard]] constexpr const T& a() const {
        return w;
    }

    [[nodiscard]] constexpr const T& s() const {
        return x;
    }
    [[nodiscard]] constexpr const T& t() const {
        return y;
    }
    [[nodiscard]] constexpr const T& p() const {
        return z;
    }
    [[nodiscard]] constexpr const T& q() const {
        return w;
    }

// swizzlers - create a subvector of specific components
#define DEFINE_SWIZZLER2(a, b, name)                                                               \
    [[nodiscard]] constexpr Vec2<T> name() const {                                                 \
        return Vec2<T>(a, b);                                                                      \
    }

    DEFINE_SWIZZLER2(x, y, xy);
    DEFINE_SWIZZLER2(x, z, xz);
    DEFINE_SWIZZLER2(x, w, xw);
    DEFINE_SWIZZLER2(y, z, yz);
    DEFINE_SWIZZLER2(y, w, yw);
    DEFINE_SWIZZLER2(z, w, zw);
    DEFINE_SWIZZLER2(r, g, rg);
    DEFINE_SWIZZLER2(r, b, rb);
    DEFINE_SWIZZLER2(r, a, ra);
    DEFINE_SWIZZLER2(g, b, gb);
    DEFINE_SWIZZLER2(g, a, ga);
    DEFINE_SWIZZLER2(b, a, ba);
    DEFINE_SWIZZLER2(s, t, st);
    DEFINE_SWIZZLER2(s, p, sp);
    DEFINE_SWIZZLER2(s, q, sq);
    DEFINE_SWIZZLER2(t, p, tp);
    DEFINE_SWIZZLER2(t, q, tq);
    DEFINE_SWIZZLER2(p, q, pq);

#define DEFINE_SWIZZLER3(a, b, c, name)                                                            \
    [[nodiscard]] constexpr Vec3<T> name() const {                                                 \
        return Vec3<T>(a, b, c);                                                                   \
    }

    DEFINE_SWIZZLER3(x, y, z, xyz);
    DEFINE_SWIZZLER3(x, y, w, xyw);
    DEFINE_SWIZZLER3(x, z, w, xzw);
    DEFINE_SWIZZLER3(y, z, w, yzw);
    DEFINE_SWIZZLER3(r, g, b, rgb);
    DEFINE_SWIZZLER3(r, g, a, rga);
    DEFINE_SWIZZLER3(r, b, a, rba);
    DEFINE_SWIZZLER3(g, b, a, gba);
    DEFINE_SWIZZLER3(s, t, p, stp);
    DEFINE_SWIZZLER3(s, t, q, stq);
    DEFINE_SWIZZLER3(s, p, q, spq);
    DEFINE_SWIZZLER3(t, p, q, tpq);

#undef DEFINE_SWIZZLER2
#undef DEFINE_SWIZZLER3
};

template <typename T, typename V>
[[nodiscard]] constexpr Vec4<T> operator*(const V& f, const Vec4<T>& vec) {
    return vec * f;
}

using Vec4f = Vec4<float>;
using Vec4i = Vec4<int>;
using Vec4u = Vec4<unsigned int>;

template <>
inline float Vec4<float>::Length() const {
#if defined(HAVE_SSE4_1)
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(simd, simd, 0xF1)));
#elif defined(HAVE_SSE2)
    __m128 sq = _mm_mul_ps(simd, simd);
    __m128 sum = _mm_add_ps(_mm_add_ps(sq, _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(2, 3, 0, 1))),
                            _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(_mm_sqrt_ss(sum));
#elif defined(HAVE_NEON)
    float32x4_t sq = vmulq_f32(simd, simd);
    float32x2_t sum = vpadd_f32(vget_low_f32(sq), vget_high_f32(sq));
    float32x2_t total = vpadd_f32(sum, sum);
    return sqrtf(vget_lane_f32(total, 0));
#else
    return std::sqrt(x * x + y * y + z * z + w * w);
#endif
}

template <>
inline Vec4<float> Vec4<float>::Normalized() const {
    const float length = Length();
    if constexpr (detail::is_vectorizable<float>::value) {
#if defined(HAVE_SSE2)
        if (length != 0) {
            Vec4 result;
            result.simd = _mm_div_ps(simd, _mm_set1_ps(length));
            return result;
        }
#elif defined(HAVE_NEON)
        if (length != 0) {
            Vec4 result;
            float32x4_t recip = vrecpeq_f32(vdupq_n_f32(length));
            recip = vmulq_f32(recip, vrecpsq_f32(vdupq_n_f32(length), recip));
            recip = vmulq_f32(recip, vrecpsq_f32(vdupq_n_f32(length), recip));
            result.simd = vmulq_f32(simd, recip);
            return result;
        }
#endif
    }
    return *this / length;
}

template <>
inline float Vec4<float>::Normalize() {
    const float length = Length();
    if constexpr (detail::is_vectorizable<float>::value) {
#if defined(HAVE_SSE2)
        if (length != 0) {
            simd = _mm_div_ps(simd, _mm_set1_ps(length));
        }
#elif defined(HAVE_NEON)
        if (length != 0) {
            float32x4_t recip = vrecpeq_f32(vdupq_n_f32(length));
            recip = vmulq_f32(recip, vrecpsq_f32(vdupq_n_f32(length), recip));
            recip = vmulq_f32(recip, vrecpsq_f32(vdupq_n_f32(length), recip));
            simd = vmulq_f32(simd, recip);
        }
#endif
    } else {
        *this /= length;
    }
    return length;
}

} // namespace Common

#endif // COMMON_VECTOR_MATH_H
