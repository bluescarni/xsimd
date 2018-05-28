/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX512_FLOAT_HPP
#define XSIMD_AVX512_FLOAT_HPP

#include <cstdint>

#include "xsimd_base.hpp"

namespace xsimd
{

    /*************************
     * batch_bool<float, 16> *
     *************************/

    template <>
    struct simd_batch_traits<batch_bool<float, 16>>
    {
        using value_type = float;
        static constexpr std::size_t size = 16;
        using batch_type = batch<float, 16>;
        static constexpr std::size_t align = 64;
    };

    template <>
    class batch_bool<float, 16> : public simd_batch_bool<batch_bool<float, 16>>
    {
    public:

        batch_bool();
        explicit batch_bool(bool b);
        batch_bool(bool b0, bool b1, bool b2, bool b3,
                   bool b4, bool b5, bool b6, bool b7,
                   bool b8, bool b9, bool b10, bool b11,
                   bool b12, bool b13, bool b14, bool b15);
        batch_bool(const __m512& rhs);
        batch_bool& operator=(const __m512& rhs);

        operator __m512() const;

        bool operator[](std::size_t index) const;

    private:

        __m512 m_value;
    };

    batch_bool<float, 16> operator&(const batch_bool<float, 16>& lhs, const batch_bool<float, 16>& rhs);
    batch_bool<float, 16> operator|(const batch_bool<float, 16>& lhs, const batch_bool<float, 16>& rhs);
    batch_bool<float, 16> operator^(const batch_bool<float, 16>& lhs, const batch_bool<float, 16>& rhs);
    batch_bool<float, 16> operator~(const batch_bool<float, 16>& rhs);
    batch_bool<float, 16> bitwise_andnot(const batch_bool<float, 16>& lhs, const batch_bool<float, 16>& rhs);

    batch_bool<float, 16> operator==(const batch_bool<float, 16>& lhs, const batch_bool<float, 16>& rhs);
    batch_bool<float, 16> operator!=(const batch_bool<float, 16>& lhs, const batch_bool<float, 16>& rhs);

    bool all(const batch_bool<float, 16>& rhs);
    bool any(const batch_bool<float, 16>& rhs);

    /********************
     * batch<float, 16> *
     ********************/

    template <>
    struct simd_batch_traits<batch<float, 16>>
    {
        using value_type = float;
        static constexpr std::size_t size = 16;
        using batch_bool_type = batch_bool<float, 16>;
        static constexpr std::size_t align = 64;
    };

    template <>
    class batch<float, 16> : public simd_batch<batch<float, 16>>
    {
    public:

        batch();
        explicit batch(float f);
        batch(float f0, float f1, float f2, float f3,
              float f4, float f5, float f6, float f7,
              float f8, float f9, float f10, float f11,
              float f12, float f13, float f14, float f15);
        explicit batch(const float* src);
        batch(const float* src, aligned_mode);
        batch(const float* src, unaligned_mode);
        batch(const __m512& rhs);
        batch& operator=(const __m512& rhs);

        operator __m512() const;

        batch& load_aligned(const float* src);
        batch& load_unaligned(const float* src);

        batch& load_aligned(const double* src);
        batch& load_unaligned(const double* src);

        batch& load_aligned(const int32_t* src);
        batch& load_unaligned(const int32_t* src);

        batch& load_aligned(const int64_t* src);
        batch& load_unaligned(const int64_t* src);

        void store_aligned(float* dst) const;
        void store_unaligned(float* dst) const;

        void store_aligned(double* dst) const;
        void store_unaligned(double* dst) const;

        void store_aligned(int32_t* dst) const;
        void store_unaligned(int32_t* dst) const;

        void store_aligned(int64_t* dst) const;
        void store_unaligned(int64_t* dst) const;

        float operator[](std::size_t index) const;

    private:

        __m512 m_value;
    };

    batch<float, 16> operator-(const batch<float, 16>& rhs);
    batch<float, 16> operator+(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> operator-(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> operator*(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> operator/(const batch<float, 16>& lhs, const batch<float, 16>& rhs);

    batch_bool<float, 16> operator==(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch_bool<float, 16> operator!=(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch_bool<float, 16> operator<(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch_bool<float, 16> operator<=(const batch<float, 16>& lhs, const batch<float, 16>& rhs);

    batch<float, 16> operator&(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> operator|(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> operator^(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> operator~(const batch<float, 16>& rhs);
    batch<float, 16> bitwise_andnot(const batch<float, 16>& lhs, const batch<float, 16>& rhs);

    batch<float, 16> min(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> max(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> fmin(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> fmax(const batch<float, 16>& lhs, const batch<float, 16>& rhs);

    batch<float, 16> abs(const batch<float, 16>& rhs);
    batch<float, 16> fabs(const batch<float, 16>& rhs);
    batch<float, 16> sqrt(const batch<float, 16>& rhs);

    batch<float, 16> fma(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z);
    batch<float, 16> fms(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z);
    batch<float, 16> fnma(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z);
    batch<float, 16> fnms(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z);

    float hadd(const batch<float, 16>& rhs);
    batch<float, 16> haddp(const batch<float, 16>* row);

    batch<float, 16> select(const batch_bool<float, 16>& cond, const batch<float, 16>& a, const batch<float, 16>& b);

    batch_bool<float, 16> isnan(const batch<float, 16>& x);

    /****************************************
     * batch_bool<float, 16> implementation *
     ****************************************/

    inline batch_bool<float, 16>::batch_bool()
    {
    }

    inline batch_bool<float, 16>::batch_bool(bool b)
        : m_value(_mm512_castsi512_ps(_mm512_set1_epi32(-(int)b)))
    {
    }

    inline batch_bool<float, 16>::batch_bool(bool b0, bool b1, bool b2, bool b3,
                                             bool b4, bool b5, bool b6, bool b7,
                                             bool b8, bool b9, bool b10, bool b11,
                                             bool b12, bool b13, bool b14, bool b15)
        : m_value(_mm512_castsi512_ps(
              _mm512_setr_epi32(-(int)b0, -(int)b1, -(int)b2, -(int)b3,
                                -(int)b4, -(int)b5, -(int)b6, -(int)b7,
                                -(int)b8, -(int)b9, -(int)b10, -(int)b11,
                                -(int)b12, -(int)b13, -(int)b14, -(int)b15)))
    {
    }

    inline batch_bool<float, 16>::batch_bool(const __m512& rhs)
        : m_value(rhs)
    {
    }

    inline batch_bool<float, 16>& batch_bool<float, 16>::operator=(const __m512& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch_bool<float, 16>::operator __m512() const
    {
        return m_value;
    }

    inline bool batch_bool<float, 16>::operator[](std::size_t index) const
    {
        alignas(64) float x[16];
        _mm512_store_ps(x, m_value);
        return static_cast<bool>(x[index & 15]);
    }

    inline batch_bool<float, 16> operator&(const batch_bool<float, 16>& lhs, const batch_bool<float, 16>& rhs)
    {
        return _mm512_and_ps(lhs, rhs);
    }

    inline batch_bool<float, 16> operator|(const batch_bool<float, 16>& lhs, const batch_bool<float, 16>& rhs)
    {
        return _mm512_or_ps(lhs, rhs);
    }

    inline batch_bool<float, 16> operator^(const batch_bool<float, 16>& lhs, const batch_bool<float, 16>& rhs)
    {
        return _mm512_xor_ps(lhs, rhs);
    }

    inline batch_bool<float, 16> operator~(const batch_bool<float, 16>& rhs)
    {
        return _mm512_xor_ps(rhs, _mm512_castsi512_ps(_mm512_set1_epi32(-1)));
    }

    inline batch_bool<float, 16> bitwise_andnot(const batch_bool<float, 16>& lhs, const batch_bool<float, 16>& rhs)
    {
        return _mm512_andnot_ps(lhs, rhs);
    }

    inline batch_bool<float, 16> operator==(const batch_bool<float, 16>& lhs, const batch_bool<float, 16>& rhs)
    {
        // Explanation: init two constant vectors m1 = 0xFFF... and zero = 0. Then, we run
        // the comparison between lhs and rhs, which returns a mask. Lastly,
        // we pick values from zero and m1 according to the mask.
        const auto m1 = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
        const auto zero = _mm512_castsi512_ps(_mm512_set1_epi32(0));
        return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(lhs, rhs, _CMP_EQ_OQ), zero, m1);
    }

    inline batch_bool<float, 16> operator!=(const batch_bool<float, 16>& lhs, const batch_bool<float, 16>& rhs)
    {
        const auto m1 = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
        const auto zero = _mm512_castsi512_ps(_mm512_set1_epi32(0));
        return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(lhs, rhs, _CMP_NEQ_OQ), zero, m1);
    }

    inline bool all(const batch_bool<float, 16>& rhs)
    {
        const auto zero = _mm512_castsi512_ps(_mm512_set1_epi32(0));
        // Explanation: this will create a mask (i.e., a 16bit unsigned value)
        // in which the bit at index i is 1 if rhs[i] != 0, 0 otherwise. Use
        // the _CMP_NEQ_UQ predicate so that NaNs are considered different from zero.
        return _mm512_cmp_ps_mask(zero, rhs, _CMP_NEQ_UQ) == std::uint16_t(-1);
    }

    inline bool any(const batch_bool<float, 16>& rhs)
    {
        // Same as above.
        const auto zero = _mm512_castsi512_ps(_mm512_set1_epi32(0));
        return _mm512_cmp_ps_mask(zero, rhs, _CMP_NEQ_UQ);
    }

    /***********************************
     * batch<float, 16> implementation *
     ***********************************/

    inline batch<float, 16>::batch()
    {
    }

    inline batch<float, 16>::batch(float f)
        : m_value(_mm512_set1_ps(f))
    {
    }

    inline batch<float, 16>::batch(float f0, float f1, float f2, float f3,
                                   float f4, float f5, float f6, float f7,
                                   float f8, float f9, float f10, float f11,
                                   float f12, float f13, float f14, float f15)
        : m_value(_mm512_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                                 f11, f12, f13, f14, f15))
    {
    }

    inline batch<float, 16>::batch(const float* src)
        : m_value(_mm512_loadu_ps(src))
    {
    }

    inline batch<float, 16>::batch(const float* src, aligned_mode)
        : m_value(_mm512_load_ps(src))
    {
    }

    inline batch<float, 16>::batch(const float* src, unaligned_mode)
        : m_value(_mm512_loadu_ps(src))
    {
    }

    inline batch<float, 16>::batch(const __m512& rhs)
        : m_value(rhs)
    {
    }

    inline batch<float, 16>& batch<float, 16>::operator=(const __m512& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<float, 16>::operator __m512() const
    {
        return m_value;
    }

    inline batch<float, 16>& batch<float, 16>::load_aligned(const float* src)
    {
        m_value = _mm512_load_ps(src);
        return *this;
    }

    inline batch<float, 16>& batch<float, 16>::load_unaligned(const float* src)
    {
        m_value = _mm512_loadu_ps(src);
        return *this;
    }

    inline batch<float, 16>& batch<float, 16>::load_aligned(const double* src)
    {
        __m256 tmp1 = _mm512_cvtpd_ps(_mm512_load_pd(src));
        __m256 tmp2 = _mm512_cvtpd_ps(_mm512_load_pd(src + 8));
        m_value = _mm512_castps256_ps512(tmp1);
        m_value = _mm512_insertf32x8(m_value, tmp2, 1);
        return *this;
    }

    inline batch<float, 16>& batch<float, 16>::load_unaligned(const double* src)
    {
        __m256 tmp1 = _mm512_cvtpd_ps(_mm512_loadu_pd(src));
        __m256 tmp2 = _mm512_cvtpd_ps(_mm512_loadu_pd(src + 8));
        m_value = _mm512_castps256_ps512(tmp1);
        m_value = _mm512_insertf32x8(m_value, tmp2, 1);
        return *this;
    }

    inline batch<float, 16>& batch<float, 16>::load_aligned(const int32_t* src)
    {
        m_value = _mm512_cvtepi32_ps(_mm512_load_si512(src));
        return *this;
    }

    inline batch<float, 16>& batch<float, 16>::load_unaligned(const int32_t* src)
    {
        m_value = _mm512_cvtepi32_ps(_mm512_loadu_si512(src));
        return *this;
    }

    inline batch<float, 16>& batch<float, 16>::load_aligned(const int64_t* src)
    {
        alignas(64) float tmp[16];
        tmp[0] = float(src[0]);
        tmp[1] = float(src[1]);
        tmp[2] = float(src[2]);
        tmp[3] = float(src[3]);
        tmp[4] = float(src[4]);
        tmp[5] = float(src[5]);
        tmp[6] = float(src[6]);
        tmp[7] = float(src[7]);
        tmp[8] = float(src[8]);
        tmp[9] = float(src[9]);
        tmp[10] = float(src[10]);
        tmp[11] = float(src[11]);
        tmp[12] = float(src[12]);
        tmp[13] = float(src[13]);
        tmp[14] = float(src[14]);
        tmp[15] = float(src[15]);
        m_value = _mm512_load_ps(tmp);
        return *this;
    }

    inline batch<float, 16>& batch<float, 16>::load_unaligned(const int64_t* src)
    {
        return load_aligned(src);
    }

    inline void batch<float, 16>::store_aligned(float* dst) const
    {
        _mm512_store_ps(dst, m_value);
    }

    inline void batch<float, 16>::store_unaligned(float* dst) const
    {
        _mm512_storeu_ps(dst, m_value);
    }

    inline void batch<float, 16>::store_aligned(double* dst) const
    {
        alignas(64) float tmp[16];
        _mm512_store_ps(tmp, m_value);
        dst[0] = static_cast<double>(tmp[0]);
        dst[1] = static_cast<double>(tmp[1]);
        dst[2] = static_cast<double>(tmp[2]);
        dst[3] = static_cast<double>(tmp[3]);
        dst[4] = static_cast<double>(tmp[4]);
        dst[5] = static_cast<double>(tmp[5]);
        dst[6] = static_cast<double>(tmp[6]);
        dst[7] = static_cast<double>(tmp[7]);
        dst[8] = static_cast<double>(tmp[8]);
        dst[9] = static_cast<double>(tmp[9]);
        dst[10] = static_cast<double>(tmp[10]);
        dst[11] = static_cast<double>(tmp[11]);
        dst[12] = static_cast<double>(tmp[12]);
        dst[13] = static_cast<double>(tmp[13]);
        dst[14] = static_cast<double>(tmp[14]);
        dst[15] = static_cast<double>(tmp[15]);
    }

    inline void batch<float, 16>::store_unaligned(double* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<float, 16>::store_aligned(int32_t* dst) const
    {
        _mm512_store_si512(dst, _mm512_cvtps_epi32(m_value));
    }

    inline void batch<float, 16>::store_unaligned(int32_t* dst) const
    {
        _mm512_storeu_si512(dst, _mm512_cvtps_epi32(m_value));
    }

    inline void batch<float, 16>::store_aligned(int64_t* dst) const
    {
        alignas(64) float tmp[16];
        _mm512_store_ps(tmp, m_value);
        dst[0] = static_cast<int64_t>(tmp[0]);
        dst[1] = static_cast<int64_t>(tmp[1]);
        dst[2] = static_cast<int64_t>(tmp[2]);
        dst[3] = static_cast<int64_t>(tmp[3]);
        dst[4] = static_cast<int64_t>(tmp[4]);
        dst[5] = static_cast<int64_t>(tmp[5]);
        dst[6] = static_cast<int64_t>(tmp[6]);
        dst[7] = static_cast<int64_t>(tmp[7]);
        dst[8] = static_cast<int64_t>(tmp[8]);
        dst[9] = static_cast<int64_t>(tmp[9]);
        dst[10] = static_cast<int64_t>(tmp[10]);
        dst[11] = static_cast<int64_t>(tmp[11]);
        dst[12] = static_cast<int64_t>(tmp[12]);
        dst[13] = static_cast<int64_t>(tmp[13]);
        dst[14] = static_cast<int64_t>(tmp[14]);
        dst[15] = static_cast<int64_t>(tmp[15]);
    }

    inline void batch<float, 16>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline float batch<float, 16>::operator[](std::size_t index) const
    {
        alignas(64) float x[16];
        store_aligned(x);
        return x[index & 15];
    }

    inline batch<float, 16> operator-(const batch<float, 16>& rhs)
    {
        return _mm512_xor_ps(rhs, _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000)));
    }

    inline batch<float, 16> operator+(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_add_ps(lhs, rhs);
    }

    inline batch<float, 16> operator-(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_sub_ps(lhs, rhs);
    }

    inline batch<float, 16> operator*(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_mul_ps(lhs, rhs);
    }

    inline batch<float, 16> operator/(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_div_ps(lhs, rhs);
    }

    inline batch_bool<float, 16> operator==(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        const auto m1 = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
        const auto zero = _mm512_castsi512_ps(_mm512_set1_epi32(0));
        return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(lhs, rhs, _CMP_EQ_OQ), zero, m1);
    }

    inline batch_bool<float, 16> operator!=(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        const auto m1 = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
        const auto zero = _mm512_castsi512_ps(_mm512_set1_epi32(0));
        return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(lhs, rhs, _CMP_NEQ_OQ), zero, m1);
    }

    inline batch_bool<float, 16> operator<(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        const auto m1 = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
        const auto zero = _mm512_castsi512_ps(_mm512_set1_epi32(0));
        return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(lhs, rhs, _CMP_LT_OQ), zero, m1);
    }

    inline batch_bool<float, 16> operator<=(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        const auto m1 = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
        const auto zero = _mm512_castsi512_ps(_mm512_set1_epi32(0));
        return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(lhs, rhs, _CMP_LE_OQ), zero, m1);
    }

    inline batch<float, 16> operator&(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_and_ps(lhs, rhs);
    }

    inline batch<float, 16> operator|(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_or_ps(lhs, rhs);
    }

    inline batch<float, 16> operator^(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_xor_ps(lhs, rhs);
    }

    inline batch<float, 16> operator~(const batch<float, 16>& rhs)
    {
        return _mm512_xor_ps(rhs, _mm512_castsi512_ps(_mm512_set1_epi32(-1)));
    }

    inline batch<float, 16> bitwise_andnot(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_andnot_ps(lhs, rhs);
    }

    inline batch<float, 16> min(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_min_ps(lhs, rhs);
    }

    inline batch<float, 16> max(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_max_ps(lhs, rhs);
    }

    inline batch<float, 16> fmin(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return min(lhs, rhs);
    }

    inline batch<float, 16> fmax(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return max(lhs, rhs);
    }

    inline batch<float, 16> abs(const batch<float, 16>& rhs)
    {
        return _mm512_abs_ps(rhs);
    }

    inline batch<float, 16> fabs(const batch<float, 16>& rhs)
    {
        return abs(rhs);
    }

    inline batch<float, 16> sqrt(const batch<float, 16>& rhs)
    {
        return _mm512_sqrt_ps(rhs);
    }

    inline batch<float, 16> fma(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z)
    {
        return _mm512_fmadd_ps(x, y, z);
    }

    inline batch<float, 16> fms(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z)
    {
        return _mm512_fmsub_ps(x, y, z);
    }

    inline batch<float, 16> fnma(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z)
    {
        return _mm512_fnmadd_ps(x, y, z);
    }

    inline batch<float, 16> fnms(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z)
    {
        return _mm512_fnmsub_ps(x, y, z);
    }

    inline float hadd(const batch<float, 16>& rhs)
    {
        return _mm512_reduce_add_ps(rhs);
    }

    inline batch<float, 16> haddp(const batch<float, 16>* row)
    {
        alignas(64) float tmp[16];
        for (std::size_t i = 0; i < 16u; ++i) {
            tmp[i] = _mm512_reduce_add_ps(row[i]);
        }
        return batch<float, 16>(tmp, aligned_mode{});
    }

    inline batch<float, 16> select(const batch_bool<float, 16>& cond, const batch<float, 16>& a, const batch<float, 16>& b)
    {
        // Explanation: we need to transform the input cond into a mask. We can do this
        // by creating a vector of zeroes, and then comparing it to cond. Each non-zero
        // element in cond (including NaN) will then have the corresponding bit in mask
        // set to 1. We can then use the blend with mask instruction.
        const auto zero = _mm512_castsi512_ps(_mm512_set1_epi32(0));
        const auto mask = _mm512_cmp_ps_mask(cond, zero, _CMP_NEQ_UQ);
        return _mm512_mask_blend_ps(mask, b, a);
    }

    inline batch_bool<float, 16> isnan(const batch<float, 16>& x)
    {
        const auto m1 = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
        const auto zero = _mm512_castsi512_ps(_mm512_set1_epi32(0));
        return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q), zero, m1);
    }
}

#endif
