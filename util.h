//
// Created by anxin on 2020-02-25.
//

#ifndef TEXTTRUTH_UTIL_H
#define TEXTTRUTH_UTIL_H

#include <immintrin.h>
#include <smmintrin.h>
#include <vector>

namespace hpc {
    inline bool array_equal(std::vector<int> &a, std::vector<int> &b);

    inline float hsum_ps_sse3(__m128 v) {
        __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
        __m128 sums = _mm_add_ps(v, shuf);
        shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }

    float dot_product(const std::vector<float> &p, const std::vector<float> &q);

    float dot_product(float *p, float *q, int length);

    void vector_mul_inplace(std::vector<float> &p, float v);

    // add p+q and store result in p
    void vector_add_inplace(std::vector<float> &p, const std::vector<float> &q);

    // sub p-q and store result in p
    void vector_sub_inplace(std::vector<float> &p, const std::vector<float> &q);

    // sub p-q and store result in p
    std::vector<float> vector_sub(const std::vector<float> &p, const std::vector<float> &q);

    inline float hsum256_ps_avx(__m256 v) {
        __m128 vlow = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
        vlow = _mm_add_ps(vlow, vhigh);     // add the low 128
        return hsum_ps_sse3(vlow);         // and inline the sse3 version, which is optimal for AVX
        // (no wasted instructions, and all of them are the 4B minimum)
    }

    bool vector_cmpeq(std::vector<int> &p, std::vector<int> &q);
}
#endif //TEXTTRUTH_UTIL_H
