//
// Created by anxin on 2020/4/9.
//

#ifndef TEXTTRUTH_OBLIVIOUS_PRIMITIVE_H
#define TEXTTRUTH_OBLIVIOUS_PRIMITIVE_H


#include <immintrin.h>
#include <emmintrin.h>

using uint32_t = unsigned int;
using uint8_t = unsigned char;

uint32_t oblivious_assign_CMOV(uint8_t pred, uint32_t t_val, uint32_t f_val);

//void oblivious_assign_AVX(uint8_t pred, void *dst, void *t_val, void  *f_val);


#endif //TEXTTRUTH_OBLIVIOUS_PRIMITIVE_H
