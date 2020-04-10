//
// Created by anxin on 2020/4/9.
//

#include "oblivious_primitive.h"

uint32_t oblivious_assign_CMOV(uint8_t pred, uint32_t t_val, uint32_t f_val) {
    uint32_t result;
    __asm__ volatile (
    "mov %2, %0;"
    "test %1, %1;"
    "cmovz %3, %0;"
    "test %2, %2;"
    :"=r" (result)
    :"r"(pred), "r"(t_val), "r"(f_val)
    :"cc"
    );
    return result;
}