//
// Created by anxin on 2020/4/9.
//

#ifndef TEXTTRUTH_OBLIVIOUS_PRIMITIVE_H
#define TEXTTRUTH_OBLIVIOUS_PRIMITIVE_H


#include <immintrin.h>
#include <emmintrin.h>
#include <vector>
#include <cmath>
#include "Member.h"

const int KEYWORD_LENGTH_ALIGN = 4;
const char KEYWORD_PAD_CHAR = ' ';

using namespace std;
//#include <omp.h>

using uint32_t = unsigned int;
using uint8_t = unsigned char;

uint32_t oblivious_assign_CMOV(uint8_t pred, uint32_t t_val, uint32_t f_val);

template<typename T>
void compare_and_swap(T &a, T &b, int sort_direction) {
    if (sort_direction == 0) {
        if (a > b) std::swap(a, b);
    } else {
        if (a < b) std::swap(a, b);
    }
}

template<typename T>
void __bitonic_sort(vector<T> &arr, int start_index, int end_index, int sort_direction) {
    int ele = end_index - start_index;
    for (int j = ele; j >= 2; j /= 2) {
        for (int k = start_index; k < end_index; k += j) {
            for (int i = k; i < k + j / 2; i++) {
                compare_and_swap(arr[i], arr[j / 2 + i], sort_direction);
            }
        }
    }
}

template<typename T>
void bitonic_sort(vector<T> &arr, int sort_direction) {
    for (int j = 2; j <= arr.size(); j *= 2) {
        for (int i = 0; i < arr.size(); i += j) {         // can use openmp accelerate this part
            __bitonic_sort(arr, i, i + j, (i / j % 2) ^ sort_direction);
        }
    }
}

void oblivious_bitonic_sort(vector<uint32_t > &arr, int sort_direction);


// offer less operator for cmp
void oblivious_bitonic_sort(vector<Keyword> &arr, int sort_direction,function<bool(Keyword&,Keyword&)> &cmp);

string oblivious_assign_string(uint8_t pred, const string &a, const string&b);

void oblivious_assign_keyword(uint8_t pred, Keyword &dst, Keyword &t, Keyword &f);

void oblivious_sort(vector<Keyword> &arr, int sort_direction);
void oblivious_sort(vector<pair<double, int>> &arr, int sort_direction);
void oblivious_shuffle(vector<Keyword> &arr);

void keywords_padding(vector<Keyword>& keywords);
void keywords_remove_padding(vector<Keyword>& keywords);

vector<Keyword> oblivious_vocabulary_decide(vector<Keyword>&keywords);

void oblivious_dummy_words_addition(vector<Keyword> &padded_vocabulary, vector<Keyword> &keywords);
#endif //TEXTTRUTH_OBLIVIOUS_PRIMITIVE_H
