//
// Created by anxin on 2020/4/9.
//

#include "oblivious_primitive.h"

const int SCORE_SCALE_FACTOR = 1e6;

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


void oblivious_compare_and_swap(uint32_t &a, uint32_t &b, int sort_direction) {
    bool flag = sort_direction == 0 ? a > b : a < b;
    uint32_t tmp = a;
    a = oblivious_assign_CMOV(flag, b, a);
    b = oblivious_assign_CMOV(flag, tmp, b);
}

void __oblivious_bitonic_sort(vector<uint32_t> &arr, int start_index, int end_index, int sort_direction) {
    int ele = end_index - start_index;
    for (int j = ele; j >= 2; j /= 2) {
        for (int k = start_index; k < end_index; k += j) {
            for (int i = k; i < k + j / 2; i++) {
                oblivious_compare_and_swap(arr[i], arr[j / 2 + i], sort_direction);
            }
        }
    }
}

void oblivious_bitonic_sort(vector<uint32_t> &arr, int sort_direction) {
    for (int j = 2; j <= arr.size(); j *= 2) {
        for (int i = 0; i < arr.size(); i += j) {         // can use openmp accelerate this part
            __oblivious_bitonic_sort(arr, i, i + j, (i / j % 2) ^ sort_direction);
        }
    }
}

inline void oblivious_assign_score_pair(uint8_t pred, pair<double, int> &dst, pair<double, int> &t, pair<double, int> &f) {
    dst.second = oblivious_assign_CMOV(pred, t.second, f.second);
    double tmp_first = oblivious_assign_CMOV(pred, t.first*SCORE_SCALE_FACTOR, f.first*SCORE_SCALE_FACTOR);
    dst.first = tmp_first/SCORE_SCALE_FACTOR;
}

void oblivious_compare_and_swap(pair<double, int> &a, pair<double, int> &b, int sort_direction,
                                function<bool(pair<double, int> &a, pair<double, int> &b)> &cmp) {
    // 0 ? a>b: a<b

    bool flag = sort_direction == 0 == !cmp(a, b);
    pair<double, int> tmp(a);
    oblivious_assign_score_pair(flag, a, b, a);
    oblivious_assign_score_pair(flag, b, tmp, b);
}

bool score_pair_cmp(pair<double, int> &a, pair<double, int> &b) {
    return a.first < b.first;
}

void __oblivious_bitonic_sort(vector<pair<double, int>> &arr, int start_index, int end_index, int sort_direction,
                              function<bool(pair<double, int> &a, pair<double, int> &b)> &cmp) {
    int ele = end_index - start_index;
    for (int j = ele; j >= 2; j /= 2) {
        for (int k = start_index; k < end_index; k += j) {
            for (int i = k; i < k + j / 2; i++) {
                oblivious_compare_and_swap(arr[i], arr[j / 2 + i], sort_direction, cmp);
            }
        }
    }
}

void oblivious_bitonic_sort(vector<pair<double, int>> &arr, int sort_direction,
                            function<bool(pair<double, int> &a, pair<double, int> &b)> &cmp) {
    // pad with dummy to fit 2^n;
    int level = log2(arr.size());
    if (pow(2, level) != arr.size()) {
        level += 1;
    }
    vector<pair<double, int>> tmp_arr;
    tmp_arr.reserve(arr.size());
    int to_pad_size = pow(2, level) - arr.size();
    pair<double, int> dummy = arr[rand() % arr.size()];
    dummy.second = -1;
    for (int i = 0; i < to_pad_size; i++) {
        arr.push_back(dummy);
    }

    //sort
//    cout<<arr.size()<<endl;
    for (int j = 2; j <= arr.size(); j *= 2) {
        for (int i = 0; i < arr.size(); i += j) {         // can use openmp accelerate this part
            __oblivious_bitonic_sort(arr, i, i + j, (i / j % 2) ^ sort_direction, cmp);
        }
    }

    //remove dummy
    for (auto &p: arr) {
        if (p.second == -1) continue;
        tmp_arr.push_back(std::move(p));
    }
    arr = std::move(tmp_arr);
}

void oblivious_sort(vector<pair<double, int>> &arr, int sort_direction) {
    function<bool(pair<double, int> &a, pair<double, int> &b)> cmp(score_pair_cmp);
    oblivious_bitonic_sort(arr, sort_direction, cmp);
}

string oblivious_assign_string(uint8_t pred, const string &a, const string &b) {
    const char *ap = &a[0];
    const char *bp = &b[0];
    string res(a.size(), ' ');
    char *resp = &res[0];
    for (int i = 0; i < a.size(); i += 4) {
        *(uint32_t *) resp = oblivious_assign_CMOV(pred, *(uint32_t *) ap, *(uint32_t *) bp);
        ap += 4;
        bp += 4;
        resp += 4;
    }
    return res;
}

void oblivious_assign_keyword(uint8_t pred, Keyword &dst, Keyword &t, Keyword &f) {
    dst.shuffle_tag = oblivious_assign_CMOV(pred, t.shuffle_tag, f.shuffle_tag);
    dst.question_id = oblivious_assign_CMOV(pred, t.question_id, f.question_id);
    dst.cluster_assignment = oblivious_assign_CMOV(pred, t.cluster_assignment, f.cluster_assignment);
    dst.owner_id = oblivious_assign_CMOV(pred, t.owner_id, f.owner_id);
    dst.content = oblivious_assign_string(pred, t.content, f.content);
}

void oblivious_compare_and_swap(Keyword &a, Keyword &b, int sort_direction, function<bool(Keyword &, Keyword &)> &cmp) {
    // 0 ? a>b: a<b
    // not need to oblivious swap padding words for bitonic sort, assume the padded elements are always biggest, or smallest
    if(a.question_id == -1 && b.question_id == -1) return;
    if(a.question_id == -1) {
        std::swap(a, b);
    } else if(b.question_id == -1) {
        return ;
    } else {
        bool flag = sort_direction == 0 == !cmp(a, b);
        Keyword tmp(a);
        oblivious_assign_keyword(flag, a, b, a);
        oblivious_assign_keyword(flag, b, tmp, b);
    }
//    if(a.question_id == -1 && b.question_id == -1) return;


}

void __oblivious_bitonic_sort(vector<Keyword> &arr, int start_index, int end_index, int sort_direction,
                              function<bool(Keyword &, Keyword &)> cmp) {
    int ele = end_index - start_index;
    for (int j = ele; j >= 2; j /= 2) {
        for (int k = start_index; k < end_index; k += j) {
            for (int i = k; i < k + j / 2; i++) {
                oblivious_compare_and_swap(arr[i], arr[j / 2 + i], sort_direction, cmp);
            }
        }
    }
}


void oblivious_bitonic_sort(vector<Keyword> &arr, int sort_direction, function<bool(Keyword &, Keyword &)> &cmp) {
    // pad with dummy to fit 2^n;
    int level = log2(arr.size());
    if (pow(2, level) != arr.size()) {
        level += 1;
    }
    vector<Keyword> tmp_arr;
    tmp_arr.reserve(arr.size());
    int to_pad_size = pow(2, level) - arr.size();
    Keyword dummy(arr[rand() % arr.size()]);
    dummy.question_id = -1;
    for (int i = 0; i < to_pad_size; i++) {
        arr.push_back(dummy);
    }

    //sort
//    cout<<arr.size()<<endl;
    for (int j = 2; j <= arr.size(); j *= 2) {
        for (int i = 0; i < arr.size(); i += j) {         // can use openmp accelerate this part
            __oblivious_bitonic_sort(arr, i, i + j, (i / j % 2) ^ sort_direction, cmp);
        }
    }

    //remove dummy
    for (auto &keyword: arr) {
        if (keyword.question_id == -1) break;
        tmp_arr.push_back(std::move(keyword));
    }
    arr = std::move(tmp_arr);
}


bool keyword_sort_compare(Keyword &a, Keyword &b) {
    return a.content < b.content;
}

bool keyword_shuffle_compare(Keyword &a, Keyword &b) {
    return a.shuffle_tag < b.shuffle_tag;
}

void oblivious_sort(vector<Keyword> &arr, int sort_direction) {
    function<bool(Keyword &, Keyword &)> kw_sort_cmp(keyword_sort_compare);
    oblivious_bitonic_sort(arr, sort_direction, kw_sort_cmp);
}

void oblivious_shuffle(vector<Keyword> &arr) {
    function<bool(Keyword &, Keyword &)> kw_shuffle_cmp(keyword_shuffle_compare);
    // set random tag before oblivious shuffle
    for (auto &keyword:arr) {
        keyword.shuffle_tag = rand();
    }
    oblivious_bitonic_sort(arr, 0, kw_shuffle_cmp);
}

void keywords_padding(vector<Keyword> &keywords) {
    // get maximum length of keywords
    int max_keyword_length = 0;
    for (auto &keyword: keywords) {
        int length = keyword.content.size();
        if (length > max_keyword_length) max_keyword_length = length;
    }
    // set to the multiply of 4
    if (max_keyword_length % KEYWORD_LENGTH_ALIGN != 0)
        max_keyword_length =
                (max_keyword_length / KEYWORD_LENGTH_ALIGN + 1) *
                KEYWORD_LENGTH_ALIGN;
    // pad all keywords to max_keyword_length
    for (auto &keyword:keywords) {
        keyword.content += string(max_keyword_length - keyword.content.size(), KEYWORD_PAD_CHAR);
    }
}

void keywords_remove_padding(vector<Keyword> &keywords) {
    for (auto &keyword:keywords) {
        int loc = keyword.content.find(KEYWORD_PAD_CHAR);
        if (loc == -1) continue;
        keyword.content.resize(loc);
    }
}

vector<Keyword> oblivious_vocabulary_decide(vector<Keyword> &keywords) {
    const int FIRST_OCCUR = -1;
    const int NON_FIRST_OCCUR = 1;
    oblivious_sort(keywords, 0);
    function<bool(Keyword &, Keyword &)> kw_shuffle_cmp(keyword_shuffle_compare);
    keywords[0].shuffle_tag = FIRST_OCCUR;
    for (int i = 1; i < keywords.size(); i++) {
        int tag = oblivious_assign_CMOV(keywords[i].content != keywords[i - 1].content,
                                        FIRST_OCCUR, NON_FIRST_OCCUR);
        keywords[i].shuffle_tag = tag;
    }
    oblivious_bitonic_sort(keywords, 0, kw_shuffle_cmp);
    // find the first position of 1
    int lo = 0;
    int hi = keywords.size();
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (keywords[mid].shuffle_tag < NON_FIRST_OCCUR) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    vector<Keyword> padded_vocabulary;
    padded_vocabulary.reserve(lo);
    for (int i = 0; i < lo; i++) {
        padded_vocabulary.push_back(keywords[i]);
    }
    return padded_vocabulary;
}

int sample_gamma(float p, int alpha) {
    const float base_prob = 0.5 * (1 - p / 2) * pow(1 - p, alpha);
    float pr = (double) rand() / RAND_MAX;
    while (pr < base_prob || pr > 1 - base_prob) {
        pr = rand() / RAND_MAX;
    }
    int k = 0;
    // k<0

    double upper_bound = 0.5 * (1 - 0.5 * p);
    double tmp = -(log(0.5 * (1 - 0.5 * p)) / log(1 - p) + 1);
    int tmp_k = tmp;

    k = oblivious_assign_CMOV(pr <= upper_bound, tmp_k, k);
    // k = 0
    upper_bound = 0.5 * (1 + 0.5 * p);
    double bottom_bound = 0.5 * (1 - 0.5 * p);
    k = oblivious_assign_CMOV(pr <= upper_bound && pr > bottom_bound, 0, k);
    // k > 0
    tmp = log((1 - pr) / (0.5 * (1 - 0.5 * p))) / log(1 - p);
    tmp_k = tmp;
    if (tmp - tmp_k > 1e-8) tmp_k += 1;
    bottom_bound = 0.5 * (1 + 0.5 * p);
    k = oblivious_assign_CMOV(pr > bottom_bound, tmp_k, k);
    int gamma = k + alpha;
    if (gamma < 0 || gamma > 2 * alpha) {
        cout << "gamma out of range" << endl;
        exit(-1);
    }
    return gamma;
}

double oblivious_dummy_words_addition(vector<Keyword> &padded_vocabulary, vector<Keyword> &keywords, float epsilon, float delta) {
    const int DUMMY_TAG = -1;
    const int DUMMY_DUMMY_TAG = 1;

//    const float epsilon = 3;
//    const float delta = -32; // 2^(-32);
    float p = 1 - exp(-epsilon);
    double tmp = (delta - log2(0.5 - p / 4) - log2(padded_vocabulary.size())) / log2(1 - p);
    int alpha = tmp;
    if (tmp - alpha > 1e-8) alpha += 1;

    int true_words_size = keywords.size();
    double overhead = 0;
    for(int i=0; i<keywords.size();i++) {
        keywords[i].shuffle_tag = DUMMY_TAG;
    }

    for (int i = 0; i < padded_vocabulary.size(); ++i) {
        int gamma = sample_gamma(p, alpha);
        // generate 2*alpha dummy keywords
        overhead+=gamma;
        Keyword dummy(padded_vocabulary[i]);
        for (int j = 0; j < 2 * alpha; j++) {
            dummy.owner_id = -1;
            dummy.shuffle_tag = oblivious_assign_CMOV(j < gamma, DUMMY_TAG, DUMMY_DUMMY_TAG);
            keywords.push_back(dummy);
        }
    }
    // oblivious sort
    function<bool(Keyword &, Keyword &)> kw_shuffle_cmp(keyword_shuffle_compare);
    oblivious_bitonic_sort(keywords, 0, kw_shuffle_cmp);


    int real_count = 0;
    vector<Keyword> tmp_arr;
    for (int i = 0; i < keywords.size(); i++) {
        if (keywords[i].shuffle_tag != DUMMY_TAG) break;
        tmp_arr.push_back(std::move(keywords[i]));
//        keywords.push_back(std::move(dummy_words[i]));
    }
//    cout<<"alpha "<<alpha<<endl;
//    cout<<"vocabulary size "<<padded_vocabulary.size()<<endl;
//    cout<<"overhad "<<overhead<<endl;
//    cout<<"true words size "<<true_words_size<<endl;
    keywords = std::move(tmp_arr);
    return overhead/true_words_size;
    // obliviou shuffle
//    oblivious_shuffle(keywords);
}