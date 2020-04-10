//
// Created by anxin on 2020/4/9.
//

#ifndef TEXTTRUTH_DATASET_H
#define TEXTTRUTH_DATASET_H

#include "Member.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "boost/filesystem.hpp"

class AnswerGradingData {
public:
    static unordered_set<string> words_filter;
    vector<Question> questions;
    vector<vector<Answer>> answers;
    vector<vector<float>> scores;
    vector<int> question_per_exam;
    static const int key_factor_number;
    AnswerGradingData() {
        question_per_exam = vector<int> {
            7, 7, 7, 7,  4, 7, 7, 7, 7, 7, 10, 10
        };
    }
    void load_dataset(const string &dir_path);

private:
    void load_question(const string &question_file_path);

    void load_answers(const string &answer_file_path);

    void load_scores(const string &score_file_path);
};

#endif //TEXTTRUTH_DATASET_H

