//
// Created by anxin on 2020-04-07.
//

#ifndef TEXTTRUTH_MEMBER_H
#define TEXTTRUTH_MEMBER_H

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include "math.h"

using namespace std;
using PriorCount = vector<int>;
using Observation = vector<int>;


class Keyword {
public:
    int question_id;
    int cluster_assignment;
    int owner_id;
    string content;

    explicit Keyword(int owner_id, int question_id, string &content) : owner_id(owner_id),question_id(question_id), content(content) {};

    inline void set_cluster_assignment(int cluster_assignment) {
        this->cluster_assignment = cluster_assignment;
    }
};

class Answer {
public:
    int question_id;
    int score;
    int owner_id;
    string raw_answer;

    explicit Answer(int owner_id, int question_id, string &raw_answer) : owner_id(owner_id),question_id(question_id), raw_answer(raw_answer) {};

    vector<Keyword> to_keywords(const unordered_set<string> &words_filter, const unordered_set<string> &dictionary);

    static vector<string> string_split(const string &s);

    inline void update_score(int score) {
        this->score = score;
    }
};

class Question {
public:
    string content;
    int question_id;
    int key_factor_number;
    vector<int> truth_indicator;
    static const int BETA[2];

    Question(int question_id, string &content) : question_id(question_id), content(content) {};

    inline void set_key_factor_number(int num) {
        this->key_factor_number = num;
        truth_indicator = vector<int>(num);
    }

    inline void set_truth_indicator(int factor_index, int indicator) {
        truth_indicator[factor_index] = indicator;
    }

    inline int get_truth_indicator(int factor_index) {
        return truth_indicator[factor_index];
    }

};

const int Question::BETA[2] = {10,10};

class User {
public:
    PriorCount priorCount;
    unordered_map<int, Answer> answers;
    unordered_map<int, Observation> observations;
    static const int ALPHA[4];

    User() {priorCount = PriorCount(4);};

//    User(PriorCount & priorCount, Observation &observation):priorCount(priorCount),observation(observation){};
    inline void update_prior_count(bool key_factor_indicator, bool observation) {
        priorCount[key_factor_indicator * 2 + observation] += 1;
    }

    inline void clear_prior_count(){
        int *start = &priorCount[0];
        memset(start, 0, sizeof(int)*4);
    }

    inline int get_prior_count(bool key_factor_indicator, bool observation) {
        return priorCount[key_factor_indicator * 2 + observation];
    }

    static inline int get_alpha(bool key_factor_indicator, bool observation) {
        return ALPHA[key_factor_indicator * 2 + observation];
    }

    inline void set_answer(int question_id, Answer &ans) {
        answers[question_id] = ans;
    }

    inline void initialize_observation(int question_id, int cluster_num) {
        observations[question_id] = Observation(cluster_num);
    }

    inline void update_observation(int question_id, int cluster_num, bool observation) {
        observations[question_id][cluster_num] = observation;
    }

    inline bool get_observation(int question_id, int cluster_num, bool observation) {
        return observations[question_id][cluster_num];
    }
};

//false negative, false positive, true negative, true positive
const int User::ALPHA[4] = {50, 10, 90, 50};

#endif //TEXTTRUTH_MEMBER_H


