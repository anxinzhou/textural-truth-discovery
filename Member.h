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
#include <fstream>
#include <iostream>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "boost/filesystem.hpp"
#include "math.h"
#include "util.h"

using namespace std;
using PriorCount = vector<int>;
using Observation = vector<int>;
using WordVec = vector<float>;

class VMF {
public:
    WordVec mu;
    float alpha;
    int dimension;

    VMF(WordVec &mu, float alpha, float k, int dimension) : mu(mu), alpha(alpha), dimension(dimension) {
        set_k(k);
    };

    VMF(WordVec &&mu, float alpha, float k, int dimension) : mu(mu), alpha(alpha), dimension(dimension) {
        set_k(k);
    };

    float log_prob(const WordVec &x) {
        return log(alpha) + log_c + kappa * hpc::dot_product(mu, x);
    }

    void set_k(float k) {
        this->kappa = k;
        this->recalculate_c();
    }

    float get_k() {
        return kappa;
    }

private:
    float log_c;
    float kappa;

    /*Compute the Amos-type upper bound asymptotic approximation on H where
    log(H_\nu)(\kappa) = \int_0^\kappa R_\nu(t) dt.
    See "lH_asymptotic <-" in movMF.R and utility function implementation notes
            from https://cran.r-project.org/web/packages/movMF/index.html
            */
    void recalculate_c();

    /*Compute the antiderivative of the Amos-type bound G on the modified
    Bessel function ratio.
    Note:  Handles scalar kappa, alpha, and beta only.
    See "S <-" in movMF.R and utility function implementation notes from
    https://cran.r-project.org/web/packages/movMF/index.html
            */
    static double S(double k, double a, double b);
};

class WordModel {
public:
    unordered_map<string, WordVec> model;
    int dimension;

    WordModel() = default;

    WordModel(unordered_map<string, WordVec> &model, int dimension) : model(model), dimension(dimension) {};

    WordModel(unordered_map<string, WordVec> &&model, int dimension) : model(model), dimension(dimension) {};

    bool contain(string &word) {
        return model.count(word) != 0;
    }

    const WordVec &get_vec(string &word) {
        return model[word];
    }

    void load_raw_file(const string &path);

    void load_model(const string &path);

    void save_model(const string &path);
};

class Keyword {
public:
    int question_id;
    int cluster_assignment;
    int owner_id;
    string content;
    WordModel *word_model;

    explicit Keyword(int owner_id, int question_id, const string &content,
                     WordModel *word_model) :
            owner_id(owner_id),
            question_id(question_id),
            content(content),
            word_model(word_model) {};

    explicit Keyword(int owner_id, int question_id, string &&content,
                     WordModel *word_model) :
            owner_id(owner_id),
            question_id(question_id),
            content(content),
            word_model(word_model) {};

    const WordVec &vec() {
        return word_model->get_vec(content);
    }

    int get_dimension() {
        return word_model->dimension;
    }
};

class Answer {
public:
    int question_id;
    int owner_id;
    string raw_answer;
    float truth_score;

    explicit Answer(int owner_id, int question_id, string &raw_answer) : owner_id(owner_id), question_id(question_id),
                                                                         raw_answer(raw_answer) { truth_score = 0; };

    explicit Answer(int owner_id, int question_id, string &&raw_answer) : owner_id(owner_id), question_id(question_id),
                                                                          raw_answer(raw_answer) { truth_score = 0; };

    vector<Keyword>
    to_keywords(const unordered_set<string> &words_filter, WordModel &word_model);

    static vector<string> string_split(const string &s);

};

class Question {
public:
    string content;
    int key_factor_number;
    vector<int> truth_indicator;
    static const int BETA[2];

    vector<Answer> top_k_results;

    Question(string &content, int key_factor_number) : content(content), key_factor_number(key_factor_number),
                                                       truth_indicator(vector<int>(key_factor_number)) {};

    Question(string &&content, int key_factor_number) : content(content), key_factor_number(key_factor_number),
                                                        truth_indicator(vector<int>(key_factor_number)) {};

    void set_key_factor_number(int num) {
        this->key_factor_number = num;
    }

    void set_truth_indicator(int factor_index, int indicator) {
        truth_indicator[factor_index] = indicator;
    }

    int get_truth_indicator(int factor_index) {
        return truth_indicator[factor_index];
    }
};

class User {
public:
    PriorCount priorCount;
    unordered_map<int, Answer> answers;
    unordered_map<int, Observation> observations;
    //false negative, false positive, true negative, true positive
    static const int ALPHA[4];

    float truth_score;

    User() {
        priorCount = PriorCount(4);
        truth_score = 0;
    };

//    User(PriorCount & priorCount, Observation &observation):priorCount(priorCount),observation(observation){};
    void update_prior_count(int key_factor_indicator, int observation) {
        priorCount[key_factor_indicator * 2 + observation] += 1;
    }

    void clear_prior_count() {
        int *start = &priorCount[0];
        memset(start, 0, sizeof(int) * 4);
    }

    int get_prior_count(int key_factor_indicator, int observation) {
        return priorCount[key_factor_indicator * 2 + observation];
    }

    void update_answer(Question &question, Answer &answer) {
        answers.emplace(answer.question_id, answer);
        initialize_observation(answer.question_id, question.key_factor_number);
    }

    void update_answer(Question &question, Answer &&answer) {
        answers.emplace(answer.question_id, std::move(answer));
        initialize_observation(answer.question_id, question.key_factor_number);
    }

    static int get_alpha(int key_factor_indicator, int observation) {
        return ALPHA[key_factor_indicator * 2 + observation];
    }

private:
    void initialize_observation(int question_id, int cluster_num) {
        observations.emplace(question_id, Observation(cluster_num));
    }
};

#endif //TEXTTRUTH_MEMBER_H


