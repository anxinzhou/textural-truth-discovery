//
// Created by anxin on 2020/4/11.
//

#ifndef TEXTTRUTH_OBLIVIOUS_TTRUTH_H
#define TEXTTRUTH_OBLIVIOUS_TTRUTH_H

#include "Member.h"
#include <random>
#include "util.h"
#include "oblivious_primitive.h"
#include <iostream>
#include <queue>


void oblivious_ttruth(vector<Question> &questions, vector<User> &users,const unordered_set<string> &words_filter ,WordModel &word_model, int top_k=5);
void oblivious_obervation_update(vector<User> &users, vector<Keyword> &keywords);

void oblivious_sphere_kmeans(vector<Keyword> &keywords, WordModel &word_model,int cluster_num, int max_iter=100, double tol= 1e-12);

void oblivious_latent_truth_model(vector<Question> &questions, vector<User> &users, int max_iter=100);

// helper function
vector<WordVec> oblivious_kmeans_init(vector<Keyword> &keywords,WordModel &word_model, int cluster_num);


#endif //TEXTTRUTH_OBLIVIOUS_TTRUTH_H
