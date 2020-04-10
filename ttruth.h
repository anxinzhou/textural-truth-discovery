//
// Created by anxin on 2020-04-07.
//

#ifndef TEXTTRUTH_TTRUTH_H
#define TEXTTRUTH_TTRUTH_H


#include "Member.h"
#include <random>
#include "util.h"
#include <iostream>
#include <queue>

void ttruth(vector<Question> &questions, vector<User> &users,const unordered_set<string> &words_filter ,WordModel &word_model, int top_k=5);

void obervation_update(vector<User> &users, vector<Keyword> &keywords);

// define two  clustering method
void hard_movMF(vector<Keyword> &keywords, int cluster_num, int max_iter=100, double tol = 1e-12);
void sphere_kmeans(vector<Keyword> &keywords, int cluster_num, int max_iter=100, double tol= 1e-12);

void latent_truth_model(vector<Question> &questions, vector<User> &users, int max_iter=20);

// helper function
vector<WordVec> kmeans_init(vector<Keyword> &keywords, int cluster_num);


#endif //TEXTTRUTH_TTRUTH_H

