//
// Created by anxin on 2020-02-21.
//

#ifndef ANONYMOUS_PAYMENT_TEXTTRUTH_H
#define ANONYMOUS_PAYMENT_TEXTTRUTH_H

#include <vector>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <cmath>
#include <unordered_set>
#include <queue>
#include <string>
#include <immintrin.h>
#include "util.h"

using namespace std;


typedef vector<float> Keyword;  // A single keyword
typedef vector<Keyword> Answer;
typedef vector<int> AnswerLabel; // cluster label of a answer, sizeof keyword length
typedef vector<int> TruthLabel; // indicator truth label for user, sizeof cluster
typedef vector<int> FactorLabel; // for cluster factor  , size of k clusters
typedef vector<float> Score;  // scores for user for a questions, sizeof user num

typedef vector<float> Cluster;  // A cluster
typedef vector<vector<Answer >> Dataset;
typedef vector<vector<string> > RawDataset;
typedef vector<int> PriorCount; // n(0,0), n(0,1), n(1,0), n(1,1)

vector<AnswerLabel> sphere_kmeans(vector<Answer> &answers, int cluster_number, const string & strategy="kmeans++",int max_iter = 40, float tol = 1e-8);

vector<Score> texttruth(Dataset &dataset, int try_round = 1);

void word_embedding(RawDataset &raw_dataset, Dataset &dataset, unordered_map<string, Keyword> &dic);

void
top_k_result(RawDataset &rawDataset, vector<Score> &all_score, vector<Score> &baseline_score, RawDataset &top_results,
             vector<Score> &top_scores, int top_k = 3);
float similarity(vector<float> &a, vector<float> &b);
#endif //ANONYMOUS_PAYMENT_TEXTTRUTH_H
