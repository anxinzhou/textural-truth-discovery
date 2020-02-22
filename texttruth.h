//
// Created by anxin on 2020-02-21.
//

#ifndef ANONYMOUS_PAYMENT_TEXTTRUTH_H
#define ANONYMOUS_PAYMENT_TEXTTRUTH_H

#include <vector>

typedef vector<float> Keyword;
typedef vector<Keyword >  Answer;
typedef vector<int> AnswerLabel;
typedef vector<int> TruthLabel;

typedef vector<float> Cluster;
typedef vector<vector<Answer >> Dataset;
typedef uint[4] PriorCount;

vector <AnswerLabel> sphere_kmeans(vector <Answer> &answers, int cluster_number, int max_iter = 300, float tol = 1e-4);
#endif //ANONYMOUS_PAYMENT_TEXTTRUTH_H
void texttruth(Dataset &dataset);