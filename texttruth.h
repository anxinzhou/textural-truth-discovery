//
// Created by anxin on 2020-02-21.
//

#ifndef ANONYMOUS_PAYMENT_TEXTTRUTH_H
#define ANONYMOUS_PAYMENT_TEXTTRUTH_H

#include <vector>

typedef vector<float> Keyword;
typedef vector<Keyword >  Answer;
typedef vector<int> AnswerLabel;

typedef vector<float> Cluster;

vector <AnswerLabel> sphere_kmeans(vector <Answer> &answers, int cluster_number, int max_iter = 300, float tol = 1e-4);
void texttruth(float ***all_data, int question_num, int dimension, int user_num, int k);
#endif //ANONYMOUS_PAYMENT_TEXTTRUTH_H
