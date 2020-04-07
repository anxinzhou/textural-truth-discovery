//
// Created by anxin on 2020-04-07.
//

#ifndef TEXTTRUTH_TTRUTH_H
#define TEXTTRUTH_TTRUTH_H


#include "Member.h"

void ttruth(vector<Question> &questions, vector<User> &users);

void obervation_update(vector<User> &users, vector<Keyword> &keywords);


// define two  clustering method
void hard_movMF(vector<Keyword> &keywords);

void bayesian_movMF(vector<Keyword> &keywords);

void latent_truth_model(vector<Question>&questions, vector<User>&users,int max_iter);



#endif //TEXTTRUTH_TTRUTH_H

