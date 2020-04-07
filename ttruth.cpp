//
// Created by anxin on 2020-04-07.
//

#include "ttruth.h"

void ttruth(vector<Question> &questions, vector<User> &users) {

}

void hard_movMF(vector<Keyword> &keywords) {

}

void bayesian_movMF(vector<Keyword> &keywords) {

}

void obervation_update(vector<User> &users, vector<Keyword> &keywords) {
    //random initialize truth indicator of each question

}

void latent_truth_model(vector<Question> &questions, vector<User> &users, int max_iter) {
    //random initialize truth indicator of each question
    for (auto &question: questions) {
        vector<int> &truth_indicator = question.truth_indicator;
        for_each(truth_indicator.begin(), truth_indicator.end(), [](int &indicator) {
            indicator = rand() % 2;
        });
    }
    // gibbs sampling to infer truth
    for (int i = 0; i < max_iter; i++) {
        // clear prior count
        vector<PriorCount>old_prior_counts;

        for (int i=0; i<users.size(); ++i) {
            auto &user=users[i];
            old_prior_counts.push_back(user.priorCount);
            user.clear_prior_count();
        }

        // set new prior count
        for (auto &question:questions) {
            int question_id = question.question_id;
            auto &truth_indicator = question.truth_indicator;
            for (int j = 0; j < question.key_factor_number; j++) {
                int indicator = truth_indicator[j];
                for (auto &user: users) {
                    if(user.observations.count(question_id)==0) continue; //user doest not offer observation for this question
                    int observation = user.observations[question_id][j];
                    user.update_prior_count(indicator, observation);
                }
            }
        }

        // check if has update, so can early stop
        for(int i=0; i<users.size();++i) {

        }


        // update truth indicator
        for (auto &question:questions) {
            int question_id = question.question_id;
            auto &truth_indicator = question.truth_indicator;
            for (int j = 0; j < question.key_factor_number; j++) {
                // set initial prob to 1/2, 1/2
                double p[2] = {log(Question::BETA[0]), log(Question::BETA[1])};
                int old_indicator = truth_indicator[j];
                for(int indicator=0; indicator<=1; ++indicator) {
                    for (auto &user: users) {
                        if(user.observations.count(question_id)==0) continue; //user doest not offer observation
                        int observation = user.observations[question_id][j];
                        int alpha_observe  = User::get_alpha(indicator, observation);
                        int alpha_observe0 = User::get_alpha(indicator, 0);
                        int alpha_observe1 = User::get_alpha(indicator, 1);
                        int prior_count_observe = user.get_prior_count(indicator, observation);
                        int prior_count_observe0 = user.get_prior_count(indicator, 0);
                        int prior_count_observe1 = user.get_prior_count(indicator, 1);
                        if(old_indicator == indicator) {
                            prior_count_observe -=1;
                        }
                        p[indicator] += log((prior_count_observe+alpha_observe)/
                                (alpha_observe0+alpha_observe1+prior_count_observe0+prior_count_observe1));
                    }
                }
                // normalize probability, adjust the larger one to log 1
                int diff = 0;
                diff = p[0]>p[1]? -p[0]: -p[1];
                p[0] += diff;
                p[1] += diff;
                p[0] = exp(p[0]);
                p[1] = exp(p[1]);
                double threshold = rand()%RAND_MAX;
                if (threshold<p[0]/(p[0]+p[1])) {
                   truth_indicator[j] = 0;
                } else {
                   truth_indicator[j] = 1;
                }
            }
        }
    }
}