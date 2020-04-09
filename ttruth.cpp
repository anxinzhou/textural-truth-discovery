//
// Created by anxin on 2020-04-07.
//

#include "ttruth.h"

void ttruth(vector<Question> &questions, vector<User> &users, const unordered_set<string> &words_filter,
            WordModel &word_model, int top_k) {
    // prepare keywords
    vector<vector<Keyword>> question_keywords;
    for (int question_id = 0; question_id < questions.size(); question_id++) {
        vector<Keyword> keywords;
        for (auto &user: users) {
            if (user.answers.count(question_id) == 0) continue;
            auto &answer = user.answers.at(question_id);
            auto user_keywords = answer.to_keywords(words_filter, word_model);
            keywords.insert(keywords.end(), user_keywords.begin(), user_keywords.end());
        }
        // vmf clustering
        hard_movMF(keywords, questions[question_id].key_factor_number);
        // update observation
        obervation_update(users, keywords);
        question_keywords.push_back(std::move(keywords));
    }

    //latent truth discovery
    latent_truth_model(questions, users);

    // calculate score of each answer
    for (auto &keywords: question_keywords) {
        for (auto &keyword: keywords) {
            int question_id = keyword.question_id;
            int owner_id = keyword.owner_id;
            int cluster_assignment = keyword.cluster_assignment;
            auto &answer = users[owner_id].answers.at(question_id);
            if (questions[question_id].truth_indicator[cluster_assignment] == 1) {
                answer.truth_score += 1;
            }
        }
    }
    // normalize score for each answer
    for (int i = 0; i < questions.size(); i++) {
        double total_score = 0;
        for (auto &user:users) {
            if (user.answers.count(i) == 0) continue; // not offer solution, skip
            total_score += user.answers.at(i).truth_score;
        }
        for (auto &user:users) {
            if (user.answers.count(i) == 0) continue; // not offer solution, skip
            if (total_score == 0) continue;
            user.answers.at(i).truth_score /= total_score;
        }
    }

    // calculate truth score of user
    for (auto &user:users) {
        auto &answers = user.answers;
        for (auto it = answers.begin(); it != answers.end(); it++) {
            auto &answer = it->second;
            user.truth_score += answer.truth_score;
        }
    }

    // calculate truth of question
    auto cmp = [](pair<double, int> &a, pair<double, int> &b) { return a.first < b.first; };
    for (int i = 0; i < questions.size(); i++) {
        vector<Answer> top_k_results;
        priority_queue<pair<double, int>, vector<pair<double, int>>, decltype(cmp)> q(cmp);
        for (int j = 0; j < users.size(); ++j) {
            auto &user = users[j];
            if (user.answers.count(i) == 0) continue; // not offer solution, skip
            q.emplace(user.answers.at(i).truth_score, j);
        }
        for (int j = 0; j < top_k; ++j) {
            int user_index = q.top().second;
            auto &answer = users[user_index].answers.at(i);
            top_k_results.push_back(answer);
            q.pop();
        }
        questions[i].top_k_results = std::move(top_k_results);
    }

}

inline float distance_metrics(const WordVec &a, const WordVec &b) {
    return 2 - 2 * hpc::dot_product(a, b);
}

void hard_movMF(vector<Keyword> &keywords, int cluster_num, int max_iter, double tol) {
    int dimension = 0;
    if (keywords.size() != 0) dimension = keywords[0].get_dimension();
    // init with kmeans ++
    vector<WordVec> mus = kmeans_init(keywords, cluster_num, dimension);
    vector<VMF> movMF;


    for (int i = 0; i < cluster_num; i++) {
        movMF.emplace_back(std::move(mus[i]), 1.0 / cluster_num, 1, dimension);
    }

    for (int t = 0; t < max_iter; t++) {
        // e step
        // assign new cluster
        for (auto &keyword: keywords) {
            float max_prob = INT_MIN;
            int max_index = -1;
            for (int i = 0; i < movMF.size(); i++) {
                auto &vmf = movMF[i];
                float prob = vmf.log_prob(keyword.vec());
                if (prob > max_prob) {
                    max_prob = prob;
                    max_index = i;
                }
            }
            keyword.cluster_assignment = max_index;
        }
        // M step
        // update parameters
        vector<float> new_alpha(cluster_num, 0);
        vector<WordVec> new_mus(cluster_num, WordVec(dimension, 0));
        for (auto &keyword: keywords) {
            int cluster_assignment = keyword.cluster_assignment;
            new_alpha[cluster_assignment] += 1;
            hpc::vector_add_inplace(new_mus[cluster_assignment], keyword.vec());
        }
        // normalize alpha
        for (int i = 0; i < cluster_num; i++) {
            auto &alpha = new_alpha[i];
            alpha = alpha / keywords.size();
            movMF[i].alpha = alpha;
        }
        // update k
        for (int i = 0; i < cluster_num; i++) {
            auto alpha = new_alpha[i];
            auto &mu = new_mus[i];
            if (alpha == 0) continue;
            double mu_l2 = sqrt(hpc::dot_product(mu, mu));
            double r = mu_l2 / (keywords.size() * alpha);
            float k; // set a large k if r == 1
            if (abs(r - 1) < 1e-10) k = 1e10;
            else {
                k = (r * dimension - pow(r, 3)) / (1 - pow(r, 2));
            }
            movMF[i].set_k(k);
        }
        //normalize mu
        for (int i = 0; i < cluster_num; i++) {
            auto &mu = new_mus[i];
            float mu_l2 = sqrt(hpc::dot_product(mu, mu));
            if (mu_l2 == 0) continue; // empty cluster
            hpc::vector_mul_inplace(mu, 1.0 / mu_l2);
        }
        // check stop tol
        double cur_tol = 0;
        for (int i = 0; i < cluster_num; i++) {
            WordVec diff = hpc::vector_sub(new_mus[i], movMF[i].mu);
            double square_norm = sqrt(hpc::dot_product(diff, diff));
            cur_tol += square_norm;
        }

        if (cur_tol < tol) {
//            cout << "vmf clustering converge at iteration " << t << endl;
            break;
        }

        //set new mu
        for (int i = 0; i < cluster_num; i++) {
            auto &mu = new_mus[i];
            movMF[i].mu = std::move(mu);
        }
    }

}

void obervation_update(vector<User> &users, vector<Keyword> &keywords) {
    for (auto &keyword: keywords) {
        int owner_id = keyword.owner_id;
        int question_id = keyword.question_id;
        int cluster_assignment = keyword.cluster_assignment;
        users[owner_id].observations.at(question_id)[cluster_assignment] = 1;
    }
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
    for (int t = 0; t < max_iter; t++) {
        // clear prior count
//        vector<PriorCount> old_prior_counts;

        for (auto &user : users) {
//            old_prior_counts.push_back(user.priorCount);
            user.clear_prior_count();
        }

        // set new prior count
        for (int i = 0; i < questions.size(); i++) {
            auto &question = questions[i];
            int question_id = i;
            auto &truth_indicator = question.truth_indicator;
            for (int j = 0; j < question.key_factor_number; j++) {
                int indicator = truth_indicator[j];
                for (auto &user: users) {
                    if (user.observations.count(question_id) == 0)
                        continue; //user doest not offer observation for this question
                    int observation = user.observations.at(question_id)[j];
                    user.update_prior_count(indicator, observation);
                }
            }
        }

        // check if has update, so can early stop
//        bool equal = true;
//        int total_diff =0;
//        for (int i = 0; i < users.size(); ++i) {
//            auto &old_prior = old_prior_counts[i];
//            auto &new_prior = users[i].priorCount;
//            for (int j = 0; j < 4; j++) {
//                total_diff += abs(old_prior[j] - new_prior[j]);
//            }
//        }
//        if (equal) {
//            cout<<"latent truth discovery converge at iteration "<<t<<endl;
//            break;
//        };


        // update truth indicator
        for (int i = 0; i < questions.size(); i++) {
            auto &question = questions[i];
            int question_id = i;
            auto &truth_indicator = question.truth_indicator;
            for (int j = 0; j < question.key_factor_number; j++) {

                double p[2] = {log(Question::BETA[0]), log(Question::BETA[1])};
                int old_indicator = truth_indicator[j];
                for (int indicator = 0; indicator <= 1; ++indicator) {
                    for (auto &user: users) {
                        if (user.observations.count(question_id) == 0) continue; //user doest not offer observation
                        int observation = user.observations.at(question_id)[j];
                        int alpha_observe = User::get_alpha(indicator, observation);
                        int alpha_observe0 = User::get_alpha(indicator, 0);
                        int alpha_observe1 = User::get_alpha(indicator, 1);
                        int prior_count_observe = user.get_prior_count(indicator, observation);
                        int prior_count_observe0 = user.get_prior_count(indicator, 0);
                        int prior_count_observe1 = user.get_prior_count(indicator, 1);
                        if (old_indicator == indicator) {
                            prior_count_observe -= 1;
                        }
                        p[indicator] += log((double) (prior_count_observe + alpha_observe) /
                                            (alpha_observe0 + alpha_observe1 + prior_count_observe0 +
                                             prior_count_observe1));
                    }
                }
//                truth_indicator[j] = p[0] > p[1] ? 0:1;
//                 normalize probability, adjust the larger one to log 1
                double diff = 0;
                diff = p[0] > p[1] ? -p[0] : -p[1];
                p[0] += diff;
                p[1] += diff;
                p[0] = exp(p[0]);
                p[1] = exp(p[1]);
                double threshold = (double) (rand()) / RAND_MAX;
                if (threshold < p[0] / (p[0] + p[1])) {
                    truth_indicator[j] = 0;
                } else {
                    truth_indicator[j] = 1;
                }
            }
        }
    }
}

vector<WordVec> kmeans_init(vector<Keyword> &keywords, int cluster_num, int dimension) {
    auto rnd1 = std::mt19937(std::random_device{}());

    vector<WordVec> clusters;
    int first_index = rnd1() % keywords.size();
    clusters.push_back(keywords[first_index].vec());

    vector<float> distance_cache(keywords.size(), INT_MAX);


    for (int i = 1; i < cluster_num; i++) {
        float max_dis = INT_MIN;
        int max_index = -1;
        for (int j = 0; j < keywords.size(); j++) {
            auto &keyword = keywords[j];
            float cache_min = distance_cache[j];
            float cur_dis = distance_metrics(keyword.vec(), clusters[clusters.size() - 1]);
            cache_min = min(cache_min, cur_dis);
            distance_cache[j] = cache_min;
            if (cache_min > max_dis) {
                max_dis = cache_min;
                max_index = j;
            }
        }
        clusters.push_back(keywords[max_index].vec());
    }
    return clusters;
}