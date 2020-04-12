//
// Created by anxin on 2020/4/11.
//

#include "oblivious_ttruth.h"


void oblivious_ttruth(vector<Question> &questions, vector<User> &users, const unordered_set<string> &words_filter,
                      WordModel &word_model, int top_k) {
    auto keyword_shuffle_compare = [](Keyword &a, Keyword &b) -> bool {
        return a.shuffle_tag < b.shuffle_tag;
    };

    // prepare keywords
    vector<vector<Keyword>> question_keywords;
    for (int question_id = 0; question_id < questions.size(); question_id++) {
        vector<Keyword> keywords;
        // process keywords first
        for (auto &user: users) {
            if (user.answers.count(question_id) == 0) continue;
            auto &answer = user.answers.at(question_id);
            auto user_keywords = answer.to_keywords(word_model);
            keywords.insert(keywords.end(), user_keywords.begin(), user_keywords.end());
        }
        // pad keywords to same length
        keywords_padding(keywords);

        // keywords space decide
        auto padded_vocabulary = oblivious_vocabulary_decide(keywords);
        // oblivious dummy words addition
        oblivious_dummy_words_addition(padded_vocabulary, keywords);
        // remove padding
        keywords_remove_padding(keywords);
        // filter keywords;
        {
            vector<Keyword> tmp_keywords;
            for (auto &keyword: keywords) {
                if (words_filter.count(keyword.content) == 1) continue;
                if (!word_model.contain(keyword.content)) continue;
                tmp_keywords.push_back(std::move(keyword));
            }
            keywords = std::move(tmp_keywords);
        }

        // vmf clustering
        oblivious_sphere_kmeans(keywords, word_model, questions[question_id].key_factor_number);

        // remove dummy words
        //shuffle to hide how many counts dummy for each word
        keywords_padding(keywords);
        oblivious_shuffle(keywords);
        {
            vector<Keyword> tmp;
            for (auto &keyword: keywords) {
                if (keyword.owner_id == -1) continue;
                tmp.push_back(std::move(keyword));
            }
            keywords = std::move(tmp);
        }
        keywords_remove_padding(keywords);
        // update observation
        oblivious_obervation_update(users, keywords);

        question_keywords.push_back(std::move(keywords));
    }

    //latent truth discovery
    oblivious_latent_truth_model(questions, users);

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
        for (auto & it : answers) {
            auto &answer = it.second;
            user.truth_score += answer.truth_score;
        }
    }

    // calculate truth of question
    auto cmp = [](pair<double, int> &a, pair<double, int> &b) { return a.first < b.first; };
    for (int i = 0; i < questions.size(); i++) {
        vector<Answer> top_k_results;
        vector<pair<double, int>> q;
        for (int j = 0; j < users.size(); ++j) {
            auto &user = users[j];
            if (user.answers.count(i) == 0) continue; // not offer solution, skip
            q.emplace_back(user.answers.at(i).truth_score, j);
        }
        oblivious_sort(q, 1);
        for (int j = 0; j < top_k; ++j) {
            int user_index = q[j].second;
            auto &answer = users[user_index].answers.at(i);
            top_k_results.push_back(answer);
        }
        questions[i].top_k_results = std::move(top_k_results);
    }
}

void
oblivious_sphere_kmeans(vector<Keyword> &keywords, WordModel &word_model, int cluster_num, int max_iter, double tol) {
    int dimension = 0;
    if (keywords.size() != 0) dimension = word_model.dimension;
    // init with kmeans ++
    vector<WordVec> clusters = oblivious_kmeans_init(keywords, word_model, cluster_num);

    vector<const WordVec *> keyword_vec;
    keyword_vec.reserve(keywords.size());

    for (int i = 0; i < keywords.size(); i++) {
        keyword_vec.push_back(&(word_model.get_vec(keywords[i].content)));
    }


    for (int t = 0; t < max_iter; t++) {
        // e step
        // assign new cluster
        for (int i = 0; i < keywords.size(); i++) {
            auto &keyword = keywords[i];
            float max_prob = INT_MIN;
            int max_index = -1;
            for (int k = 0; k < clusters.size(); k++) {
                float prob = hpc::dot_product(clusters[k], *keyword_vec[i]);
                if (prob > max_prob) {
                    max_prob = prob;
                    max_index = k;
                }
            }
            keyword.cluster_assignment = max_index;
        }
        // M step
        // update parameters
        vector<WordVec> new_clusters(cluster_num, WordVec(dimension, 0));
        for (int i = 0; i < keywords.size(); i++) {
            auto &keyword = keywords[i];
            int cluster_assignment = keyword.cluster_assignment;
            WordVec tmp(*keyword_vec[i]);
            hpc::vector_mul_inplace(tmp, keyword.owner_id != -1);
            hpc::vector_add_inplace(new_clusters[cluster_assignment], tmp);
        }

        //normalize mu
        for (int i = 0; i < cluster_num; i++) {
            auto &mu = new_clusters[i];
            float mu_l2 = sqrt(hpc::dot_product(mu, mu));
            if (mu_l2 < 1e-8) continue; //
            hpc::vector_mul_inplace(mu, 1.0 / mu_l2);
        }
        // check stop tol
        double cur_tol = 0;
        for (int i = 0; i < cluster_num; i++) {
            WordVec diff = hpc::vector_sub(new_clusters[i], clusters[i]);
            double square_norm = sqrt(hpc::dot_product(diff, diff));
            cur_tol += square_norm;
        }

        if (cur_tol < tol) {
//            cout << "sphere clustering converge at iteration " << t << endl;
            break;
        }

        //set new cluster
        for (int i = 0; i < cluster_num; i++) {
            clusters[i] = std::move(new_clusters[i]);
        }
    }

//     see how many words each cluster have
//    vector<int> cluster_histogram(cluster_num, 0);
//    for (int k = 0; k < keywords.size(); k++) {
//        if (keywords[k].owner_id == -1) continue;
//        cluster_histogram[keywords[k].cluster_assignment] += 1;
//    }
//    for (int k = 0; k < cluster_histogram.size(); k++) {
//        cout << cluster_histogram[k] << " ";
//    }
//    cout << endl;
}

inline float distance_metrics(const WordVec &a, const WordVec &b) {
    return abs(2 - 2 * hpc::dot_product(a, b));
}

vector<WordVec> oblivious_kmeans_init(vector<Keyword> &keywords, WordModel &word_model, int cluster_num) {
    auto rnd1 = std::mt19937(std::random_device{}());

    vector<const WordVec *> keyword_vec;
    keyword_vec.reserve(keywords.size());

    for (int i = 0; i < keywords.size(); i++) {
        keyword_vec.push_back(&(word_model.get_vec(keywords[i].content)));
    }

    vector<WordVec> clusters;
    int first_index = rnd1() % keywords.size();
    clusters.push_back(*keyword_vec[first_index]);
    vector<float> min_dis_cache(keywords.size(), INT_MAX);
    vector<float> cluster_distance(keywords.size());
//    vector<int> clusters_index(cluster_num);
//    clusters_index[0] = first_index;
    for (int i = 1; i < cluster_num; i++) {
        float total_dis = 0;
        for (int j = 0; j < keywords.size(); j++) {
            auto &keyword = keywords[j];
            float min_dis = min_dis_cache[j];
            float dis = distance_metrics(*keyword_vec[j], clusters[i - 1]);
            min_dis = min(min_dis, dis);
            min_dis_cache[j] = min_dis;
            min_dis *= (keyword.owner_id != -1);
            cluster_distance[j] = min_dis;
            total_dis += min_dis;
        }
        // normalize distance
        for (auto &cd: cluster_distance) {
            cd /= total_dis;
        }
        // sample from distribution
        for (int j = 1; j < cluster_distance.size(); j++) {
            cluster_distance[j] += cluster_distance[j - 1];
        }
        cluster_distance[cluster_distance.size() - 1] = 1;
        double prob = (double) rand() / RAND_MAX;
        int cluster_index = -1;
        cluster_index = oblivious_assign_CMOV(prob <= cluster_distance[0], 0, cluster_index);
        bool larger = false;
        bool first_occur = true;
        for (int j = 1; j < cluster_distance.size(); j++) {
            larger |= prob > cluster_distance[j - 1];
            bool flag = larger && prob < cluster_distance[j] && keywords[j].owner_id != -1 && first_occur;
            first_occur = oblivious_assign_CMOV(flag, false, first_occur);
            cluster_index = oblivious_assign_CMOV(flag, j, cluster_index);
        }
//        clusters_index[i] = cluster_index;
        clusters.push_back(*keyword_vec[cluster_index]);
    }
    return clusters;
}

void oblivious_obervation_update(vector<User> &users, vector<Keyword> &keywords) {

    for (auto &keyword: keywords) {
        int owner_id = keyword.owner_id;
        int question_id = keyword.question_id;
        int cluster_assignment = keyword.cluster_assignment;
        users[owner_id].observations.at(question_id)[cluster_assignment] = 1;
    }
}

void oblivious_latent_truth_model(vector<Question> &questions, vector<User> &users, int max_iter) {
    //random initialize truth indicator of each question and user prior count
    for (int i = 0; i < questions.size(); i++) {
        auto &question = questions[i];
        vector<int> &truth_indicator = question.truth_indicator;
        for (int j = 0; j < truth_indicator.size(); j++) {
            int indicator = rand() % 2;
            truth_indicator[j] = indicator;
            for (auto &user:users) {
                if (user.observations.count(i) == 0) continue;
                int index = indicator * 2 + user.observations.at(i)[j];
                for (int k = 0; k < 4; k++) {
                    user.priorCount[k] += 1 * (index == k);
                }
            }
        }
    }
    // gibbs sampling to infer truth
    for (int t = 0; t < max_iter; t++) {
        // clear prior count
//        vector<PriorCount> old_prior_counts;

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
                        int alpha_observe0 = User::get_alpha(indicator, 0);
                        int alpha_observe1 = User::get_alpha(indicator, 1);
                        int alpha_observe = oblivious_assign_CMOV(observation, alpha_observe1, alpha_observe0);
                        int prior_count_observe0 = user.get_prior_count(indicator, 0);
                        int prior_count_observe1 = user.get_prior_count(indicator, 1);
                        int prior_count_observe = oblivious_assign_CMOV(observation, prior_count_observe1,
                                                                        prior_count_observe0);
                        int tmp = 1 * (old_indicator == indicator);
                        prior_count_observe -= tmp;
                        p[indicator] += log((double) (prior_count_observe + alpha_observe) /
                                            (alpha_observe0 + alpha_observe1 + prior_count_observe0 +
                                             prior_count_observe1));
                    }
                }
//                truth_indicator[j] = p[0] > p[1] ? 0:1;
//                 normalize probability, adjust the larger one to log 1
                double diff = 0;
                const int scale_factor = 1e8;
                int diff_tmp = oblivious_assign_CMOV(p[0] > p[1], p[0] * scale_factor, p[1] * scale_factor);
                diff = -(double) diff_tmp / scale_factor;
                p[0] += diff;
                p[1] += diff;
                p[0] = exp(p[0]);
                p[1] = exp(p[1]);
                double threshold = (double) (rand()) / RAND_MAX;

                int new_indicator = oblivious_assign_CMOV(threshold < p[0] / (p[0] + p[1]), 0, 1);
                truth_indicator[j] = new_indicator;
                for (auto &user:users) {
                    if (user.observations.count(i) == 0) continue;
                    int add_index = new_indicator * 2 + user.observations.at(i)[j];
                    int decline_index = old_indicator * 2 + user.observations.at(i)[j];
                    for (int k = 0; k < 4; k++) {
                        user.priorCount[k] += 1 * (add_index == k);
                        user.priorCount[k] -= 1 * (decline_index == k);
                    }
                }

            }
        }
    }
}