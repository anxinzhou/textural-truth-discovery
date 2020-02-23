//
// Created by anxin on 2020-02-21.
//

#include "texttruth.h"
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

vector<int>::iterator get_prior_pointer(PriorCount &prior_count, bool is_true, bool is_positive) {
    if (!is_true && is_positive) {
        return prior_count.begin() + 2;
    } else if (is_true && !is_positive) {
        return prior_count.begin();
    } else if (is_true && is_positive) {
        return prior_count.begin() + 3;
    } else {
        return prior_count.begin() + 1;
    }
}

// default prior count is zero
vector<Score> texttruth(Dataset &dataset, int try_round) {
    int cluster_num = 10;
    int question_num = dataset.size();
    int user_num = dataset[0].size();
    vector<vector<TruthLabel >> all_truth_label(question_num);
    vector<vector<PriorCount >> all_prior_count(question_num);
    vector<PriorCount> user_prior_count(user_num, vector<int>(4));
    vector<FactorLabel> question_factor_label(question_num);
    vector<vector<AnswerLabel>> all_answer_label(question_num);
    // randomly set


    // calculate user truth labels and prior counts
    for (int i = 0; i < question_num; i++) {
        vector<Answer> &answers = dataset[i];
        vector<AnswerLabel> user_answer_label = sphere_kmeans(answers, cluster_num);
        vector<TruthLabel> user_truth_label(user_num);
        for (int j = 0; j < user_num; ++j) {
            AnswerLabel &answer_label = user_answer_label[j];
            int key_word_length = answer_label.size();
            TruthLabel truth_label(cluster_num);
            for (int k = 0; k < key_word_length; ++k) {
                int label = answer_label[k];
                truth_label[label] = 1;
            }
            user_truth_label.push_back(truth_label);
        }
        all_truth_label.push_back(user_truth_label);
        all_answer_label.push_back(user_answer_label);
    }

    // initialize factor truth label to 0 or 1
    for (int i = 0; i < question_num; ++i) {
        FactorLabel &factor_label = question_factor_label[i];
        for (int j = 0; j < cluster_num; ++j) {
            factor_label.push_back(rand() % 2);
        }
    }


    // // calculate factor truth label by MCMC with fix rounds
    for (int m = 0; m < try_round; m++) {
        // set prior count
        for (int i = 0; i < user_num; ++i) {
            // x,
            PriorCount &prior_count = user_prior_count[i];
            for (int j = 0; j < question_num; ++j) {
                FactorLabel &factor_label = question_factor_label[j];
                for (int k = 0; k < cluster_num; ++k) {
                    int x = factor_label[k];
                    int y = all_truth_label[j][i][k];
                    auto it = get_prior_pointer(prior_count, x, y);
                    *it = *it + 1;
                }
            }

        }
        // calculate new factor label
        for (int i = 0; i < question_num; i++) {
            FactorLabel factor_label(cluster_num);
            for (int j = 0; j < cluster_num; j++) {
                // to avoid underflow, use log
                float p[2];
                p[0] = p[1] = 0;
                for (int x = 0; x <= 1; x++) {
                    // get y(u,qk)
                    for (int k = 0; k < user_num; ++k) {
                        int y = all_truth_label[i][k][j];
                        PriorCount &prior_count = user_prior_count[k];
                        p[x] = log(*get_prior_pointer(prior_count, x, y) /
                                   (*get_prior_pointer(prior_count, x, 0) + *get_prior_pointer(prior_count, x, 1)));
                    }
                }
                if (p[0] > p[1]) {
                    factor_label[j] = 0;
                } else {
                    factor_label[j] = 1;
                }
            }
        }
    }

    // calculate user core
    vector<Score> user_score(question_num);
    for (int i = 0; i < question_num; ++i) {
        vector<AnswerLabel> &user_answer_label = all_answer_label[i];
        FactorLabel &factor_label = question_factor_label[i];
        Score score(user_num);
        for (int j = 0; j < user_num; ++j) {
            int s = 0;
            AnswerLabel &answer_label = user_answer_label[j];
            for (int k = 0; k < answer_label.size(); ++k) {
                int label = answer_label[k];
                s += factor_label[label];
            }
        }
        user_score.push_back(score);
    }
    return user_score;
}

float similarity(vector<float> &a, vector<float> &b) {
    int n = a.size();
    float sim = 0;
    for (int i = 0; i < n; ++i) {
        sim += a[i] * b[i];
    }
    return sim;
}


vector<AnswerLabel> sphere_kmeans(vector<Answer> &answers, int cluster_number, int max_iter, float tol) {
    int user_num = answers.size();
    int dimension = answers[0][0].size();
    vector<AnswerLabel> user_answer_label;

    // randomly assign cluster label;
    for (int i = 0; i < user_num; i++) {
        int answer_size = answers[i].size();
        AnswerLabel answer_label(answer_size);
        for (int j = 0; j < answer_size; ++j) {
            answer_label[j] = rand() % cluster_number;
        }
        user_answer_label.push_back(answer_label);
    }

    vector<Cluster> clusters(cluster_number, Cluster(dimension));
    vector<Cluster> next_clusters(cluster_number, Cluster(dimension));

    for (int i = 0; i < max_iter; ++i) {
        // clear next cluster
        for (int j = 0; j < cluster_number; ++j) {
            for (int k = 0; k < dimension; ++k) {
                next_clusters[j][k] = 0;
            }
        }

        // calculate new cluster
        for (int j = 0; j < user_num; j++) {
            int answer_size = answers[j].size();
            for (int k = 0; k < answer_size; k++) {
                Keyword &keyword = answers[j][k];
                int label = user_answer_label[j][k];
                for (int p = 0; p < dimension; p++) {
                    next_clusters[k][p] += keyword[p];
                }
            }
        }

        // normalize each cluster
        for (int j = 0; j < cluster_number; j++) {
            float total = 0;
            for (int p = 0; p < dimension; ++p) {
                total += next_clusters[j][p];
            }
            total = sqrt(total);
            for (int p = 0; p < dimension; ++p) {
                next_clusters[j][p] /= total;
            }
        }

        // test stop condition
        bool stop = true;
        for (int j = 0; j < cluster_number; j++) {
            float sim = 0;
            for (int p = 0; p < dimension; ++p) {
                if (similarity(clusters[j], next_clusters[j]) < (1 - tol)) {
                    stop = false;
                    break;
                }
            }
        }
        if (stop) {
            cout << "stop at iteration " << i << endl;
            break;
        }

        // change cluster pointer
        swap(clusters, next_clusters);
        // assign new cluster //TODO

        for (int j = 0; j < user_num; ++j) {
            int answer_size = answers[j].size();
            for (int k = 0; k < answer_size; ++k) {
                float max_sim = -1;
                int max_sim_index = -1;
                Keyword &keyword = answers[j][k];
                for (int p = 0; p < cluster_number; ++p) {
                    float sim = similarity(keyword, clusters[p]);
                    if (sim > max_sim) {
                        max_sim = sim;
                        max_sim_index = p;
                    }
                }
                user_answer_label[j][k] = max_sim_index;
            }
        }
    }

    return user_answer_label;
}
