//
// Created by anxin on 2020-02-21.
//

#include "texttruth.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <cmath>
#include <unordered_set>
#include <queue>
#include <string>

using namespace std;

vector<int>::iterator get_prior_pointer(PriorCount &prior_count, bool is_true, bool is_positive) {
    if (!is_true && !is_positive) return prior_count.begin();
    if (!is_true && is_positive) return prior_count.begin() + 1;
    if (is_true && !is_positive) return prior_count.begin() + 2;
    if (is_true && is_positive) return prior_count.begin() + 3;
}

// default prior count is zero
vector<Score> texttruth(Dataset &dataset, int try_round) {
    int cluster_num = 15;
    int question_num = dataset.size();
    int user_num = dataset[0].size();
    vector<vector<TruthLabel >> all_truth_label(question_num);
//    vector<vector<PriorCount >> all_prior_count(question_num);
    vector<PriorCount> user_prior_count(user_num, vector<int>(4));
    vector<FactorLabel> question_factor_label(question_num);
    vector<vector<AnswerLabel>> all_answer_label(question_num);
    // randomly set


    // calculate user truth labels
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
            user_truth_label[j] = (truth_label);
        }
        all_truth_label[i] = (user_truth_label);
        all_answer_label[i] = (user_answer_label);
    }

    // initialize factor truth label to 0 or 1
    for (int i = 0; i < question_num; ++i) {
        FactorLabel &factor_label = question_factor_label[i];
        for (int j = 0; j < cluster_num; ++j) {
            factor_label.push_back(rand() % 2);
        }
    }

    vector<PriorCount> next_user_prior_count(user_num, PriorCount(4));
    // // calculate factor truth label by MCMC with fix rounds
    for (int m = 0; m < try_round; m++) {
        // set prior count
        // clear first
        for(int i=0;i<user_num;++i) {
            for(int j=0; j<4;++j) {
                next_user_prior_count[i][j] = 0;
            }
        }

        for (int i = 0; i < user_num; ++i) {
            // x,
            PriorCount &prior_count = next_user_prior_count[i];
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
        // test if any change, early quit
        bool equal = true;
        for(int i=0;i<user_num;++i) {
            for(int j=0; j<4;++j) {
                if(user_prior_count[i][j]!=next_user_prior_count[i][j]) {
                    equal = false;
                    break;
                }
            }
        }
        if(equal) {
            cout<<"no change to prior_count, quit at round " <<m<< endl;
            break;
        }

        swap(user_prior_count, next_user_prior_count);

        // calculate new factor label
        for (int i = 0; i < question_num; i++) {
            FactorLabel &factor_label = question_factor_label[i];
            for (int j = 0; j < cluster_num; j++) {
                // to avoid underflow, use log
                float p[2];
                p[0] = p[1] = 0;
                for (int x = 0; x <= 1; x++) {
                    // get y(u,qk)
                    for (int k = 0; k < user_num; ++k) {
                        int y = all_truth_label[i][k][j];
                        PriorCount &prior_count = user_prior_count[k];
                        float prob = (*get_prior_pointer(prior_count, x, y) + 1.0) /
                                     (*get_prior_pointer(prior_count, x, 0) + *get_prior_pointer(prior_count, x, 1) +
                                      2.0);
                        p[x] += log(prob);
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
            score[j] = s;
        }
        user_score[i] = (score);
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

vector<string> string_split(string &s) {
    vector<string> tokens;
    istringstream iss(s);
    copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(tokens));
    return tokens;
}

void word_embedding(RawDataset &raw_dataset, Dataset &dataset, unordered_map<string, Keyword> &dic) {
    unordered_set<string> stop_words = {"with", "who", "s", "above", "of", "are", "nor", "both", "have", "i",
                                        "yourself", "after", "those", "is", "their", "we", "same", "each", "she", "be",
                                        "yours", "when", "all", "only", "had", "your", "own", "against", "don", "again",
                                        "does", "other", "whom", "off", "by", "its", "her", "too", "that", "out",
                                        "through", "am", "why", "our", "between", "being", "for", "yourselves", "his",
                                        "so", "having", "under", "them", "most", "did", "there", "will", "such", "the",
                                        "as", "and", "herself", "some", "ourselves", "at", "these", "how", "than", "a",
                                        "were", "myself", "they", "an", "which", "further", "but", "not", "on", "been",
                                        "now", "more", "was", "while", "do", "to", "this", "once", "no", "during",
                                        "himself", "itself", "any", "doing", "into", "it", "here", "me", "him", "then",
                                        "what", "very", "few", "before", "if", "just", "where", "themselves", "hers",
                                        "can", "has", "below", "should", "from", "t", "in", "you", "he", "up", "until",
                                        "because", "my", "down", "or", "over", "about", "theirs", "ours"};
    int question_num = raw_dataset.size();
    int user_num = raw_dataset[0].size();
    int count = 0;
    int total = 0;
    for (int i = 0; i < question_num; ++i) {
        vector<Answer> answers;
        for (int j = 0; j < user_num; ++j) {
            Answer answer;
            string &raw_answer = raw_dataset[i][j];
            // replace punc
            replace_if(raw_answer.begin(), raw_answer.end(),
                       [](const char &c) { return std::ispunct(c); }, ' ');
            // lower alphabetic
            transform(raw_answer.begin(), raw_answer.end(), raw_answer.begin(),
                      [](unsigned char c) { return std::tolower(c); });
            auto tokens = string_split(raw_answer);
            for (auto token: tokens) {
                total += 1;
                // pass token not exist in dictionary
                if (dic.count(token) == 0) {
//                    cout << token << endl;
                    count += 1;
                    continue;
                } else {
                    answer.push_back(dic[token]);
                }
            }
            answers.push_back(answer);
        }
        dataset.push_back(answers);
    }
    cout << "tokens skipped" << " " << count << endl;
    cout << "total token" << " " << total << endl;
}

void
top_k_result(RawDataset &rawDataset, vector<Score> &all_score, vector<Score> &baseline_score, RawDataset &top_results,
             vector<Score> &top_scores, int top_k) {
    std::vector<double> test = {0.2, 1.0, 0.01, 3.0, 0.002, -1.0, -20};
    int question_num = rawDataset.size();
    std::priority_queue<std::pair<int, int>> q;
    for (int i = 0; i < question_num; i++) {
        Score &score = all_score[i];
        // ge top k index
        vector<string> question_top_answer;
        Score base_score;
        for (int j = 0; j < score.size(); ++j) {
            q.push(std::pair<int, int>(score[j], j));
        }
        for (int j = 0; j < top_k; ++j) {
            int index = q.top().second;
            question_top_answer.push_back(rawDataset[i][index]);
            base_score.push_back(baseline_score[i][index]);
            q.pop();
        }
        top_results.push_back(question_top_answer);
        top_scores.push_back(base_score);
    }
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
                    next_clusters[label][p] += keyword[p];
                }
            }
        }

        // normalize each cluster
        for (int j = 0; j < cluster_number; j++) {
            float total = 0;
            for (int p = 0; p < dimension; ++p) {
                total += next_clusters[j][p] * next_clusters[j][p];
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
                }
            }
        }

        if (stop) {
            if (i < 2) {
                cout << "early stop, stop at iteration " << i << endl;
                break;
            }
        }

        // change cluster pointer
        swap(clusters, next_clusters);
        // assign new cluster //TODO

        for (int j = 0; j < user_num; ++j) {
            int answer_size = answers[j].size();
            for (int k = 0; k < answer_size; ++k) {
                float max_sim = -2;
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