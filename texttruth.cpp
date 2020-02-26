//
// Created by anxin on 2020-02-21.
//

#include "texttruth.h"


using namespace std;

float similarity(vector<float> &a, vector<float> &b) {
    int n = a.size();
    float sim = 0;
    for (int i = 0; i < n; ++i) {
        sim += a[i] * b[i];
    }

    return sim;
}

vector<int>::iterator get_prior_pointer(PriorCount &prior_count, bool is_true, bool is_positive) {
    if (!is_true && !is_positive) return prior_count.begin();
    if (!is_true && is_positive) return prior_count.begin() + 1;
    if (is_true && !is_positive) return prior_count.begin() + 2;
    if (is_true && is_positive) return prior_count.begin() + 3;
    throw "unexpected position";
}

// default prior count is zero
vector<Score> texttruth(Dataset &dataset, int try_round) {
    int cluster_num = 8;
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
        vector<AnswerLabel> user_answer_label = sphere_kmeans(answers, cluster_num, "kmeans++");
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
    // // calculate factor truth label by EM
    for (int m = 0; m < try_round; m++) {
        // set prior count
        // clear first
        for (int i = 0; i < user_num; ++i) {
            for (int j = 0; j < 4; ++j) {
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
        for (int i = 0; i < user_num; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (user_prior_count[i][j] != next_user_prior_count[i][j]) {
                    equal = false;
                    break;
                }
            }
        }
        if (equal) {
            cout << "no change to prior_count, quit at round " << m << endl;
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

void kmeans_init(vector<Answer> &answers, int cluster_number, vector<Keyword> &clusters, string strategy) {
    // initialize by kmeans++
    int user_num = answers.size();
    int dimension = answers[0][0].size();
    if (strategy == "kmeans++") {
        // random chooose first point
        int user;
        int size = 0;
        do {
            user = rand() % user_num;
            size = answers[user].size();
        } while (size == 0);
        int index = rand() % size;
        memcpy(&clusters[0][0], &answers[user][index][0], sizeof(float) * dimension);

        // init cache
        vector<vector<float>> min_distance_cache(user_num);
        vector<vector<int>> min_distance_index_cache(user_num);

        for(int i=0;i<user_num;i++){
            int answer_size = answers[i].size();
            min_distance_cache[i]= vector<float>(answer_size);
            min_distance_index_cache[i] = vector<int>(answer_size);
        }

        const int MIN_SIM = 10;
        const int MAX_SIM = -10;
        for (int i = 0; i < cluster_number-1; i++) {
            int global_max_user_index = -1;
            float global_max_distance = MAX_SIM;
            int global_max_keywrod_index = -1;
            for (int j = 0; j < user_num; ++j) {
                int answer_size = answers[j].size();
                auto &user_distance_min_cache = min_distance_cache[j];
                auto & user_min_distance_index_cache = min_distance_index_cache[j];
                for (int k = 0; k < answer_size; ++k) {
                    float min_sim = MIN_SIM;
                    int min_keyword_index = -1;
                    if (i != 0) {
                        min_sim = user_distance_min_cache[k];
                        min_keyword_index = user_min_distance_index_cache[k];
                    }

                    Keyword &keyword = answers[j][k];
                    float sim = 1 - hpc::dot_product(keyword, clusters[i]);
                    if (min_sim > sim) {
                        min_sim = sim;
                        min_keyword_index = k;
                    }
                    if (min_sim > global_max_distance) {
                        global_max_distance = min_sim;
                        global_max_user_index = j;
                        global_max_keywrod_index = min_keyword_index;
                    }

                    user_distance_min_cache[k] = min_sim;
                    user_min_distance_index_cache[k] = min_keyword_index;
                }
            }
//            cout<< global_max_distance <<" "<<global_max_user_index << " "<<global_max_keywrod_index<<endl;
            memcpy(&clusters[i+1][0], &answers[global_max_user_index][global_max_keywrod_index][0],
                   sizeof(float) * dimension);
        }

    } else {
        // use random
        for (int i = 0; i < cluster_number; i++) {

            int user;
            int size = 0;
            do {
                user = rand() % user_num;
                size = answers[user].size();
            } while (size == 0);
            int index = rand() % size;
            memcpy(&clusters[i][0], &answers[user][index][0], sizeof(float) * dimension);
        }
    }
}

vector<AnswerLabel>
sphere_kmeans(vector<Answer> &answers, int cluster_number, const string &strategy, int max_iter, float tol) {
    int user_num = answers.size();
    int dimension = answers[0][0].size();
    vector<AnswerLabel> user_answer_label(user_num);
    vector<Cluster> clusters(cluster_number, Cluster(dimension));
    vector<Cluster> next_clusters(cluster_number, Cluster(dimension));
    for (int i = 0; i < user_num; ++i) {
        user_answer_label[i] = AnswerLabel(answers[i].size());
    }

    kmeans_init(answers, cluster_number, clusters, strategy);


    for (int i = 0; i < max_iter; ++i) {
        // assign new cluster label

        for (int j = 0; j < user_num; ++j) {
            int answer_size = answers[j].size();
            for (int k = 0; k < answer_size; ++k) {
                float max_sim = -2;
                int max_sim_index = -1;
                Keyword &keyword = answers[j][k];
                for (int p = 0; p < cluster_number; ++p) {
                    float sim = hpc::dot_product(keyword, clusters[p]);
                    if (sim > max_sim) {
                        max_sim = sim;
                        max_sim_index = p;
                    }
                }
                user_answer_label[j][k] = max_sim_index;
            }
        }

        // clear next cluster
        for (int j = 0; j < cluster_number; ++j) {
            memset(&next_clusters[j][0], 0, dimension * sizeof(float));
        }

        // calculate new cluster
        for (int j = 0; j < user_num; j++) {
            int answer_size = answers[j].size();
            for (int k = 0; k < answer_size; k++) {
                Keyword &keyword = answers[j][k];
                int label = user_answer_label[j][k];
                hpc::vector_add_inplace(next_clusters[label], keyword);
            }
        }

        // normalize each cluster
        for (int j = 0; j < cluster_number; j++) {
            float total = hpc::dot_product(next_clusters[j], next_clusters[j]);
            if (total == 0) {
                // no item belongs to this cluter, use last one
                memcpy(&next_clusters[j][0], &clusters[j][0], sizeof(float) * dimension);
                total = hpc::dot_product(next_clusters[j], next_clusters[j]);
            }
            total = sqrt(total);
            hpc::vector_mul_inplace(next_clusters[j], 1.0 / total);
        }

        // test stop condition
        bool stop = true;
        for (int j = 0; j < cluster_number; j++) {
            if (hpc::dot_product(clusters[j], next_clusters[j]) < (1 - tol)) {
                stop = false;
                break;
            }
//            float result = (hpc::dot_product(clusters[j], next_clusters[j]));
//            if (isnan(result)) {
//                for (int k = 0; k < dimension; k++) {
//                    cout << clusters[j][k] << " " << next_clusters[j][k] << endl;
//                }
//                exit(0);
//            }
        }

        if (stop) {
            cout << "early stop, stop at iteration " << i << endl;
            break;
//            if (i < 2) {
//                cout << "early stop, stop at iteration " << i << endl;
//                break;
//            }
        }

        // change cluster pointer
        swap(clusters, next_clusters);
    }

    return user_answer_label;
}