//
// Created by anxin on 2020-02-21.
//

#include "texttruth.h"


void texttruth(Dataset &dataset) {
    int cluster_number = 10;
    int question_num = dataset.size();
    int user_num = dataset[0].size();
    vector <vector<TruthLabel >> all_truth_label(question_num);
    vector <vector<PriorCount >> all_prior_count(question_num);

    for (int i = 0; i < question_num; i++) {
        vector <Answer> &answers = dataset[i];
        vector <AnswerLabel> user_answer_label = sphere_kmeans(answers, cluster_number);
        vector <TruthLabel> user_truth_label(user_num);
        for (int j = 0; j < user_num; ++j) {
            AnswerLabel &answer_label = user_answer_label[j];
            int key_word_length = answer_label.size();
            TruthLabel truth_label(cluster_num);
            for (int k = 0; k < key_word_length; ++k) {
                int label = answer_label[k];
                truth_label[label] = 1;
            }
            user_truth_label.append(truth_label);
        }
        all_truth_label.append(user_truth_label);
    }


}

int *GetPriorPointer(uint[4] prior_count, bool is_true, bool is_positive) {
    if (!is_true && is_positive) {
        return &prior_count[2];
    } else if (is_true && !is_positive) {
        return &prior_count[0];
    } else if (is_true && is_positive) {
        return &prior_count[3];
    } else {
        return &prior_count[1];
    }
}

float similarity(vector<float> &a, vector<float> &b) {
    int n = a.size();
    float sim = 0;
    for (int i = 0; i < n; ++i) {
        sim += a[i] * b[i];
    }
    return sim;
}


vector <AnswerLabel> sphere_kmeans(vector <Answer> &answers, int cluster_number, int max_iter = 300, float tol = 1e-4) {
    int user_num = answers.size();
    int dimension = answers[0][0].size();
    vector <AnswerLabel> user_answer_label;

    // randomly assign cluster label;
    for (int i = 0; i < user_num; i++) {
        int answer_size = answers[i].size();
        AnswerLabel answer_label(answer_size);
        for (int j = 0; j < answer_size; ++j) {
            answer_label[j] = rand() % cluster_number;
        }
        user_answer_label.append(answer_label);
    }

    vector <Cluster> clusters(cluster_number, Cluster(dimension));
    vector <Cluster> next_clusters(cluster_number, Cluster(dimension));

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
                    next_clusters[][p] += keyword[p];
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
                keyword &keyword = answers[j][k];
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
