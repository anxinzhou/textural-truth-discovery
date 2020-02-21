//
// Created by anxin on 2020-02-21.
//

#include "texttruth.h"


void texttruth(float ***all_data, int question_num, int dimension, int user_num, int k) {
    // init variable
    float **cluster = new float *[k];
    for (int i = 0; i < k; i++) {
        cluster[i] = new float[dimension]();
    }
    int *cluster_label = new int[user_num];


    int ***user_truth_label = new int **[question_num];
    for (int i = 0; i < question_num; i++) {
        user_truth_label[i] = new int *[user_num];
        for (int j = 0; i < user_num; ++i) {
            ser_truth_label[i][j] = new int[k]();
        }
    }

    int **count =


            int * *question_truth_label = new int *[question_num];
    for (int i = 0; i < k; ++i) {
        question_truth_label[i] = new int[k];
    }

    // begin texttruth
    for (int i = 0; i < question_num; ++i) {
        sphere_kmeans(all_data[i], dimension, user_num, k, cluster, cluster_label);
        for (int j = 0; j < user_num; j++) {
            user_truth_label[i][j][(cluster_label[j]] = 1;
        }
    }



    // free memory


    for (int i = 0; i < question_num; i++) {
        delete user_truth_label[i];
        for (int j = 0; i < user_num; ++i) {
            delete ser_truth_label[i][j];
        }
    }

    for (int i = 0; i < k; i++) {
        delete cluster[i];
        delete question_truth_label[i];
    }

    delete[]cluster;
    delete[]cluster_label;
    delete[]user_truth_label;
    delete[]question_truth_label;
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
