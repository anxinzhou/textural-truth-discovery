//
// Created by anxin on 2020 4/9.
//
#include <iostream>
#include "dataset.h"
#include "Member.h"
#include "ttruth.h"

using namespace std;

void test_time() {
    auto t1 = std::chrono::high_resolution_clock::now();
    // func()
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << duration / 1000.0 / 1000 << "s" << endl;
}

int main() {
    // load dataset answer grading
    const string dataset_path = "../answer_grading";
    const string model_path = "../word_model/golve6B50d";
    AnswerGradingData answer_grade_data;
    answer_grade_data.load_dataset(dataset_path);
    WordModel word_model;
    auto s_w = std::chrono::high_resolution_clock::now();
    word_model.load_model(model_path);
    auto e_w = std::chrono::high_resolution_clock::now();
    auto dw = std::chrono::duration_cast<std::chrono::microseconds>(e_w - s_w).count();
    std::cout << "load model time "<<dw / 1000.0 / 1000 << "s" << endl;

    // set user
    auto &questions_answers = answer_grade_data.answers;
    auto &questions = answer_grade_data.questions;
    auto &baseline_scores = answer_grade_data.scores;
    auto &question_per_exam = answer_grade_data.question_per_exam;

    // decide user number
    int user_num = INT_MIN;
    for(auto &answer:questions_answers) {
        int answer_num = answer.size();
        if (answer_num>user_num) {
            user_num = answer_num;
        }
    }
    if(user_num <0) {
        cout<<"empty dataset"<<endl;
        return -1;
    }

    vector<User> users(user_num);
    for(int question_id=0; question_id<questions_answers.size(); question_id++) {
        auto &question = questions[question_id];
        auto &answers = questions_answers[question_id];
        for(int user_id = 0; user_id < answers.size(); ++user_id) {
            users[user_id].update_answer(question,std::move(answers[user_id]));
        }
    }

    // benchmark texttruth
    for(int top_k = 1; top_k <=10; top_k+=1) {
        cout<<"top " <<top_k<<endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        ttruth(questions, users, AnswerGradingData::words_filter, word_model,top_k);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << duration / 1000.0 / 1000 << "s" << endl;

        double total_score = 0;
        int num = 0;
        vector<double>exam_score(question_per_exam.size(),0);
        int count = 0;
        int exam_id = 0;
        for(auto &question: questions) {
            if(count == question_per_exam[exam_id]) {
                count = 0;
                exam_id += 1;
            }
            for(auto &result: question.top_k_results) {
                int question_id = result.question_id;
                int owner_id = result.owner_id;
                double score = baseline_scores[question_id][owner_id];
                total_score += score;
                num += 1;
                exam_score[exam_id] += score;
            }
            count += 1;
        }
        // average exam_score;
        for(int i=0; i<exam_score.size();i++) {
            exam_score[i] /= top_k * question_per_exam[i];
            cout<<"exam "<<i+1<<" avg score: "<< exam_score[i]<<endl;
        }
        double average_score = total_score/ num;
//        cout<<"top "<<top_k<<" average score: "<< average_score<<endl;
    }
}


