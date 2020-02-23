//
// Created by anxin on 2020-02-20.
//
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include "texttruth.h"
#include <unordered_map>
#include <chrono>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "boost/filesystem.hpp"

using namespace std;

void load_answer_grading_data(vector<string> &questions, Dataset &dataset, vector<Score> &baseline_score) {
    // load questions
    ifstream question_file("../answer_grading/questions/questions");
    if (!question_file.is_open()) {
        cout << strerror(errno) << endl;
        exit(0);
    }
    int count = 0;
    string line;
    while (getline(question_file, line)) {
        questions.push_back(line);
    }
    question_file.close();


    // load dataset
//    data_file_name
    // load scores
    boost::filesystem::directory_iterator end_itr;

    // cycle through the directory
    int question_num = questions.size();
    baseline_score = vector<Score>(question_num);
    for (boost::filesystem::directory_iterator itr("../answer_grading/scores"); itr != end_itr; ++itr) {
        // If it's not a directory, list it. If you want to list directories too, just remove this check.
        if (boost::filesystem::is_regular_file(itr->path())) {
            // assign current file name to current_file and echo it out to the console.
            auto current_file = itr->path();
            auto file_name = current_file.filename();
            ifstream score_file(current_file.string());
            Score scores;
            if (!score_file.is_open()) {
                cout << strerror(errno) << endl;
                exit(0);
            }
            float score;
            while (score_file >> score) {
                scores.push_back(score);
            }
            baseline_score[atof(file_name.c_str())] = scores;
        }
    }

    auto raw_dataset = vector<vector<string> >(question_num);
    // read answer
    for (boost::filesystem::directory_iterator itr("../answer_grading/answers"); itr != end_itr; ++itr) {
        // If it's not a directory, list it. If you want to list directories too, just remove this check.
        if (boost::filesystem::is_regular_file(itr->path())) {
            // assign current file name to current_file and echo it out to the console.
            auto current_file = itr->path();
            auto file_name = current_file.filename();
            ifstream answers_file(current_file.string());
            vector<string> answers;
            if (!answers_file.is_open()) {
                cout << strerror(errno) << endl;
                exit(0);
            }
            string answer;
            while (getline(answers_file, answer)) {
                answers.push_back(answer);
            }
            raw_dataset[atof(file_name.c_str())] = answers;
        }
    }
}

void load_word_embedding(unordered_map<string, Keyword> &dic) {
    ifstream word_embedding_file("../glove.6B/glove.6B.50d.txt");
    if (!word_embedding_file.is_open()) {
        cout << strerror(errno) << endl;
        exit(0);
    }
    int count = 0;
    string line;
    while (getline(word_embedding_file, line)) {
        istringstream iss(line);
        string key;
        iss >> key;
        Keyword keyword;
        float v;
        while (iss >> v) {
            keyword.push_back(v);
        }
        dic[key] = keyword;
//        cout<<key<<endl;
//        for(int i=0; i<keyword.size();i++){
//            cout<<keyword[i]<<" ";
//        }
//        cout<<endl;
    }
    word_embedding_file.close();
}

void save_keyword_to_file(unordered_map<string, Keyword> &dic, string path) {
    ofstream dictionary(path);
    if (!dictionary.is_open()) {
        cout << strerror(errno) << endl;
        exit(0);
    }
    boost::archive::binary_oarchive oa(dictionary);
    oa << dic;
    dictionary.close();
}

void load_keyword_from_file(unordered_map<string, Keyword> &dic, string path) {
    ifstream dictionary(path);
    if (!dictionary.is_open()) {
        cout << strerror(errno) << endl;
        exit(0);
    }
    boost::archive::binary_iarchive ia(dictionary);
    ia >> dic;
    dictionary.close();
}

void test_time() {
    auto t1 = std::chrono::high_resolution_clock::now();
    // func()
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << duration / 1000.0 / 1000 << "s" << endl;
}

int main() {
    vector<string> questions;
    Dataset dataset;
    vector<Score> baseline_score;
    unordered_map<string, Keyword> dic;
//    load_keyword_from_file(dic, "../dictionary");
//    cout<<"dictionary size"<<dic.size()<<endl;

    load_answer_grading_data(questions, dataset, baseline_score);
//    for (int i=0;i<questions.size();i++) {
//        cout<<questions[i]<<endl;
//    }
    return 0;
}


