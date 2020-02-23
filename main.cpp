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

using namespace std;

void load_answer_grading_data(vector<string> &questions, Dataset &dataset, vector<Score> baseline_score) {
    // load questions
    ifstream question_file("../answer_grading/data/questions");
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

int main() {
    vector<string> questions;
    Dataset dataset;
    vector<Score> baseline_score;
    unordered_map<string, Keyword> dic;

    auto t1 = std::chrono::high_resolution_clock::now();
    load_word_embedding(dic);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    std::cout << duration/1000.0/1000<<endl;

    cout<<dic.size()<<endl;


    return 0;
}


