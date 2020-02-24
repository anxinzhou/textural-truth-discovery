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

void load_answer_grading_data(vector<string> &questions, RawDataset &raw_dataset, vector<Score> &baseline_score) {
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

    raw_dataset = RawDataset(question_num);
    // get maximum size of score
    int max_score_size = 0;
    for(auto score: baseline_score) {
        if (score.size()>max_score_size) {
            max_score_size = score.size();
        }
    }
    int user_num = max_score_size;

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
            int answer_count = 0;
            while (getline(answers_file, answer)) {
                answers.push_back(answer);
                answer_count += 1;
            }
            // fill not answered question
            for (int i = answer_count; i < user_num; ++i) {
                int choice = rand() % i;
                answers.push_back(answers[choice]);
                Score &score = baseline_score[atof(file_name.c_str())];
                score.push_back(score[choice]);
            }
            raw_dataset[atof(file_name.c_str())] = answers;
        }
    }


}

void load_word_embedding(unordered_map<string, Keyword> &dic) {
    ifstream word_embedding_file("../glove.twitter.27B/glove.twitter.27B.50d.txt");
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
    // initialize
    const int TEXT_TRUTH_TRY_ROUND = 1000;
    vector<string> questions;
    Dataset dataset;
    RawDataset raw_dataset;
    vector<Score> baseline_score;
    unordered_map<string, Keyword> dic;
    //------------------------------------------------------------------------
    // load dictionary

    cout << "load dictionary" << endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    // func()
//    load_word_embedding(dic);
//    save_keyword_to_file(dic,"../dictionary");
    load_keyword_from_file(dic, "../dictionary");
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << duration / 1000.0 / 1000 << "s" << endl;


    //------------------------------------------------------------------------
    cout << "dictionary size: " << dic.size() << endl;
    load_answer_grading_data(questions, raw_dataset, baseline_score);

    // word embedding ----------------------
    cout << "word embedding" << endl;
    t1 = std::chrono::high_resolution_clock::now();
    word_embedding(raw_dataset, dataset, dic);
    t2 = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    cout << duration / 1000.0 / 1000 << "s" << endl;

    //------------------------------------------------------------------------
    cout << "begin texttruth" << endl;
    cout << "raw dataset size: " << raw_dataset.size() << endl;
    cout << "dataset size: " << dataset.size() << endl;
    t1 = std::chrono::high_resolution_clock::now();
    vector<Score> score = texttruth(dataset,  TEXT_TRUTH_TRY_ROUND);
    // func()

    t2 = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    cout << duration / 1000.0 / 1000 << "s" << endl;


//    for(int i=0;i< score.size();i++) {
//        cout<<"question: "<<i<<endl;
//        for(int j=0; j<score[i].size();j++) {
//            cout<<score[i][j]<<endl;
//        }
//    }
    // get top k results
    RawDataset top_results;
    vector<Score> top_scores;
    const int top_k = 5;
    top_k_result(raw_dataset, score, baseline_score, top_results, top_scores, top_k);
    float total_score = 0;
    int sample_count = 0;
    for (int i = 0; i < top_results.size(); ++i) {
//        cout << "question: " << i << endl;
//        cout << "question:" << questions[i] << endl;
//        cout << " top 3 results:" << endl;
        for (int j = 0; j < top_results[i].size(); ++j) {
//            cout<< "Number "<<j+1<<" result:"<<endl;
//            cout << top_results[i][j] << endl;
//            cout << "actual point: " << top_scores[i][j]<<endl;
            total_score += top_scores[i][j];
            sample_count+=1;
        }
    }
    cout<<"sample count:" << sample_count <<endl;
    cout<<"average score: "<<total_score/sample_count<<endl;

    return 0;
}


