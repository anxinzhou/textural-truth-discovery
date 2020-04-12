//
// Created by anxin on 2020/4/9.
//

#include "dataset.h"

const int AnswerGradingData::key_factor_number = 30;

unordered_set<string> AnswerGradingData::words_filter = {"with", "who", "s", "above", "of", "are", "nor", "both",
                                                         "have", "i",
                                                         "yourself", "after", "those", "is", "their", "we", "same",
                                                         "each", "she", "be",
                                                         "yours", "when", "all", "only", "had", "your", "own",
                                                         "against", "don", "again",
                                                         "does", "other", "whom", "off", "by", "its", "her", "too",
                                                         "that", "out",
                                                         "through", "am", "why", "our", "between", "being", "for",
                                                         "yourselves", "his",
                                                         "so", "having", "under", "them", "most", "did", "there",
                                                         "will", "such", "the",
                                                         "as", "and", "herself", "some", "ourselves", "at", "these",
                                                         "how", "than", "a",
                                                         "were", "myself", "they", "an", "which", "further", "but",
                                                         "not", "on", "been",
                                                         "now", "more", "was", "while", "do", "to", "this", "once",
                                                         "no", "during",
                                                         "himself", "itself", "any", "doing", "into", "it", "here",
                                                         "me", "him", "then",
                                                         "what", "very", "few", "before", "if", "just", "where",
                                                         "themselves", "hers",
                                                         "can", "has", "below", "should", "from", "t", "in", "you",
                                                         "he", "up", "until",
                                                         "because", "my", "down", "or", "over", "about", "theirs",
                                                         "ours"};

void AnswerGradingData::load_dataset(const string &dir_path) {
    const string question_file_path = (boost::filesystem::path(dir_path) / "questions/questions").string();
    const string answers_file_path = (boost::filesystem::path(dir_path) / "answers").string();
    const string score_file_path = (boost::filesystem::path(dir_path) / "scores").string();
    load_question(question_file_path);
    load_answers(answers_file_path);
    load_scores(score_file_path);
}

void AnswerGradingData::load_question(const string &question_file_path) {
    ifstream question_file(question_file_path);
    if (!question_file.is_open()) {
        cout<<"can not open question file"<<endl;
        cout << strerror(errno) << endl;
        exit(0);
    }

    string line;
    while (getline(question_file, line)) {
        questions.emplace_back(std::move(line),key_factor_number);
    }
    question_file.close();
}

void AnswerGradingData::load_answers(const string &answer_file_path) {
    // read answer
    int question_num = questions.size();
    answers = vector<vector<Answer>> (question_num,vector<Answer>());
    boost::filesystem::directory_iterator end_itr;
    for (boost::filesystem::directory_iterator itr(answer_file_path); itr != end_itr; ++itr) {
        // If it's not a directory, list it. If you want to list directories too, just remove this check.
        if (boost::filesystem::is_regular_file(itr->path())) {
            // assign current file name to current_file and echo it out to the console.
            auto current_file = itr->path();
            int question_id = atof(current_file.filename().c_str());
            ifstream answers_file(current_file.string());
            vector<Answer> sub_answers;
            if (!answers_file.is_open()) {
                cout<< "can not open answer file"<<endl;
                cout << strerror(errno) << endl;
                exit(0);
            }
            string answer;
            int count = 0;
            while (getline(answers_file, answer)) {
                sub_answers.emplace_back(count, question_id, std::move(answer));
                count+=1;
            }

            answers[question_id] = sub_answers;
        }
    }
}

void AnswerGradingData::load_scores(const string &score_file_path) {
    int question_num = questions.size();
    scores = vector<vector<float>> (question_num,vector<float>());
    boost::filesystem::directory_iterator end_itr;
    for (boost::filesystem::directory_iterator itr(score_file_path); itr != end_itr; ++itr) {
        // If it's not a directory, list it. If you want to list directories too, just remove this check.
        if (boost::filesystem::is_regular_file(itr->path())) {
            // assign current file name to current_file and echo it out to the console.
            auto current_file = itr->path();
            int question_id = atof(current_file.filename().c_str());
            ifstream score_file(current_file.string());
            vector<float> sub_scores;
            if (!score_file.is_open()) {
                cout<< "can not open score file"<<endl;
                cout << strerror(errno) << endl;
                exit(0);
            }
            float score;
            while (score_file >> score) {
                sub_scores.push_back(score);
            }
            scores[question_id] = sub_scores;
        }
    }
}