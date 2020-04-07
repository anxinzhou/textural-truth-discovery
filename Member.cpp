//
// Created by anxin on 2020-04-07.
//

#include "Member.h"


vector<string> Answer::string_split(const string &s) {
    vector<string> tokens;
    istringstream iss(s);
    copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(tokens));
    return tokens;
}

vector<Keyword>
Answer::to_keywords(const unordered_set<string> &words_filter, const unordered_set<string> &dictionary) {
    // replace punc
    vector<Keyword> keywords;
    replace_if(raw_answer.begin(), raw_answer.end(),
               [](const char &c) { return std::ispunct(c); }, ' ');

    // lower alphabetic
    transform(raw_answer.begin(), raw_answer.end(), raw_answer.begin(),
              [](unsigned char c) { return std::tolower(c); });

    auto tokens = string_split(raw_answer);

    // extract keywords
    // only remove stopwords here, may use doman specific dictionary
    // here skip words not in dictionary
    for (auto token: tokens) {
        if (words_filter.count(token) == 0 || dictionary.count(token) == 0) {
//                    cout << token << endl;
            continue;
        }
        keywords.emplace_back(owner_id, question_id, token);
    }
}