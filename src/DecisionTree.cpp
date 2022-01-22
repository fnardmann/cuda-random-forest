#include <DecisionTree.h>

#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <set>
#include <numeric>

#include <math.h>

DecisionTree::DecisionTree(const std::string& impurity_function)
    : _ifunction(impurity_function)
{
    
}

void DecisionTree::fit(
    const std::vector<Feature>& features,
    const Labels& labels) 
{
    if (features.empty() || labels.empty())
        throw std::invalid_argument("received invalid input");

    if (features[0].size() != labels.size())
        throw std::invalid_argument("received invalid input");

    const unsigned int num_samples = labels.size();
    const unsigned int num_features = features.size();

    // init unique labels
    const auto& labelset = std::set<unsigned int>(labels.begin(), labels.end());
    _uniquelabels = Labels(labelset.begin(), labelset.end());

    std::pair<Labels, Labels> bestsplit;
    double bestscore = __DBL_MAX__;

    // test impurity for each possible split
    for (const Feature& feature : features)
    {
        const std::pair<Labels, Labels>& split = impurity_split(feature, labels);
        const double score = impurity_score(split);

        if (score > bestscore)
        {
            bestscore = score;
            bestsplit = split;
        }
    }
}

std::pair<Labels, Labels> DecisionTree::impurity_split(
    const Feature& feature,
    const Labels& labels) 
{
    if (_ifunction == "Entropy") return entropy_split(feature, labels);

    else throw std::invalid_argument("no valid impurity function");
}   

double DecisionTree::impurity_score(const std::pair<Labels, Labels>& split) 
{
    if (_ifunction == "Entropy")
    {
        return entropy_score(split.first) + entropy_score(split.second);
    } 

    else throw std::invalid_argument("no valid impurity function");

}

std::pair<Labels, Labels> DecisionTree::entropy_split(
    const Feature& feature,
    const Labels& labels) 
{
    Labels lhs;
    Labels rhs;

    for (unsigned int i = 0; i < feature.size(); i++)
    {
        feature[i] ? lhs.push_back(labels[i]) : rhs.push_back(labels[i]);
    }

    return std::make_pair(lhs, rhs);
}

double DecisionTree::entropy_score(const Labels& leaf) 
{
    // a low entropy score is good !
    const double totalsize = leaf.size();

    double score = 0.0;

    for (unsigned int label : _uniquelabels)
    {
        // get number of occurences for this label
        const double labelsize = std::count(leaf.begin(), leaf.end(), label);

        if (labelsize != 0.0) // log(0) is undefined
            // log to the base e
            score += ((labelsize/totalsize) * log(labelsize/totalsize)); 
    }

    return -score;
}

