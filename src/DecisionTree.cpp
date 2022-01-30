#include <DecisionTree.h>

#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <set>
#include <numeric>
#include <cmath>

Split::Split(
    const Indices& indices,
    const Labels& l,
    const std::vector<Feature>& f)
    : labels(), features()
{
    // add (empty) features vectors
    for(unsigned int i = 0; i < f.size(); i++)
    {
        features.push_back(Feature());
    }

    for(const unsigned int idx : indices)
    {
        labels.push_back(l[idx]);

        // add each feature at this index
        for(unsigned int i = 0; i < f.size(); i++)
        {
            features[i].push_back(f[i][idx]);
        }
    }
}

DecisionNode::DecisionNode()
    : _splitfeatureID(-1), _children()
{
    
}


DecisionNode::DecisionNode(
    const unsigned int splitfeatureID) 
    : _splitfeatureID(splitfeatureID), _children()
{
    
}

DecisionTree::DecisionTree(
    const std::string& impurity_function,
    const std::string& maxfeatures,
    const unsigned int maxdepth,
    const unsigned int maxleafs)
    : _ifunction(impurity_function), _maxfeatures(maxfeatures), 
      _maxdepth(maxdepth), _maxleafs(maxleafs), _numfeatures(0), 
      _g((std::random_device())()), _root()
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

    // init unique labels
    const auto& labelset = std::set<unsigned int>(labels.begin(), labels.end());
    _uniquelabels = Labels(labelset.begin(), labelset.end());

    // init number to features to actually use for each split
    // _numfeatures <= features.size()
    if (_maxfeatures == "sqrt") _numfeatures = std::ceil(std::sqrt(features.size()));
    else throw std::invalid_argument("no valid max features function");

    // at this stage pass all features and handle selecton inside
    split(features, labels, _root);

}

void DecisionTree::split(
    const std::vector<Feature>& features,
    const Labels& labels,
    std::shared_ptr<DecisionNode> parent) 
{
    // recursion anchors to avoid splitting further
    // at least three samples
    if (labels.size() < 3) return;
    // at leat one feature
    if (features.empty()) return;
    // TODO: function to get actual depth of tree
    // if (depth > _maxdepth) return;

    Indices indices(features.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), _g);
    // indices represents randomly shuffled indices of features

    std::pair<Indices, Indices> bestsplit;
    double bestscore = __DBL_MAX__;
    int bestsplitfeature = -1; // index of feature producing the best split

    // loop till _numfeatures is reached
    // test impurity for possible splits
    for (unsigned int i = 0; i < _numfeatures - 1; i++)
    {
        // get index from randomly sampled indices vector
        const unsigned idx = indices[i];

        const std::pair<Indices, Indices>& split = impurity_split(features[idx], labels);
        const double score = impurity_score(split);

        if (score < bestscore)
        {
            bestscore = score;
            bestsplit = split;
            bestsplitfeature = i; 
        }
    }

    // create node with best possible split
    // shared pointer to keep track of tree
    std::shared_ptr<DecisionNode> node = std::make_shared<DecisionNode>(bestsplitfeature);

    if (!parent) parent = node;
    else parent->addChild(*node.get());

    // create Splits with indices from bestsplit
    Split a(bestsplit.first, labels, features);
    Split b(bestsplit.second, labels, features);

    // call this method recursively for splitted labels
    split(a.features, a.labels, node);
    split(b.features, b.labels, node);
}

std::pair<Indices, Indices> DecisionTree::impurity_split(
    const Feature& feature,
    const Labels& labels) 
{
    if (_ifunction == "entropy") return entropy_split(feature, labels);

    else throw std::invalid_argument("no valid impurity function");
}   

double DecisionTree::impurity_score(
    const std::pair<Labels, Labels>& split) 
{
    if (_ifunction == "entropy")
    {
        return entropy_score(split.first) + entropy_score(split.second);
    } 

    else throw std::invalid_argument("no valid impurity function");

}

std::pair<Indices, Indices> DecisionTree::entropy_split(
    const Feature& feature,
    const Labels& labels) 
{
    Indices lhs;
    Indices rhs;

    for (unsigned int i = 0; i < feature.size(); i++)
    {
        feature[i] ? lhs.push_back(i) : rhs.push_back(i);
    }

    return std::make_pair(lhs, rhs);
}

double DecisionTree::entropy_score(
    const Labels& leaf) 
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
            score += ((labelsize/totalsize) * std::log(labelsize/totalsize)); 
    }

    return -score;
}