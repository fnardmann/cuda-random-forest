#include <IO.h>

#include <utility>
#include <memory>
#include <random>

typedef std::vector<unsigned int> Indices;

class Split
{
public:

    Split(
        const Indices& indices,
        const std::vector<Label>& l,
        const std::vector<std::vector<Feature>>& f
    );

    std::vector<std::vector<Feature>> features;
    std::vector<Label> labels;
};

class DecisionNode 
{
public:

    DecisionNode();

    DecisionNode(
        const unsigned int splitfeatureID
    );

    void addChild(const DecisionNode& node){ _children.push_back(node); };

    bool empty(){ return _splitfeatureID == -1; };

private:

    int _splitfeatureID; // feature to get splitted children
    std::vector<DecisionNode> _children;
};

class DecisionTree
{
public:

    DecisionTree(
        const std::string& impurity_function = "entropy",
        const std::string& maxfeatures = "sqrt",
        const unsigned int maxdepth = INT8_MAX,
        const unsigned int maxleafs = 2, // binary for now
        const unsigned int minsplitsamples = 2
    );

    void fit(
        const Data& data
    );

    std::vector<unsigned int> predict(
        const std::vector<Feature>& features
    );

    
private:

    void split(
        const std::vector<std::vector<Feature>>& features,
        const std::vector<Label>& labels,
        std::shared_ptr<DecisionNode> parent
    );

    std::pair<Indices, Indices> impurity_split(
        const std::vector<Feature>& feature
    );

    double impurity_score(
        const std::pair<std::vector<Label>, std::vector<Label>>& split
    );

    std::pair<Indices, Indices> entropy_split(
        const std::vector<Feature>&
    );

    double entropy_score(
        const std::vector<Label>& leaf
    );

    std::string _ifunction;
    std::string _maxfeatures;
    unsigned int _maxdepth;
    unsigned int _maxleafs;
    unsigned int _minsplitsamples;

    unsigned int _numfeatures;

    std::vector<Label> _uniquelabels;
    std::shared_ptr<DecisionNode> _root;

    std::mt19937 _g; // random generator
};