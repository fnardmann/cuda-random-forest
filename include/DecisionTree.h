#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <random>

typedef std::vector<unsigned int> Labels;
typedef std::vector<bool> Feature;

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
        const unsigned int maxleafs = 2 // binary for now
    );

    void fit(
        const std::vector<Feature>& features,
        const Labels& labels
    );

    std::vector<unsigned int> predict(
        const std::vector<Feature>& features
    );

    
private:

    void split(
        const std::vector<Feature>& features,
        const Labels& labels,
        std::shared_ptr<DecisionNode> parent
    );

    std::pair<Labels, Labels> impurity_split(
        const Feature& feature,
        const Labels& labels
    );

    double impurity_score(
        const std::pair<Labels, Labels>& split
    );

    std::pair<Labels, Labels> entropy_split(
        const Feature& ,
        const Labels& labels
    );

    double entropy_score(
        const Labels& leaf
    );

    std::string _ifunction;
    std::string _maxfeatures;
    unsigned int _maxdepth;
    unsigned int _maxleafs;

    unsigned int _numfeatures;

    Labels _uniquelabels;
    std::shared_ptr<DecisionNode> _root;

    std::mt19937 _g; // random generator
};