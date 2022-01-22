#include <vector>
#include <utility>
#include <string>

typedef std::vector<unsigned int> Labels;
typedef std::vector<bool> Feature;

class DecisionTree
{
public:

    DecisionTree(
        const std::string& impurity_function = "entropy",
        const std::string& maxfeatures = "sqrt",
        const unsigned int maxdepth = INT8_MAX
    );

    void fit(
        const std::vector<Feature>& features,
        const Labels& labels
    );

    std::vector<unsigned int> predict(
        const std::vector<Feature>& features
    );

    
private:

    void split();

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

    Labels _uniquelabels;

};