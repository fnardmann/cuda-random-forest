#include <iostream>
#include <vector>

#include <DecisionTree.h>

int main(int argc, char** argv)
{
    // test data with 2 classes and 2 binary features each
    std::vector<unsigned int> labels = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1};

    std::vector<std::vector<bool>> features = 
    {
        {true, true, true, false, true, true, false, false, false, false},
        {true, false, true, false, true, false, true, false, true, false}
    };

    DecisionTree dt("entropy");

    dt.fit(features, labels);

}