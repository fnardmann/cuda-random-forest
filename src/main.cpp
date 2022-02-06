#include <iostream>
#include <vector>

#include <DecisionTree.h>

int main(int argc, char** argv)
{
    const std::string path = "/home/felix/Desktop/data/haberman.data";

    const Data& data = IO::read(path);

    DecisionTree dt;

    dt.fit(data);

}