#include <vector>
#include <string>

typedef unsigned int Label;
typedef double Feature; // TODO: template

class DataEntry
{
public:

    DataEntry(const Label l, const std::vector<Feature>& f);

    Label label;
    std::vector<Feature> features; 
};

class Data
{
public:

    Data(const std::vector<DataEntry>& e);

    bool empty() const { return entries.empty(); };
    bool size() const { return entries.size(); };

    // get all labels in data set
    std::vector<Label> getLabels() const;

    // get feature at index
    std::vector<Feature> getFeature(const unsigned int index) const;

    // get all features
    std::vector<std::vector<Feature>> getFeatures() const;

    std::vector<DataEntry> entries;
};

class IO
{
public:

    static Data read(const std::string& path);

private:
};