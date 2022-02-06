#include <IO.h>

#include <fstream>
#include <iostream>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

DataEntry::DataEntry(
    const Label l, 
    const std::vector<Feature>& f)
    : label(l), features(f)
{
    
}


Data::Data(const std::vector<DataEntry>& e)
    : entries(e)
{
    
}

std::vector<Label> Data::getLabels() const
{
    if (this->entries.empty())
        throw std::invalid_argument("no data entries");
        
    std::vector<Label> labels; labels.reserve(this->size());

    for (const DataEntry& entry : this->entries)
    {
        labels.push_back(entry.label);
    }

    return labels;
}

std::vector<Feature> Data::getFeature(const unsigned int index) const
{
    if (this->entries.empty())
        throw std::invalid_argument("no data entries");

    if (this->entries[0].features.size() <= index)
        throw std::invalid_argument("no feature with this index");

    std::vector<Feature> features; features.reserve(this->size());
    
    for (const DataEntry& entry : this->entries)
    {
        features.push_back(entry.features[index]);
    }

    return features;
}

std::vector<std::vector<Feature>> Data::getFeatures() const
{
    if (this->entries.empty())
        throw std::invalid_argument("no data entries");

    const unsigned int num_features = this->entries[0].features.size();

    std::vector<std::vector<Feature>> features(num_features, std::vector<Feature>());

    for (unsigned int i = 0; i < num_features; i++)
    {
        features[i] = this->getFeature(i);
    }

    return features;
}

Data IO::read(const std::string& path) 
{
    std::ifstream infile(path);

    // first elements are features; last element is labels
    std::vector<DataEntry> entries;

    // TODO: convert bool to double

    std::string line;
    while (std::getline(infile, line))
    {
        std::vector<std::string> elements;
        boost::split(elements, line, boost::is_any_of(","), boost::token_compress_on);

        std::vector<Feature> features;

        for (unsigned int i = 0; i < elements.size() - 1; i++)
        {
            features.push_back(std::stod(elements[i]));
        }
        
        const Label label = std::stoi(elements[elements.size() - 1]);
        entries.push_back(DataEntry(label, features));
    }
    
    return Data(entries);
}