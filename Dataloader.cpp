//
// Created by Han Cao on 4/28/20.
//

#include <sstream>
#include <vector>
#include "Dataloader.h"
#include "Neuron.h"

void Dataloader::getTopology(vector<unsigned int> &topology) {
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;

    if (this->isEof() || label.compare("topology:")!= 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }
}

void Dataloader::getType(vector<NeuronType> &t) {
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;

    if (this->isEof() || label.compare("type:")!= 0) {
        abort();
    }

    while (!ss.eof()) {
        std::string n;
        ss >> n;
        if (n == "tanh") {
            t.push_back(NeuronType::Tanh);
        } else if (n == "relu") {
            t.push_back(NeuronType::ReLU);
        } else if (n == "input") {
            t.push_back(NeuronType::Input);
        } else if (n == "softmax") {
            t.push_back(NeuronType::Softmax);
        }

    }
}

Dataloader::Dataloader(const string datapath) {
    m_trainingDataFile.open(datapath.c_str());
}

unsigned Dataloader::getNext(vector<double> &inputVals) {
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned Dataloader::getTargetOutputs(vector<double> &targetOutputVals) {
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}