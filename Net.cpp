#include "Net.h"

using namespace std;

void Net::getResult(vector<double> &resultVals) const {
    resultVals.clear();
    for (unsigned n = 0; n < m_layers.size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::feedForward(const vector<double> &inputVals) {
    assert(inputVals.size() == m_layers[0].size() - 1);

    // add input data to each of the neurons at the first layer
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // actually do the feed forward
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
};

void Net::backProp(const std::vector<double> &targetVals) {
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }

    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);

    m_recentAvgErr = (m_recentAvgErr * m_recentAvgSmoothingFactor + m_error)
                     / (m_recentAvgSmoothingFactor + 1.0);

    for (unsigned int n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // calculate the gradient
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcOutputGradients(targetVals[n]);
        }
    }

    // update connection weights
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

Net::Net(const vector<unsigned> &topology) {
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        for (unsigned neuroNum = 0; neuroNum <= topology[layerNum]; ++neuroNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuroNum));
            cout << "Neuron Got Created" << endl;
        }
    }

    m_layers.back().back().setOutputVal(1.0);
}

double Net::getRecentAverageError() const { return m_recentAvgErr; }