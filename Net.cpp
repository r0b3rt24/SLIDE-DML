#include "Net.h"

using namespace std;

void Net::getResult(vector<double> &resultVals) const {
    resultVals.clear();
    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
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

        if (layerNum == m_layers.size()-1) {
            Layer &thisLayer = m_layers[layerNum];
            for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
                m_layers[layerNum][n].softmax(thisLayer);
            }
        }
    }
};

void Net::backProp(const std::vector<double> &targetVals) {
    Layer &outputLayer = m_layers.back();

    // TODO: change to cross-entropy

//  cross-entropy
    m_error = 0.0;
    for (unsigned  n = 0; n < outputLayer.size() - 1; ++n) {
        double temp = targetVals[n] * log(outputLayer[n].getOutputVal());
        m_error += temp;
    }

//    m_error = -m_error/outputLayer.size();
    m_error = -m_error;

    m_recentAvgErr = m_error;

    // sum exp
    double sumExp = 0;
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        sumExp += exp(outputLayer[n].getOutputVal());
    }

    // calc and set output gradient: no need to update
    vector<double> gradient_output;
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        // dE/dO_in
        double o_gradient = (outputLayer[n].getOutputVal() - targetVals[n]) * (outputLayer[n].getOutputVal()*(targetVals[n] - outputLayer[n].getOutputVal()));
        outputLayer[n].calcOutputGradients(o_gradient);
    }

    // update input weights for output layer
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        // dE/dO_in
        outputLayer[n].updateInputWeights(m_layers[m_layers.size()-2]);
    }

    // calculate the gradient for hidden layers
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {  // hidden layers
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
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
//            cout << "Neuron Got Created" << endl;
        }
    }
    Net::m_recentAvgSmoothingFactor = 100.0; // Number of training samples to average over
    m_layers.back().back().setOutputVal(1.0);

    // make last layer become softmax
    Layer &layer = m_layers.back();
    for (unsigned n = 0; n < layer.size(); n++) {
        layer[n].if_sotfmax = true;
    }
}

double Net::getRecentAverageError() const { return m_recentAvgErr; }