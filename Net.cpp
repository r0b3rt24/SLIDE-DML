#include "Net.h"

using namespace std;

void Net::getResult(vector<double> &resultVals) const {
    resultVals.clear();
#pragma omp parallel for
    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::feedForward(const vector<double> &inputVals) {
    assert(inputVals.size() == m_layers[0].size() - 1);
    auto start = std::chrono::high_resolution_clock::now();
    // add input data to each of the neurons at the first layer
#pragma omp parallel for
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // actually do the feed forward
#pragma omp parallel for
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
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    total_forward_time += elapsed.count();
};

void Net::backProp(const std::vector<double> &targetVals) {
    auto start = std::chrono::high_resolution_clock::now();
    Layer &outputLayer = m_layers.back();

//  cross-entropy
    m_error = 0.0;
    #pragma omp parallel for
    for (unsigned  n = 0; n < outputLayer.size() - 1; ++n) {
        double temp = targetVals[n] * log(outputLayer[n].getOutputVal());
        m_error += temp;
    }

    m_error = -m_error;

    m_recentAvgErr = m_error;

    // sum exp
    double sumExp = 0;
    #pragma omp parallel for
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        sumExp += exp(outputLayer[n].getOutputVal());
    }

    // calc and set output layer gradient: no need to update
    vector<double> gradient_output;
    #pragma omp parallel for
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {  // loop thru every node on the output layer
        // dE/logit_out
        double o_gradient = outputLayer[n].softmax_o - targetVals[n];
        outputLayer[n].calcOutputGradients(o_gradient);
    }

    // calculate the gradient for hidden layers
    #pragma omp parallel for
    for (unsigned layerNum = m_layers.size() - 2; layerNum >= 1; --layerNum) {  // hidden layers
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // update connection weights
    #pragma omp parallel for
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    total_backprop_time += elapsed.count();
}

Net::Net(const vector<unsigned> &topology, const vector<NeuronType> &t) {
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        NeuronType myType = t[layerNum];
        for (unsigned neuroNum = 0; neuroNum <= topology[layerNum]; ++neuroNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuroNum, myType));
//            cout << "Neuron Got Created" << endl;
        }
    }
    Net::m_recentAvgSmoothingFactor = 100.0; // Number of training samples to average over
    m_layers.back().back().setOutputVal(1.0);

    total_backprop_time = 0.0;
    total_forward_time = 0.0;
}

double Net::getRecentAverageError() const { return m_recentAvgErr; }