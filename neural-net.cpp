#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;


struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron {};

typedef vector<Neuron> Layer;


//================================================================
class Neuron
{
public:
    Neuron(unsigned numOutput);
    void feedForward(const Layer &prevLayer)

    void setOutputVal(double val) {m_outputVal = val;}
    double getOutputVal(void) {return m_outputVal;}

private:
    static double activation(double x);
    static double activationDerivative(double x);
    double m_outputVal;
    vector<Connection> m_outputWeights;
    static double randomWeight(void) {
        return rand() / double(RAND_MAX);
    }
    unsigned m_myIndex;
};

Neuron:: activation(double x)
{
    return tanh(x);
}

Neuron:: activationDerivative(double x)
{
    return 1 - x * x;  // an aproxiamation to the derivative (faster)
}

Neuron::Neuron(unsigned numOutput, unsigned myIndex)
{
    for (unsigned i = 0; i < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}

Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() * 
                prevLayer[m_myIndex].m_outputWeights[].weight;
    }

    m_outputVal = Neuron::activation(sum);
}


//================================================================
class Net
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals){};
    void backProp(const vector<double> &targetVals){};
    void getResult(vector<double> &resultVals) const {};

private:
    vector<Layer> m_layers;  //m_layers[layerNum][neuroNum]
    double m_error;
    double m_recentAvgErr;
    double m_recentAvgSmoothingFactor;

void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);

    // add input data to each of the neurons at the first layer
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setInput(inputVals[i]);
    }

    // actually do the feed forward
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() -1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
};

void Net::backProp(const std::vector<double> &targetVals)
{
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }

    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);

    m_recentAvgErr = (m_recentAvgErr * m_recentAvgSmoothingFactor + m_error)
                     / (m_recentSmoothingFactor + 1.0);
    
    for (unsigned int n = 0; n < outputLayer.size() - 1; ++n)
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layters[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcOutputGradients(nextLayer);
        }
    }

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum-1];

        for 
    }
}

Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ?  0 : topology[layerNum + 1];

        for (unsigned neuroNum = 0; neuroNum <= topology[layerNum]; ++neuroNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuroNum));
            cout << "Neuron Got Created" << endl;
        }
    }
}

int main()
{
    vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);

    Net myNet(topology);

    vector<double> inputVals;
    myNet.feedForward(inputVals);

    vector<double> targetVals;
    myNet.backProp(targetVals);

    vector<double> resultVals;
    myNet.getResult(resultVals);
}