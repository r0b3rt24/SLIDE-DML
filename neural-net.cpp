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

};

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