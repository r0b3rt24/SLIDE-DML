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
class Neuron;
typedef vector<Neuron> Layer;


//================================================================
class Neuron
{
public:
    Neuron(unsigned numOutput, unsigned myIndex);
    void feedForward(const Layer &prevLayer);

    void setOutputVal(double val) {m_outputVal = val;}
    double getOutputVal(void) const {return m_outputVal;}
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer); 

private:
    static double eta;
    static double alpha;
    static double activation(double x);
    static double activationDerivative(double x);
    double m_outputVal;
    vector<Connection> m_outputWeights;
    static double randomWeight(void) {
        return rand() / double(RAND_MAX);
    }
    unsigned m_myIndex;
    double m_gradient;
    double sumDOW(const Layer& nextLayer) const;
};

double Neuron::eta = 0.15;    // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.5;   // momentum, multiplier of last deltaWeight, [0.0..1.0]


double Neuron::activation(double x)
{
    return tanh(x);
}

double Neuron:: activationDerivative(double x)
{
    return 1 - x * x;  // an aproxiamation to the derivative (faster)
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];

        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
    double sum = 0.0;

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::activationDerivative(m_outputVal);

}

void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::activationDerivative(m_outputVal);
}



Neuron::Neuron(unsigned numOutput, unsigned myIndex)
{
    for (unsigned i = 0; i < numOutput; ++i) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() * 
                prevLayer[m_myIndex].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::activation(sum);
}


//================================================================

class Net
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResult(vector<double> &resultVals) const;

private:
    vector<Layer> m_layers;  //m_layers[layerNum][neuroNum]
    double m_error;
    double m_recentAvgErr;
    double m_recentAvgSmoothingFactor;
};

void Net::getResult(vector<double> &resultVals) const 
{
    resultVals.clear();
    for (unsigned n = 0; n < m_layers.size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);

    // add input data to each of the neurons at the first layer
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
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
                     / (m_recentAvgSmoothingFactor + 1.0);
    
    for (unsigned int n = 0; n < outputLayer.size() - 1; ++n)
    {
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

    m_layers.back().back().setOutputVal(1.0);
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
};