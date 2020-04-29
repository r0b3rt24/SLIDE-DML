#ifndef NEURON_H
#define NEURON_H

#include "Connection.h"
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;

class Neuron;

typedef vector<Neuron> Layer;

class Neuron {
public:
    Neuron(unsigned numOutput, unsigned myIndex);

    void feedForward(const Layer &prevLayer);

    void setOutputVal(double val) { m_outputVal = val; }

    double getOutputVal() const { return m_outputVal; }

    void calcOutputGradients(double targetVal);

    void calcHiddenGradients(const Layer &nextLayer);

    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;
    static double alpha;

    static double activation(double x);

    static double activationDerivative(double x);

    double m_outputVal{};
    vector<Connection> m_outputWeights;

    static double randomWeight() {
        return rand() / double(RAND_MAX);
    }

    unsigned m_myIndex;
    double m_gradient{};

    double sumDOW(const Layer &nextLayer) const;
};

#endif