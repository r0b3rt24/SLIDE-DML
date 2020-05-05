#ifndef NEURON_H
#define NEURON_H

#include "Connection.h"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <string>

using namespace std;

class Neuron;

typedef vector<Neuron> Layer;

enum class NeuronType{ReLU, Tanh, Softmax, Input};

class Neuron {
public:
    Neuron(unsigned numOutput, unsigned myIndex, NeuronType t);

    void feedForward(const Layer &prevLayer);

    void setOutputVal(double val) { m_outputVal = val; }

    double getOutputVal() const { return m_outputVal; }

    double softmax(Layer &prevLayer);

    void calcOutputGradients(double o_gradient);

    void calcHiddenGradients(const Layer &nextLayer);

    void updateInputWeights(Layer &prevLayer);

    double logit(double z);
    double logit_d(double z);

    double relu(double z);
    double relu_d(double z);

    double m_inputVal;
    double softmax_o;
    double m_d_weight;

private:
    static double eta;
    static double alpha;

    static double activation(double x);

    static double activationDerivative(double x);

    double m_outputVal;
    double m_logit_op;
    vector<Connection> m_outputWeights;

    static double randomWeight() {
        return rand()/double(RAND_MAX);
//        return 2;
//        return rand()%5;
    }

    unsigned m_myIndex;

    double d_bias;


    NeuronType n_type;
};

#endif