#include "Neuron.h"
#include "Connection.h"
#include <chrono>

double Neuron::eta = 0.05;    // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0;   // momentum, multiplier of last deltaWeight, [0.0..1.0]

double Neuron::activation(double x) {
    return tanh(x);
}

double Neuron::activationDerivative(double x) {
    return 1.0 - x * x;  // an approximation to the derivative (faster)
}

void Neuron::feedForward(const Layer &prevLayer) {
    auto start = std::chrono::high_resolution_clock::now();  // timing start
    double sum = 0.0;

    // TODO: bias??
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
               prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
//    sum += prevLayer.back().m_outputWeights[m_myIndex].weight;  // Bias

    if (if_sotfmax) {
        m_inputVal = sum;  //m_outputVal = inputVal
    } else {
        m_outputVal = Neuron::activation(sum);
    }
    auto finish = std::chrono::high_resolution_clock::now();  // timing ends
    std::chrono::duration<double> elapsed = finish - start;
}

double Neuron::softmax(Layer &thisLayer) {
    double sumExp = 0;

    for (unsigned n = 0; n < thisLayer.size()-1; ++n) {
        Neuron &neuron = thisLayer[n];
        sumExp += exp(neuron.m_inputVal);
    }

    m_outputVal = exp(thisLayer[m_myIndex].m_inputVal)/sumExp;
}

void Neuron::updateInputWeights(Layer &prevLayer) {
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                eta
                * neuron.getOutputVal()
                * m_gradient
                + alpha
                  * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;  // updated the w
    }
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double sum = 0;
    for (unsigned n = 0; n < nextLayer.size()-1; n++) {
        sum += nextLayer[n].m_gradient * m_outputWeights[n].weight * Neuron::activationDerivative(m_outputVal);
    }
    m_gradient = sum;
}

void Neuron::calcOutputGradients(double o_gradient) {
    m_gradient = o_gradient;
}


Neuron::Neuron(unsigned numOutput, unsigned myIndex) {
    for (unsigned i = 0; i < numOutput; ++i) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}