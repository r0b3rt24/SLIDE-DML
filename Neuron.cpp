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

double Neuron::logit(double z) {
//    return 1/(1 + exp(-z));
    return z;
}

double Neuron::logit_d(double z) {
//    double sigmoid = 1/(1 + exp(-z));
//    return sigmoid * (1 - sigmoid);
    return z;
}

double Neuron::relu(double z) {
    return std::max(z,0.0);

}

double Neuron::relu_d(double z) {
    if (z > 0) {
        return 1;
    } else {
        return 0;
    }
}

void Neuron::feedForward(const Layer &prevLayer) {
    auto start = std::chrono::high_resolution_clock::now();  // timing start
    double sum = 0.0;
    Neuron bias = prevLayer.back();

    #pragma omp parallel for reduction(+:sum)
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
               prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    sum += bias.m_outputWeights[m_myIndex].weight;  // added the bias

    if (n_type == NeuronType::Softmax) {
        m_inputVal = logit(sum);  //m_outputVal = inputVal
    } else if (n_type == NeuronType::ReLU) {
        m_outputVal = Neuron::relu(sum);
    } else if (n_type == NeuronType::Tanh) {
        m_outputVal = Neuron::activation(sum);
    }

    auto finish = std::chrono::high_resolution_clock::now();  // timing ends
    std::chrono::duration<double> elapsed = finish - start;
}

double Neuron::softmax(Layer &thisLayer) {
    double sumExp = 0;

    #pragma omp parallel for reduction(+:sumExp)
    for (unsigned n = 0; n < thisLayer.size()-1; ++n) {
        sumExp += exp(thisLayer[n].m_inputVal);
    }

    m_outputVal = exp(thisLayer[m_myIndex].m_inputVal)/sumExp;
    softmax_o = m_outputVal;
    return m_outputVal;
}

void Neuron::updateInputWeights(Layer &prevLayer) {

    Neuron bias = prevLayer.back();
    #pragma omp parallel for
    for (unsigned n = 0; n < prevLayer.size() - 1; ++n) {
        double oldDeltaWeight = prevLayer[n].m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                eta
                * prevLayer[n].getOutputVal()
                * m_d_weight
                + alpha
                  * oldDeltaWeight;

        prevLayer[n].m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        prevLayer[n].m_outputWeights[m_myIndex].weight -= newDeltaWeight;  // updated the w
        bias.m_outputWeights[m_myIndex].weight -= m_d_weight * eta;  // update b
    }
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    NeuronType t = n_type;
    double sum = 0;

    if (t == NeuronType::ReLU) {
        #pragma omp parallel for reduction(+:sum)
        for (unsigned n = 0; n < nextLayer.size() - 1; n++) {
            sum += nextLayer[n].m_d_weight * m_outputWeights[n].weight;
        }
        sum = sum * Neuron::relu_d(m_outputVal);
    } else if (t == NeuronType::Tanh) {
        #pragma omp parallel forreduction(+:sum)
        for (unsigned n = 0; n < nextLayer.size()-1; n++) {
            sum += nextLayer[n].m_d_weight * m_outputWeights[n].weight;
        }
        sum = sum * Neuron::activationDerivative(m_outputVal);
    }

    m_d_weight = -sum;
}

void Neuron::calcOutputGradients(double o_gradient) {  // this is for softmax
//    m_d_weight = o_gradient * logit_d(m_inputVal);  // with respect to weight
    m_d_weight = o_gradient;
}

Neuron::Neuron(unsigned numOutput, unsigned myIndex, NeuronType t) {
    for (unsigned i = 0; i < numOutput; ++i) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    n_type = t;
    m_myIndex = myIndex;
}