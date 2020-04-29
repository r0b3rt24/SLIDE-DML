#ifndef NET_H
#define NET_H

#include <vector>
#include <cassert>
#include <iostream>
#include "Neuron.h"

using namespace std;

class Net {
public:
    explicit Net(const vector<unsigned> &topology);

    void feedForward(const vector<double> &inputVals);

    void backProp(const vector<double> &targetVals);

    void getResult(vector<double> &resultVals) const;

private:
    vector<Layer> m_layers;  //m_layers[layerNum][neuroNum]
    double m_error;
    double m_recentAvgErr;
    double m_recentAvgSmoothingFactor;
};

#endif