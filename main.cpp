#include "Net.h"
#include <vector>
#include "Dataloader.h"

using namespace std;

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}


int main() {

    // init dataloader
    Dataloader dataloader("./dataset.txt");
    vector<unsigned> topology;

    dataloader.getTopology(topology);
    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainPasses = 0;

    while (!dataloader.isEof()) {
        ++trainPasses;

        if (dataloader.getNext(inputVals) != topology[0]) {
            cout << "input dimension doesn't match" << endl;
            break;
        }

        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        myNet.getResult(resultVals);
        showVectorVals("Outputs:", resultVals);

        dataloader.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        cout << "Net recent average error: "
             << myNet.getRecentAverageError() << endl;
    }

    cout << endl << "Done" << endl;
};