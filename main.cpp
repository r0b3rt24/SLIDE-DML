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
    int epoch = 0;
    Dataloader dataloader("./data/e_data.txt");
    vector<unsigned> topology;
    dataloader.getTopology(topology);
    Net myNet(topology);
    while (epoch <= 1) {
        epoch++;
        cout << "Epoch: " << epoch << endl;
        Dataloader dataloader("./data/e_data.txt");
        vector<unsigned> topology;

        dataloader.getTopology(topology);
//        Net myNet(topology);

        vector<double> inputVals, targetVals, resultVals;
        int trainPasses = 0;

        // train the model
        while (!dataloader.isEof()) {
            ++trainPasses;
            auto next = dataloader.getNext(inputVals);
            if (next != topology[0]) {
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
    }

//    cout << endl << "==> TESTING" << endl;
//    vector<double> inputVals, targetVals, resultVals;
//   // test the mode
//    Dataloader testloader("./ball_test.txt");
//    testloader.getTopology(topology);
//    int total_correct = 0;
//    int testPasses = 0;
//    while (!testloader.isEof()) {
//        ++testPasses;
//        testloader.getNext(inputVals);
//        myNet.feedForward(inputVals);
//        myNet.getResult(resultVals);
//        testloader.getTargetOutputs(targetVals);
//        if (resultVals[0] > 3.9 && resultVals[0] < 4.1) total_correct++;
////        cout << resultVals[0] << ";" << targetVals[0] << endl;
////        if (resultVals[0] >= 0.5 && targetVals[0] == 1.0) {
////            ++total_correct;
////        } else if (resultVals[0] < 0.5 && targetVals[0] == 0.0) {
////            ++total_correct;
////        }
//    }
//    cout << endl << "Acc: " << 1.0*total_correct/(1.0*testPasses) << endl;
    cout << endl << "Done" << endl;
};