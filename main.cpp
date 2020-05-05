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
//    omp_set_num_treads(4);
    auto start = std::chrono::high_resolution_clock::now();
    int epoch = 1;
    Dataloader dataloader("./data/e_data.txt");
    vector<unsigned> topology;
    vector<NeuronType> t;
    dataloader.getTopology(topology);
    dataloader.getType(t);
    Net myNet(topology, t);
    while (epoch <= 1) {
        cout << "Epoch: " << epoch << endl;
        epoch++;
        Dataloader dataloader("./data/e_data.txt");

        dataloader.getTopology(topology);
        dataloader.getType(t);

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

            cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;
        }
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    double total_time = elapsed.count();
    cout << "Total Forward Time: " << myNet.total_forward_time << endl;
    cout << "Total Backprop Time: " << myNet.total_backprop_time << endl;
    cout << "Total Training Time: " << total_time << endl;
    cout << endl << "Done" << endl;
};