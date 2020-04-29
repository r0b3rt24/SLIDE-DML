//
// Created by Han Cao on 4/28/20.
//

#ifndef SLIDE_DML_DATALOADER_H
#define SLIDE_DML_DATALOADER_H

#include <string>
#include <fstream>

using namespace std;
class Dataloader
{
public:
    Dataloader(const string datapath);
    bool isEof(void) {return m_trainingDataFile.eof();}
    void getTopology(vector<unsigned> &topology);

    unsigned  getNext(vector<double> &inputVals);
    unsigned  getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

#endif //SLIDE_DML_DATALOADER_H
