# SLIDE-DML
Final Project for CS759

# TO RUN THE PROGRAM
1. Download the formatted dataset from https://drive.google.com/open?id=1KAmKIO-Jy5a_zyBDHea6GYdw_gny7ft2

2. Move it under the folder called data. So the path should look like ./data/train.txt

3. Add the desired structe of the model on the top of the train.txt for example

topology: 728 200 100 10
type: input tanh relu output 

It means the first layer has 728 input, the second layer has 200 nodes using tanh activation function, the third layer using relu with 100 nodes, and the output layer will always be softmax. 

4. adding the desired learning rate and momentum constant in Neuron.c

5. run the program with specific thread number 

./network 16

In this case, it means run the program with 16 threads. 
