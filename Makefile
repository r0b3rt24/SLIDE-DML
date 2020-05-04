all:
	g++ Net.cpp Neuron.cpp Dataloader.cpp main.cpp -Wall -O3 -o network

run:
	sbatch ./run.sh