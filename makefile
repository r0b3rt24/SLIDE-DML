all:
	g++ Net.cpp Neuron.cpp Dataloader.cpp main.cpp -Wall -O3 -o network -fopenmp

run:
	sbatch ./run.sh