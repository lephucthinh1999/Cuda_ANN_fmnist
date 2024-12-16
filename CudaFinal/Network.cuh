#ifndef NETWORK_CUH
#define NETWORK_CUH

#include "include.h"
#include "Node.cuh"
#include "Linear.cuh"

typedef std::vector<Node> Layer;
typedef std::vector<std::vector<double>> DataFrame;

class Network
{
public:
	Network(double = 0.01);
	DataFrame forward(const DataFrame&);
	void backward();

	double learningRate = 0.01;
	Linear inputLayer = Linear(784, 128, learningRate);
	Linear hiddenLayer = Linear(128, 128, learningRate);
	Linear outputLayer = Linear(128, 10, learningRate);
};

#include "Network.cu"

#endif