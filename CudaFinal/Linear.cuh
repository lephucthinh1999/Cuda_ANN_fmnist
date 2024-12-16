#ifndef LINEAR_CUH
#define LINEAR_CUH

#include "include.h"
#include "Node.cuh"

typedef std::vector<Node> Layer;
typedef std::vector<std::vector<double>> DataFrame;

class Linear
{
public:
	Linear(int, int, double learningRate);
	DataFrame forward(const DataFrame&);
	void backward(const Layer&);

	double learningRate = 0.01;
	Layer inputs;
	Layer outputs;

private:
	int inDim;
	int outDim;
};

#include "Linear.cu"

#endif

