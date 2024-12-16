#ifndef NODE_CUH
#define NODE_CUH

#include "include.h"

class Node;
typedef std::vector<Node> Layer;

struct Link
{
	double weight;
};

class Node
{
public:
	Node(int, int, double);
	void initializeWeights(double = 0.01);
	void forward(const Layer&);
	void backward(const Layer&);
	void updateWeight(const Layer&);

	int idx;
	double value = 0.;
	double gradient = 0.;
	double learningRate = 0.01;
	std::vector<Link> links;
};

#include "Node.cu"

#endif