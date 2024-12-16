#include "Node.cuh"

Node::Node(int idx, int weightDim, double learningRate) : idx(idx), links(weightDim), learningRate(learningRate)
{
	initializeWeights();
}

void Node::initializeWeights(double range)
{
	thread_local std::random_device rd;
	thread_local std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-range, range);

	for (auto& link : links)
	{
		link.weight = dis(gen);
	}
}

void Node::forward(const Layer& prevLayer)
{
	auto sum = 0.;
	for (const auto& node : prevLayer)
	{
		sum += node.value * node.links[idx].weight;
	}
	value = sum;
}

void Node::backward(const Layer& nextLayer)
{
	auto sum = nextLayer.size() > 0 ? 0. : 1.;
	for (const auto& node : nextLayer)
	{
		sum += node.gradient * links[node.idx].weight;
	}
	gradient = sum;
}

void Node::updateWeight(const Layer& nextLayer)
{
	for (const auto& node : nextLayer)
	{
		links[node.idx].weight -= learningRate * value * node.gradient;
	}
}
