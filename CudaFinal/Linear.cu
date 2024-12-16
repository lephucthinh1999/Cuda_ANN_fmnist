#include "Linear.cuh"

Linear::Linear(int inDim, int outDim, double learningRate) : inDim(inDim), outDim(outDim), learningRate(learningRate)
{
	inputs.reserve(inDim);
	for (auto i = 0; i < inDim; ++i)
	{
		inputs.emplace_back(i, outDim, learningRate);
	}

	outputs.reserve(outDim);
	for (auto i = 0; i < outDim; ++i)
	{
		outputs.emplace_back(i, 0, learningRate);
	}
}

DataFrame Linear::forward(const DataFrame& x)
{
    for (auto i = 0; i < x.size(); ++i)
    {
        for (auto j = 0; j < inDim; ++j)
        {
            inputs[j].value = x[i][j];
        }
    }

    DataFrame result(x.size(), std::vector<double>(outDim));
    for (auto i = 0; i < x.size(); ++i)
    {
        for (auto& node : outputs)
        {
            node.forward(inputs);
            result[i][node.idx] = node.value;
        }
    }

    return result;
}

void Linear::backward(const Layer& nextLayer)
{
    for (auto& outputNode : outputs)
    {
        outputNode.backward(nextLayer);
    }
    
    for (auto& inputNode : inputs)
    {
        inputNode.backward(outputs);
    }

    for (auto& inputNode : inputs)
    {
        inputNode.updateWeight(outputs);
    }
}