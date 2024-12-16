#include "Network.cuh"

Network::Network(double learningRate) : learningRate(learningRate) {}

DataFrame Network::forward(const DataFrame& x)
{
    auto hidden1 = inputLayer.forward(x);
    auto hidden2 = hiddenLayer.forward(hidden1);
    auto output = outputLayer.forward(hidden2);

    return output;
}

void Network::backward()
{
    outputLayer.backward({});
    hiddenLayer.backward(outputLayer.inputs);
    inputLayer.backward(hiddenLayer.inputs);
}