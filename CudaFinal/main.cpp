#include "Network.cuh"
#include <iostream>
#include <iomanip>

int main() {
    // Create 10 sample inputs (10 samples, each with 784 features)
    DataFrame input(10, std::vector<double>(784));

    // Initialize each sample with different values for testing
    for (int sample = 0; sample < 10; ++sample) {
        for (int feature = 0; feature < 784; ++feature) {
            // Generate some varying test data
            input[sample][feature] = (sample + feature % 10) / 10.0;
        }
    }

    // Create network
    Network network;

    try {
        // Forward pass
        DataFrame output = network.forward(input);

        // Print output values for each sample
        std::cout << std::fixed << std::setprecision(4);
        for (size_t sample = 0; sample < output.size(); ++sample) {
            std::cout << "\nSample " << sample << " outputs:" << std::endl;
            for (size_t node = 0; node < output[sample].size(); ++node) {
                std::cout << "Class " << node << ": " << output[sample][node] << "\t";
                if (node % 5 == 4) std::cout << std::endl;  // Line break every 5 values
            }
            std::cout << "\n";
        }
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}