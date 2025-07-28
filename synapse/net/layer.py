import numpy as np

import synapse

class FullyConnectedLayer:
    def __init__(self, n_inputs, n_outputs):
        """
        Initialize the weights and biases. Weights are initialized using He initialization, while biases are sampled from a Gaussian distribution
        with mean 0 and standard deviation 1. Biases are initialized as a column vector
        """
        self.weights = synapse.Tensor(np.random.randn(n_outputs, n_inputs) * np.sqrt(2 / n_inputs))
        self.biases = synapse.Tensor(np.random.randn(n_outputs, 1))

    def forward(self, inputs) -> np.array:
        """
        Forward pass through this layer. (Expects each batch to be a column vector)
        """
        return self.weights @ inputs + self.biases

    