import numpy as np

class Sigmoid:
    """
    Sigmoid activation function, squishes all input values between zero and one according to the following
    equation: s(z) = 1/(1 + e^-z)
    """
    def __init__(self):
        pass
    
    def forward(self, z):
        return (np.exp(-z) + 1.0)**-1.0
    
    def backward_pass(self, z, next_layer_error):
        """
        Backpropagate the errors through the sigmoid layer, by evaluating the derivative of the sigmoid
        function at the inputs
        """
        pass