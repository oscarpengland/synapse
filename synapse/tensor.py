import numpy as np

class Tensor:
    # Set the array priority higher than numpy's default to ensure that operations are handled by the tensor class.
    __array_priority__ = 1000.0

    def __init__(self, data):
        arr = np.asarray(data)
        self.data = arr.copy()
    
    def __add__(self, other):
        """
        Elementwise sum of two Tensor objects.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data)
        return out
    
    def __mul__(self, other):
        """
        Elementwise multiplication of two Tensor objects.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data)
        return out
    
    def __matmul__(self, other):
        """
        Matrix multiplication of two Tensor objects.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data)
        return out
    
    def __pow__(self, other):
        """
        Elementwise raise to the power of.
        """
        assert isinstance(other, (int, float)), 'Only supporting int/float powers for now'
        out = Tensor(self.data**other)
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rmatmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other @ self
    
    def __neg__(self):
        return Tensor(-self.data)
    
    def exp(self):
        out = Tensor(np.exp(self.data))
        return out
    
    def __repr__(self):
        """
        Display tensor as an input string.
        """
        return f'Value : \n{self.data}'