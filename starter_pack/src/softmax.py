import numpy as np
from lite_torch import Tensor

class SoftmaxRegression:
    """
    @brief Implements a softmax regression model for multi-class classification
    @param input_dim: Dimensionality of the input features
    @param target_dim: Number of classes for classification
    @param seed: Random seed for reproducibility of parameter initialization
    """
    
    def __init__(self, input_dim, target_dim, seed: int = 0):
        rng = np.random.default_rng(seed)
        
        scale = 1.0 / np.sqrt(input_dim)
        W = rng.uniform(-scale, scale, size=(input_dim, target_dim))
        b = np.zeros((1, target_dim), dtype=np.float64)
 
        self.W = Tensor(W)
        self.b = Tensor(b)
        
    def __call__(self, X):
        S = X @ self.W + self.b          
        P = S.softmax(axis=-1)                    
        return P

    def predict(self, X):
        P = self(X)
        return np.argmax(P.data, axis=1)

    def parameters(self): 
        return [self.W, self.b]