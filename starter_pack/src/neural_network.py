import numpy as np
from lite_torch import Tensor

class Linear:
    """
    @brief A simple 2-layer feedforward neural network for classification.
    The network consists of a hidden layer with tanh activation and an output layer with softmax activation. The forward pass computes the class probabilities, and the parameters are updated using backpropagation.
    
    """
    
    def __init__(self, input_dim, hidden_dim, target_dim, seed=0):
        """
        @brief Initializes the Linear layer with random weights and biases.
        @param input_dim: The dimensionality of the input features.
        @param hidden_dim: The number of neurons in the hidden layer.
        @param target_dim: The number of output classes.
        @param seed: Random seed for reproducible parameter initialization.
        """
        
        rng = np.random.default_rng(seed)
        W1 = rng.uniform(low=-1, high=1, size=(input_dim, hidden_dim)) / (input_dim ** 0.5)
        b1 = np.zeros(shape=(1, hidden_dim))

        W2 = rng.uniform(low=-1, high=1, size=(hidden_dim, target_dim)) / (input_dim ** 0.5)
        b2 = np.zeros(shape=(1, target_dim))

        self.W1 = Tensor(W1)
        self.b1 = Tensor(b1)
        self.W2 = Tensor(W2)
        self.b2 = Tensor(b2)
    
    def __call__(self, x):
        z = x @ self.W1 + self.b1
        h = z.tanh()
        s = h @ self.W2 + self.b2
        p = s.softmax()
        return p
    
    def parameters(self,):
        return [self.W1, self.b1, self.W2, self.b2]
    
    def predict(self, x):
        probs = self(x)
        target = np.argmax(probs.data, axis=-1)
        return target