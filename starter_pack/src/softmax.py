import numpy as np
from math_utils import softmax, one_hot, cross_entropy_loss
from config import LAMBDA
"""
Credits:
https://medium.com/@preethithakur/softmax-regression-93808c02e6ac
https://d2l.ai/chapter_linear-classification/softmax-regression.html


"""


class SoftmaxRegression:
    def __init__(self, d: int, k: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        
        scale = 1.0 / np.sqrt(d)
        self.W = rng.uniform(-scale, scale, size=(k, d))
        self.b = np.zeros(k, dtype=np.float64)
 
        self.d = d
        self.k = k
        
    def forward(self, X: np.ndarray):
        S = X @ self.W.T + self.b          
        P = softmax(S)                     
        return S, P
    
    def backward(self, X: np.ndarray, P: np.ndarray, Y_oh: np.ndarray, lam: float = LAMBDA):
        n = X.shape[0]
        dS = (P - Y_oh) / n
        dW = dS.T @ X + lam * self.W
        db = dS.T @ np.ones(n)
        return {"W": dW, "b": db}
    
    def loss(self, P: np.ndarray, Y_oh: np.ndarray,lam: float = LAMBDA):
        return cross_entropy_loss(P, Y_oh, [self.W], lam=lam)

    def predict(self, X: np.ndarray):
        _, P = self.forward(X)
        return np.argmax(P, axis=1)
    
    # Used by optimizer to get and set parameters
    def get_params(self) -> dict:
        return {"W": self.W, "b": self.b} 
    def set_params(self, params: dict) -> None:
        self.W = params["W"]
        self.b = params["b"]