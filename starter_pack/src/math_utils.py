"""
math_utils.py
-------------
Core math primitives used by both models.
 
Functions
---------
softmax(S)                          numerically stable row-wise softmax
one_hot(y, k)                       integer labels to one-hot matrix
cross_entropy_loss(P, Y_oh, weights, lam)  mean cross-entropy + L2 reg


https://medium.com/@preethithakur/softmax-regression-93808c02e6ac
https://d2l.ai/chapter_linear-classification/softmax-regression.html
"""
import numpy as np
 

def softmax(S):
    S_shifted = S - S.max(axis=1, keepdims=True)
    exp_S = np.exp(S_shifted)
    P = exp_S / exp_S.sum(axis=1, keepdims=True)
    return P
 

def one_hot(y, k):
    n = y.shape[0]
    Y_oh = np.zeros((n, k), dtype=np.float64)
    Y_oh[np.arange(n), y] = 1.0
    return Y_oh
 
 
def cross_entropy_loss(P, Y_oh, weights, lam):
    n = P.shape[0]
    P_clipped = np.clip(P, 1e-15, 1.0)
    ce = -np.sum(Y_oh * np.log(P_clipped)) / n
    l2 = sum(np.sum(W ** 2) for W in weights)
    l2_term = (lam / 2.0) * l2
    return ce + l2_term     

