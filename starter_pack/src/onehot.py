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
 

def one_hot(y, k):
    n = y.shape[0]
    Y_oh = np.zeros((n, k), dtype=np.float64)
    Y_oh[np.arange(n), y] = 1.0
    return Y_oh
 
