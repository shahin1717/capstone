import numpy as np
from config import Config as cfg


class Optimizer:
    def __init__(self, parameters, learning_rate, weight_decay=cfg.LAMBDA):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def zero_grad(self, ):
        for p in self.parameters:
            p.grad = np.zeros_like(p.grad)


class SGD(Optimizer):
    def __init__(self, parameters, learning_rate, weight_decay=cfg.LAMBDA):
        super().__init__(parameters, learning_rate, weight_decay)

    def step(self, ):
        for p in self.parameters:
            p.grad += self.weight_decay * p.data
            p.data -= self.learning_rate * p.grad


class MomentumOptimizer(Optimizer):
    def __init__(self, parameters, learning_rate, momentum_coeff=0.9, weight_decay=cfg.LAMBDA):
        super().__init__(parameters, learning_rate, weight_decay)
        self.momentum_coeff = momentum_coeff
        self.velocities = [np.zeros_like(p.data) for p in parameters]

    def step(self, ):
        for p, v in zip(self.parameters, self.velocities):
            p.grad += self.weight_decay * p.data
            v[:] = self.momentum_coeff * v + p.grad
            p.data -= self.learning_rate * v


class Adam(Optimizer):
    def __init__(self, parameters, learning_rate, b1=0.9, b2=0.999, eps=1e-8, weight_decay=cfg.LAMBDA):
        super().__init__(parameters, learning_rate, weight_decay)
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        self.t = 0

    def step(self, ):
        self.t += 1
        for i, p in enumerate(self.parameters):
            p.grad += self.weight_decay * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (p.grad ** 2)

            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)

            p.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
