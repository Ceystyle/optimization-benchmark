"""This is a module containing optimizers."""

import numpy as np

class StupidGradientDescent:

    def __init__(self, alpha, initstate, fgrad):
        self.state = initstate
        self.alpha = alpha
        self.fgrad = fgrad

    def step(self):
        self.state -= self.alpha * self.fgrad(self.state)

class MomentumGradientDescent:

    def __init__(self, alpha, gamma, initstate, fgrad):
        self.state = initstate
        self.alpha = alpha
        self.gamma = gamma
        self.v = np.zeros((len(initstate),))
        self.fgrad = fgrad

    def step(self):
        self.v *= self.gamma*self.v
        self.v += self.alpha*self.fgrad(self.state)
        self.state -= self.v