"""This is a module containing functions and their gradients."""

import numpy as np

class Rosenbrock:

    def __init__(self, dim=2, b=4.):
        self.dim = dim
        self.b = b
        self.a = 1.

    def eval(self, x):
        return np.sum(self.b * (x[1:] - x[:-1 ** 2]) ** 2 + (self.a - x[:-1]) ** 2)

    def grad(self, x):
        out = np.empty((self.dim,))
        if self.dim > 2:
            out[1:-1] = (-4) * self.b * (x[2:] - x[1:-1] ** 2) * x[1:-1] - 2 * (self.a - x[1:-1]) \
                        + 2 * self.b * (x[1:-1] - x[:-2] ** 2)
        out[0] = (-4) * self.b * (x[1] - x[0] ** 2) * x[0] - 2 * (self.a - x[0])
        out[-1] = 2 * self.b * (x[-1] - x[-2] ** 2)
        return out

    def optimal(self):
        return (np.ones((self.dim,)), 0.)


class StyblinskiTang:

    def __init__(self, dim=2):
        self.dim = dim

    def eval(self, x):
        return (1./2.)*np.sum(x**4-16.*x**2+5.*x)

    def grad(self, x):
        return 2.*x**3-16.*x+2.5

    def optimal(self):
        return (-2.903534*np.ones((self.dim,)), -39.16599*self.dim)