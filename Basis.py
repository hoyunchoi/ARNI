import numpy as np
import itertools
from abc import ABC, abstractmethod


class Basis(ABC):
    """ Basis expansion evalued on all points of a multivariate time series """

    def __init__(self, dynamics: np.ndarray, nodeIdx: int, maxOrder: int) -> None:
        """
            dynamics: Dynamics to be expanded
            nodeIdx: target index of node to be expanded
            maxOrder: maximum order of basis to be used
        """
        self.dynamics = dynamics
        self.nodeIdx = nodeIdx
        self.maxOrder = maxOrder
        self.numData, self.numVariable = dynamics.shape

    @abstractmethod
    def expand(self) -> np.ndarray:
        pass

    def getRelativeDynamics(self) -> np.ndarray:
        nodeDynamics = self.dynamics[:, self.nodeIdx].reshape(-1, 1)
        return self.dynamics - np.repeat(nodeDynamics, self.numVariable, axis=1)


class Polynomial(Basis):
    def expand(self) -> np.ndarray:
        """
            h_k (x_j) = x_j^^k
        """
        expansion = np.zeros((self.maxOrder + 1, self.numData, self.numVariable))

        for neighbor, order in itertools.product(range(self.numVariable), range(self.maxOrder)):
            expansion[order, :, neighbor] = np.power(self.dynamics[:, neighbor], order)

        return expansion


class PolynomialDiff(Basis):
    def expand(self) -> np.ndarray:
        """
            h_k (x_i, x_j) = (x_j - x_i)**k
        """
        expansion = np.zeros((self.maxOrder + 1, self.numData, self.numVariable))
        relativeDynamics = self.getRelativeDynamics()

        for neighbor, order in itertools.product(range(self.numVariable), range(self.maxOrder)):
            expansion[order, :, neighbor] = np.power(relativeDynamics[:, neighbor], order)
        return expansion


class Fourier(Basis):
    def expand(self) -> np.ndarray:
        expansion = np.zeros((2 * self.maxOrder, self.numData, self.numVariable))

        for neighbor, order in itertools.product(range(self.numVariable), range(self.maxOrder)):
            expansion[2 * order, :, neighbor] = np.sin(order * self.dynamics[:, neighbor])
            expansion[2 * order + 1, :, neighbor] = np.cos(order * self.dynamics[:, neighbor])
        return expansion


class FourierDiff(Basis):
    def expand(self) -> np.ndarray:
        expansion = np.zeros((2 * self.maxOrder, self.numData, self.numVariable))
        relativeDynamics = self.getRelativeDynamics()

        for neighbor, order in itertools.product(range(self.numVariable), range(self.maxOrder)):
            expansion[2 * order, :, neighbor] = np.sin(order * relativeDynamics[:, neighbor])
            expansion[2 * order + 1, :, neighbor] = np.cos(order * relativeDynamics[:, neighbor])
        return expansion


class Power(Basis):
    def expand(self) -> np.ndarray:
        expansion = np.zeros((self.maxOrder**2, self.numData, self.numVariable))

        for neighbor, order1, order2, data in itertools.product(range(self.numVariable),
                                                                range(self.maxOrder),
                                                                range(self.maxOrder),
                                                                range(self.numData)):
            expansion[self.maxOrder * order1 + order2, data, neighbor] = (self.dynamics[data, self.nodeIdx]**order1) * (self.dynamics[data, neighbor] ** order2)
        return expansion

class RBF(Basis):
    def expand(self) -> np.ndarray:
        expansion = np.zeros((self.maxOrder, self.numData, self.numVariable))

        for neighbor in range(self.numVariable):
            A = np.vstack((self.dynamics[:, neighbor], self.dynamics[:, self.nodeIdx]))
            for m1, m2 in itertools.product(range(self.maxOrder), range(self.numData)):
                expansion[m1, m2, neighbor] = np.sqrt(2.0 + np.linalg.norm(A[:, m1] - A[:, m2], 2.0) ** 2.0)
        return expansion

if __name__ == "__main__":
    pass