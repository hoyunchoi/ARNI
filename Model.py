import os
import numpy as np
import itertools
from scipy.integrate import odeint
from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, adjacency: np.ndarray, *args, **kwargs) -> None:
        # Information about underlying network
        self.adjacency = adjacency
        self.networkSize = adjacency.shape[0]

        # Default time resolution
        self.resolution = 1

        # Information about real network
        self.numVariable: int = None

    @classmethod
    @abstractmethod
    def getTrueAdjacency(cls, adjacency: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def getInitialCondition(self) -> np.ndarray:
        pass

    @abstractmethod
    def dydt(self, variable: np.ndarray, t: np.ndarray) -> np.ndarray:
        pass

    def __singleRun(self, length: int) -> np.ndarray:
        """
            Run simulation
            Args
                length: Time length of simulation
        """
        self.time = np.arange(0, length * self.resolution, self.resolution)

        # Initialize condition
        initalCondition = self.getInitialCondition()

        # Single run
        return odeint(self.dydt, initalCondition, self.time)

    def run(self, length: int, ensemble: int) -> np.ndarray:
        """
            Run simulation ensemble times and save the data
            Args
                length: Time length of each simulation ensemble
                ensemble: Number of ensembles
        """
        # Variable to store simulation data
        self.data = np.empty((ensemble, length, self.numVariable))

        for e in range(ensemble):
            self.data[e] = self.__singleRun(length)
        return self.data

    def save(self, dataDirectory: str) -> None:
        """
            Save following file in dataDirectory
            'data.npy' : Simulated time series in a stacked form.
        """
        # Save dynamics
        np.save(os.path.join(dataDirectory, 'data'), self.data)


class KuramotoModel(Model):
    def __init__(self, adjacency: np.ndarray, *args, **kwargs) -> None:
        super().__init__(adjacency, *args, **kwargs)
        self.numVariable = self.networkSize
        self.trueAdjacency = self.adjacency

        # Initialize natural frequency
        self.naturalFrequency = np.random.uniform(low=-2.0, high=2.0, size=self.numVariable)

    def getInitialCondition(self) -> np.ndarray:
        return np.random.uniform(low=-np.pi, high=np.pi, size=self.numVariable)

    @classmethod
    def getTrueAdjacency(cls, adjacency: np.ndarray) -> np.ndarray:
        return adjacency

    @staticmethod
    def normalizePhase(variables: np.ndarray) -> np.ndarray:
        variables = np.mod(variables, 2*np.pi)
        variables[variables > np.pi] -= 2 * np.pi
        return variables

    def run(self, length: int, ensemble: int) -> np.ndarray:
        super().run(length, ensemble)
        self.data = self.normalizePhase(self.data)
        return self.data

class ParticleModel(Model):
    def __init__(self, adjacency: np.ndarray, *args, **kwargs) -> None:
        super().__init__(adjacency, *args, **kwargs)
        self.numVariable = 3 * self.networkSize

    @classmethod
    def getTrueAdjacency(cls, adjacency: np.ndarray) -> np.ndarray:
        trueAdjacency = np.zeros((cls.networkSize, cls.numVariable))

        # for each node, x1 regulated by x2 & x3. Modify adjacency matrix
        for i,j in itertools.product(range(cls.networkSize), range(cls.networkSize)):
            cls.trueAdjacency[i, 3*j] = adjacency[i, j]
        for i in range(cls.networkSize):
            cls.trueAdjacency[i, 3*i+1] = 1
            cls.trueAdjacency[i, 3*i+2] = 1

        return trueAdjacency

class Kuramoto1(KuramotoModel):
    def __init__(self, adjacency: np.ndarray, *args, **kwargs) -> None:
        super().__init__(adjacency, *args, **kwargs)

    def dydt(self, variable: np.ndarray, t: np.ndarray) -> np.ndarray:
        interaction = np.empty(self.numVariable)
        for i in range(self.numVariable):
            interaction[i] = sum(self.adjacency[i, j] * np.sin(variable[j] - variable[i]) for j in range(self.numVariable))

        return self.naturalFrequency + interaction


class Kuramoto2(KuramotoModel):
    def __init__(self, adjacency: np.ndarray, *args, **kwargs) -> None:
        super().__init__(adjacency, *args, **kwargs)

    def dydt(self, variable: np.ndarray, t: np.ndarray) -> np.ndarray:
        interaction = np.empty(self.numVariable)
        for i in range(self.numVariable):
            interaction[i] = sum(self.adjacency[i, j] * np.sin(variable[j] - variable[i] - 1.05)
                                 + 0.33 * np.sin(2 * (variable[j] - variable[i])) for j in range(self.numVariable))
        return self.naturalFrequency + interaction


class Michaelis_Menten(Model):
    def __init__(self, adjacency: np.ndarray, *args, **kwargs) -> None:
        super().__init__(adjacency, *args, **kwargs)
        self.numVariable = self.networkSize
        self.trueAdjacency = self.adjacency

    def getInitialCondition(self) -> np.ndarray:
        return 1.0 + np.random.uniform(0.0, 1.0, size=self.numVariable)

    def dydt(self, variable: np.ndarray, t: np.ndarray) -> np.ndarray:
        interaction = np.empty(self.numVariable)
        for i in range(self.numVariable):
            interaction[i] = sum(self.adjacency[i, j] * variable[j] / (1 + variable[j]) for j in range(self.numVariable))
        return -variable + interaction

    @classmethod
    def getTrueAdjacency(cls, adjacency: np.ndarray) -> np.ndarray:
        return adjacency

class Roessler(ParticleModel):
    def __init__(self, adjacency: np.ndarray, *args, **kwargs) -> None:
        super().__init__(adjacency, *args, **kwargs)

    def getInitialCondition(self) -> np.ndarray:
        return np.random.uniform(low=-5.0, high=5.0, size=self.numVariable)

    def dydt(self, variable: np.ndarray, t: np.ndarray) -> np.ndarray:
        dot = np.empty_like(variable)
        for i in range(self.networkSize):
            interaction = sum(self.adjacency[i, j] * np.sin(variable[3 * j]) for j in range(self.networkSize))
            dot[3 * i] = -variable[3 * i + 1] - variable[3 * i + 2] + interaction
            dot[3 * i + 1] = variable[3 * i] + 0.1 * variable[3 * i + 1]
            dot[3 * i + 2] = 0.1 + variable[3 * i + 2] * (variable[3 * i] - 18.0)
        return dot

if __name__ == "__main__":
    a = np.ndarray([7, 5])
    a = np.mod(a, 2*np.pi)
    a[a>np.pi] -= 2*np.pi
    print(a)