from enum import Enum
from Network import ER, SF
from Model import Kuramoto1, Kuramoto2, Michaelis_Menten, Roessler
from Basis import Polynomial, PolynomialDiff, Fourier, FourierDiff, Power, RBF

class NetworkList(Enum):
    """ Available network topologies"""
    ER = ER
    SF = SF

class ModelList(Enum):
    """List of available models"""
    kuramoto1 = Kuramoto1, 'kuramoto1'
    kuramoto2 = Kuramoto2, 'kuramoto2'
    michaelis_menten = Michaelis_Menten, 'michelis_menten'
    roessler = Roessler, 'roessler'

class BasisList(Enum):
    """Available basis"""
    polynomial = Polynomial
    polynomialDiff = PolynomialDiff
    fourier = Fourier
    fourierDiff = FourierDiff
    power = Power
    RBF = RBF

if __name__ == "__main__":
    import numpy as np
    adj = np.empty((2,2))
    model = ModelList.kuramoto1.value(adj)
    print(model.getInitialCondition())


