import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from Common import NetworkList, ModelList, BasisList
from Reconstruction import Reconstruction
from Evaluation import Evaluation

rootDirectory = '/pds/pds11/hoyun/ARNI/data'

#* Network parameter
networkSize = 25
meanDegree = 6
network_cls = NetworkList.ER.value
directed = True

#* Model parameters
model_cls, model_name = ModelList.kuramoto1.value

#* Simulation parameters
length = 10
ensemble = 50
dataDirectory = os.path.join(rootDirectory, model_name)
Path(dataDirectory).mkdir(parents=True, exist_ok=True)

network = network_cls(networkSize, meanDegree)        # Create network instance

#*  Generate network
######################################################
# adjacency = network.getAdjacency(directed)            # Create network (adjacency matrix)
# network.save(dataDirectory)                           # Save adjacency matrix
######################################################


#* Read network
######################################################
adjPath = os.path.join(dataDirectory, 'adjacency.dat')
adjacency = np.loadtxt(adjPath)
adjacency = np.loadtxt('/pds/pds11/hoyun/ARNI/Data/connectivity.dat')
######################################################

model = model_cls(adjacency)                            # Create model instance

#* Generate data
######################################################
# data = model.run(length, ensemble)                      # Run model
# model.save(dataDirectory)                               # Save model time series data
######################################################


#* Read data
######################################################
dataPath = os.path.join(dataDirectory, 'data.npy')
data = np.load(dataPath)
data = np.loadtxt('/pds/pds11/hoyun/ARNI/Data/data.dat')
######################################################

######################################################
#* Reconstruction parameters
nodeIdx = 1
basis_cls = BasisList.RBF.value
maxOrder = 6
trueNeighbors = adjacency[nodeIdx]
numNeighbors = np.count_nonzero(trueNeighbors)
trueNeighbors = np.argsort(trueNeighbors)[::-1]
trueNeighbors = trueNeighbors[:numNeighbors]
######################################################

arni = Reconstruction(data, nodeIdx)            # Reconstruction instance is created
dynamics = arni.getDynamics()                   # Dynamics is created from raw data

# Expand dynamics with basis
basis = basis_cls(dynamics, nodeIdx, maxOrder)  # Basis instance is created
basisExpansion = basis.expand()                 # Dynamics is expanded by basis

# Reconstruct interaction with basis expansion
neighbors, adjacencyProb, costArray = arni.reconstruct(basisExpansion)

evaluator = Evaluation(adjacency, model)

fpr, tpr, aucScore = evaluator.evaluate(nodeIdx, adjacencyProb)
print(f'AUC score: {aucScore}')