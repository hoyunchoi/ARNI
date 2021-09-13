from ast import NodeVisitor
import numpy as np
from sklearn.metrics import roc_curve, auc
from typing import Optional

from Model import Model

class Evaluation:
    def __init__(self, adjacency: np.ndarray, model:Model):
        # Network information
        self.binaryAdjacency = adjacency
        self.binaryAdjacency[self.binaryAdjacency != 0] = 1
        self.binaryAdjacency = model.getTrueAdjacency(self.binaryAdjacency)

    def evaluate(self, nodeIdx: int, adjacencyProb: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
        if not np.sum(adjacencyProb):
            print('WARNING: no predicted regulators - check that NODE abundance varies in the data!')
            return None

        fpr, tpr, _ = roc_curve(self.binaryAdjacency[nodeIdx, :], adjacencyProb)
        aucScore = auc(fpr, tpr)

        # Add origin point to fpr, tpr
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)

        return fpr, tpr, aucScore

if __name__ == "__main__":
    from Model import Kuramoto1
    adj = np.empty((2,2))
    model = Kuramoto1(adj)

    e = Evaluation(adj, model)
    print(e.binaryAdjacency)


