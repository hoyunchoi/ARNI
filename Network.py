import os
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod

class Network(ABC):
    def __init__(self, networkSize: int, meanDegree: int):
        self.networkSize = networkSize
        self.meanDegree = meanDegree
        self.adjacency: np.ndarry = None

    @abstractmethod
    def getAdjacency(self, directed:bool = True) -> np.ndarray:
        pass

    def save(self, dataDirectory: str) -> None:
        """
            Save following file in dataDirectory
            'adjacency.dat': Adjacency matrix of network underlying the dynamics
        """
        # Save adjacency matrix
        np.savetxt(os.path.join(dataDirectory, 'adjacency.dat'), self.adjacency)


class ER(Network):
    def getAdjacency(self, directed: bool = True) -> np.ndarray:
        # Create ER random network
        numEdge = self.networkSize * self.meanDegree if directed else self.networkSize * self.meanDegree / 2
        graph = nx.gnm_random_graph(self.networkSize, numEdge, directed=directed)
        graph.remove_edges_from(nx.selfloop_edges(graph))

        # Give random weight to edges
        for u,v in graph.edges():
            graph.edges[u,v]['weight'] = np.random.uniform(0.5, 1.0) / self.meanDegree

        self.adjacency = nx.to_numpy_array(graph)
        return self.adjacency


class SF(Network):
    def getAdjacency(self, directed: bool) -> np.ndarray:
        if directed:
            # Create directed scale free network
            beta, gamma = (self.meanDegree - 1)/self.meanDegree, 0.05
            alpha = 1-beta-gamma
            graph = nx.scale_free_graph(n=self.networkSize, alpha=alpha, beta=beta, gamma=gamma)
            graph.remove_edges_from(nx.selfloop_edges(graph))
            graph = nx.DiGraph(graph)
        else:
            # Create undirected scale free network: barabasi-albert network
            graph = nx.barabasi_albert_graph(self.networkSize, int(self.meanDegree/2))
            graph.remove_edges_from(nx.selfloop_edges(graph))

        # Give random weight to edges
        for u,v in graph.edges():
            graph.edges[u,v]['weight'] = np.random.uniform(0.5, 1.0) / self.meanDegree

        self.adjacency = nx.to_numpy_array(graph)
        return self.adjacency

if __name__ == "__main__":
    pass
