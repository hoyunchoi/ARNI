from collections import deque
import numpy as np

class Reconstruction:
    def __init__(self, rawData: np.ndarray, nodeIdx: int) -> None:
        """
            rawData: data points of multi-variative time series
            nodeIdx: target node to be reconstructed
        """
        self.rawData = rawData.reshape(50, 10, 25)
        # self.rawData = rawData
        self.ensemble, self.length, self.numVariable = self.rawData.shape
        self.numData = self.ensemble * (self.length -1)
        self.nodeIdx = nodeIdx

        # (averaged) data points
        self.dynamics: np.ndarray = np.empty((self.ensemble, self.length - 1, self.numVariable))
        # Time derivatives of dynamics of target node
        self.nodeDerivative: np.ndarray = np.empty((self.ensemble, self.length - 1))

        # Default threshold to check if algorithm is converged or not
        self.threshold = 1e-4


    def getDynamics(self) -> np.ndarray:
        """
            Get (averaged) data points and time derivatives of raw data
        """
        for e, dyn in enumerate(self.rawData):
            self.dynamics[e] = (dyn[1:, :] + dyn[:-1, :]) / 2.0
            self.nodeDerivative[e] = dyn[1:, self.nodeIdx] - dyn[:-1, self.nodeIdx]     #! Resolution is 1 for default

        # Concatenate ensembles of dynamics/nodeDerivative
        self.dynamics = self.dynamics.reshape((self.numData, self.numVariable))
        self.nodeDerivative = self.nodeDerivative.reshape(self.numData)

        return self.dynamics


    def reconstruct(self, basisExpansion: np.ndarray) -> tuple[np.ndarray]:
        """
            Reconstruct dynamics with basis expansion
            Args
                basisExpansion: basis expansion of dynamics under certain basis
            Return
                neighbor: array of most probable neighbors
                adjacencyProb: probability of each neighbors
                cost: array of MSE loss between time derivatives when considering i-th neighbor
        """
        remainingSet = set(range(self.numVariable))    # Remaining variable sets not yet classifed as neighbor
        neighborDeque = deque()                        # variable deque classifed as neighbor
        costDeque = deque()                            # Cost deque w.r.t neighbor deque
        adjacencyProb = np.zeros(self.numVariable)     # Probability of active link for each neighbor

        approx = np.array([])            # Approximation of dynamics using variables at neighbor deque

        while remainingSet:
            # projection on remaining composite spaces
            projectionErr: dict[int, float] = dict.fromkeys(remainingSet)   # value: error of projection
            projectionCost: dict[int, float] = dict.fromkeys(remainingSet)  # value: MSE loss

            for variable in remainingSet:
                newApprox = self.__getNewApprox(approx, basisExpansion[:, :, variable])
                difference = self.__getDifference(newApprox)

                # Save projection error and cost
                projectionErr[variable] = np.std(difference)
                projectionCost[variable] = np.linalg.norm(difference) / self.numData

            # break if all candidates equivalent
            if self.__equivalentErr(projectionErr):
                break

            # Otherwise, save the best neighbor
            bestNeighbor = min(projectionErr, key=projectionErr.get)                # get best neighbor
            approx = self.__getNewApprox(approx, basisExpansion[:, :, bestNeighbor])  # update approximation
            neighborDeque.append(bestNeighbor)                                      # Append best neighbor
            remainingSet.remove(bestNeighbor)                                       # Remove best neighbor
            adjacencyProb[bestNeighbor] = projectionErr[bestNeighbor]               # Update adjacency probability
            costDeque.append(projectionCost[bestNeighbor])                          # Update cost

        return np.array(neighborDeque), adjacencyProb, np.array(costDeque)


    def __getNewApprox(self, approx: np.ndarray, candidate: np.ndarray):
        """
            Get new approx: candidate concatenated after current approx
            Args
                approx: Approximation of dynamics using nodes at neighbor deque
                candidate: basis expansion of some neighbor at remaining set
        """
        try:
            return np.vstack([approx, candidate])
        except ValueError:
            # When approx is empty: This is first candidate
            return candidate

    def __getDifference(self, approx: np.ndarray) -> np.ndarray:
        """
            Get difference between true label: self.nodeDerivative and input approximation
            difference: error of projection on approximated space
        """
        temp = np.dot(self.nodeDerivative, np.linalg.pinv(approx))
        return self.nodeDerivative - np.dot(temp, approx)

    def __equivalentErr(self, projectionErr: dict[int, float]) -> bool:
        """
            Return true when all variables at remaining set are equivalent
        """
        return np.std(list(projectionErr.values())) < self.threshold

if __name__ == "__main__":
    a = deque([1,2,3])
    b = np.array([5,7,9,9,4,2,1,35,76,1])

    print(b[a])

    # a = np.array([[1,2,3], [4,5,6]])
    # print(a)
    # print(a.shape)

    # index = np.array([])
    # print(a[:, index])
