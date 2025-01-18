import numpy as np
import torch
from scipy.spatial import distance

def calcADJ(coord, k=4, distanceType='euclidean', pruneTag='NA'):
    spatialMatrix = coord
    nodes = spatialMatrix.shape[0]
    edges = []

    for i in range(nodes):
        tmp = spatialMatrix[i, :].reshape(1, -1)
        distMat = distance.cdist(tmp, spatialMatrix, distanceType)
        if k == 0:
            k = spatialMatrix.shape[0] - 1
        res = distMat.argsort()[:k + 1]
        tmpdist = distMat[0, res[0][1:k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in range(1, k + 1):
            if pruneTag == 'NA' or (pruneTag == 'STD' and distMat[0, res[0][j]] <= boundary) or (pruneTag == 'Grid' and distMat[0, res[0][j]] <= 2.0):
                edges.append((i, res[0][j]))

    edge_index = torch.tensor(edges).t()
    return edge_index
