
import numpy as np

from Evaluation.Evaluation import RMSE


def applyThreshold(landmarkList, thresh):
    coords = np.zeros((len(landmarkList), len(landmarkList[0]), 2))

    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            if (landmarkList[i][j].likelihood >= thresh):
                coords[i, j, :] = np.array([landmarkList[i][j].X, landmarkList[i][j].Y])
            else:
                coords[i, j, :] = np.array([np.nan, np.nan])
    return coords


def calculateManualError(arrays, setMeanAsGroundTruth: bool):
    error = []
    mean = np.zeros(arrays.shape[1], arrays.shape[2], arrays.shape[3])

    if setMeanAsGroundTruth:
        for array in arrays:
            mean += array
        mean = mean/arrays.shape[0]

        for array in arrays:
            error += RMSE(array, mean)
        error = error/arrays.shape[0]
    else:
        for array in arrays[1:]:
            error += RMSE(array, arrays[0])
        error = error/arrays.shape[0]
    return error
