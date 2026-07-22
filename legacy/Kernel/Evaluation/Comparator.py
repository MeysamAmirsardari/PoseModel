import numpy as np

from Kernel.Evaluation.Evaluation import RMSE


def applyThreshold(landmarkList, thresh):
    coords = np.zeros((len(landmarkList), len(landmarkList[0]), 2))

    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            if (float(landmarkList[i][j].likelihood) >= thresh):
                coords[i, j, :] = np.array([landmarkList[i][j].X, landmarkList[i][j].Y])
            else:
                coords[i, j, :] = np.array([np.nan, np.nan])
    return coords


def calculateManualError(arrays, setMeanAsGroundTruth: bool):
    ## arrays: A list of numpy arrays
    error = np.zeros((1, arrays[0].shape[1]))
    print(error.shape)
    mean = np.zeros_like(arrays[0])

    if setMeanAsGroundTruth:
        for array in arrays:
            mean += array
        mean = mean / len(arrays)

        for array in arrays:
            step = np.array(array)
            error += RMSE(step, mean)
        error = error / len(arrays)
    else:
        for array in arrays[1:]:
            step = np.array(array)
            error += RMSE(step, arrays[0])
        error = error / len(arrays)
    return error


def calculatePredictionError(predTensor, targetTensor, frameIndexList):
    forAnalyze = np.zeros((len(frameIndexList), predTensor.shape[1], predTensor.shape[2]))

    for i in range(len(frameIndexList)):
        forAnalyze[i, :, :] = predTensor[frameIndexList[i], :, :]

    error = RMSE(forAnalyze, targetTensor)
    return error