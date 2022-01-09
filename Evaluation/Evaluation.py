import numpy as np
# from distributed.protocol import torch

from fileHandler import Landmark

def MPJPE(predicted, target, bb_width):
    mpjpe = []
    assert predicted.shape == target.shape
    assert len(bb_width) == predicted.shape[0]

    for j in range(predicted.shape[1]):
        step = 0
        for i in range(predicted[0]):
            step += np.linalg.norm(predicted[i, j, :] - target[i, j, :]) / bb_width[i]
        mpjpe.append(step/predicted.shape[0])
    return mpjpe


def PCK(predicted, target, bb_width, e):
    assert predicted.shape == target.shape
    assert bb_width.shape[0] == predicted.shape[0]

    step = 0
    for j in range(predicted.shape[1]):
        for i in range(predicted[0]):
            step += int((np.linalg.norm(predicted[i, j, :] - target[i, j, :]) / bb_width(j)) > e)
    return step/(predicted.shape[0]*predicted.shape[1])


def AP(predicted, target, bb_width, e, k):
    assert predicted.shape == target.shape
    assert bb_width.shape[0] == predicted.shape[0]

    step = 0
    for j in range(predicted.shape[1]):
        for i in range(predicted[0]):
            step += int(OKS(predicted[i, j, :], target[i, j, :], bb_width, k[j]) > e)
    return step/(predicted.shape[0]*predicted.shape[1])

def OKS(predicted, target, bb_width, k):
    # OKS measures keypoint similarity
    return np.exp(-np.square(np.linalg.norm(predicted-target)/(bb_width*k))/2)


def RMSE(predicted, target):
    assert predicted.shape == target.shape

    error = []

    for j in range(predicted.shape[1]):
        step = 0
        counter = 1
        for i in range(predicted.shape[0]):
            if not np.isnan(predicted[i, j]).any():
                step += np.square(np.linalg.norm(predicted[i, j, :] - target[i, j, :]))
                counter += 1
        error.append(step/(counter*predicted.shape[0]))
    return error

def extractSubData(data, frameIdxList):
    subData = np.zeros((len(frameIdxList), data.shape[1], data.shape[2]))
    for i in range(len(frameIdxList)):
        subData[i] = data[frameIdxList[i]]
    return subData

def tensorIndexFromFrmaeIndex(frameIdxList: list, inputIdxList):
    outList = []
    for idx in inputIdxList:
        outList.append(frameIdxList.index(idx))
    return outList
