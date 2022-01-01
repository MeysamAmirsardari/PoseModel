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

    step = 0
    error = []

    for j in range(predicted.shape[1]):
        for i in range(predicted[0]):
            if(~np.isnan(predicted[i, j, :])):
                step += np.square(np.linalg.norm(predicted[i, j, :] - target[i, j, :]))
            error.append(step/predicted.shape[0])
    return error
