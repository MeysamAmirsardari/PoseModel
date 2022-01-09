
import math
import numpy as np
# import applyThreshold, calculateManualError, calculatePredictionError
# import RMSE, PCK, AP
from fileHandler import Landmark as lm, Landmark

predictedFileName = 'F:/DLC/Test6.1-MMM-2021-12-18/labeled-data/test1/CollectedData_MMM.csv'
targetFileName = 'F:/DLC/Test6.1-MMM-2021-12-18/labeled-data/test1/CollectedData_MMM.csv'
manualTestNum = 3
thresh = 0.5

manuals = np.zeros()
predicted = np.zeros()

for i in range(manualTestNum):
    path = 'C:/Users/EMINENT/Desktop/manualTests/manualTest' + str(i) + '.csv'
    manuals[i], targetFrameNames = lm.csvReader(path)

predRows, predFrameNames = lm.csvReader(predictedFileName)
targetRows, targetFrameNames = lm.csvReader(targetFileName)

labelNames = lm.extractLabelNames(manuals[0, 1])

targetTensor, targetScores = lm.landmark2array(targetRows)
predTensor, predScores = lm.landmark2array(predRows)

predArray = applyThreshold(predRows, thresh)
frameIndexList = lm.extractFrameIndex(targetFrameNames)
pred_RMSE = calculatePredictionError(predArray, targetTensor, frameIndexList)


manual_RMSE = calculateManualError(manuals, 0)

import matplotlib.pyplot as plt


labels = ['nose', 'L_eye', 'R eye', 'L ear', 'R ear', 'L shoulder', 'R shoulder', 'L elbow',
          'R elbow', 'L wrist', 'R wrist', 'L hip', 'R hip', 'L knee', 'R knee', 'L ankle',
          'R ankle'] # = labelNames

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, pred_RMSE, width, label='Prediction')
rects2 = ax.bar(x + width/2, manual_RMSE, width, label='Manual annotation')

ax.set_ylabel('Normalized Error')
ax.set_title(' Averaged error of predicted')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()

pckList = []
root = 'C:/Users/EMINENT/Desktop/networkOuts'
targetFileName = root + 'target.csv'

for i in range(7):
    predictedFileName = root+'filtered'+str(i)+'.csv'
    thresh = 0.5

    predRows, predFrameNames = lm.csvReader(predictedFileName)
    targetRows, targetFrameNames = lm.csvReader(targetFileName)

    targetTensor, targetScores = lm.landmark2array(targetRows)
    predTensor, predScores = lm.landmark2array(predRows)

    e = 0.1
    bb_width = math.sqrt((150^2)+(50^2))
    pck = PCK(predTensor, targetTensor, bb_width, e)
    pckList.append(pck)

apList = []
root = 'C:/Users/EMINENT/Desktop/networkOuts'
targetFileName = root + 'target.csv'
e = 0.1
bb_width = math.sqrt((150^2)+(50^2))
k = 10

for i in range(7):
    predictedFileName = root+'filtered'+str(i)+'.csv'
    thresh = 0.5

    predRows, predFrameNames = lm.csvReader(predictedFileName)
    targetRows, targetFrameNames = lm.csvReader(targetFileName)

    targetTensor, targetScores = lm.landmark2array(targetRows)
    predTensor, predScores = lm.landmark2array(predRows)

    ap = AP(predTensor, targetTensor, bb_width, e, k)
    apList.append(ap)

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
        mean = mean / arrays.shape[0]

        for array in arrays:
            error += RMSE(array, mean)
        error = error / arrays.shape[0]
    else:
        for array in arrays[1:]:
            error += RMSE(array, arrays[0])
        error = error / arrays.shape[0]
    return error


def calculatePredictionError(predTensor, targetTensor, frameIndexList):
    error = []
    forAnalyze = np.zeros(len(frameIndexList), predTensor.shape[1], predTensor.shape[2])

    for i in range(len(frameIndexList)):
        forAnalyze[i, :, :] = predTensor[frameIndexList(i), :, :]

    error = RMSE(forAnalyze, targetTensor)
    return error