import csv
import re

import pandas as pd
import numpy as np


class Landmark:
    lmCount = 0

    def __init__(self, name, X_coord, Y_coord, Likelihood):
        self.name = name
        self.X = X_coord
        self.Y = Y_coord
        self.likelihood = Likelihood
        Landmark.lmCount += 1

    # initializing the titles and rows list and file name:
    fields = []
    rows = []
    landmarks = []
    lmRows = []

    def landmark2array(landmarkList):
        coords = np.zeros((len(landmarkList), len(landmarkList[0]), 2))
        scores = np.zeros((len(landmarkList), len(landmarkList[0])))

        for i in range(coords.shape[0]):
            for j in range(coords.shape[1]):
                coords[i, j, :] = np.array([landmarkList[i][j].X, landmarkList[i][j].Y])
                scores[i, j] = landmarkList[i][j].likelihood
        return coords, scores

    def csvReader(fileName):
        with open(fileName, 'r') as csvfile:
            csvIn = pd.read_csv(fileName)
            csvIn.fillna(np.nan)
            # tensor = torch.tensor(csvIn.values)
            inputData = np.array(csvIn.values)

            #csvreader = csv.reader(csvfile)
            #scorers = next(csvreader)

            landmarks = []
            lmRows = []
            frameNames = []

        for row in inputData[1:]:
            landmarks.clear()
            frameNames.append(row[0])
            for i in range(1, len(row)-1, 3):
                step = Landmark(inputData[0, i], row[i], row[i + 1], row[i + 2])
                landmarks.append(step)
            lmRows.append(landmarks)
        frameNames.pop(0)
        return lmRows, frameNames

    def csvReaderForLabeled (fileName):
        csvIn = pd.read_csv(fileName)
        csvIn.fillna(np.nan)
        inputData = np.array(csvIn.values)

        landmarks = []
        lmRows = []
        frameNames = []

        for row in inputData[1:]:
            landmarks.clear()
            frameNames.append(re.sub("[\\\]", "/", row[0]))
            for i in range(1, len(row)-1, 2):
                step = Landmark(inputData[0, i], row[i], row[i + 1], np.nan)
                landmarks.append(step)
            lmRows.append(landmarks)
        frameNames.pop(0)
        return lmRows, frameNames

    def extractLabelNames(row):
        labelNames = []
        for element in row:
            labelNames.append(element.name)
        return labelNames


    def saveOutputCSV(self, list):
        # TODO: saving outputs as a .CSV file.
        pass
