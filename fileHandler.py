import csv
import pandas as pd


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

def csvReader(fileName):
    with open(fileName, 'r') as csvfile:
        # csvIn = pd.read_csv(fileName, "UTF-8")
        # tensor = torch.tensor(csvIn.values)

        csvreader = csv.reader(csvfile)
        #scorers = next(csvreader)

        for row in csvreader:
            rows.append(row)
        rows.pop(1)

    for row in rows[1:]:
        landmarks.clear()
        for i in range(1, len(row), 3):
            lm = Landmark(rows[0][i], row[i], row[i + 1], row[i + 2])
            landmarks.append(lm)
        lmRows.append(landmarks)
    return lmRows


def saveOutputCSV(list):
    #TODO: saving outputs as a .CSV file.
    pass