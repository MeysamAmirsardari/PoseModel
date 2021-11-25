import csv

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
        csvreader = csv.reader(csvfile)
        scorers = next(csvreader)

        for row in csvreader:
            rows.append(row)
        rows.pop(1)

    for row in rows[1:]:
        for i in range(1, len(row), 3):
            lm = landmark.Landmark(rows[0][i], row[i], row[i + 1], row[i + 2])
            landmarks.append(lm)
        lmRows.append(landmarks)
    return lmRows, scorers

def saveOutput(list):
    #TODO: saving outputs as a .CSV file.
    pass