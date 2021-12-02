import csv
import pandas as pd
import click
import numpy as np

from Evaluation.Evaluation import landmark2array
from fileHandler import csvReader, landmarks, lmRows, rows, Landmark

filePath = ''
predictedFileName = 'Predict.csv'
evaluationFileName = 'Test1.csv'
Video_FILE = "/Mon_vid.mp4"
maskFile = '4.png'

DEFAULT_CONFIG = {
    'video_extension': 'avi',
    'converted_video_speed': 1,
    'calibration': {
        'animal_calibration': False,
        'calibration_init': None,
        'fisheye': False
    },
    'manual_verification': {
        'manually_verify': False
    },
    'triangulation': {
        'ransac': False,
        'optim': False,
        'scale_smooth': 2,
        'scale_length': 2,
        'scale_length_weak': 1,
        'reproj_error_threshold': 5,
        'score_threshold': 0.8,
        'n_deriv_smooth': 3,
        'constraints': [],
        'constraints_weak': []
    },
    'pipeline': {
        'videos_raw': 'videos-raw',
        'videos_raw_mp4': 'videos-raw-mp4',
        'pose_2d': 'pose-2d',
        'pose_2d_filter': 'pose-2d-filtered',
        'pose_2d_projected': 'pose-2d-proj',
        'pose_3d': 'pose-3d',
        'pose_3d_filter': 'pose-3d-filtered',
        'videos_labeled_2d': 'videos-labeled',
        'videos_labeled_2d_filter': 'videos-labeled-filtered',
        'calibration_videos': 'calibration',
        'calibration_results': 'calibration',
        'videos_labeled_3d': 'videos-3d',
        'videos_labeled_3d_filter': 'videos-3d-filtered',
        'angles': 'angles',
        'summaries': 'summaries',
        'videos_combined': 'videos-combined',
        'videos_compare': 'videos-compare',
        'videos_2d_projected': 'videos-2d-proj',
    },
    'filter': {
        'enabled': False,
        'type': 'medfilt',
        'medfilt': 13,
        'offset_threshold': 25,
        'score_threshold': 0.05,
        'spline': True,
        'n_back': 5,
        'multiprocessing': False
    },
    'filter3d': {
        'enabled': False
    }
}

# csvIn = pd.read_csv("Test1.csv", "UTF-8")
# tensor = np.array(csvIn.values)

# with open(evaluationFileName, 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     # scorers = next(csvreader)
#
#     for row in csvreader:
#         rows.append(row)
#     rows.pop(1)
#
# for row in rows[1:]:
#     landmarks.clear()
#     for i in range(1, len(row), 3):
#         lm = Landmark(rows[0][i], row[i], row[i + 1], row[i + 2])
#         landmarks.append(lm)
#     lmRows.append(landmarks)

targetRows = csvReader(evaluationFileName)
predRows = csvReader(predictedFileName)
targetTensor = landmark2array(predRows)
predTensor = landmark2array(targetRows)
