import csv
import re

import pandas as pd
import click
import numpy as np

from Preprocessing.augmentor import Augmentation
from fileHandler import Landmark as lm, Landmark

from Evaluation.Visualization import compare
from Filters.Viterbi import viterbi_path

filePath = ''
predictedFileName = 'Predict.csv'
targetFileName = 'Target.csv'
test = 'F:/DLC/Test6.1-MMM-2021-12-18/labeled-data/test1/CollectedData_MMM.csv'
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

# targetRows, targetFrameNames = lm.csvReader(targetFileName)
# predRows, predFrameNames = lm.csvReader(predictedFileName)
#
# labelNames = lm.extractLabelNames(targetRows[1])
# targetTensor, targetScores = lm.landmark2array(predRows)
# predTensor, predScores = lm.landmark2array(targetRows)

### Just for Test:
# df = pd.read_csv(test)
# ten = np.array(df)

testRows, testFrameNames = lm.csvReaderForLabeled(test)
testLabelNames = lm.extractLabelNames(testRows[1])
testTensor, testScores = lm.landmark2array(testRows)

rootPath = 'F:/DLC/Test6.1-MMM-2021-12-18'
Augmentation.augmenter(testFrameNames, rootPath)


# outPoint, outScores = viterbi_path(predTensor, predScores, 3, 30)
# compare(predTensor[:, 5, 0], outPoint[:, 0])
