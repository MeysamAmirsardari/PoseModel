import itertools
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from Kernel.fileHandler import Landmark as lm


# img = cv2.imread('C:/Users/EMINENT/Desktop/img02900.png')
# mask = cv2.imread('C:/Users/EMINENT/Desktop/mask1.png')
# res = cv2.bitwise_and(img, img, mask=mask[:, :, 2])
#
# #equ = cv2.equalizeHist(res[:, :, 2])
#
# img_yuv = cv2.cvtColor(res, cv2.COLOR_BGR2YUV)
# img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
# img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
#
# img_yuv2 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# img_yuv2[:, :, 0] = cv2.equalizeHist(img_yuv2[:, :, 0])
# img_output2 = cv2.cvtColor(img_yuv2, cv2.COLOR_YUV2BGR)
#
# res2 = cv2.bitwise_and(img_output2, img_output2, mask=mask[:, :, 2])
#
# img = cv2.imread('C:/Users/EMINENT/Desktop/img02079.png')
#
# img_yuv3 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# img_yuv3[:, :, 0] = cv2.equalizeHist(img_yuv3[:, :, 0])
# img_output3 = cv2.cvtColor(img_yuv3, cv2.COLOR_YUV2BGR)
#
# cv2.imshow('Color input image', res)
# cv2.imshow('Histogram equalized_without mask', res2)
# cv2.imshow('Histogram equalized', img_output)
# cv2.imshow('Histogram equalized2', img_output3)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def equalize(img, mask, name):
    #img = cv2.imread('C:/Users/EMINENT/Desktop/img02900.png')
    #mask = cv2.imread('C:/Users/EMINENT/Desktop/mask1.png')
    res = img#cv2.bitwise_and(img, img, mask=mask[:, :, 2])

    img_ycrcb = cv2.cvtColor(res, cv2.COLOR_BGR2YCrCb)
    img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])
    img_output = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)

    #dst = cv2.detailEnhance(img_output, 10, 0.15)
    # imout = cv2.edgePreservingFilter(img_output, flags=cv2.RECURS_FILTER);
    # cv2.imwrite("C:/Users/EMINENT/Desktop/Test6.0-MMM-2021-12-14/edge-preserving-recursive-filter.png", imout);
    #
    # imout = cv2.detailEnhance(img_output);
    # cv2.imwrite("C:/Users/EMINENT/Desktop/Test6.0-MMM-2021-12-14/enh.png", imout);

    root = 'C:/Users/EMINENT/Desktop/Test6.0-MMM-2021-12-14/BB/'
    cv2.imwrite(root+name, img_output)
    pass


targetFileName = 'C:/Users/EMINENT/Desktop/Test6.0-MMM-2021-12-14/labeled-data/test1/CollectedData_MMM.csv'
root = 'C:/Users/EMINENT/Desktop/Test6.0-MMM-2021-12-14/'
_, targetFrameNames = lm.csvReaderForLabeled(targetFileName)

##################################
##################################

mask = cv2.imread('C:/Users/EMINENT/Desktop/mask1.png')
imgs = [cv2.imread(fn, cv2.IMREAD_COLOR) for fn in glob(root+'labeled-data/Test103-1/*.png')]
list(map(equalize, imgs, itertools.repeat(mask, len(imgs)), targetFrameNames))

