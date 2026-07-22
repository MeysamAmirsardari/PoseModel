import os
import random
import re

import numpy as np
import imgaug as ia
import imageio
import imgaug.augmenters as iaa


class Augmentation:

    def augmenter(pathList, rootPath):
        fullPath = []

        for path in pathList:
            fullPath.append(rootPath + "\\" + path)

        inputImages = Augmentation.path2image(fullPath)
        print(inputImages.shape)

        ## setting a augmentation type for each file to save:
        for augType in range(4):
            # shuffle = np.random.shuffle(inputImages)
            shuffle = sorted(inputImages, key=lambda k: random.random())
            inArr = np.array(shuffle)
            parts = np.split(inArr, 4)
            augmented = []

            for level in range(1):
                # step = Augmentation.apply(parts[level], type=augType, level=level)
                step = Augmentation.testApply(parts[level], type=augType, level=level)
                augmented.append(step)
            # Augmentation.saveInFile(inputImages, rootPath, pathList, FileIndex=type)
        pass

    def path2image(pathList):
        image = np.array(imageio.imread(pathList[0]))
        images = np.empty([len(pathList), image.shape[0], image.shape[1], image.shape[2]])

        i = 0
        for path in pathList:
            images[i] = np.array(imageio.imread(path), dtype=np.uint8)
            i += 1
        return images

    def apply(inputImages, type=0, level=0):
        ia.seed(1)
        seqList = []

        seqList.append(iaa.Sequential([
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),

            iaa.LinearContrast((0.75, 1.5)),

            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

            iaa.Multiply((0.8, 1.2), per_channel=0.2),

            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True))

        seqList.append(iaa.Sequential([
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),

            iaa.LinearContrast((0.75, 1.5)),

            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

            iaa.Multiply((0.8, 1.2), per_channel=0.2),

            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True))

        seqList.append(iaa.Sequential([
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),

            iaa.LinearContrast((0.75, 1.5)),

            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

            iaa.Multiply((0.8, 1.2), per_channel=0.2),

            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True))

        seq = iaa.Sequential([
            # iaa.Sometimes(
            #             #     0.5,
            #             #     iaa.GaussianBlur(sigma=(0, 0.5))
            #             # ),

            iaa.LinearContrast((0.75, 1.5)),

            # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

            # iaa.Multiply((0.8, 1.2), per_channel=0.2),

            # iaa.Affine(
            #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            #     rotate=(-25, 25),
            #     shear=(-8, 8)
            # )
        ], random_order=True)

        ini = np.array(inputImages)
        augmented = seq(ini, dtype=np.uint8)
        return augmented

    def testApply(inputImages, type=0, level=0):
        ia.seed(1)
        seqList = []

        seqList.append(iaa.Sequential([
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),

            iaa.LinearContrast((0.75, 1.5)),

            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

            iaa.Multiply((0.8, 1.2), per_channel=0.2)
        ], random_order=True))

        ## Blend group:
        seqList.append(iaa.Sequential([
            iaa.BlendAlphaFrequencyNoise(
                foreground=iaa.Multiply(iaa.iap.Choice([0.5, 1.5]), per_channel=True)
            )
        ], random_order=True))

        ## Weather group
        seqList.append(iaa.Sequential([
            iaa.FastSnowyLandscape(
                lightness_threshold=140,
                lightness_multiplier=2.5
            ),

            iaa.Clouds(),

            iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),

            iaa.Rain()
        ], random_order=True))

        ## Color group:
        seqList.append(iaa.Sequential([
            iaa.WithColorspace(
                to_colorspace="HSV",
                from_colorspace="RGB",
                children=iaa.WithChannels(
                    0,
                    iaa.Add((0, 50))
                )),

            iaa.WithBrightnessChannels(iaa.Add((-50, 50))),

            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),

            iaa.AddToHueAndSaturation((-50, 50), per_channel=True)
        ], random_order=True))

        ini = np.array(inputImages, dtype=np.uint8)
        images_aug = seqList[level](images=ini)
        return images_aug

    def saveInFile(augmentedImgs, rootPath=None, pathList=None, FileIndex=0):
        assert len(augmentedImgs) == len(pathList)

        for idx, img in enumerate(augmentedImgs):
            txtList = re.split('/', pathList[idx])
            name = rootPath + '/' + txtList[0] + '/' + 'test' + str(FileIndex + 2) + '/' + txtList[2]
            imageio.imwrite(name, img)
        pass
