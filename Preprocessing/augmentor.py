import os
import numpy as np
import imgaug as ia
import imageio
import imgaug.augmenters as iaa

class Augmentation:

    def augmenter(pathList, rootPath):
        inputImages = Augmentation.path2image(pathList)

        ## setting a augmentation type for each file to save:
        for type in range(4):
            shuffle = np.random.shuffle(inputImages)
            parts = np.split(shuffle, 4)
            augmented = []

            for level in range(4):
                step = Augmentation.aplly(parts[level], type=type, level=level)
                augmented.concatinate((step, augmented), axis=0)
            Augmentation.saveInFile(augmented, rootPath, pathList, FileIndex=type)
        pass

    def path2image(pathList):
        image = np.array(imageio.imread(pathList[0]))
        images = np.empty([len(pathList), image.shape])

        i = 0
        for path in pathList:
            images[i] = np.array(imageio.imread(path), dtype=np.uint8)
            i += 1
        return images

    def apply(inputImages, type=0, level=0):
        ia.seed(1)
        seq = []

        seq[0] = iaa.Sequential([
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),

            iaa.LinearContrast((0.75, 1.5)),

            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

            iaa.Multiply((0.8, 1.2), per_channel=0.2),

            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True)

        seq[1] = iaa.Sequential([
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
        ], random_order=True)

        seq[2] = iaa.Sequential([
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
        ], random_order=True)

        seq[3] = iaa.Sequential([
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
        ], random_order=True)

        augmented = seq[type](inputImages)
        return augmented

    def saveInFile(augmentedImgs, rootPath=None, pathList=None, FileIndex=0):
        assert len(augmentedImgs) == len(pathList)

        for idx, img in enumerate(augmentedImgs):
            name = os.path.join(str(rootPath), str(FileIndex), '/', str(pathList[idx]), '.png')
            imageio.imwrite(name+"%d.png" % (idx,), img)
        pass