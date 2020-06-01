import cv2
import numpy as np
import os
import subprocess


class ImageManip:

    def __init__(self, width, height):
        self.WANTED_WIDTH = width
        self.WANTED_HEIGHT = height

        # Keep track of how large the images become when rotating
        self.maxheightImageSofar = -1
        self.maxwidthImageSofar = -1

    def addBackgroundToTransparentPng(self, indir, outdir):
        """
        Use imagemagick to flatten images.
        This is needed for example when we have transparent pngs.
        Parameters
        ----------
        indir : string
            Path to directory with files to process
        outdir : string
            Path to directory to save the output
        """
        for filename in os.listdir(indir):
            filenameFull = os.path.join(indir, filename)
            outfile = os.path.join(outdir, filename)
            subprocess.run(["convert", "-flatten", filenameFull, outfile])

    def shrinkDown(self, img):
        """
        Remove any whitespace around the image and
        shrink it down to the actual content
        """
        (_, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_OTSU)
        points = np.argwhere(im_bw == 0)
        points = np.fliplr(points)
        x, y, w, h = cv2.boundingRect(points)
        crop = img[y:y+h, x:x+w]
        return crop

    def resizeImg(self, img, percent):
        """
        Parameters
        ----------
        percent : int
            percentage to resize it,
            e.g. 98 to make slightly smaller,
            102 to slightly larger
        """

        width = int(img.shape[1] * percent / 100)
        height = int(img.shape[0] * percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        resized = self.shrinkDown(resized)
        resized = self.padImage(
            resized,
            self.WANTED_WIDTH,
            self.WANTED_HEIGHT,
            0,
            0)

        return resized

    def padImage(self, img, width, height, padSide, padTop):
        """
        Parameters
        ----------
        padSide : int
            0 even, 1 = pad left, 2 = pad right
        padTop : int
            0 even, 1= pad top, 2 pad bottom
        """

        # border around, fill it so it becomes desired width/height
        paddingNeededHeight = height - img.shape[0]

        if 0 == padTop:
            topSide = int(paddingNeededHeight/2)
            bottomSide = paddingNeededHeight - topSide
        elif 1 == padTop:
            topSide = paddingNeededHeight
            bottomSide = 0
        else:
            topSide = 0
            bottomSide = paddingNeededHeight

        paddingNeededWidth = width - img.shape[1]

        if 0 == padSide:
            leftSide = int(paddingNeededWidth/2)
            rightSide = paddingNeededWidth - leftSide
        elif 1 == padSide:
            leftSide = paddingNeededWidth
            rightSide = 0
        else:
            leftSide = 0
            rightSide = paddingNeededWidth

        if topSide < 0:
            topSide = 0
        if bottomSide < 0:
            bottomSide = 0
        if leftSide < 0:
            leftSide = 0
        if rightSide < 0:
            rightSide = 0

        borderImg = cv2.copyMakeBorder(
            img,
            top=topSide,
            bottom=bottomSide,
            left=leftSide,
            right=rightSide,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        return borderImg

    def rotateImage(self, img, degrees):
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D((width/2, height/2), degrees, 1)
        img = cv2.warpAffine(
            img,
            M,
            (width, height),
            borderValue=(255, 255, 255))

        img = self.shrinkDown(img)

        if img.shape[0] > self.maxheightImageSofar:
            self.maxheightImageSofar = img.shape[0]

        if img.shape[1] > self.maxwidthImageSofar:
            self.maxwidthImageSofar = img.shape[1]

        img = self.padImage(img, width, height, 0, 0)
        return img
