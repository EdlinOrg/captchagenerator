import cv2
import numpy as np
import os
import random
import time
import math

import PILasOPENCV

import captchagenerator.imagemanip as imagemanip
import captchagenerator.PILasOPENCVFix as PILasOPENCVFix


class CapGenerator:

    def __init__(
            self,
            width,
            height,
            symbols="abcdefghijklmnopqrstuvwxyz",
            font={}):
        self.WANTED_WIDTH = width
        self.WANTED_HEIGHT = height
        self.symbols = symbols

        if font:
            self.font = PILasOPENCV.truetype(font['face'], font['size'])
        else:
            self.font = cv2.FONT_HERSHEY_TRIPLEX

        self.imageManipObj = imagemanip.ImageManip(width, height)
        # Number of characters wanted in the image
        self.num_of_chars = 5

    def generateOneLetter(self, mychar):
        """
        Generate an image consisting of one character
        """
        width, height = 340, 340

        img = np.ones((height, width, 1), dtype='uint8')
        img *= 255

        img = PILasOPENCVFix.getmaskFix(mychar, self.font)

        img = 255 - img

        img = self.imageManipObj.padImage(img, width, height, 0, 0)

        rows, cols = height, width
        degrees = random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), degrees, 1)
        img = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))

        # shrink down to just contain that character
        return self.imageManipObj.shrinkDown(img)

    def stichTogetherLetters(self, img, img2):
        # Overlap 2 pixels
        overlap = 2

        newh = 100
        neww = 200

        newimg = np.ones((newh, neww), dtype='uint8')
        newimg *= 255

        offseth = 10
        offsetw = 30

        endImg1y = offseth+img.shape[0]
        endImg1x = offsetw+img.shape[1]

        newimg[offseth:endImg1y, offsetw:endImg1x] = img

        starty = offseth
        startx = offsetw + img.shape[1] - overlap

        # revert to turn black to zero and white to 255
        newimg = (255 - newimg)
        img2 = (255 - img2)

        img_bwo = cv2.bitwise_or(
            newimg[
                starty:starty + img2.shape[0],
                startx:startx + img2.shape[1]
                ],
            img2)

        newimg[
            starty:starty + img2.shape[0],
            startx:startx + img2.shape[1]
            ] = img_bwo

        # revert back
        newimg = (255 - newimg)

        return self.imageManipObj.shrinkDown(newimg)

    def stichString(self, mystring):
        """
        Return an image consisting of the characters
        provided in the string stitched together
        """
        mychars = list(mystring)

        first = True
        for x in mychars:
            img = self.generateOneLetter(x)

            if first:
                first = False
                imgMerge = img
                continue

            imgMerge = self.stichTogetherLetters(
                imgMerge.reshape(imgMerge.shape[0], imgMerge.shape[1]),
                img.reshape(img.shape[0], img.shape[1]))

        return imgMerge

    def splitImage(self, img, width, height, numchars):
        """
        Split the image into numchars sections by width,
        scale up to the desired width/height
        pad with whitespace if needed
        """
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        sectionWidth = math.floor(imgWidth / numchars)
        lastSectionWidth = imgWidth - (sectionWidth * (numchars - 1))

        sectionWidths = [sectionWidth] * (numchars - 1)
        sectionWidths.append(lastSectionWidth)

        imgSections = []

        starty = 0
        startx = 0

        partNo = 0
        for myWidth in sectionWidths:
            part = img[starty:imgHeight, startx:startx + myWidth]

            """
            if we want to pad differently depending on position
            if partNo == 0:
                padSide = 1 # pad on left side
            elif partNo == 4:
                padSide = 2 # pad on right side
            else:
                padSide = 0 # pad on both sides
            """

            shrunkImg = self.imageManipObj.shrinkDown(part)
            paddedPart = self.imageManipObj.padImage(
                shrunkImg, width, height, 0, 0)

            imgSections.append(paddedPart)
            startx += myWidth
            partNo += 1

        return imgSections

    def generateImages(self, numimages, outputdir, width, height, augment=False):
        mydictlist = list(self.symbols)

        while numimages > 0:
            mystring = ""
            want = self.num_of_chars

            thechars = []
            while want > 0:
                achar = random.choice(mydictlist)
                mystring += achar
                thechars.append(achar)
                want -= 1

            pic_target = mystring[0:5]

            imgMerge = self.stichString(pic_target)

            splitIntoChars = False

            if splitIntoChars:
                imgSections = self.splitImage(imgMerge, 37, 45)
                tmpi = 0
                for img in imgSections:
                    cv2.imwrite(outputdir + thechars[tmpi] + "_" + str(time.time()) + ".png", img)
                    tmpi += 1

            # pad with whitespace
            paddedImg = self.imageManipObj.padImage(imgMerge, width, height, 0, 0)

            outfilename = mystring + "_" + str(time.time()) + ".png"

            cv2.imwrite(outputdir + outfilename, paddedImg)

            if augment:
                for i in range(-5, 6):
                    if i == 0:
                        # will not augment the default
                        continue
                    myimg = self.imageManipObj.rotateImage(paddedImg, i)

                    outfilename = mystring + "_aug" + str(i) + '_' + str(time.time()) + ".png"
                    cv2.imwrite(outputdir + outfilename, myimg)

            numimages -= 1

    def prepateInputImg(self, filename, wantedWidth, wantedHeight, invert=False):
        """
        Load an image, cut away all whitespace around the image,
        then add whitespace to the correct width/height.
        That means the content will always be centered in the middle 

        Returns
        -------
        An image object
        """
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        if invert:
            img = 255 - img

        shrunkImg = self.imageManipObj.shrinkDown(img)

        currentHeight = shrunkImg.shape[0]
        currentWidth = shrunkImg.shape[1]

        hasResized = False
        redoForHeight = 0

        if currentWidth > wantedWidth:

            hasResized = True

            divBy = math.floor(currentWidth / wantedWidth) + 1

            newWidth = math.floor(currentWidth / divBy)
            newHeight = math.floor(currentHeight / divBy)

            redoForHeight = 1

            if newHeight > wantedHeight:
                redoForHeight = 2

        if (0 == redoForHeight and currentHeight > wantedHeight) or (2 == redoForHeight):

            hasResized = True

            # we add +1 to get a bit of padding
            divBy = math.floor(currentHeight / (wantedHeight + 1)) + 1

            newWidth = math.floor(currentWidth / divBy)
            newHeight = math.floor(currentHeight / divBy)

        if hasResized:
            dim = (newWidth, newHeight)
            shrunkImg = cv2.resize(shrunkImg, dim)

        return self.imageManipObj.padImage(shrunkImg, wantedWidth, wantedHeight, 0, 0)

    def prepateInput(self, filename, outputfile, wantedWidth, wantedHeight, invert=False):
        """
        Prepare the image and saves it to a file
        (see prepateInputImg)
        """
        paddedImg = self.prepateInputImg(filename, wantedWidth, wantedHeight, invert)
        cv2.imwrite(outputfile, paddedImg)

    def prepareDir(self, indir, outdir, wantedWidth, wantedHeight, invert, augment, doResize):
        """
        Read image files from a directory,
        prepare the image and optionally created augmented/resized copies and save in outdir
        Parameters
        ----------
        invert : bool
            set to invert the images
        augment : bool
            set for augmentation
        doResize : bool
            set to create images of different sizes
        """

        for _, pic in enumerate(os.listdir(indir)):
            filename = os.path.join(indir, pic)
            outfile = os.path.join(outdir, pic)
            self.prepateInput(filename, outfile, wantedWidth, wantedHeight, invert)

            pic_target = pic[0:5]
            if augment:
                paddedImg = cv2.imread(outfile, cv2.IMREAD_GRAYSCALE)
                for i in range(-2, 3):
                    if i == 0:
                        continue

                    myimg = self.imageManipObj.rotateImage(paddedImg, i)

                    outfilename = pic_target + "_aug" + str(i) + '_' + str(time.time()) + ".png"
                    outfile = os.path.join(outdir, outfilename)
                    cv2.imwrite(outfile, myimg)

                    if doResize:
                        for j in range(97, 99):
                            if j == 100:
                                continue

                            resimg = self.imageManipObj.resizeImg(myimg, j)
                            outfilename = pic_target + "_augrez" + str(j) + '_' + str(time.time()) + ".png"
                            outfile = os.path.join(outdir, outfilename)
                            cv2.imwrite(outfile, resimg)

            if doResize:
                paddedImg = cv2.imread(outfile, cv2.IMREAD_GRAYSCALE)
                for i in range(97, 99):
                    if i == 100:
                        continue

                    myimg = self.imageManipObj.resizeImg(paddedImg, i)
                    outfilename = pic_target + "_rez" + str(i) + '_' + str(time.time()) + ".png"
                    outfile = os.path.join(outdir, outfilename)
                    cv2.imwrite(outfile, myimg)

    def prepareDirSplit(self, indir, outdir, wantedWidth, wantedHeight, numchar):
        """
        Split into numchar chunks and we will use that to predict characters
        we assume prepareDir has been executed first
        """
        for _, pic in enumerate(os.listdir(indir)):
            filename = os.path.join(indir, pic)

            paddedImg = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            shrunkImg = self.imageManipObj.shrinkDown(paddedImg)

            imgSections = self.splitImage(shrunkImg, wantedWidth, wantedHeight, numchar)
            tmpi = 0
            for img in imgSections:
                outfilename = os.path.join(outdir, pic[tmpi])
                cv2.imwrite(outfilename + "_" + str(time.time()) + ".png", img)
                tmpi += 1
