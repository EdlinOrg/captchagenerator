
import cv2
import operator
import subprocess

import captchagenerator.capgenerator as capgenerator
import captchagenerator.classify as classify
import captchagenerator.imagemanip as imagemanip


class ClassifySplit:
    """
    Split up the image in X different parts and predict the outcome
    for each of those parts
    """

    def __init__(self, width, height, modelfile, symbols, num_chars):
        self.width = width
        self.height = height
        self.numchars = num_chars

        self.tmpfile = "/tmp/classifysplittmp.png"

        self.cg = capgenerator.CapGenerator(width, height)
        self.imm = imagemanip.ImageManip(width, height)

        self.classifier = classify.Classify(width, height, modelfile, symbols, 1)

    def aggRes(self, letter):
        if letter not in self.letterVote:
            self.letterVote[letter] = 0
        self.letterVote[letter] += 1

    def predictAugment(self, img):

        self.letterVote = {}

        outcomeOrg = self.classifier.predictImg(img)
        self.aggRes(outcomeOrg)

        for j in range(97, 99):
            tmpim = self.imm.resizeImg(img, j)

            outcome = self.classifier.predictImg(tmpim)
            self.aggRes(outcome)

        for k in range(-5, 6):
            if k == 0:
                continue
            newim = self.imm.rotateImage(img, k)

            outcome = self.classifier.predictImg(newim)
            self.aggRes(outcome)

            for j in range(97, 99):
                tmpim = self.imm.resizeImg(newim, j)

                outcome = self.classifier.predictImg(tmpim)
                self.aggRes(outcome)

        letterVoteWinner = max(self.letterVote.items(), key=operator.itemgetter(1))[0]

        return [outcomeOrg, letterVoteWinner]

    def predict(self, filename):
        subprocess.run(["convert", "-flatten", filename, self.tmpfile])

        img = cv2.imread(self.tmpfile, cv2.IMREAD_GRAYSCALE)
        shrunkImg = self.imm.shrinkDown(img)

        imgSections = self.cg.splitImage(
                        shrunkImg,
                        self.width,
                        self.height,
                        self.numchars
                        )

        stringwinners = ['', '']
        for img in imgSections:
            winners = self.predictAugment(img)
            stringwinners[0] += winners[0]
            stringwinners[1] += winners[1]
        return stringwinners
