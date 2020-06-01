# CLASSIFY THE IMAGES
import operator
import numpy as np
import subprocess

from keras.models import load_model


import captchagenerator.capgenerator as capgenerator
import captchagenerator.imagemanip as imagemanip


class Classify:

    def __init__(self, width, height, modelfile, symbols, num_chars):
        """
        Parameters
        ----------
        symbols : string
            the symbols that are possible, e.g. "abdefghij"
        num_chars: int
            the number of symbols in the captcha, e.g. 6
        """
        self.width = width
        self.height = height
        self.model = load_model(modelfile)
        self.symbols = symbols
        self.numsymbols = len(self.symbols)
        self.numchars = num_chars

        self.tmpfile = "/tmp/predicttmp.png"

        self.cg = capgenerator.CapGenerator(width, height)
        self.imm = imagemanip.ImageManip(width, height)

    def loadAndPrep(self, filepath):
        """
        Load and prepare an image, keep the prepared image in a property
        """
        subprocess.run(["convert", "-flatten", filepath, self.tmpfile])
        self.img = self.cg.prepateInputImg(
            self.tmpfile,
            self.width,
            self.height)

    def predict(self, filepath):
        """
        Load, prepare and classify one image, no augmentation
        """
        self.loadAndPrep(filepath)
        return self.predictImg(self.img)

    def predictImg(self, img):
        """
        Classify image, the image has to be prepared already
        """
        if img is not None:
            img = img / 255.0
        else:
            print("Error reading image")
            return

        res = np.array(self.model.predict(img[np.newaxis, :, :, np.newaxis]))
        ans = np.reshape(res, (self.numchars, self.numsymbols))
        l_ind = []
        probs = []
        for a in ans:
            l_ind.append(np.argmax(a))
            probs.append(np.max(a))

        capt = ''
        for l in l_ind:
            capt += self.symbols[l]
        return capt

    def aggRes(self, outcome):
        """
        Helper to aggregate the result when using augmentation
        """
        tmplist = list(outcome)
        i = 0
        for letter in tmplist:
            if letter not in self.letterVote[i]:
                self.letterVote[i][letter] = 0
            self.letterVote[i][letter] += 1
            i += 1

        if outcome not in self.fullResultVote:
            self.fullResultVote[outcome] = 0
        self.fullResultVote[outcome] += 1

    def predictAugment(self, filepath):
        """
        Classify an image from file using augmentation
        """
        self.loadAndPrep(filepath)

        self.letterVote = []

        i = 0
        while i < self.numchars:
            self.letterVote.append({})
            i += 1

        self.fullResultVote = {}

        outcomeOrg = self.predictImg(self.img)
        self.aggRes(outcomeOrg)

        for j in range(97, 99):
            tmpim = self.imm.resizeImg(self.img, j)

            outcome = self.predictImg(tmpim)
            self.aggRes(outcome)

        for k in range(-5, 6):
            if k == 0:
                continue
            newim = self.imm.rotateImage(self.img, k)

            outcome = self.predictImg(newim)
            self.aggRes(outcome)

            for j in range(97, 99):
                tmpim = self.imm.resizeImg(newim, j)

                outcome = self.predictImg(tmpim)
                self.aggRes(outcome)

        letterVoteWinner = ''
        for tmpdict in self.letterVote:
            thisWinner = max(tmpdict.items(), key=operator.itemgetter(1))[0]
            letterVoteWinner += thisWinner

        fullResultWinner = max(
            self.fullResultVote.items(),
            key=operator.itemgetter(1))[0]

        return [outcomeOrg, letterVoteWinner, fullResultWinner]
