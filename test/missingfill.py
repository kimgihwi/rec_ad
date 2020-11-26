import cv2

import numpy as np


class imgFill:
    def __init__(self, input1, input2):
        self.img1 = cv2.imread('./' + str(input1) + '.png')
        self.img2 = cv2.imread('./' + str(input2) + '.png')
        self.img1 = cv2.resize(self.img1, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        self.img2 = cv2.resize(self.img1, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

    def imgMean(self):
        # mean = np.mean([list(self.img1), list(self.img2)], axis=0)
        # print(self.img1.shape)
        # print(self.img2.shape)
        mean = (self.img1 + self.img2) / 2.
        return mean

    def getImage(self):
        return self.imgMean()

    def saveImage(self):
        saveImg = self.getImage()
        cv2.imwrite('./fillImg.png', saveImg)


if __name__ == '__main__':
    imgFill(49, 51).saveImage()
