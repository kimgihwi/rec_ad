import os

import cv2

import numpy as np

from tqdm import tqdm, tqdm_notebook


class imgDiff:
    def __init__(self, user, video):
        self.user = user
        self.video = video
        self.path = './Data/crop/CNN based/preprocessing/video' + str(video) + '/' + str(user) + '/'
        self.i_list = os.listdir(self.path)

    def calDiff(self):
        for i in range(len(self.i_list) - 1):
            # img1 = np.array(cv2.imread(self.path + str(self.user) + '/' + str(i) + '.png'))
            # img2 = np.array(cv2.imread(self.path + str(self.user) + '/' + str(i+1) + '.png'))
            img1 = np.array(cv2.imread(self.path + str(i) + '.png'))
            img2 = np.array(cv2.imread(self.path + str(i+1) + '.png'))
            diff = img2 - img1
            cv2.imwrite('./Data/crop/CNN based/diff/video' + str(self.video) + '/' + str(self.user) + '/' +
                        str(i+1) + '.png', diff)


if __name__ == '__main__':
    for u in tqdm(range(1, 38)):
        imgDiff(u, 1).calDiff()
