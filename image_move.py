import os
import shutil

from tqdm import tqdm, tqdm_notebook


class imgMove:
    def __init__(self, video, user, rating, time):
        self.video = video
        self.user = user
        self.time = time
        self.rating = rating

    def imgCopy(self):
        shutil.copy('./Data/diff/video' + str(self.video) + '/' + str(self.user) + '/' + str(self.time) + '.png',
                    './Data/inputData/video' + str(self.video) + '/time' + str(self.time)
                    + '/rating' + str(self.rating) + '/')
        os.rename('./Data/inputData/video' + str(self.video) + '/time' + str(self.time)
                  + '/rating' + str(self.rating) + '/' + str(self.time) + '.png',
                  './Data/inputData/video' + str(self.video) + '/time' + str(self.time)
                  + '/rating' + str(self.rating) + '/user' + str(self.user) + '.png')


if __name__ == '__main__':
    rating1 = [2, 25]
    rating2 = [14, 17, 27]
    rating3 = [3, 4, 6, 8, 9, 11, 13, 16, 19, 20, 21, 22, 29, 31, 36]
    rating4 = [1, 5, 7, 18, 26, 28, 30, 32, 33, 34, 37]
    rating5 = [10, 12, 15, 23, 24, 35]

    for t in tqdm(range(1, 61)):
        for r1 in rating1:
            imgMove(1, r1, 1, t).imgCopy()
        for r2 in rating2:
            imgMove(1, r2, 2, t).imgCopy()
        for r3 in rating3:
            imgMove(1, r3, 3, t).imgCopy()
        for r4 in rating4:
            imgMove(1, r4, 4, t).imgCopy()
        for r5 in rating5:
            imgMove(1, r5, 5, t).imgCopy()
