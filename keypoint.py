import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm, tqdm_notebook
from sklearn.preprocessing import MinMaxScaler


class keypointScore:

    def __init__(self, video, user):
        self.video = video
        self.user = user
        self.path = './Data/inputData/similar/video{video}'.format(video=video)

        self.sharpening = np.array([[-1, -1, -1, -1, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, 2, 9, 2, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, -1, -1, -1, -1]]) / 9.0

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def imgRead(self):
        kps_list = []
        for time in range(len(os.listdir(self.path))-1):
            img1 = cv2.imread('{path}/time{time}/user{user}.png'.format(path=self.path, time=time, user=self.user), 0)
            img2 = cv2.imread('{path}/time{time}/user{user}.png'.format(path=self.path, time=time+1, user=self.user), 0)
            img2 = cv2.filter2D(img2, -1, self.sharpening)
            kps_list.append(self.getKPS(img1, img2))

        len_ = len(kps_list)
        kps_list = np.array(kps_list).reshape(-1, 1)
        # transformer = MinMaxScaler()
        # transformer.fit(kps_list)
        # kps = transformer.transform(kps_list).reshape(1, -1)
        kps_list = kps_list.reshape(1, -1)

        # return kps.reshape(len_)
        return kps_list.reshape(len_)


    def getKPS(self, img1, img2):
        sift = cv2.xfeatures2d.SIFT_create()
        # orb = cv2.ORB_create()

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        if des2 is None:
            kp2, des2 = sift.detectAndCompute(img1, None)
        elif des1 is None:
            kp1, des1 = sift.detectAndCompute(img2, None)

        # kp1, des1 = orb.detectAndCompute(img1, None)
        # kp2, des2 = orb.detectAndCompute(img2, None)

        matches = self.flann.knnMatch(des1, des2, k=2)
        # bf = cv2.BFMatcher()
        # matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.3 * n.distance:
                good.append([m])

        return len(good)


if __name__ == '__main__':

    user_ = range(1, 78)

    for video in tqdm(range(1, 21)):
        kps_list = []
        for user in tqdm(user_):
            kps_list.append(keypointScore(video, user).imgRead())

        df_kps = pd.DataFrame(kps_list, index=['user{}'.format(u) for u in user_])
        df_kps.to_csv('./result/kps_video{video}.csv'.format(video=video))
