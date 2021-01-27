import pandas as pd
import numpy as np
import random
from tqdm import tqdm, tqdm_notebook

random.seed(1)


class TopNRecommender:

    def __init__(self):
        # self.user = user
        self.bestseller = [12, 11, 5, 18, 20, 17, 6, 1, 3, 7, 4, 8, 13, 9, 10, 14, 19, 15, 2, 16]

    def sortRating(self, user, top):
        """
        :param user: number of user
        :param top: N of top-N
        :return: top-N list
        """
        df = self.readRatingData()
        tmp_list = df.loc['user{}'.format(user)]

        r5 = []
        r4 = []
        r3 = []
        r2 = []
        r1 = []
        tmp_r5 = list(tmp_list[tmp_list == 5].index)
        tmp_r4 = list(tmp_list[tmp_list == 4].index)
        tmp_r3 = list(tmp_list[tmp_list == 3].index)
        tmp_r2 = list(tmp_list[tmp_list == 2].index)
        tmp_r1 = list(tmp_list[tmp_list == 1].index)

        for idx in self.bestseller:
            if 'video{}'.format(idx) in tmp_r5:
                r5.append(idx)
            elif 'video{}'.format(idx) in tmp_r4:
                r4.append(idx)
            elif 'video{}'.format(idx) in tmp_r3:
                r3.append(idx)
            elif 'video{}'.format(idx) in tmp_r2:
                r2.append(idx)
            elif 'video{}'.format(idx) in tmp_r1:
                r1.append(idx)
        video_ranking = r5 + r4 + r3 + r2 + r1

        return video_ranking[:top]

    def readRatingData(self):
        df = pd.read_csv('./Data/rating.csv', index_col=0)
        return df

    def readKpsData(self, video):
        df = pd.read_csv('./result/similar user/kps/kps_video{}.csv'.format(video), index_col=0)
        return df

    def readCFDAta(self):
        df = pd.read_csv('./result/similar user/user cf/user_best_similarity.csv', index_col=0)
        return df

    # def topN(self, user, top):
    #     return self.sortRating(user, top)

    def bestSeller(self, top):
        return self.bestseller[:top]

    def random(self, top):
        rand_list = list(range(1, 21))
        random.shuffle(rand_list)
        return rand_list[:top]

    def kps(self, video, time, user, top):
        df_kps = self.readKpsData(video)
        sim_user = df_kps.loc['user{}'.format(user)][str(time)]
        kpsTopN = self.sortRating(sim_user, top)
        return kpsTopN

    def cf(self, user, top):
        df_cf = self.readCFDAta()
        sim_user = int(df_cf['user{}'.format(user)].loc['user'])
        return self.sortRating(sim_user, top)

    def calAccuracy_kps(self, video, time, user, top):

        kps_inter = list(set(self.sortRating(user, top)) & set(self.kps(video, time, user, top)))

        return len(kps_inter)/top

    def calAccuracy_etc(self, user, top):
        cf_inter = list(set(self.sortRating(user, top)) & set(self.cf(user, top)))
        best_inter = list(set(self.sortRating(user, top)) & set(self.bestSeller(top)))
        rand_inter = list(set(self.sortRating(user, top)) & set(self.random(top)))

        return len(cf_inter)/top, len(best_inter)/top, len(rand_inter)/top


if __name__ == '__main__':
    # print(TopNRecommender().kps(1, 1, 2, 5))    # video, time, user, top
    # print(TopNRecommender().sortRating(2, 5))
    # print(TopNRecommender().calAccuracy(1, 1, 2, 5))
    # print(TopNRecommender().calAccuracy_kps(1, 4, 1, 5))    # video, time, user, top / kps, cf, best, rand

    video1 = [2]  # 11 sec
    video2 = [3, 4, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19]  # 15 sec
    video3 = [1, 5, 7, 12, 14, 20]  # 30 sec
    video4 = [6]  # 50 sec

    # video_ = 1
    # user = 1

    df_top = pd.DataFrame()

    for top in tqdm(range(1, 21)):
        acc_ = 0.
        idx = 0.
        for video_ in video1:
            for user in range(1, 78):
                for time_ in range(1, 12):
                    acc_ += TopNRecommender().calAccuracy_kps(video_, time_, user, top)
                    idx += 1
        for video_ in video2:
            for user in range(1, 78):
                for time_ in range(1, 16):
                    acc_ += TopNRecommender().calAccuracy_kps(video_, time_, user, top)
                    idx += 1
        for video_ in video3:
            for user in range(1, 78):
                for time_ in range(1, 31):
                    acc_ += TopNRecommender().calAccuracy_kps(video_, time_, user, top)
                    idx += 1
        for video_ in video4:
            for user in range(1, 78):
                for time_ in range(1, 51):
                    acc_ += TopNRecommender().calAccuracy_kps(video_, time_, user, top)
                    idx += 1
        df_top.loc['top{}'.format(top)] = acc_/idx

    df_top.to_csv('./result/similar user/Top-N accuracy.csv')
