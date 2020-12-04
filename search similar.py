import numpy as np
from numpy.linalg import norm
import pandas as pd
from tqdm import tqdm, tqdm_notebook


class searchSim:

    def __init__(self, video, user):
        self.video = video
        self.user = user-1
        self.df_kps = self.readKPS()
        self.user_list = self.df_kps.index.tolist()
        self.user_list.remove('user{user}'.format(user=user))

    def readKPS(self):
        df_kps = pd.read_csv('./result/kps/kps_video{video}.csv'.format(video=self.video), index_col=0, header=0)
        return df_kps

    def calSim(self, u1, u2):
        u1 = np.array(u1) + 1e-15
        u2 = np.array(u2) + 1e-15
        return np.dot(u1, u2)/(norm(u1) * norm(u2))

    def searchUser(self, time):
        best_sim = -1
        best_user = '-1'
        for u in self.user_list:
            tmp_sim = self.calSim(self.df_kps.iloc[self.user][:time+1], self.df_kps.loc[u][:time+1])
            if tmp_sim > best_sim:
                best_sim = tmp_sim
                best_user = u[4:]
        return best_user

    def getSimList(self):
        sim_list = []
        for t in range(len(self.df_kps.T)):
            b_user = self.searchUser(t)
            sim_list.append(b_user)
        return sim_list


if __name__ == '__main__':
    for video in tqdm(range(1, 21)):
        tmp_list = []
        for user in range(1, 78):
            tmp_list.append(searchSim(video, user).getSimList())
        tmp_df = pd.DataFrame(tmp_list, index=['user{}'.format(u) for u in range(1, 78)])
        tmp_df.to_csv('./result/similar user/kps_video{video}.csv'.format(video=video))
