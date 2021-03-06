import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import csv
import random
from tqdm import tqdm, tqdm_notebook


class EvaluationPredictRating:

    def __init__(self, video, time):
        self.video = video
        self.time = time

    def getPredictData(self, pred_path, time):
        pred = pd.read_csv('{path}/video{video}/time{time}.csv'.format(path=pred_path, video=self.video, time=time),
                           index_col=0)
        # pred.index(['user{}'.format(u) for u in range(1, 78)])

        # pred = pd.DataFrame(index=range(1, 11))
        # f = open('{path}/video{video}/time{time}.csv'.format(path=pred_path, video=self.video, time=self.time),
        #          'r', encoding='utf-8')
        # rdr = csv.reader(f)
        # idx = 0
        # for line in rdr:
        #     user = idx
        #     if idx == 0:
        #         continue
        #     # pred.loc['user{}'.format(user)] = line[1:]
        #     pred.loc[idx] = line[1:]
        #     idx += 1
        return pred

    def getActualData(self):
        act = pd.read_csv('./Data/rating.csv', index_col=0)
        return act['video' + str(self.video)]

    def getMAE(self):
        df_act = self.getActualData()
        # df_eval = self.getPredictData('./result/predict rating/predict')
        # df_MAE = pd.DataFrame(index=['user{}'.format(user) for user in range(1, 78)])

        best_acc = 100.
        best_epoch = -1
        MAE_list = []

        df_eval = self.getPredictData('./result/predict rating/resnet34/predict', time=1)
        # df_eval = self.getPredictData('./result/predict rating/predict', time=1)
        if self.time > 1:
            for time in range(1, self.time):
                # df_eval += self.getPredictData('./result/predict rating/predict', time=time)
                df_eval += self.getPredictData('./result/predict rating/resnet34/predict', time=time)
        df_eval = df_eval/self.time

        for epoch in range(0, 10):

            # print(list(df_eval[str(epoch)] - df_act.T.iloc[0]))
            error = list(df_eval[str(epoch)] - df_act.T.iloc[0])
            # df_MAE['epoch{}'.format(epoch + 1)] = [abs(e) for e in error]
            mae = sum([abs(e) for e in error])/77.
            if mae < best_acc:
                best_acc = mae
                best_epoch = epoch + 1
            MAE_list = [best_acc, best_epoch]
        return MAE_list


class EvaluationAverageRating:

    def __init__(self):
        print('start to evaluate average rating for mean absolute error')

    def getActualData(self):
        act = pd.read_csv('./Data/rating.csv', index_col=0)
        return act

    def getMAE(self):
        avg_rating = []
        act = self.getActualData()

        for video in range(1, 21):
            tmp_avg = np.mean(act['video' + str(video)])
            avg_rating.append(tmp_avg)
        avg_rating = np.array(avg_rating)

        MAE_list = []
        for video_ in range(1, 21):
            error = avg_rating[video_ - 1] - act['video' + str(video_)]
            MAE = np.mean(np.abs(error))
            MAE_list.append(MAE)
        return MAE_list


class EvaluationRandomRating:

    def __init__(self):
        print('start to evaluate random rating for mean absolute error')

    def getActualData(self):
        act = pd.read_csv('./Data/rating.csv', index_col=0)
        return act

    def getMAE(self):
        random_rating = []
        act = self.getActualData()

        for video in range(1, 21):
            i = random.randint(1, 5)
            random_rating.append(i)

        MAE_list = []
        for video_ in range(1, 21):
            error = random_rating[video_ - 1] - act['video' + str(video_)]
            MAE = np.mean(np.abs(error))
            MAE_list.append(MAE)
        return MAE_list


class EvaluationItemCFRating:

    def __init__(self, video, user):
        self.video = video
        self.user = user
        self.df = self.getActualData()

    def getActualData(self):
        act = pd.read_csv('./Data/rating.csv', index_col=0)
        return act

    def calSim(self, input1, input2):
        # input1 += 0.
        # input2 += 0.
        return dot(input1, input2)/(norm(input1)*norm(input2))

    def itemSim(self, com):
        target = self.df['video'+str(self.video)]
        compare = self.df['video'+str(com)]
        target = target.drop('user{}'.format(self.user))
        compare = compare.drop('user{}'.format(self.user))

        return self.calSim(target, compare)

    def calRating(self):
        video_list = list(range(1, 21))
        video_list.remove(self.video)
        sum_cf_rating = 0.
        sum_weight = 0.
        for idx in video_list:
            # sum_cf_rating += self.itemSim(idx) * self.df['video'+str(self.video)].loc['user{}'.format(self.user)]
            sum_cf_rating += self.itemSim(idx) * self.df['video'+str(idx)].loc['user{}'.format(self.user)]
            sum_weight += self.itemSim(idx)
        # mean_cf_rating = sum_cf_rating/len(video_list)
        # mean_cf_rating = mean_cf_rating/sum_weight
        mean_cf_rating = sum_cf_rating/sum_weight
        return mean_cf_rating


class EvaluationUserCFRating:

    def __init__(self, video, user):
        self.video = video
        self.user = user
        self.df = self.getActualData()

    def getActualData(self):
        act = pd.read_csv('./Data/rating.csv', index_col=0)
        return act

    def calSim(self, input1, input2):
        # input1 += 0.
        # input2 += 0.
        return dot(input1, input2)/(norm(input1)*norm(input2))

    def userSim(self, com):
        target = self.df.loc['user'+str(self.user)]
        compare = self.df.loc['user'+str(com)]
        target = target.drop('video{}'.format(self.video))
        compare = compare.drop('video{}'.format(self.video))

        return self.calSim(target, compare)

    def calRating(self):
        user_list = list(range(1, 78))
        user_list.remove(self.user)
        sum_cf_rating = 0.
        sum_weight = 0.
        for idx in user_list:
            # sum_cf_rating += self.itemSim(idx) * self.df['video'+str(self.video)].loc['user{}'.format(self.user)]
            sum_cf_rating += self.userSim(idx) * self.df['video'+str(self.video)]['user{}'.format(idx)]
            sum_weight += self.userSim(idx)
        # mean_cf_rating = sum_cf_rating/len(video_list)
        # mean_cf_rating = mean_cf_rating/sum_weight
        mean_cf_rating = sum_cf_rating/sum_weight
        return mean_cf_rating


if __name__ == '__main__':

    ## evaluation to predict rating
    # video_ = [2]
    # video_ = [3, 4, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19]
    # video_ = [1, 5, 7, 12, 14, 20]
    # video_ = [6]
    video_ = [13]

    for video in video_:
        df_MAE = pd.DataFrame()
        for time in range(1, 16):
            df_MAE['time{}'.format(time)] = EvaluationPredictRating(video=video, time=time).getMAE()
            df_MAE = df_MAE.rename(index={0: 'MAE', 1: 'epoch'})
        # df_MAE.to_csv('./result/predict rating/predict/mae/model/total/video{}.csv'.format(video))
        df_MAE.to_csv('./result/predict rating/resnet34/predict/mae_video{}.csv'.format(video))


    ### evaluation to average rating
    # with open('./result/predict rating/predict/mae/avg/avg_mae.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(EvaluationAverageRating().getMAE())


    ### evaluation to random rating
    # random.seed(4)
    # with open('./result/predict rating/predict/mae/random/random_mae.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(EvaluationRandomRating().getMAE())

    ### evaluation to Item based CF rating
    # df_cf = pd.DataFrame(columns=['video{}'.format(video) for video in range(1, 21)])
    # for user_ in tqdm(range(1, 78)):
    #     tmp_rating = []
    #     for video_ in range(1, 21):
    #         tmp_rating.append(EvaluationItemCFRating(video_, user_).calRating())
    #     df_cf.loc['user{}'.format(user_)] = tmp_rating
    # # df_cf.to_csv('./result/predict rating/predict/mae/cf/cf_predict.csv')
    # df_actual = EvaluationItemCFRating(1, 1).getActualData()
    # df_error = df_cf.sub(df_actual)
    # df_error.to_csv('./result/predict rating/predict/mae/cf/cf_error.csv')
    # df_sub_abs = df_error.abs()
    # mae_list = df_sub_abs.mean()
    # with open('./result/predict rating/predict/mae/cf/cf_mae.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(mae_list)

    ### evaluation to User based CF rating
    # df_cf = pd.DataFrame(columns=['video{}'.format(video) for video in range(1, 21)])
    # for user_ in tqdm(range(1, 78)):
    #     tmp_rating = []
    #     for video_ in range(1, 21):
    #         tmp_rating.append(EvaluationUserCFRating(video_, user_).calRating())
    #     df_cf.loc['user{}'.format(user_)] = tmp_rating
    # # df_cf.to_csv('./result/predict rating/predict/mae/cf/cf_predict.csv')
    # df_actual = EvaluationUserCFRating(1, 1).getActualData()
    # df_error = df_cf.sub(df_actual)
    # df_error.to_csv('./result/predict rating/predict/mae/user based cf/user_cf_error.csv')
    # df_sub_abs = df_error.abs()
    # mae_list = df_sub_abs.mean()
    # with open('./result/predict rating/predict/mae/user based cf/user_cf_mae.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(mae_list)

    ### calculating to user similarity
    # df_sim = pd.DataFrame(columns=['user{}'.format(video) for video in range(1, 78)])
    # for u1 in tqdm(range(1, 78)):
    #     tmp_sim_list = []
    #     for u2 in range(1, 78):
    #         tmp_sim_list.append(EvaluationUserCFRating(1, u2).userSim(u1))
    #     df_sim.loc['user{}'.format(u1)] = tmp_sim_list
    #
    # df_sim.to_csv('./result/predict rating/predict/user_similarity.csv')

    # df_sim = pd.DataFrame(columns=['user{}'.format(video) for video in range(1, 78)])
    # best_sim_list = []
    # best_user_list = []
    # for u1 in tqdm(range(1, 78)):
    #     best_sim = 0.
    #     best_user = 0.
    #     for u2 in range(1, 78):
    #         if u1 == u2:
    #             continue
    #         tmp_sim = EvaluationUserCFRating(1, u2).userSim(u1)
    #         if tmp_sim > best_sim:
    #             best_sim = tmp_sim
    #             best_user = u2
    #     best_sim_list.append(best_sim)
    #     best_user_list.append(best_user)
    # df_sim.loc['user'] = best_user_list
    # df_sim.loc['sim'] = best_sim_list
    #
    # df_sim.to_csv('./result/predict rating/predict/user_best_similarity.csv')
