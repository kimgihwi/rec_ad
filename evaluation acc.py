import pandas as pd
import numpy as np


class EvaluationPredictAccuracy:
    """
    this class is used at calculation of best accuracy
    """

    def __init__(self, video, time, epoch):
        self.video = video
        self.time = time
        self.epoch = epoch

    def getAcc(self, train):
        df = pd.read_csv('./result/predict rating/{train} acc/video{video}/time{time}.csv'.format(train=train,
                                                                                                  video=self.video,
                                                                                                  time=self.time))
        best_acc = -1.
        best_epoch = -1
        for epoch in range(0, 10):
            tmp_acc = np.mean(df[str(epoch)])
            if best_acc < tmp_acc:
                best_acc = tmp_acc
                best_epoch = epoch + 1
        return best_acc, best_epoch

    def getLoss(self, train):
        df = pd.read_csv('./result/predict rating/{train}} loss/video{video}/time{time}.csv'.format(train=train,
                                                                                                    video=self.video,
                                                                                                    time=self.time))
        best_loss = 100.
        best_epoch = -1
        for epoch in range(0, 10):
            tmp_loss = np.mean(df[epoch])
            if best_loss > tmp_loss:
                best_loss = tmp_loss
                best_epoch = epoch + 1
        return best_loss, best_epoch


if __name__ == '__main__':
    print(EvaluationPredictAccuracy(video=1, time=1).getAcc('train'))
