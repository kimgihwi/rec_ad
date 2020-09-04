import os
import numpy as np
import pandas as pd
import random

from tqdm import tqdm
import matplotlib.pyplot as plt

import cv2
from skimage import transform

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class KiwiDataset(Dataset):
    def __init__(self):
        self.face = self.load_face()
        self.label = self.load_label()

    def __len__(self):
        return len(self.face)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.face[idx])
        y = torch.tensor(self.label[idx])
        return x, y

    def load_face(self):
        face_list = []
        for user_num in range(32):
            tmp_time = []
            for time in range(15):
                tmp_feature = []
                for feature in range(68):
                    tmp_feature.append(float(random.randrange(-5, 5)))
                tmp_time.append(tmp_feature)
            face_list.append(tmp_time)
            # for feature in range(68):
            #     tmp_time = []
            #     for time in range(15):
            #         tmp_time.append(float(random.randrange(-5, 5)))
            #     # tmp_feature.append(np.array(tmp_time))
            #     tmp_feature.append(tmp_time)
            # # face_list.append(np.array(tmp_feature))
            # face_list.append(tmp_feature)
        return np.array(face_list)

    def load_label(self):
        rate_list = []
        for user_num in range(32):
            rate_list.append(float(random.randrange(1, 6)))
        #
        # tmp_np = np.array(rate_list)
        # print(tmp_np)
        # print(torch.Tensor(tmp_np))

        return np.array(rate_list)


dataset = KiwiDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# model = torch.nn.Linear(3, 1)
model = torch.nn.Linear(68, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 4

for epoch in range(nb_epochs + 1):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # print(batch_idx)
        x_train, y_train = inputs, targets
        print("input : ", inputs.size())
        print("target : ", targets)

        prediction = model(x_train)
        # prediction = F.softmax(prediction, 1)
        #
        # print("prediction", prediction)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        #     epoch, nb_epochs, batch_idx+1, len(dataloader),
        #     cost.item()
        #     ))

#
# if __name__ == '__main__':
#
