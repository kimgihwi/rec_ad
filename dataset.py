import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import cv2
from skimage import transform

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class KiwiDataset(Dataset):
    def __init__(self):
        self.face = self.load_face('./Data/Item1/')
        self.label = self.load_label('./Data/Item1/')

    def __len__(self):
        return len(self.face)

    def __getitem__(self, idx):
        x = torch.tensor(self.face[idx])
        y = torch.tensor(self.label[idx])
        return x, y

    def load_face(self, path):
        csv_list = []
        user_list = os.listdir()

        for i in user_list:
            tmp_df = pd.read_csv('i1_u' + str(i[4]) + 'csv', sep=',')
            csv_list.append(tmp_df)

        return np.array(csv_list)

    def load_label(self, path):
        tmp_df = pd.read_csv('label.csv', sep=',')

        label_list = list(tmp_df)

        return np.array(label_list)
