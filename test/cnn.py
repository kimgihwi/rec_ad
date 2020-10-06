import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2

import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(10 * 155 * 155, 50)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 10*155*155)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


from numpy import dot
from numpy.linalg import norm


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


from PIL import Image


if __name__ == '__main__':
    img_list = []

    for i in range(1, 30):
        tmp = cv2.imread('./data/' + str(i) + '.png')
        tmp = cv2.resize(tmp, dsize=(163, 163), interpolation=cv2.INTER_AREA)
        if i == 1:
            first_img = tmp
        # print(tmp.shape)
        # tmp = Image.open('./data/' + str(i) + '.png')
        # tmp = tmp.resize((155, 155))
        # tmp = np.array(tmp)
        img_list.append(tmp - first_img)

    img_arr = np.array(img_list)
    img_arr = np.transpose(img_arr, axes=(0, 3, 1, 2))
    # print(img_arr.shape)
    intorch = torch.FloatTensor(img_arr)

    cnn = CNN()
    output = cnn(intorch)

    # print(output)
    # print(output[0] - output[1])

    sim_list = []

    for i in range(1, 29):
        # print(cos_sim(output[i].detach().numpy(), output[0].detach().numpy()))
        print(output[i].detach().numpy())
        sim_list.append(cos_sim(output[i].detach().numpy(), output[0].detach().numpy()))

    import matplotlib.pyplot as plt
    plt.plot(sim_list)
    plt.show()
    # import matplotlib.pyplot as plt
    # for i in range(29):
    #     plt.imshow(img_list[i])
    #     plt.show()

    # print(output[1].detach().numpy())
