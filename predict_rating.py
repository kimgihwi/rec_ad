from __future__ import print_function, division
import os
import time
import copy
from tqdm import tqdm, tqdm_notebook
import csv

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils, models


import cv2
from skimage import io, transform
from PIL import Image

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# 경고 메시지 무시하기
import warnings
warnings.filterwarnings("ignore")


# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class RatingDataset(Dataset):

    def __init__(self, video, time, user, val=False, transform=None):
        self.val = val
        self.user = user-1
        self.root_dir = './Data/inputData/rating/video{video}/time{time}/'.format(video=video, time=time)
        self.transform = transform

        label_list = list(pd.read_csv('./Data/rating.csv', index_col='user').astype(int)['video{0}'.format(video)])
        label_list = [i-1 for i in label_list]
        # self.label = list(pd.read_csv('./Data/rating.csv', index_col='user').astype(int)['video{0}'.format(video)])

        img_list = []
        # for file in range(self.__len__()):
        for file in range(len(label_list)):
            # img_list.append(cv2.imread(self.root_dir + '/user{file}.png'.format(file=file+1)))
            img_list.append(Image.open(self.root_dir + 'user{file}.png'.format(file=file+1)))
        # self.img = img_list

        if self.val:
            self.label = [label_list[self.user]]
        else:
            del label_list[self.user]
            self.label = label_list

        if self.val:
            self.img = [img_list[self.user]]
        else:
            del img_list[self.user]
            self.img = img_list

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        """
        :param idx: the number of image(user)
        :return: {image, label}
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.transform:
            return self.transform(self.img[idx]), self.label[idx]

        return self.img[idx], self.label[idx]


# train_dataset = RatingDataset(video=1, time=1, user=1, transform=data_transforms['train'])
# val_dataset = RatingDataset(video=1, time=1, user=1, val=True, transform=data_transforms['val'])
#
# dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0),
#                'val': torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)}
#
# dataset_size = {'train': train_dataset.__len__(),
#                 'val': val_dataset.__len__()}
#
# class_names = (1, 2, 3, 4, 5)


# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean   # 정규화를 해제
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(1)


# # 한개의 배치(batch)만큼 이미지를 불러온다. 배치 사이즈를 4로 했으니 사진 4장이 로드된다.
# inputs, classes = next(iter(dataloaders['train']))
#
# # 로드된 데이터에 make_grid 함수를 통해 그리드를 추가한다.
# out = torchvision.utils.make_grid(inputs)
#
# # 이미지를 출력한다.
# # imshow(out)
# imshow(out, title=[class_names[x] for x in classes])

#
# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     since = time.time()
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
#
#         # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # 모델을 학습 모드로 설정
#             else:
#                 model.eval()   # 모델을 평가 모드로 설정
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             # 데이터를 반복
#             for inputs, labels in dataloaders[phase]:
#                 labels = [x for x in labels]
#                 labels = torch.tensor(labels)
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 # 매개변수 경사도를 0으로 설정
#                 optimizer.zero_grad()
#
#                 # 순전파
#                 # 학습 시에만 연산 기록을 추적
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)
#
#                     # 학습 단계인 경우 역전파 + 최적화
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#
#                 # 통계
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#
#                 if phase == 'val':
#                     print('model predict : ', preds)
#                     print('real label : ', labels.data)
#
#             if phase == 'train':
#                 scheduler.step()
#             epoch_loss = running_loss / dataset_size[phase]
#             epoch_acc = running_corrects.double() / dataset_size[phase]
#
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))
#
#             # 모델을 깊은 복사(deep copy)함
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#
#         print()
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))
#
#     # 가장 나은 모델 가중치를 불러옴
#     model.load_state_dict(best_model_wts)
#     return model


# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = torch.nn.Linear(num_ftrs, 5)
#
# model_ft = model_ft.to(device)
#
# criterion = torch.nn.CrossEntropyLoss()
#
# # Observe that all parameters are being optimized
# optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#
#
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)


class modelEvaluation:

    def __init__(self, video, time, user, epoch=25):
        self.video = video
        self.time = time
        self.user = user
        self.answer = []
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []
        self.pred_list = []

        train_dataset = RatingDataset(video=self.video, time=self.time, user=self.user,
                                      transform=data_transforms['train'])
        val_dataset = RatingDataset(video=self.video, time=self.time, user=self.user, val=True,
                                    transform=data_transforms['val'])
        self.dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                                                 shuffle=True, num_workers=0),
                            'val': torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                               shuffle=False, num_workers=0)}

        self.dataset_size = {'train': train_dataset.__len__(),
                             'val': val_dataset.__len__()}

        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, 5)

        model_ft = model_ft.to(device)

        criterion = torch.nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        # model_ft = self.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epoch)
        self.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epoch)

    def getAnswer(self):
        return self.answer

    def resultValue(self):
        return self.pred_list, self.train_acc_list, self.val_acc_list, self.train_loss_list, self.val_loss_list

    def train_model(self, model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        # best_model_wts = copy.deepcopy(model.state_dict())
        # best_acc = 0.0

        for epoch in range(num_epochs):
            train_loss_list = []
            val_loss_list = []
            train_acc_list = []
            val_acc_list = []
            pred_list = []

            # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            # print('-' * 10)

            # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 모델을 학습 모드로 설정
                else:
                    model.eval()  # 모델을 평가 모드로 설정

                running_loss = 0.0
                running_corrects = 0

                # 데이터를 반복
                for inputs, labels in self.dataloaders[phase]:
                    labels = [x for x in labels]
                    labels = torch.tensor(labels)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 매개변수 경사도를 0으로 설정
                    optimizer.zero_grad()

                    # 순전파
                    # 학습 시에만 연산 기록을 추적
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # 학습 단계인 경우 역전파 + 최적화
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 통계
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    if phase == 'val':
                        pred_list.append(preds.tolist()[0]+1)
                        # if preds == labels.data:
                        #     self.answer.append(1)
                        # else:
                        #     self.answer.append(0)

                if phase == 'train':
                    scheduler.step()
                epoch_loss = running_loss / self.dataset_size[phase]
                epoch_acc = running_corrects.double() / self.dataset_size[phase]

                if phase == 'train':
                    train_loss_list.append(epoch_loss)
                    train_acc_list.append(epoch_acc)
                else:
                    val_loss_list.append(epoch_loss)
                    val_acc_list.append(epoch_acc)

                # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                #     phase, epoch_loss, epoch_acc))

            self.train_loss_list = train_loss_list
            self.val_loss_list = val_loss_list
            self.train_acc_list = train_acc_list
            self.val_acc_list = val_acc_list
            self.pred_list = pred_list

                # # 모델을 깊은 복사(deep copy)함
                # if phase == 'val' and epoch_acc > best_acc:
                #     best_acc = epoch_acc
                #     best_model_wts = copy.deepcopy(model.state_dict())

            # with open('user{user}_video{video}_time{time}.csv'.format(user=self.user, video=self.video, time=self.time),
            #           'w', newline='') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(pred_list)

        time_elapsed = time.time() - since
        print('Video{video} User{user} Time{time} Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60, video=self.video, user=self.user, time=self.time))
        # print('Best val Acc: {:4f}'.format(best_acc))

        # # 가장 나은 모델 가중치를 불러옴
        # model.load_state_dict(best_model_wts)
        #
        # return model


if __name__ == '__main__':

    video = 1

    pred_list = []
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    for t in tqdm(range(1, 61)):
        for user in tqdm(range(1, 78)):
            pred, train_acc, test_acc, train_loss, test_loss \
                = modelEvaluation(video=video, time=t, user=user, epoch=10).resultValue()
            pred_list.append(pred)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            train_loss_list.append(train_loss)
            train_loss_list.append(test_loss)

        df_pred = pd.DataFrame(pred_list)
        df_train_acc = pd.DataFrame(train_acc_list)
        df_test_acc = pd.DataFrame(test_acc_list)
        df_train_loss = pd.DataFrame(train_loss_list)
        df_test_loss = pd.DataFrame(test_loss_list)

        df_pred.to_csv('./result/predict rating/predict/video{video}/time{time}.csv'.format(video=video, time=t))
        df_train_acc.to_csv('./result/predict rating/train acc/video{video}/time{time}.csv'.format(video=video, time=t))
        df_train_loss.to_csv('./result/predict rating/train loss/video{video}/time{time}.csv'.format(video=video, time=t))
        df_test_acc.to_csv('./result/predict rating/test acc/video{video}/time{time}.csv'.format(video=video, time=t))
        df_test_loss.to_csv('./result/predict rating/test loss/video{video}/time{time}.csv'.format(video=video, time=t))
