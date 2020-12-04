from PIL import Image
import cv2

import os

from tqdm import tqdm, tqdm_notebook

#
# image_path = './Data/crop/test/video1/5'
#
# img_num = len(os.walk(image_path).__next__()[2])
#
# for i in range(img_num):
#     tmp = Image.open(image_path + '/' + str(i) + '.png')
#     resize = tmp.resize((32, 32))
#     resize.save('./Data/crop/test/test_img/' + str(i) + '.png')


def imgResize(path, user, video):
    img_list = os.listdir(path)
    for img in img_list:
        tmp_img = cv2.imread(path + '/' + img)
        img_ = cv2.resize(tmp_img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite('./Data/crop/CNN based/resize/' + 'video' + str(video) + '/' + str(user) + '/' + str(img), img_)


# num_user = 38
# num_video = 20
#
# for v in tqdm(range(1, num_video+1)):
#     for u in range(1, num_user+1):
#         tmp_path = './Data/crop/CNN based/video' + str(v) + '/' + str(u)
#         imgResize(tmp_path, u, v)

for v in tqdm(range(1, 20+1)):
    for u in range(73, 79):
        # tmp_path = './Data/crop/CNN based/video' + str(v) + '/' + str(u)
        tmp_path = './Data/crop/CNN based/preprocessing/video' + str(v) + '/' + str(u)
        imgResize(tmp_path, u, v)
