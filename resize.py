from PIL import Image

import os


image_path = './Data/crop/test/video1/5'

img_num = len(os.walk(image_path).__next__()[2])

for i in range(img_num):
    tmp = Image.open(image_path + '/' + str(i) + '.png')
    resize = tmp.resize((32, 32))
    resize.save('./Data/crop/test/test_img/' + str(i) + '.png')
