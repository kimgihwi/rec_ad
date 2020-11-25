import cv2

import os



class ImageDiff:
    def __init__(self, path, time):
        self.path = path
        self.time = time
        self.img = self.imgValue()

    def imgList(self):
        file_list = os.listdir(self.path)

        not_exist = range(self.time)

        for idx in file_list:
            not_exist.remove(idx)

        return file_list, not_exist

    def imgMissingFill(self):
        print('a')


    def imgDiff(self):
        for idx in range(self.time):
            print('a')



if __name__=='__main__':
    print('쿄쿄')
