import cv2
import dlib
import face_recognition
from PIL import Image

import os

import numpy as np
import pandas as pd

from tqdm import tqdm


# def video_capture(file, user, path='./', savepath='./', color='color', mode='save'):
def video_capture(file, user, path='./', savepath='./', color='color'):
    """
    :param file: video file name, type == str()
    :param user: user number, type == int()
    :param path: video file path, type == str()
    :param savepath: crop image saving path, type == str()
    :param color: color default value is 'color'. If you want to use gray image, then set mode to 'gray'.
    # :param mode: mode default value is 'save'. If you want to get image coordinate, then set mode to 'save'.
    :return: If you set mode to 'coordinate', then you could get coordinates of cropped image.
    """

    # face_cascade = cv2.CascadeClassifier('./detector/haarcascade_frontalface_alt.xml')

    vid = cv2.VideoCapture(path + '/' + file)
    iterator = 0        # while iterator
    img_idx = 0     # cropped image index

    # # face coordinate list for each image
    # x_co_list = []
    # y_co_list = []
    # height_list = []
    # width_list = []

    while vid.isOpened():
        ret, img = vid.read()

        if ret is False:
            vid.release()
            break

        if color == 'gray':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # control capture interval
        # video fps == 30, so 30/15 -> crop 2frame for 1sec
        if iterator % 15 != 0:
            iterator += 1
            continue

        # faces = face_cascade.detectMultiScale(img, 1.3, 5)
        # for (x, y, w, h) in faces:
        #     # cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]  # cropped image
        #     cropped = img[y:y + h, x:x + w]
        #     # cv2.imwrite(savepath + '/' + str(user) + '_' + str(img_idx) + '.png', cropped)    # save cropped image
        #     cv2.imwrite(savepath + '/' + str(img_idx) + '.png', cropped)

            # # if you want to get face recognition coordinate for each image
            # if mode == 'coordinate':
            #     x_co_list.append(x)
            #     y_co_list.append(y)
            #     height_list.append(h)
            #     width_list.append(w)

        face_locations = face_recognition.face_locations(img, model="cnn")
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = img[top:bottom, left:right]
            # pil_image = Image.fromarray(face_image)
            cv2.imwrite(savepath + '/' + str(img_idx) + '.png', face_image)

        iterator += 1
        img_idx += 1

        # if mode == 'coordinate':
        #     return np.array([x_co_list, y_co_list, height_list, width_list])

        # print("image " + str(img_idx) + " cropped success")


def face_landmark(path, mode='normal'):
    """
    This function is a function for obtaining 68 of face landmark.
    :param path: face image directory
    :param mode: If you want to save dot image about 68 of face landmark, then set mode to 'save'.
    :return: 68 of face landmark coordinate list about every image in directory. / type == numpy.array
    """

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./detector/shape_predictor_68_face_landmarks.dat')

    img_path = os.getcwd() + '/data/' + path   # type == str
    img_file_list = os.listdir(img_path)   # type == list
    img_num = len(img_file_list)              # amount of images in directory

    landmark_list = []

    idx = 0
    err_num = 0

    while True:
        file = img_path + '/crop_img_' + str(idx) + '.png'

        if os.path.isfile(file):
            tmp_img = cv2.imread(file)
        else:
            tmp_img = cv2.imread(img_path + '/crop_img_' + str(idx-1) + '.png')
            err_num += 1

        # detector = dlib.get_frontal_face_detector()
        # predictor = dlib.shape_predictor('./detector/shape_predictor_68_face_landmarks.dat')
        faces = detector(tmp_img)

        tmp_landmark_list = []

        if faces == dlib.rectangles([]):
            landmark_list.append(landmark_list[len(landmark_list)-1])
            idx += 1
            continue

        for face in faces:
            # face rectangular coordinate (not using)
            # x1 = face.left()
            # y1 = face.top()
            # x2 = face.right()
            # y2 = face.bottom()

            landmarks = predictor(tmp_img, face)

            for n in range(0, 68):
                x = landmarks.part(n).x     # x-coordinate of face landmark
                y = landmarks.part(n).y     # y-coordinate of face landmark

                tmp_landmark_list.append([x, y])

                if mode == 'save':
                    cv2.circle(tmp_img, (x, y), 4, (255, 0, 0), -1)

        landmark_list.append(tmp_landmark_list)

        if mode == 'save':
            cv2.imwrite(os.getcwd() + './data/face_landmark/landmark_' + str(idx) + '.png', tmp_img)

        idx += 1

        if idx == (img_num + err_num):
            break

    return np.array(landmark_list)


def landmark_table(arr, coor, mode=''):
    """
    :param arr: landmarks numpy array
    :param coor: x or y coordinate / type==str
    :param mode: if you want to save it, then mode='save'
    :return: landmark DataFrame (about x or y coordinate)
    """

    if coor == 'x':
        coor = 0
    elif coor == 'y':
        coor = 1
    else:
        print('invalid value')

    coor_list = []

    for i in range(len(arr)):
        coor_list.append(list(np.array(arr[i]).T[coor]))

    df_coor = pd.DataFrame(np.array(coor_list))

    if mode == 'save':
        df_coor.to_csv('test_coor_data3.csv', mode='w')

    return df_coor


def landmark_table_diff(df, mode=''):
    """
    :param df: landmark pandas DataFrame
    :param mode: if you want to save itm then mode='save'
    :return: landmark's differential DataFrame
    """

    diff_df = df - df.iloc[0]

    if mode == 'save':
        diff_df.to_csv('test_diff_df3.csv', mode='w')

    return diff_df


if __name__ == '__main__':
    print("excute main")
    # video_capture('happy.avi', color='gray')
    # landmarks = face_landmark('crop3')
    #
    # tmp_table = landmark_table(landmarks, 'x', mode='save')
    #
    # landmark_table_diff(tmp_table, mode='save')

    # User = 9
    # Video = 8
    # video_capture('User' + str(User) + '_Video' + str(Video) + '.avi', user=User, path='./Data/' + str(User),
    #               savepath='./data/crop', mode='save')

    # video = 1
    # user = 100
    # numUser = 33
    # numUser += 1
    # for i in tqdm(range(1, numUser)):
    #     video_capture(str(i) + '.avi', user=i, path='./Data/video' + str(video),
    #                   savepath='./Data/crop/video' + str(video) + '/' + str(i))
    # for video in [1, 2, 4]:
    #     for i in tqdm(range(1, numUser)):
    #         video_capture(str(i) + '.avi', user=i, path='./Data/video' + str(video),
    #                         savepath='./Data/crop/video' + str(video) + '/' + str(i))
        # print("remain " + str(i) + "/" + str(numUser-1))
        # video_capture(str(i) + '.avi', user=i, path='./Data/video' + str(video),
        #               savepath='./Data/crop/video' + str(video) + '/' + str(i), mode='save')

    # video_capture(str(user) + '.avi', user=user, path='./Data/video' + str(video),
    #               savepath='./Data/crop/video' + str(video) + '/' + str(user), mode='save')
    # video_capture(str(user) + '.avi', user=user, path='./Data/video' + str(video),
    #               savepath='./Data/crop/video' + str(video) + '/' + str(user))

    # for user in tqdm(range(1, 38)):
    #     for video in range(1, 21):
    #         video_capture(str(user) + '.avi', user=user, path='./Data/video' + str(video),
    #                       savepath='./Data/crop/video' + str(video) + '/' + str(user))

    for user in range(38, 39):
        for video in tqdm(range(1, 21)):
            video_capture(str(user) + '.avi', user=user, path='./Data/video' + str(video),
                          savepath='./Data/crop/video' + str(video) + '/' + str(user))
