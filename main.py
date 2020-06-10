import cv2
import numpy as np
import dlib
import os


def video_capture(path, color='color', mode='save'):
    """
    :param path: video file path + video file name
    :param color: color default value is 'color'. If you want to use gray image, then set mode to 'gray'.
    :param mode: mode default value is 'save'. If you want to get image coordinate, then set mode to 'save'.
    :return: If you set mode to 'coordinate', then you could get coordinates of cropped image.
    """

    face_cascade = cv2.CascadeClassifier('./detector/haarcascade_frontalface_alt.xml')

    vid = cv2.VideoCapture(path)
    iterator = 0        # while iterator
    img_idx = 0     # cropped image index

    # face coordinate list for each image
    x_co_list = []
    y_co_list = []
    height_list = []
    width_list = []

    while vid.isOpened():
        ret, img = vid.read()

        if ret is False:
            vid.release()
            break

        if color == 'gray':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(img, 1.3, 5)  # image scale

        # control capture interval
        if iter % 4 != 0:
            iter += 1
            continue

        for (x, y, w, h) in faces:
            cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]  # cropped image
            cv2.imwrite('./data/crop/crop_img_' + str(img_idx) + '.png', cropped)    # save cropped image

            # if you want to get face recognition coordinate for each image
            if mode == 'coordinate':
                x_co_list.append(x)
                y_co_list.append(y)
                height_list.append(h)
                width_list.append(w)

        iterator += 1
        img_idx += 1

        if mode == 'coordinate':
            return np.array([x_co_list, y_co_list, height_list, width_list])

        print("image cropped success")


def face_landmark(path, mode='normal'):
    """
    This function is a function for obtaining 68 of face landmark.
    :param path: face image directory
    :param mode: If you want to save dot image about 68 of face landmark, then set mode to 'save'.
    :return: 68 of face landmark coordinate list about every image in directory. / type == numpy.array
    """


    img_path = os.getcwd() + '/data' + path   # type == str
    img_file_list = os.listdir(os.getcwd())   # type == list
    img_num = len(img_file_list)              # amount of images in directory

    landmark_list = []

    for i in img_num:
        tmp_img = cv2.imread(img_path + 'crop_img_' + str(i) + 'png')
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('./detector/shape_predictor_68_face_landmarks.dat')
        faces = detector(tmp_img)

        tmp_landmark_list = []

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
            cv2.imwrite(os.getcwd() + 'data/face_landmark/landmark_' + str(i) + 'png', tmp_img)

    return np.array(landmark_list)


if __name__ == '__main__':
    print("main")
