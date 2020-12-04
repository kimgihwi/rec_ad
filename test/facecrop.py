import numpy as np
import matplotlib.pyplot as plt

import cv2


face_cascade = cv2.CascadeClassifier('./detector/haarcascade_frontalface_alt.xml')

video_cap = cv2.VideoCapture('./Data/1/User1_Video1.avi')

while True:
    ret, frame = video_cap.read()

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
