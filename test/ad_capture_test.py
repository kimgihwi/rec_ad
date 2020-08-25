import cv2

import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui

import os

import time

import json
from collections import OrderedDict


abs_path = os.getcwd()


def video_open(video, path):
    file = "AD" + str(video) + ".mp4"
    os.system(path+file)

    vid_cap = cv2.VideoCapture(path+file)
    video_frame = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
    vid_cap.release()

    return video_frame/video_fps


def video_close():
    os.system('taskkill /f /im Video.UI.exe')
    return True


def video_player(video, user):
    cam_cap = cv2.VideoCapture(0)
    cam_width = int(cam_cap.get(3))
    cam_height = int(cam_cap.get(4))
    path = abs_path + '/video/'
    video_open(video, path)
    video_time = video_open(video, path)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    save_out = cv2.VideoWriter((abs_path + '/capture/' + 'User' + str(user) + '_Video' + str(video) + '.avi'),
                               fourcc, 30.0, (cam_width, cam_height))

    # idx = 0

    now = time.time()

    while True:
        _, frame = cam_cap.read()

        ### Capture Video Save Mode ###
        save_out.write(frame)
        diff = time.time() - now
        # if diff > 16:
        #     break
        if diff > video_time + 0.5:
            break
        ###############################

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_close()
    cam_cap.release()
    cv2.destroyAllWindows()


class App(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        # Application Window
        self.setWindowTitle("Kiwi Recommendation System")
        self.setWindowIcon((QtGui.QIcon(abs_path + '/etc/icon.png')))
        self.setFixedSize(800, 450)
        self.center()
        self.statusBar().showMessage('Ready')

        # User Information
        self.u_num = -1
        self.i_num = -1

        self.num_line = QtWidgets.QLineEdit(self)
        self.name_line = QtWidgets.QLineEdit(self)
        self.gender_line = QtWidgets.QLineEdit(self)
        self.age_line = QtWidgets.QSpinBox(self)

        self.num_line.setAlignment(QtCore.Qt.AlignCenter)
        self.name_line.setAlignment(QtCore.Qt.AlignCenter)
        self.gender_line.setAlignment(QtCore.Qt.AlignCenter)
        self.age_line.setAlignment(QtCore.Qt.AlignCenter)

        self.num_line.setFont(QtGui.QFont("나눔바른고딕", 9))
        self.name_line.setFont(QtGui.QFont("나눔바른고딕", 9))
        self.gender_line.setFont(QtGui.QFont("나눔바른고딕", 9))
        self.age_line.setFont(QtGui.QFont("나눔바른고딕", 9))

        #
        self.num_line.move(670, 50)
        self.name_line.move(670, 100)
        self.gender_line.move(670, 150)
        self.age_line.move(670, 200)

        self.num_label = QtWidgets.QLabel("No.", self)
        self.name_label = QtWidgets.QLabel("이름 : ", self)
        self.gender_label = QtWidgets.QLabel("성별 : ", self)
        self.age_label = QtWidgets.QLabel("나이 : ", self)

        self.num_label.resize(50, 30)
        self.name_label.resize(50, 30)
        self.gender_label.resize(50, 30)
        self.age_label.resize(50, 30)

        self.num_label.setFont(QtGui.QFont("나눔스퀘어라운드 Bold", 8))
        self.name_label.setFont(QtGui.QFont("나눔스퀘어라운드 Bold", 8))
        self.gender_label.setFont(QtGui.QFont("나눔스퀘어라운드 Bold", 8))
        self.age_label.setFont(QtGui.QFont("나눔스퀘어라운드 Bold", 8))

        self.num_label.move(630, 50)
        self.name_label.move(630, 100)
        self.gender_label.move(630, 150)
        self.age_label.move(630, 200)

        self.save_btn = QtWidgets.QPushButton("Save", self)
        self.save_btn.move(670, 250)
        self.save_btn.clicked.connect(self.save_clicked)

        # video start
        self.btn1 = QtWidgets.QPushButton("Video1", self)
        self.btn2 = QtWidgets.QPushButton("Video2", self)
        self.btn3 = QtWidgets.QPushButton("Video3", self)
        self.btn4 = QtWidgets.QPushButton("Video4", self)
        self.btn5 = QtWidgets.QPushButton("Video5", self)
        self.btn6 = QtWidgets.QPushButton("Video6", self)
        self.btn7 = QtWidgets.QPushButton("Video7", self)
        self.btn8 = QtWidgets.QPushButton("Video8", self)
        self.btn9 = QtWidgets.QPushButton("Video9", self)
        self.btn10 = QtWidgets.QPushButton("Video10", self)
        self.btn11 = QtWidgets.QPushButton("Video11", self)
        self.btn12 = QtWidgets.QPushButton("Video12", self)
        self.btn13 = QtWidgets.QPushButton("Video13", self)
        self.btn14 = QtWidgets.QPushButton("Video14", self)
        self.btn15 = QtWidgets.QPushButton("Video15", self)
        self.btn16 = QtWidgets.QPushButton("Video16", self)
        self.btn17 = QtWidgets.QPushButton("Video17", self)
        self.btn18 = QtWidgets.QPushButton("Video18", self)
        self.btn19 = QtWidgets.QPushButton("Video19", self)
        self.btn20 = QtWidgets.QPushButton("Video20", self)

        self.btn1.move(20, 50)
        self.btn2.move(140, 50)
        self.btn3.move(260, 50)
        self.btn4.move(380, 50)
        self.btn5.move(500, 50)
        self.btn6.move(20, 160)
        self.btn7.move(140, 160)
        self.btn8.move(260, 160)
        self.btn9.move(380, 160)
        self.btn10.move(500, 160)
        self.btn11.move(20, 270)
        self.btn12.move(140, 270)
        self.btn13.move(260, 270)
        self.btn14.move(380, 270)
        self.btn15.move(500, 270)
        self.btn16.move(20, 380)
        self.btn17.move(140, 380)
        self.btn18.move(260, 380)
        self.btn19.move(380, 380)
        self.btn20.move(500, 380)

        self.video_clicked(self.btn1, 1)
        self.video_clicked(self.btn2, 2)
        self.video_clicked(self.btn3, 3)
        self.video_clicked(self.btn4, 4)
        self.video_clicked(self.btn5, 5)
        self.video_clicked(self.btn6, 6)
        self.video_clicked(self.btn7, 7)
        self.video_clicked(self.btn8, 8)
        self.video_clicked(self.btn9, 9)
        self.video_clicked(self.btn10, 10)
        self.video_clicked(self.btn11, 11)
        self.video_clicked(self.btn12, 12)
        self.video_clicked(self.btn13, 13)
        self.video_clicked(self.btn14, 14)
        self.video_clicked(self.btn15, 15)
        self.video_clicked(self.btn16, 16)
        self.video_clicked(self.btn17, 17)
        self.video_clicked(self.btn18, 18)
        self.video_clicked(self.btn19, 19)
        self.video_clicked(self.btn20, 20)

        # exit button
        self.exit_btn = QtWidgets.QPushButton("exit", self)
        self.exit_btn.move(670, 390)
        self.exit_btn.clicked.connect(lambda: QtCore.QCoreApplication.quit())

    def center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def on_click(self):
        self.lbl.setText(self.qle.text())
        self.lbl.adjustSize()

    def video_clicked(self, btn, num):
        btn.clicked.connect(lambda: self.change_video(num))
        btn.clicked.connect(lambda: video_player(video=self.i_num, user=self.u_num))

    def change_video(self, num):
        self.i_num = int(num)

    def save_clicked(self):
        self.u_num = int(self.num_line.text())

        # saving user information to json
        file_data = OrderedDict()
        file_data["No."] = self.u_num
        file_data["Name"] = self.name_line.text()
        file_data["Gender"] = self.gender_line.text()
        file_data["Age"] = self.age_line.text()
        with open(abs_path + "/User" + str(self.u_num) + ".json", "w", encoding="utf-8") as make_file:
            json.dump(file_data, make_file, ensure_ascii=False, indent='\t')

        self.statusBar().showMessage('User' + str(self.u_num))


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    ex.show()
    app.exec_()
