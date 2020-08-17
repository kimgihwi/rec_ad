import cv2
from playsound import playsound
import av

from multiprocessing import Process
import ray

import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui

import os

import time

import json
from collections import OrderedDict


def video_open(video, path="./"):
    # os.popen("C:/kiwi/ui_basic/video1.mp4")
    # os.system("C:/kiwi/ui_basic/video1.mp4")
    file = "AD" + str(video) + ".mp4"
    os.system(path+file)
    # print(os.system('tasklist'))

    # time.sleep(15)
    # os.system('taskkill /f /im Video.UI.exe')

    vid_cap = cv2.VideoCapture(path+file)
    video_frame = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
    vid_cap.release()

    # print(video_frame)
    # print(video_fps)

    return video_frame/video_fps


def video_close():
    os.system('taskkill /f /im Video.UI.exe')
    return True
#
#
# def video_sound():
#     playsound('./video1.mp3', True)
#
#
# def video_play():
#     video = './video1.mp4'
#     video_cap = cv2.VideoCapture(video)
#     cam_cap = cv2.VideoCapture(0)
#
#     idx = 0
#
#     while True:
#         ret, frame = video_cap.read()
#         # ret, frame = cam_cap.read()
#         ret1, frame1 = cam_cap.read()
#
#         if ret:
#             cv2.imshow('video', frame)
#
#             #
#             # if video_cap.isOpened() == False:
#             #     break
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#             if idx % 10 == 0:
#                 cv2.imwrite('u1_i1_' + str(idx/10) + '.png', frame1)
#             idx += 1
#
#         else:
#             video_close()
#             break
#
#     video_cap.release()
#     cam_cap.release()
#     cv2.destroyAllWindows()


# def video_player(file, path='./', ID='Test'):
def video_player(video, user):
    cam_cap = cv2.VideoCapture(0)
    cam_width = int(cam_cap.get(3))
    cam_height = int(cam_cap.get(4))
    # video_cap = cv2.VideoCapture(path+file)
    path = "c:/kiwi/GitHub/rec_ad/Data/AD/"
    video_open(video, path)
    video_time = video_open(video, path)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    save_out = cv2.VideoWriter(('c:/kiwi/Github/rec_ad/Data/Item' + str(video) + '/User' +
                                str(user) + '/vidCap_i' + str(video) + '_u' + str(user) + '.avi'),
                               fourcc, 30.0, (cam_width, cam_height))

    # idx = 0

    now = time.time()

    while True:
        _, frame = cam_cap.read()

        #################################
        ### Capture Picture Save Mode ###
        # time.sleep(0.5)
        # cv2.imwrite('c:/kiwi/Github/rec_ad/Data/Item' + str(video) + '/User' +
        #             str(user) + '/cap_' + str(idx) + '.png', frame)
        # idx += 1
        # if idx == 30:
        #     break
        #################################

        ###############################
        ### Capture Video Save Mode ###
        save_out.write(frame)
        diff = time.time() - now
        # if diff > 16:
        #     break
        if diff > video_time:
            break
        ###############################

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #
        # _, frame = cam_cap.read()
        # ret1, _ = video_cap.read()
        #
        # if ret1:
        #     if idx % 10 == 0:
        #         cv2.imwrite('./Data/Item' + str(file[5]) + '/User1/u1_i1_' + str(int(idx/10.0)) + '.png', frame)
        #     idx += 1
        #
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        #
        # else:
        #     video_close()
        #     break

    video_close()
    # video_cap.release()
    cam_cap.release()
    cv2.destroyAllWindows()


class Show_Pic(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.lb1 = QtWidgets.QLabel()
        self.lb2 = QtWidgets.QLabel()
        self.lb3 = QtWidgets.QLabel()

        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight)
        self.setLayout(layout)

        pixmap1 = QtGui.QPixmap('C:/kiwi/GitHub/rec_ad/etc/AD1.png')
        pixmap2 = QtGui.QPixmap('C:/kiwi/GitHub/rec_ad/etc/AD2.png')
        pixmap3 = QtGui.QPixmap('C:/kiwi/GitHub/rec_ad/etc/AD3.png')

        pixmap1 = pixmap1.scaledToHeight(60)
        pixmap2 = pixmap2.scaledToHeight(60)
        pixmap3 = pixmap3.scaledToHeight(60)

        self.lb1.setPixmap(pixmap1)
        self.lb2.setPixmap(pixmap2)
        self.lb3.setPixmap(pixmap3)

        layout.addWidget(self.lb1)
        layout.addWidget(self.lb2)
        layout.addWidget(self.lb3)


class App(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        # Application Window
        self.setWindowTitle("Kiwi Recommendation System")
        # self.setGeometry(300, 300, 800, 400)
        self.setWindowIcon((QtGui.QIcon('c:/kiwi/github/rec_ad/etc/kiwi.png')))
        self.setFixedSize(800, 450)
        self.center()
        # self.setStyleSheet("background-color: #808081; border: 1.5px solid #444444;")
        self.statusBar().showMessage('Ready')

        # TODO(Kiwi) - insert picture into ui
        # # Video Picture
        # wg = Show_Pic()
        # wg.move(20, 20)
        # wg.resize(100, 100)
        # self.setCentralWidget(wg)

        # pic_path = 'C:/kiwi/GitHub/rec_ad/etc/'
        # pic1 = QtGui.QPixmap(pic_path + 'AD1.png')
        # pic2 = QtGui.QPixmap(pic_path + 'AD2.png')
        # pic3 = QtGui.QPixmap(pic_path + 'AD3.png')
        # pic4 = QtGui.QPixmap(pic_path + 'AD4.png')
        # pic5 = QtGui.QPixmap(pic_path + 'AD5.png')
        # pic6 = QtGui.QPixmap(pic_path + 'AD6.png')
        # pic7 = QtGui.QPixmap(pic_path + 'AD7.png')
        # pic8 = QtGui.QPixmap(pic_path + 'AD8.png')
        # pic9 = QtGui.QPixmap(pic_path + 'AD9.png')
        #
        # pic1_label = QtWidgets.QLabel()
        # pic2_label = QtWidgets.QLabel()
        # pic3_label = QtWidgets.QLabel()
        # pic4_label = QtWidgets.QLabel()
        # pic5_label = QtWidgets.QLabel()
        # pic6_label = QtWidgets.QLabel()
        # pic7_label = QtWidgets.QLabel()
        # pic8_label = QtWidgets.QLabel()
        # pic9_label = QtWidgets.QLabel()
        # # pic1_label.resize(30, 30)
        #
        # pic1_label.setPixmap(pic1)
        # pic2_label.setPixmap(pic2)
        # pic3_label.setPixmap(pic3)
        # pic4_label.setPixmap(pic4)
        # pic5_label.setPixmap(pic5)
        # pic6_label.setPixmap(pic6)
        # pic7_label.setPixmap(pic7)
        # pic8_label.setPixmap(pic8)
        # pic9_label.setPixmap(pic9)

        # User Information
        self.u_num = -1
        self.i_num = -1

        self.num_line = QtWidgets.QLineEdit(self)
        self.name_line = QtWidgets.QLineEdit(self)
        self.gender_line = QtWidgets.QLineEdit(self)
        # self.age_line = QtWidgets.QLineEdit(self)
        self.age_line = QtWidgets.QSpinBox(self)

        self.num_line.setAlignment(QtCore.Qt.AlignCenter)
        self.name_line.setAlignment(QtCore.Qt.AlignCenter)
        self.gender_line.setAlignment(QtCore.Qt.AlignCenter)
        self.age_line.setAlignment(QtCore.Qt.AlignCenter)

        self.num_line.setFont(QtGui.QFont("나눔바른고딕", 10))
        self.name_line.setFont(QtGui.QFont("나눔바른고딕", 10))
        self.gender_line.setFont(QtGui.QFont("나눔바른고딕", 10))
        self.age_line.setFont(QtGui.QFont("나눔바른고딕", 10))
        #
        self.num_line.move(670, 70)
        self.name_line.move(670, 120)
        self.gender_line.move(670, 170)
        self.age_line.move(670, 220)

        self.num_label = QtWidgets.QLabel("No.", self)
        self.name_label = QtWidgets.QLabel("이름 : ", self)
        self.gender_label = QtWidgets.QLabel("성별 : ", self)
        self.age_label = QtWidgets.QLabel("나이 : ", self)

        self.num_label.resize(50, 30)
        self.name_label.resize(50, 30)
        self.gender_label.resize(50, 30)
        self.age_label.resize(50, 30)

        # self.num_label.setAlignment(QtCore.Qt.AlignCenter)

        self.num_label.setFont(QtGui.QFont("나눔스퀘어라운드 Bold", 10))
        self.name_label.setFont(QtGui.QFont("나눔스퀘어라운드 Bold", 10))
        self.gender_label.setFont(QtGui.QFont("나눔스퀘어라운드 Bold", 10))
        self.age_label.setFont(QtGui.QFont("나눔스퀘어라운드 Bold", 10))

        self.num_label.move(630, 70)
        self.name_label.move(630, 120)
        self.gender_label.move(630, 170)
        self.age_label.move(630, 220)

        # #
        # # form_lbx = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight, parent=self)
        # # self.setLayout(form_lbx)
        #
        # gbox = QtWidgets.QGroupBox(self)
        # gbox.setTitle("User Information")
        # gbox.setAlignment(QtCore.Qt.AlignCenter)
        # # form_lbx.addWidget(gbox)
        # lbx = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom, parent=self)
        # lbx2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom, parent=self)
        # gbox.setLayout(lbx)
        # gbox.setLayout(lbx2)
        # gbox.move(400, 40)
        # gbox.resize(400, 300)
        # lbx.addWidget(self.num_line)
        # lbx.addWidget(self.name_line)
        # lbx.addWidget(self.gender_line)
        # lbx.addWidget(self.age_line)
        # lbx2.addWidget(self.num_label)
        # lbx2.addWidget(self.name_label)
        # lbx2.addWidget(self.gender_label)
        # lbx2.addWidget(self.age_label)
        # gbox.setLayout(lbx)
        # gbox.setLayout(lbx2)
        # self.setLayout(gbox)

        self.save_btn = QtWidgets.QPushButton("Save", self)
        self.save_btn.move(670, 280)
        self.save_btn.clicked.connect(self.save_clicked)

        #
        # # self.qle = QtWidgets.QLineEdit(self)
        # # self.lbl = QtWidgets.QLabel(self)
        # # self.qle.move(20, 20)
        # # self.lbl.move(150, 150)
        # vbox = QtWidgets.QVBoxLayout()
        # # vbox.addWidget(self.qle)
        # # # vbox.addWidget(btn2)
        # # vbox.addWidget(self.lbl)
        #
        # vbox.addWidget(self.name_line)
        # vbox.addWidget(self.gender_line)
        # vbox.addWidget(self.age_line)
        # vbox.addWidget(self.name_label)
        # vbox.addWidget(self.gender_label)
        # vbox.addWidget(self.age_label)
        # self.setLayout(vbox)

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

        self.btn1.move(20, 60)
        self.btn2.move(140, 60)
        self.btn3.move(260, 60)
        self.btn4.move(380, 60)
        self.btn5.move(500, 60)
        self.btn6.move(20, 170)
        self.btn7.move(140, 170)
        self.btn8.move(260, 170)
        self.btn9.move(380, 170)
        self.btn10.move(500, 170)
        self.btn11.move(20, 280)
        self.btn12.move(140, 280)
        self.btn13.move(260, 280)
        self.btn14.move(380, 280)
        self.btn15.move(500, 280)
        self.btn16.move(20, 390)
        self.btn17.move(140, 390)
        self.btn18.move(260, 390)
        self.btn19.move(380, 390)
        self.btn20.move(500, 390)

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
        # print(self.lbl.text())
        self.lbl.adjustSize()

    def video_clicked(self, btn, num):
        btn.clicked.connect(lambda: self.change_video(num))
        btn.clicked.connect(lambda: video_player(video=self.i_num, user=self.u_num))

    def change_video(self, num):
        self.i_num = int(num)

    def save_clicked(self):
        self.u_num = int(self.num_line.text())
        # self.i_num = self.

        # saving user information to json
        file_data = OrderedDict()
        file_data["No."] = self.u_num
        file_data["Name"] = self.name_line.text()
        file_data["Gender"] = self.gender_line.text()
        file_data["Age"] = self.age_line.text()
        with open("./userInfo/User" + str(self.u_num) + ".json", "w", encoding="utf-8") as make_file:
            json.dump(file_data, make_file, ensure_ascii=False, indent='\t')

        self.statusBar().showMessage('User' + str(self.u_num))


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    ex.show()
    app.exec_()

    # TODO(Kiwi) - Play Video in UI
    # file_num = input('The number of Video : ')
    # filename = 'video' + str(int(file_num)) +'.mp4'
    # # filename = 'video2.mp4'
    # user_num = input('User : ')
    # id = 'Kiwi'
    #
    # video_player(file=filename, path='C:/kiwi/ui_basic/', ID=id)

    # p1 = Process(target=video_sound())
    # p2 = Process(target=video_play())
    #
    # p1.start()
    # p2.start()
    #
    # p1.join()
    # p2.join()

    #
    # ray.init()
    #
    # ret_id1 = video_sound().remote()
    # ret_id2 = video_play().remote()
    #
    # ret1, ret2 = ray.get([ret_id1, ret_id2])
