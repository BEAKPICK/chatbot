# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from mytensorflow.attention_seq2seq_tf import att_seq2seq_tf as astf

import pandas as pd
from pandas import DataFrame
import os

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.only_once = 0
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.send_Btn = QtWidgets.QPushButton(self.centralwidget)
        self.send_Btn.setGeometry(QtCore.QRect(460, 520, 93, 28))
        self.send_Btn.setObjectName("send_Btn")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(10, 446, 541, 71))
        self.textEdit.setObjectName("textEdit")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(10, 10, 541, 431))
        self.listWidget.setObjectName("listWidget")

        self.listWidget.addItem('환영합니다. 로드하는데 시간이 소요되오니 기다려주세요.')

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.save_Btn = QtWidgets.QPushButton(self.centralwidget)
        self.save_Btn.setGeometry(QtCore.QRect(690, 420, 93, 28))
        self.save_Btn.setObjectName("save_Btn")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(570, 240, 81, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(570, 330, 111, 16))
        self.label_2.setObjectName("label_2")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(570, 260, 211, 61))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(570, 350, 211, 61))
        self.textEdit_3.setObjectName("textEdit_3")
        MainWindow.setCentralWidget(self.centralwidget)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.send_Btn.setText(_translate("MainWindow", "보내기"))
        self.send_Btn.setShortcut(_translate("MainWindow", "Ctrl+Return"))
        self.save_Btn.setText(_translate("MainWindow", "데이터 저장"))
        self.label.setText(_translate("MainWindow", "상대의 대화"))
        self.label_2.setText(_translate("MainWindow", "나의 추천 대답"))

    def select_conversation(self):
        str1 = 'Q : '
        str2 = 'M : '
        tstr = self.listWidget.currentItem().text()
        if str1 in tstr or str2 in tstr:
            tstr = tstr[4:]
        self.textEdit_2.setPlainText(tstr)

    def save_conversation(self):
        if self.textEdit_3.toPlainText() == '' or self.textEdit_2.toPlainText() == '':
            return

        data = {}
        data['Q'] = [self.textEdit_2.toPlainText()]
        data['A'] = [self.textEdit_3.toPlainText()]
        data_df = DataFrame.from_dict(data)
        #save csv file from data_df
        if os.path.isfile('../dataset/myChatbotData.csv'):
            data_df.to_csv('../dataset/myChatbotData.csv', mode='a', header=False, encoding='utf-8')
        else:
            data_df.to_csv('../dataset/myChatbotData.csv', encoding='utf-8')

        self.textEdit_2.clear()
        self.textEdit_3.clear()

    def send(self):
        if self.only_once != 0:
            return
        self.only_once = 1
        self.listWidget.addItem('Q : '+self.textEdit.toPlainText())
        self.att.ask_question(self.textEdit.toPlainText(), log=self.listWidget)
        self.textEdit.clear()
        self.only_once = 0

    def runChat(self, epoch=0):
        self.att = astf()
        self.att.learn(epoch=epoch, log=self.listWidget, use_loaded=True)
        self.send_Btn.clicked.connect(self.send)
        self.save_Btn.clicked.connect(self.save_conversation)
        self.listWidget.itemClicked.connect(self.select_conversation)
        self.listWidget.addItem('대화를 입력해주세요.')

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.runChat(epoch=0)
    sys.exit(app.exec_())

