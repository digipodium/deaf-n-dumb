import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
import time
import os
from sign_predictor import detect_sign

def get_images_for_text(text):
    images = []
    for letter in text:
        if os.path.exists("symbols/" + letter + ".png"):
            images.append("symbols/" + letter + ".png")
        else:
            images.append("symbols/None.png")
    return images


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("app_layout.ui", self)
        self.cw = self.centralWidget()
        self.labelStatus = self.cw.findChild(QtWidgets.QLabel, "labelStatus")

        self.btnCamera = self.cw.findChild(QtWidgets.QPushButton, "btnCamera")
        self.btnClear = self.cw.findChild(QtWidgets.QPushButton, "btnClear")
        self.btnQuit = self.cw.findChild(QtWidgets.QPushButton, "btnQuit")
        self.btnText = self.cw.findChild(QtWidgets.QPushButton, "btnText")
        self.editMsg = self.cw.findChild(QtWidgets.QTextEdit, "editMsg")
        self.imageContainer = self.cw.findChild(QtWidgets.QHBoxLayout, "cont")
        self.btnCamera.clicked.connect(self.start_camera)
        self.btnClear.clicked.connect(self.clear_text)
        self.btnQuit.clicked.connect(self.quit_app)
        self.btnText.clicked.connect(self.text_sign)
        print("Loaded UI")
        print(self.imageContainer)


    def start_camera(self):
        self.labelStatus.setText("Camera started")
        detect_sign()

    def clear_text(self):
        self.labelStatus.setText("Ready")
        self.editMsg.setText("")

    def quit_app(self):
        QtWidgets.qApp.quit()
    
    def text_sign(self):
        message = self.editMsg.toPlainText()
        # remove all images from container
        for i in reversed(range(self.imageContainer.layout().count())):
            self.imageContainer.itemAt(i).widget().setParent(None)
        
        self.labelStatus.setText("Fetching sign for " + message)
        images = get_images_for_text(message)
        for image in images:
            # add a label for image in container
            label = QtWidgets.QLabel()
            label.setPixmap(QtGui.QPixmap(image))
            self.imageContainer.layout().addWidget(label)

print("Starting app")
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()