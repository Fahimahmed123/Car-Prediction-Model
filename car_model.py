from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

class Ui_CarModelPrediction(object):
    def setupUi(self, CarModelPrediction):
        CarModelPrediction.setObjectName("CarModelPrediction")
        CarModelPrediction.resize(1129, 822)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        CarModelPrediction.setFont(font)

        self.centralwidget = QtWidgets.QWidget(CarModelPrediction)
        self.centralwidget.setStyleSheet("background-color: lightblue;")
        self.centralwidget.setAutoFillBackground(True)
        self.centralwidget.setObjectName("centralwidget")

        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(255, 20, 531, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.textBrowser.setFont(font)
        self.textBrowser.setAutoFillBackground(False)
        self.textBrowser.setStyleSheet("color: black; border: none;")
        self.textBrowser.setObjectName("textBrowser")

        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(250, 120, 541, 51))
        self.textBrowser_2.setStyleSheet("color: black; border: none;")
        self.textBrowser_2.setObjectName("textBrowser_2")

        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(250, 210, 381, 27))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit.setFont(font)
        self.lineEdit.setStyleSheet("background-color: white; color: black;")
        self.lineEdit.setObjectName("lineEdit")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(690, 210, 87, 27))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.openFileDialog)

        self.textBrowser_3 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_3.setGeometry(QtCore.QRect(210, 600, 531, 51))
        self.textBrowser_3.setStyleSheet("color: black; border: none;")
        self.textBrowser_3.setObjectName("textBrowser_3")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(190, 280, 641, 281))
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setIndent(-4)
        self.label.setStyleSheet("background-color: white; color: black;")
        self.label.setObjectName("label")

        # self.label_2 = QtWidgets.QLabel(self.centralwidget)
        # self.label_2.setGeometry(QtCore.QRect(210, 660, 501, 51))
        # self.label_2.setText("Hello")
        # self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        # self.label_2.setStyleSheet("background-color: white; color: black;")
        # self.label_2.setObjectName("label_2")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(420, 720, 101, 27))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.runPrediction)  # Connect the button to runPrediction method

        CarModelPrediction.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(CarModelPrediction)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1129, 35))
        self.menubar.setObjectName("menubar")
        self.menuTEAM_FMA = QtWidgets.QMenu(self.menubar)
        self.menuTEAM_FMA.setObjectName("menuTEAM_FMA")
        CarModelPrediction.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(CarModelPrediction)
        self.statusbar.setObjectName("statusbar")
        CarModelPrediction.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuTEAM_FMA.menuAction())

        self.retranslateUi(CarModelPrediction)
        QtCore.QMetaObject.connectSlotsByName(CarModelPrediction)

    def retranslateUi(self, CarModelPrediction):
        _translate = QtCore.QCoreApplication.translate
        CarModelPrediction.setWindowTitle(_translate("CarModelPrediction", "MainWindow"))
        self.textBrowser.setHtml(_translate("CarModelPrediction", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
    "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
    "p, li { white-space: pre-wrap; }\n"
    "</style></head><body style=\" font-family:\'Cantarell\'; font-size:18pt; font-weight:600; font-style:normal;\">\n"
    "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Car Model Prediction</p></body></html>"))
        self.textBrowser_2.setHtml(_translate("CarModelPrediction", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
    "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
    "p, li { white-space: pre-wrap; }\n"
    "</style></head><body style=\" font-family:\'Cantarell\'; font-size:18pt; font-weight:600; font-style:normal;\">\n"
    "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:400;\">Please choose a file to check the car model</span></p></body></html>"))
        self.pushButton.setText(_translate("CarModelPrediction", "Browse"))
        self.textBrowser_3.setHtml(_translate("CarModelPrediction", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
    "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
    "p, li { white-space: pre-wrap; }\n"
    "</style></head><body style=\" font-family:\'Cantarell\'; font-size:18pt; font-weight:600; font-style:normal;\">\n"
    "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:400;\">  </span></p></body></html>"))
        self.pushButton_2.setText(_translate("CarModelPrediction", "Run"))
        self.menuTEAM_FMA.setTitle(_translate("CarModelPrediction", "TEAM FMA"))

    def openFileDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(None, "Select Image File", "/home/fahim/Music/test/", 
                                                  "Images (*.png *.jpeg *.jpg *.bmp *.gif)", options=options)
        if fileName:
            self.lineEdit.setText(fileName)
            pixmap = QtGui.QPixmap(fileName)
            self.label.setPixmap(pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio))

    def load_and_preprocess_image(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        return img_array / 255.0

    def preprocess_and_predict(self, image_path):
        model_path = "./car_model_classifier_vgg16_another.h5"  # Adjust path to your model
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            self.textBrowser_3.setPlainText("Model not found.")
            return None

        img = self.load_and_preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_label_index = np.argmax(prediction)
        return predicted_label_index

    def runPrediction(self):
        image_path = self.lineEdit.text()
        if image_path:
            predicted_label_index = self.preprocess_and_predict(image_path)
            if predicted_label_index is not None:
                class_mapping = {
                    0: 'Audi',
                    1: 'Rolls Royce',
                    2: 'Toyota_celica',
                    3: 'Toyota_hilux',
                    4: 'Toyota_supra',
                    5: 'BMW',
                    6: 'Swift',
                    7: 'Toyota_corolla',
                    8: 'Toyota Innova',
                    9: 'Toyota_yaris',
                    10: 'Hyundai Creta',
                    11: 'Tata Safari',
                    12: 'Toyota_corona',
                    13: 'Toyota_prius',
                    14: 'Lamborghini',
                    15: 'Toyota_alphard',
                    16: 'Toyota_crown',
                    17: 'Toyota_rav4',
                    18: 'Mahindra Scorpio',
                    19: 'Toyota_avanza',
                    20: 'Toyota_fortuner',
                    21: 'Toyota_rush',
                    22: 'Mercedes',
                    23: 'Toyota_camry',
                    24: 'Toyota_hiace'
                }

                predicted_car_model = class_mapping[predicted_label_index]
                self.textBrowser_3.setPlainText(f"Predicted Car Model: {predicted_car_model}")
            else:
                self.textBrowser_3.setPlainText("Prediction failed. Please check your image path.")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    CarModelPrediction = QtWidgets.QMainWindow()
    ui = Ui_CarModelPrediction()
    ui.setupUi(CarModelPrediction)
    CarModelPrediction.setWindowTitle("Car Model Prediction")
    CarModelPrediction.show()
    sys.exit(app.exec_())
