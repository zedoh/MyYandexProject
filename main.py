import math
import sqlite3

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, \
    QMessageBox, QLineEdit, QFormLayout, QDialog, QFrame, QTableWidgetItem, QTableWidget
from PyQt5.QtGui import QPixmap, QWindow
from PyQt5.QtCore import QDateTime


class line_finder():
    def __init__(self, img):
        self.image = None
        self.gray_image = None
        self.lines = None
        self.edges = None
        self.image_path = img
        self.linseses = []

    def graing(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError("Image not found or unable to load.")
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def finding(self):
        self.edges = cv2.Canny(self.gray_image, 50, 150, apertureSize=3)
        self.lines = cv2.HoughLines(self.edges, 1, np.pi / 180, threshold=100)

    def lining(self):
        if self.lines is not None:
            for rho, theta in self.lines[:, 0]:  # lines is a 3D array, we need to loop through it
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)
                self.linseses.append([[x1, y1], [x2, y2]])
                cv2.line(self.image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    def painting(self):
        return self.linseses

    def checking(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Hide axis
        plt.title('Detected Circles')
        plt.show()


class circle_finder():
    def __init__(self, img):
        self.image = None
        self.gray_image = None
        self.blurred_image = None
        self.circles = None
        self.image_path = img
        self.circleses = []

    def graing1(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError("Image not found or unable to load.")
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def bluring(self, dpi, mindist, param1, param2, minrad, maxrad):
        self.blurred_image = cv2.medianBlur(self.gray_image, 5)
        self.circles = cv2.HoughCircles(self.blurred_image,
                                        cv2.HOUGH_GRADIENT,
                                        dp=dpi,
                                        minDist=mindist,
                                        param1=param1,
                                        param2=param2,
                                        minRadius=minrad,
                                        maxRadius=maxrad)

    def non_maximum_suppression(self, circles, threshold=10):
        if circles is None:
            return []

            # Convert circles to a list of (x, y, radius)
        circles = np.int64(np.around(circles))
        picked = []

        # Sort circles by their radius in descending order
        circles = sorted(circles[0], key=lambda x: x[2], reverse=True)

        while circles:
            # Pick the largest circle and add it to the list
            current_circle = circles.pop(0)
            picked.append(current_circle)
            # Remove circles that are too close to the current circle
            circles = [c for c in circles if
                       np.sqrt((c[0] - current_circle[0]) ** 2 + (c[1] - current_circle[1]) ** 2) > (
                               current_circle[2] + c[2] + threshold)]

        return picked

    def circling(self):
        if self.circles is not None:
            # Apply non-maximum suppression
            self.circles = self.non_maximum_suppression(self.circles)

            for i in self.circles:  # `circles` is now a list of circles
                cv2.circle(self.image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Green outer circle
                cv2.circle(self.image, (i[0], i[1]), 2, (0, 0, 255), 3)  # Draw the center of the circle
                self.circleses.append([[int(i[0]), int(i[1])], int(i[2])])

    def painting1(self):
        return self.circleses

    def checking1(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Hide axis
        plt.title('Detected Circles')
        plt.show()


class blocks(circle_finder, line_finder):
    def __init__(self, img):
        super().__init__(img)
        self.sp4 = None
        self.sp3 = None
        self.spisoksort3 = None
        self.spisoksort2 = None
        self.spisoksort1 = None
        self.ci = None
        self.li = None
        self.nearest_pair = None
        self.min_difference = 0
        self.difference = None
        self.spisoksort = None
        self.sp2 = None
        self.sp1 = None
        self.b = None
        self.circle = None
        self.a = None
        self.line = None

    def params(self, img, params):
        self.line = line_finder(img)
        self.line.graing()
        self.line.finding()
        self.line.lining()
        self.li = self.line.painting()
        self.circle = circle_finder(img)
        self.circle.graing1()
        if not params:
            params = [1, 45, 75, 40, 0, 0]
        self.circle.bluring(params[0], params[1], params[2], params[3], params[4], params[5])
        self.circle.circling()
        self.ci = self.circle.painting1()
        self.sp1 = []
        self.sp2 = []
        self.sp3 = []
        self.sp4 = []
        self.spisoksort = []
        self.spisoksort1 = []
        self.spisoksort2 = []
        self.spisoksort3 = []

    def correcting_lines(self):
        global numerator
        for i in range(len(self.li)):
            if not self.sp1:
                self.sp1.append(self.li[i])
            else:
                if (self.li[i][0][1] - self.li[i][1][1]) >= 0.1:
                    if (self.li[i][0][0] - self.li[i][1][0]) / (self.li[i][0][1] - self.li[i][1][1]) - (
                            self.sp1[0][0][0] - self.sp1[0][1][0]) / (
                            self.sp1[0][0][1] - self.sp1[0][1][1]) < 0.1:
                        self.sp1.append(self.li[i])
                else:
                    if not self.sp2:
                        self.sp2.append(self.li[i])
                    else:
                        if (self.li[i][0][1] - self.li[i][1][1]) / (self.li[i][0][0] - self.li[i][1][0]) - (
                                self.sp2[0][0][1] - self.sp2[0][1][1]) / (
                                self.sp2[0][0][0] - self.sp2[0][1][0]) < 0.1:
                            self.sp2.append(self.li[i])
                            # to this moment we sorted horizontal and vertical +- parallels
        for i in range(len(self.sp1)):
            if self.sp1[i][0][0] - self.sp1[i][1][0] != 0:
                self.spisoksort.append(
                    [(self.sp1[i][0][1] - self.sp1[i][1][1]) / (self.sp1[i][0][0] - self.sp1[i][1][0]), i])
            else:
                self.spisoksort.append([
                    (self.li[i][0][0] - self.li[i][1][0]) / (self.li[i][0][1] - self.li[i][1][1]), i])
        for i in range(len(self.spisoksort)):
            self.spisoksort1.append(self.spisoksort[i][0])
        for i in range(len(self.spisoksort1) - 1):
            self.difference = self.spisoksort1[i + 1] - self.spisoksort1[i]
            if self.difference < self.min_difference:
                self.min_difference = self.difference
                self.nearest_pair = (self.spisoksort1[i], self.spisoksort1[i + 1])
        for i in range(len(self.spisoksort)):
            if self.spisoksort1[0] == self.spisoksort[i][0] or self.spisoksort1[1] == self.spisoksort[i][0]:
                self.spisoksort2.append(self.spisoksort[i][1])
        for i in range(len(self.sp1)):
            if i == self.spisoksort2[0] or i == self.spisoksort2[1]:
                self.spisoksort3.append(self.sp1[i])
        self.sp1.clear()
        for i in range(len(self.spisoksort3)):
            self.sp1.append(self.spisoksort3[i])
        self.spisoksort.clear()
        self.spisoksort1.clear()
        self.spisoksort2.clear()
        self.spisoksort3.clear()
        for i in range(len(self.sp2)):
            if self.sp2[i][0][0] - self.sp2[i][1][0] != 0:
                self.spisoksort.append(
                    [(self.sp2[i][0][1] - self.sp2[i][1][1]) / (self.sp2[i][0][0] - self.sp2[i][1][0]), i])
            else:
                self.spisoksort.append([
                    (self.li[i][0][0] - self.li[i][1][0]) / (self.li[i][0][1] - self.li[i][1][1]), i])
        for i in range(len(self.spisoksort)):
            self.spisoksort1.append(self.spisoksort[i][0])
        for i in range(len(self.spisoksort1) - 1):
            self.difference = self.spisoksort1[i + 1] - self.spisoksort1[i]
            if self.difference < self.min_difference:
                self.min_difference = self.difference
                self.nearest_pair = (self.spisoksort1[i], self.spisoksort1[i + 1])
        for i in range(len(self.spisoksort)):
            if self.spisoksort1[0] == self.spisoksort[i][0] or self.spisoksort1[1] == self.spisoksort[i][0]:
                self.spisoksort2.append(self.spisoksort[i][1])
        for i in range(len(self.sp2)):
            if i == self.spisoksort2[0] or i == self.spisoksort2[1]:
                self.spisoksort3.append(self.sp2[i])
        self.sp2 = self.spisoksort3
        x1 = self.sp1[0][0][0]
        x2 = self.sp1[0][1][0]
        y1 = self.sp1[0][0][1]
        y2 = self.sp1[0][1][1]
        x3 = self.sp2[0][0][0]
        x4 = self.sp2[0][1][0]
        y3 = self.sp2[0][0][1]
        y4 = self.sp2[0][1][1]
        x11 = self.sp1[1][0][0]
        x22 = self.sp1[1][1][0]
        y11 = self.sp1[1][0][1]
        y22 = self.sp1[1][1][1]
        x33 = self.sp2[1][0][0]
        x44 = self.sp2[1][1][0]
        y33 = self.sp2[1][0][1]
        y44 = self.sp2[1][1][1]
        denom1 = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        denom2 = (x1 - x2) * (y33 - y44) - (y1 - y2) * (x33 - x44)
        denom22 = (x11 - x22) * (y3 - y4) - (y11 - y22) * (x3 - x4)
        denom11 = (x11 - x22) * (y33 - y44) - (y11 - y22) * (x33 - x44)
        int_x1 = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom1
        int_y1 = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom1
        int_x2 = ((x1 * y2 - y1 * x2) * (x33 - x44) - (x1 - x2) * (x33 * y44 - y33 * x44)) / denom2
        int_y2 = ((x1 * y2 - y1 * x2) * (y33 - y44) - (y1 - y2) * (x33 * y44 - y33 * x44)) / denom2
        int_x3 = ((x11 * y22 - y11 * x22) * (x33 - x44) - (x11 - x22) * (x33 * y44 - y33 * x44)) / denom11
        int_y3 = ((x11 * y22 - y11 * x22) * (y33 - y44) - (y11 - y22) * (x33 * y44 - y33 * x44)) / denom11
        int_x4 = ((x11 * y22 - y11 * x22) * (x3 - x4) - (x11 - x22) * (x3 * y4 - y3 * x4)) / denom22
        int_y4 = ((x11 * y22 - y11 * x22) * (y3 - y4) - (y11 - y22) * (x3 * y4 - y3 * x4)) / denom22
        line1 = math.sqrt(abs((int_x1 - int_x2) ** 2 - (int_y1 - int_y2) ** 2))
        line2 = math.sqrt(abs((int_x2 - int_x3) ** 2 - (int_y2 - int_y3) ** 2))
        line3 = math.sqrt(abs((int_x3 - int_x4) ** 2 - (int_y3 - int_y4) ** 2))
        line4 = math.sqrt(abs((int_x4 - int_x1) ** 2 - (int_y4 - int_y1) ** 2))
        numerator = 1
        if line2 != 0:
            if (line1 + line2 // 2) // line2 < 15:
                numerator = (line1 + line2 // 2) // line2
        elif line4 != 0:
            if (line1 + line4 // 2) // line4 < 15:
                numerator = (line1 + line4 // 2) // line4

        elif line2 != 0:
            if (line3 + line2 // 2) // line2 < 15:
                numerator = (line3 + line2 // 2) // line2

        elif line4 != 0:
            if (line3 + line4 // 2) // line4 < 15:
                numerator = (line3 + line4 // 2) // line4
        else:
            return "problem"
        return ((len(self.ci) + numerator // 2) // numerator), \
            ((len(self.ci) + numerator // 2) // numerator) // numerator


class MedianColor:
    def __init__(self, img, params):
        self.params = params
        self.median_color = None
        self.img = img


    def calculate_median_color(self):
        with Image.open(self.img) as img:
            img = img.convert('RGB')
            pixels = list(img.getdata())
            pixel_array = np.array(pixels)
            self.median_color = np.median(pixel_array, axis=0)
            self.median_color = tuple(map(int, self.median_color))
            return self.median_color

class PhotoViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.circle = None
        self.saveButton = None
        self.dateTimeLabel = None
        self.button = None
        self.label = None
        self.layout = None
        self.settingsButton = None
        self.List_With = []
        self.initUI()
        self.counter = 0

    def initUI(self):
        self.counter = 0
        self.setWindowTitle('Photo Viewer')
        self.setGeometry(100, 100, 600, 400)

        self.layout = QVBoxLayout()

        self.label = QLabel('Select a photo to view and get the date and time.')
        self.layout.addWidget(self.label)

        self.button = QPushButton('Open Photo')
        self.button.clicked.connect(self.openFile)
        self.layout.addWidget(self.button)

        self.settingsButton = QPushButton('Settings')
        self.settingsButton.clicked.connect(self.openSettings)
        self.layout.addWidget(self.settingsButton)

        self.saveButton = QPushButton('check your correction', self)
        self.saveButton.clicked.connect(self.parameters)
        self.layout.addWidget(self.saveButton)

        self.saveButton = QPushButton('about program', self)
        self.saveButton.clicked.connect(self.info)
        self.layout.addWidget(self.saveButton)

        self.saveButton = QPushButton('check your data_base', self)
        self.saveButton.clicked.connect(self.openCSV)
        self.layout.addWidget(self.saveButton)

        self.dateTimeLabel = QLabel('')
        self.layout.addWidget(self.dateTimeLabel)

        self.setLayout(self.layout)

    def info(self):
        self.second_window = InfoWindow()
        self.second_window.show()


    def openCSV(self):
        self.table_window = DBSample()
        self.table_window.show()


    def openSettings(self):
        settings_dialog = SettingsDialog(self)
        if settings_dialog.exec_() == QDialog.Accepted:
            self.List_With = settings_dialog.getParameters()


    def parameters(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Photo File", "",
                                                  "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)",
                                                  options=options)
        self.circle = circle_finder(fileName)
        self.circle.graing1()
        if not self.List_With:
            self.List_With = [1, 1, 75, 40, 0, 0]
        self.circle.bluring(self.List_With[0], self.List_With[1], self.List_With[2], self.List_With[3],
                            self.List_With[4], self.List_With[5])
        self.circle.circling()
        self.circle.painting1()
        self.circle.checking1()


    def data_base(self):
        try:
            connection = sqlite3.connect('my_database.sqlite')
            cursor = connection.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS my_database(id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, format TEXT,
             median TEXT,color TEXT, params TEXT, img TEXT);''')
            connection.commit()
            cursor.close()
        except sqlite3.Error as error:
            print("Failed to insert blob data into sqlite table", error)
        finally:
            if connection:
                connection.close()

    def updating_table(self, data, format, median, color, params, img):
        try:
            connection = sqlite3.connect('my_database.sqlite')
            cursor = connection.cursor()
            sqlite_insert = """ INSERT INTO my_database(date, format, median, color, params, img)
             VALUES (?, ?, ?, ?, ?, ?)"""
            data_tuple = (data, format, median, color, params, img)
            cursor.execute(sqlite_insert, data_tuple)
            connection.commit()
            cursor.close()
        except sqlite3.Error as error:
            print("Failed to insert blob data into sqlite table", error)
        finally:
            if connection:
                connection.close()


    def openFile(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Photo File", "",
                                                  "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)",
                                                  options=options)

        if fileName:
            self.counter += 1
            block = blocks(fileName)
            block.params(fileName, self.List_With)
            median_color_instance = MedianColor(fileName, self.List_With)
            median_color_value = median_color_instance.calculate_median_color()

            circle = circle_finder(fileName)
            circle.graing1()
            if not self.List_With:
                self.List_With = [1, 1, 75, 40, 0, 0]
            circle.bluring(self.List_With[0], self.List_With[1], self.List_With[2], self.List_With[3],
                                self.List_With[4], self.List_With[5])
            circle.circling()
            circles = circle.painting1()
            need = circles[0]
            x, y = need[0][0] - need[1], need[0][1] - need[1]
            with Image.open(fileName) as img:
                img = img.convert('RGB')
                r, g, b = img.getpixel((x, y))
            color = (r, g, b)

            self.label.setPixmap(QPixmap(fileName).scaled(400, 300))
            current_time = QDateTime.currentDateTime().toString()
            ab = str(block.correcting_lines())
            self.data_base()
            self.updating_table(str(current_time) ,str(ab), str(median_color_value), str(color), str(self.List_With),
                                str(fileName))
            self.dateTimeLabel.setText(
                f'Opened: {fileName}'
                f'\nDate and Time: {current_time}'
                f'\nFormat: {ab}'
                f'\nThe median color of the image is: {median_color_value}'
                f'\n The color of block {color}'
                f'\n params {self.List_With}')


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Settings')
        self.setGeometry(150, 150, 300, 200)

        self.layout = QFormLayout()
        self.inputs = []

        # Create five input fields for integers
        for i in range(6):
            self.str = ["dpi", "mindist", "param1, base = 75", "param2, base = 40", "minrad", "maxrad"]
            line_edit = QLineEdit(self)
            line_edit.setPlaceholderText(f'{self.str[i]}')  # text on the input line
            self.inputs.append(line_edit)
            self.layout.addRow(f'{self.str[i]}:', line_edit)

        self.saveButton = QPushButton('Save', self)
        self.saveButton.clicked.connect(self.saveParameters)
        self.layout.addWidget(self.saveButton)

        self.setLayout(self.layout)

    def saveParameters(self):
        try:
            params = [int(input_field.text()) for input_field in self.inputs]
            QMessageBox.information(self, 'Success', f'Your params: {params}')
            self.accept()

        except ValueError:
            QMessageBox.warning(self, 'Input Error', 'Please enter valid integers for all parameters.')

    def getParameters(self):
        if [int(input_field.text()) for input_field in self.inputs]:
            return [int(input_field.text()) for input_field in self.inputs]
        else:
            return [1, 5, 75, 40, 0, 0]


class DBSample(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Database Table")
        self.setGeometry(150, 150, 1000, 1000)

        self.table_widget = QTableWidget()
        self.load_data()

        layout = QVBoxLayout()
        layout.addWidget(self.table_widget)
        self.setLayout(layout)

    def load_data(self):
        connection = sqlite3.connect("my_database.sqlite")
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM my_database")
        data = cursor.fetchall()

        self.table_widget.setRowCount(len(data))
        self.table_widget.setColumnCount(len(data[0]))

        for row_index, row_data in enumerate(data):
            for column_index, item in enumerate(row_data):
                self.table_widget.setItem(row_index, column_index, QTableWidgetItem(str(item)))

        connection.close()

class InfoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("info")
        self.layout1 = QVBoxLayout()
        self.layout1.addWidget(QLabel("программа вычисляет размер, а также цвет кубика лего по фотографии"))
        self.layout1.addWidget(QLabel("она работает на функциях преобразованиях и пространствах Хафа с помощью которых"))
        self.layout1.addWidget(QLabel("Определяются прямые и кружочки но фотографии"))
        self.layout1.addWidget(QLabel("параметры для определния кружков по фотографии(которые важны)"))
        self.layout1.addWidget(QLabel("изначально заданы для фотографий проверки"))
        self.layout1.addWidget(QLabel("могут не работать с другими"))
        self.layout1.addWidget(QLabel("значение параметров описано в тестовом файле о программе"))
        self.layout1.addWidget(QLabel("первая кнопка - открыть ваш файл, сверху - откроется фотография,снизу вам будут даны данные о"))
        self.layout1.addWidget(QLabel("файле,дате и времени, отношении сторон(формат фигурки), цвета медианного всей картинки"))
        self.layout1.addWidget(QLabel("и цвета кубика, а также о заданных параметрах"))
        self.layout1.addWidget(QLabel("вторая кнопка - кнопка задачи параметров"))
        self.layout1.addWidget(QLabel("третья кнопка - проверка определения кружочков по фотографии"))
        self.layout1.addWidget(QLabel("четвертая кнопка - о программе"))
        self.layout1.addWidget(QLabel("пятая - база данных о прошлых проверках"))
        self.setLayout(self.layout1)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = PhotoViewer()
    viewer.show()
    sys.exit(app.exec_())