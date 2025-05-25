from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont
import sys
import os

# 尝试导入serial模块，如果不存在则设置标志
_has_serial = True
try:
    import serial.tools.list_ports  # 用于获取系统串口列表
except ImportError:
    _has_serial = False

# 参数配置
camera_ids = 2 # 总摄像头数量
camera_id = 0 # 当前摄像头编号
camera_width = 1024 # 视频宽度
camera_height = 576 # 视频高度
conf_thres = 0.5 # 置信度阈值
iou_thres = 0.5 # 交叉比阈值
scrath_detection = True # 划痕检测
dents = True # 凹坑检测

# 参数配置
class DefectDetectionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化摄像头参数
        self.camera_id = 0  # 当前摄像头编号
        self.camera_width = 1024  # 视频宽度
        self.camera_height = 576  # 视频高度
        self.conf_thres = 0.5  # 置信度阈值
        self.iou_thres = 0.5  # 交叉比阈值
        self.scratch_detection = True  # 划痕检测
        self.dents_detection = True  # 凹坑检测

        # 初始化摄像头和媒体播放器
        self.camera = None
        self.capture_session = QMediaCaptureSession()

        self.setupUi()

    def closeEvent(self, event):
        """Handle application close event"""
        # 关闭摄像头
        if self.camera is not None and self.camera.isActive():
            self.camera.stop()
        event.accept()

    def setVideoWidgetBlack(self):
        """Set the video widget background to black"""
        # 创建一个黑色背景的样式表
        self.videoWidget.setStyleSheet("background-color: black;")

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1200, 800)
        self.centralwidget = QtWidgets.QWidget(parent=self)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(parent=self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(20, 30, 1160, 780))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setEnabled(True)
        self.tab_3.setObjectName("tab_3")
        # 使用QVideoWidget显示视频
        self.videoWidget = QVideoWidget(parent=self.tab_3)
        self.videoWidget.setGeometry(QtCore.QRect(30, 60, 640, 360))
        self.videoWidget.setObjectName("videoWidget")

        # 设置视频输出
        self.capture_session.setVideoOutput(self.videoWidget)
        self.label = QtWidgets.QLabel(parent=self.tab_3)
        self.label.setGeometry(QtCore.QRect(30, 30, 111, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setUnderline(False)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(parent=self.tab_3)
        self.comboBox.setGeometry(QtCore.QRect(150, 27, 68, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.checkBox = QtWidgets.QCheckBox(parent=self.tab_3)
        self.checkBox.setGeometry(QtCore.QRect(340, 30, 79, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")
        self.checkBox_2 = QtWidgets.QCheckBox(parent=self.tab_3)
        self.checkBox_2.setGeometry(QtCore.QRect(500, 30, 79, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.checkBox_2.setFont(font)
        self.checkBox_2.setObjectName("checkBox_2")
        self.pushButton_4 = QtWidgets.QPushButton(parent=self.tab_3)
        self.pushButton_4.setGeometry(QtCore.QRect(730, 220, 180, 40))
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_2 = QtWidgets.QLabel(parent=self.tab_3)
        self.label_2.setGeometry(QtCore.QRect(740, 42, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setUnderline(False)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(parent=self.tab_3)
        self.label_3.setGeometry(QtCore.QRect(740, 92, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setUnderline(False)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(parent=self.tab_3)
        self.doubleSpinBox.setGeometry(QtCore.QRect(820, 40, 49, 22))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.doubleSpinBox.setFont(font)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(parent=self.tab_3)
        self.doubleSpinBox_2.setGeometry(QtCore.QRect(820, 90, 49, 22))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.doubleSpinBox_2.setFont(font)
        self.doubleSpinBox_2.setObjectName("doubleSpinBox_2")
        self.pushButton_5 = QtWidgets.QPushButton(parent=self.tab_3)
        self.pushButton_5.setGeometry(QtCore.QRect(940, 220, 180, 40))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(parent=self.tab_3)
        self.pushButton_6.setGeometry(QtCore.QRect(730, 290, 180, 40))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_13 = QtWidgets.QPushButton(parent=self.tab_3)
        self.pushButton_13.setGeometry(QtCore.QRect(940, 290, 180, 40))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_14 = QtWidgets.QPushButton(parent=self.tab_3)
        self.pushButton_14.setGeometry(QtCore.QRect(730, 360, 180, 40))
        self.pushButton_14.setObjectName("pushButton_14")
        self.layoutWidget = QtWidgets.QWidget(parent=self.tab_3)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 500, 641, 140))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(70)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton = QtWidgets.QPushButton(parent=self.layoutWidget)
        self.pushButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icons/car_left.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.pushButton.setIcon(icon)
        self.pushButton.setIconSize(QtCore.QSize(130, 130))
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(parent=self.layoutWidget)
        self.pushButton_2.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("icons/car_right.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.pushButton_2.setIcon(icon1)
        self.pushButton_2.setIconSize(QtCore.QSize(130, 130))
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(parent=self.layoutWidget)
        self.pushButton_3.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("icons/car_roof.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.pushButton_3.setIcon(icon2)
        self.pushButton_3.setIconSize(QtCore.QSize(130, 130))
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.label_4 = QtWidgets.QLabel(parent=self.tab_3)
        self.label_4.setGeometry(QtCore.QRect(940, 20, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setUnderline(False)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.comboBox_2 = QtWidgets.QComboBox(parent=self.tab_3)
        self.comboBox_2.setGeometry(QtCore.QRect(1000, 20, 121, 22))
        self.comboBox_2.setObjectName("comboBox_2")
        self.pushButton_7 = QtWidgets.QPushButton(parent=self.tab_3)
        self.pushButton_7.setGeometry(QtCore.QRect(940, 50, 181, 31))
        self.pushButton_7.setObjectName("pushButton_7")
        self.label_5 = QtWidgets.QLabel(parent=self.tab_3)
        self.label_5.setGeometry(QtCore.QRect(940, 90, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setUnderline(False)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.comboBox_3 = QtWidgets.QComboBox(parent=self.tab_3)
        self.comboBox_3.setGeometry(QtCore.QRect(1000, 90, 121, 22))
        self.comboBox_3.setObjectName("comboBox_3")
        self.label_6 = QtWidgets.QLabel(parent=self.tab_3)
        self.label_6.setGeometry(QtCore.QRect(940, 120, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setUnderline(False)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.comboBox_4 = QtWidgets.QComboBox(parent=self.tab_3)
        self.comboBox_4.setGeometry(QtCore.QRect(1000, 120, 121, 22))
        self.comboBox_4.setObjectName("comboBox_4")
        self.label_7 = QtWidgets.QLabel(parent=self.tab_3)
        self.label_7.setGeometry(QtCore.QRect(940, 150, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setUnderline(False)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.comboBox_5 = QtWidgets.QComboBox(parent=self.tab_3)
        self.comboBox_5.setGeometry(QtCore.QRect(1000, 150, 121, 22))
        self.comboBox_5.setObjectName("comboBox_5")
        self.textBrowser = QtWidgets.QTextBrowser(parent=self.tab_3)
        self.textBrowser.setGeometry(QtCore.QRect(730, 450, 400, 200))
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton_15 = QtWidgets.QPushButton(parent=self.tab_3)
        self.pushButton_15.setGeometry(QtCore.QRect(940, 360, 180, 40))
        self.pushButton_15.setObjectName("pushButton_15")
        self.pushButton_18 = QtWidgets.QPushButton(parent=self.tab_3)
        self.pushButton_18.setGeometry(QtCore.QRect(730, 140, 160, 40))
        self.pushButton_18.setObjectName("pushButton_18")
        self.comboBox_7 = QtWidgets.QComboBox(parent=self.tab_3)
        self.comboBox_7.setGeometry(QtCore.QRect(1000, 180, 121, 22))
        self.comboBox_7.setObjectName("comboBox_7")
        self.label_15 = QtWidgets.QLabel(parent=self.tab_3)
        self.label_15.setGeometry(QtCore.QRect(940, 180, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setUnderline(False)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.pushButton_8 = QtWidgets.QPushButton(parent=self.tab_4)
        self.pushButton_8.setGeometry(QtCore.QRect(770, 320, 141, 101))
        self.pushButton_8.setText("")
        self.pushButton_8.setIcon(icon2)
        self.pushButton_8.setIconSize(QtCore.QSize(130, 130))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(parent=self.tab_4)
        self.pushButton_9.setGeometry(QtCore.QRect(770, 190, 141, 101))
        self.pushButton_9.setText("")
        self.pushButton_9.setIcon(icon1)
        self.pushButton_9.setIconSize(QtCore.QSize(130, 130))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QtWidgets.QPushButton(parent=self.tab_4)
        self.pushButton_10.setGeometry(QtCore.QRect(770, 50, 141, 111))
        self.pushButton_10.setText("")
        self.pushButton_10.setIcon(icon)
        self.pushButton_10.setIconSize(QtCore.QSize(130, 130))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(parent=self.tab_4)
        self.pushButton_11.setGeometry(QtCore.QRect(940, 190, 141, 101))
        self.pushButton_11.setText("")
        self.pushButton_11.setIcon(icon1)
        self.pushButton_11.setIconSize(QtCore.QSize(130, 130))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_12 = QtWidgets.QPushButton(parent=self.tab_4)
        self.pushButton_12.setGeometry(QtCore.QRect(940, 50, 141, 111))
        self.pushButton_12.setText("")
        self.pushButton_12.setIcon(icon)
        self.pushButton_12.setIconSize(QtCore.QSize(130, 130))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_16 = QtWidgets.QPushButton(parent=self.tab_4)
        self.pushButton_16.setGeometry(QtCore.QRect(940, 320, 141, 101))
        self.pushButton_16.setText("")
        self.pushButton_16.setIcon(icon2)
        self.pushButton_16.setIconSize(QtCore.QSize(130, 130))
        self.pushButton_16.setObjectName("pushButton_16")
        self.label_8 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_8.setGeometry(QtCore.QRect(10, 10, 320, 180))
        self.label_8.setAutoFillBackground(True)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_9.setGeometry(QtCore.QRect(370, 10, 320, 180))
        self.label_9.setAutoFillBackground(True)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_10.setGeometry(QtCore.QRect(10, 220, 320, 180))
        self.label_10.setAutoFillBackground(True)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_11.setGeometry(QtCore.QRect(370, 220, 320, 180))
        self.label_11.setAutoFillBackground(True)
        self.label_11.setObjectName("label_11")
        self.textBrowser_2 = QtWidgets.QTextBrowser(parent=self.tab_4)
        self.textBrowser_2.setGeometry(QtCore.QRect(730, 450, 400, 200))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.listView = QtWidgets.QListView(parent=self.tab_4)
        self.listView.setGeometry(QtCore.QRect(20, 460, 681, 192))
        self.listView.setObjectName("listView")
        self.checkBox_5 = QtWidgets.QCheckBox(parent=self.tab_4)
        self.checkBox_5.setGeometry(QtCore.QRect(1000, 20, 79, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.checkBox_5.setFont(font)
        self.checkBox_5.setObjectName("checkBox_5")
        self.checkBox_6 = QtWidgets.QCheckBox(parent=self.tab_4)
        self.checkBox_6.setGeometry(QtCore.QRect(880, 20, 79, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.checkBox_6.setFont(font)
        self.checkBox_6.setObjectName("checkBox_6")
        self.checkBox_7 = QtWidgets.QCheckBox(parent=self.tab_4)
        self.checkBox_7.setGeometry(QtCore.QRect(770, 20, 79, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.checkBox_7.setFont(font)
        self.checkBox_7.setObjectName("checkBox_7")
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.scrollArea = QtWidgets.QScrollArea(parent=self.tab_5)
        self.scrollArea.setGeometry(QtCore.QRect(20, 80, 1120, 620))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1118, 618))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.label_12 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_12.setGeometry(QtCore.QRect(20, 30, 60, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.dateEdit = QtWidgets.QDateEdit(parent=self.tab_5)
        self.dateEdit.setGeometry(QtCore.QRect(210, 30, 110, 22))
        self.dateEdit.setCalendarPopup(True)
        self.dateEdit.setObjectName("dateEdit")
        self.dateEdit_3 = QtWidgets.QDateEdit(parent=self.tab_5)
        self.dateEdit_3.setGeometry(QtCore.QRect(90, 30, 91, 22))
        self.dateEdit_3.setCalendarPopup(True)
        self.dateEdit_3.setObjectName("dateEdit_3")
        self.label_13 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_13.setGeometry(QtCore.QRect(190, 30, 31, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_14.setGeometry(QtCore.QRect(430, 30, 41, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.comboBox_6 = QtWidgets.QComboBox(parent=self.tab_5)
        self.comboBox_6.setGeometry(QtCore.QRect(470, 30, 51, 22))
        self.comboBox_6.setObjectName("comboBox_6")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.pushButton_17 = QtWidgets.QPushButton(parent=self.tab_5)
        self.pushButton_17.setGeometry(QtCore.QRect(1020, 30, 75, 25))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_17.setFont(font)
        self.pushButton_17.setObjectName("pushButton_17")
        self.tabWidget.addTab(self.tab_5, "")
        # Set central widget
        self.setCentralWidget(self.centralwidget)

        # Create menu bar
        self.menubar = QtWidgets.QMenuBar(parent=self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 22))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)

        # Create status bar
        self.statusbar = QtWidgets.QStatusBar(parent=self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        # Initialize UI
        self.retranslateUi()
        self.tabWidget.setCurrentIndex(0)
        self.connectSignalsSlots()

    def retranslateUi(self):
        """Set up all the text for UI elements"""
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "车身缺陷检测系统"))
        self.label.setText(_translate("MainWindow", "当前摄像头编号为："))
        self.comboBox.setItemText(0, _translate("MainWindow", "摄像头0"))
        self.comboBox.setItemText(1, _translate("MainWindow", "摄像头1"))
        self.comboBox.setItemText(2, _translate("MainWindow", "摄像头2"))
        self.comboBox.setItemText(3, _translate("MainWindow", "摄像头3"))
        self.comboBox.setItemText(4, _translate("MainWindow", "摄像头4"))
        self.checkBox.setText(_translate("MainWindow", "划痕检测"))
        self.checkBox_2.setText(_translate("MainWindow", "凹坑检测"))
        self.pushButton_4.setText(_translate("MainWindow", "通信测试"))
        self.label_2.setText(_translate("MainWindow", "置信度阈值："))
        self.label_3.setText(_translate("MainWindow", "交叉比阈值："))
        self.pushButton_5.setText(_translate("MainWindow", "云服务测试"))
        self.pushButton_6.setText(_translate("MainWindow", "打开摄像头"))
        self.pushButton_13.setText(_translate("MainWindow", "开始检测"))
        self.pushButton_14.setText(_translate("MainWindow", "停止检测"))
        self.label_4.setText(_translate("MainWindow", "串口："))
        self.pushButton_7.setText(_translate("MainWindow", "刷新"))
        self.label_5.setText(_translate("MainWindow", "波特率："))
        self.label_6.setText(_translate("MainWindow", "数据位："))
        self.label_7.setText(_translate("MainWindow", "校验位："))
        self.pushButton_15.setText(_translate("MainWindow", "退出程序"))
        self.pushButton_18.setText(_translate("MainWindow", "选择权重文件"))
        self.label_15.setText(_translate("MainWindow", "停止位："))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "开始检测"))
        self.label_8.setText(_translate("MainWindow", "picture1"))
        self.label_9.setText(_translate("MainWindow", "picture2"))
        self.label_10.setText(_translate("MainWindow", "picture3"))
        self.label_11.setText(_translate("MainWindow", "picture4"))
        self.checkBox_5.setText(_translate("MainWindow", "查看凹坑"))
        self.checkBox_6.setText(_translate("MainWindow", "查看划痕"))
        self.checkBox_7.setText(_translate("MainWindow", "所有"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "查看结果"))
        self.label_12.setText(_translate("MainWindow", "检测日期："))
        self.label_13.setText(_translate("MainWindow", "至"))
        self.label_14.setText(_translate("MainWindow", "缺陷："))
        self.comboBox_6.setItemText(0, _translate("MainWindow", "所有"))
        self.comboBox_6.setItemText(1, _translate("MainWindow", "划痕"))
        self.comboBox_6.setItemText(2, _translate("MainWindow", "凹坑"))
        self.pushButton_17.setText(_translate("MainWindow", "开始搜索"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "历史记录"))

    def connectSignalsSlots(self):
        """Connect signals to their respective slots"""
        # 连接信号和槽
        # Camera and detection controls
        self.pushButton_6.clicked.connect(self.openCamera)
        self.pushButton_13.clicked.connect(self.startDetection)
        self.pushButton_14.clicked.connect(self.stopDetection)
        self.pushButton_15.clicked.connect(self.close)
        self.pushButton_18.clicked.connect(self.selectWeightFile)

        # Communication and service testing
        self.pushButton_4.clicked.connect(self.testCommunication)
        self.pushButton_5.clicked.connect(self.testCloudService)
        self.pushButton_7.clicked.connect(self.refreshPorts)

        # History search
        self.pushButton_17.clicked.connect(self.searchHistory)

        # 摄像头选择下拉框
        self.comboBox.currentIndexChanged.connect(self.changeCamera)

        # 设置默认值
        self.doubleSpinBox.setValue(self.conf_thres)
        self.doubleSpinBox_2.setValue(self.iou_thres)
        self.checkBox.setChecked(self.scratch_detection)
        self.checkBox_2.setChecked(self.dents_detection)

        # 设置日期控件默认值
        current_date = QtCore.QDate.currentDate()
        self.dateEdit.setDate(current_date)
        self.dateEdit_3.setDate(current_date.addDays(-7))  # 默认显示过去一周

        # 初始化串口设置
        self.refreshPorts()

        # 初始化时将视频显示区域设置为黑色
        self.setVideoWidgetBlack()

    # 各种功能方法
    def openCamera(self):
        """Open the camera"""
        if self.camera is not None and self.camera.isActive():
            # 关闭摄像头
            self.camera.stop()
            self.camera = None
            self.pushButton_6.setText("打开摄像头")
            self.textBrowser.append("已关闭摄像头")

            # 将视频显示区域变为黑色
            self.setVideoWidgetBlack()
        else:
            # 获取可用摄像头列表
            available_cameras = QMediaDevices.videoInputs()

            if not available_cameras:
                self.textBrowser.append("未找到可用摄像头")
                return

            # 确保摄像头ID在有效范围内
            if self.camera_id >= len(available_cameras):
                self.camera_id = 0

            # 确保视频窗口可见并清除黑色背景
            self.videoWidget.show()
            self.videoWidget.setStyleSheet("")

            # 创建摄像头对象
            self.camera = QCamera(available_cameras[self.camera_id])

            # 设置摄像头到媒体捕获会话
            self.capture_session.setCamera(self.camera)

            # 启动摄像头
            self.camera.start()
            self.pushButton_6.setText("关闭摄像头")
            self.textBrowser.append(f"已打开摄像头 {self.camera_id}")

    def changeCamera(self, index):
        """Change the camera based on comboBox selection"""
        # 获取新的摄像头ID (索引从0开始)
        new_camera_id = index

        # 更新摄像头ID
        self.camera_id = new_camera_id
        self.textBrowser.append(f"已切换到摄像头 {self.camera_id}")

        # 如果摄像头已经打开，则重新打开新的摄像头
        if self.camera is not None and self.camera.isActive():
            # 关闭当前摄像头
            self.camera.stop()
            self.camera = None

            # 确保视频窗口可见
            self.videoWidget.show()
            # 切换摄像头时不需要清除背景，因为会重新打开摄像头

            # 重新打开新摄像头
            self.openCamera()

    def startDetection(self):
        """Start the defect detection process"""
        # 更新检测参数
        self.conf_thres = self.doubleSpinBox.value()
        self.iou_thres = self.doubleSpinBox_2.value()
        self.scratch_detection = self.checkBox.isChecked()
        self.dents_detection = self.checkBox_2.isChecked()

        # 检查摄像头是否打开
        if self.camera is None or not self.camera.isActive():
            self.textBrowser.append("请先打开摄像头")
            return

        self.textBrowser.append("开始检测...")
        self.textBrowser.append(f"置信度阈值: {self.conf_thres}")
        self.textBrowser.append(f"交叉比阈值: {self.iou_thres}")
        self.textBrowser.append(f"划痕检测: {'开启' if self.scratch_detection else '关闭'}")
        self.textBrowser.append(f"凹坑检测: {'开启' if self.dents_detection else '关闭'}")

    def stopDetection(self):
        """Stop the defect detection process"""
        self.textBrowser.append("停止检测")

    def selectWeightFile(self):
        """Select weight file for the detection model"""
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择权重文件", "", "Weight Files (*.pt *.pth);;All Files (*)")
        if file_path:
            self.textBrowser.append(f"已选择权重文件: {file_path}")

    def testCommunication(self):
        """Test communication with external devices"""
        self.textBrowser.append("正在测试通信...")
        # 模拟通信测试
        QtCore.QTimer.singleShot(1000, lambda: self.textBrowser.append("通信测试成功"))

    def testCloudService(self):
        """Test connection to cloud services"""
        self.textBrowser.append("正在测试云服务连接...")
        # 模拟云服务测试
        QtCore.QTimer.singleShot(1500, lambda: self.textBrowser.append("云服务连接测试成功"))

    def update_port_list(self):
        """Update the list of available serial ports"""
        self.textBrowser.append("正在获取串口列表...")

        # 获取系统中的所有串口
        ports = []

        # 检查是否安装了pyserial库
        if not _has_serial:
            self.textBrowser.append("未安装pyserial库，无法获取串口列表。请安装: pip install pyserial")
            # 添加一些默认的串口作为占位
            default_ports = ["COM1", "COM2", "COM3", "COM4"]
            self.comboBox_2.clear()
            for port in default_ports:
                self.comboBox_2.addItem(port)
            return default_ports

        try:
            # 使用serial.tools.list_ports获取系统串口列表
            ports_list = list(serial.tools.list_ports.comports())

            if not ports_list:
                self.textBrowser.append("未发现串口设备")
            else:
                # 清空当前串口列表
                self.comboBox_2.clear()

                # 添加发现的串口
                for port in ports_list:
                    # 获取串口名称
                    port_name = port.device
                    # 添加到下拉列表
                    self.comboBox_2.addItem(port_name)
                    ports.append(port_name)

                self.textBrowser.append(f"已发现{len(ports)}个串口设备")

                # 如果有串口，选中第一个
                if ports:
                    self.comboBox_2.setCurrentIndex(0)
        except Exception as e:
            self.textBrowser.append(f"获取串口列表时出错: {str(e)}")
            # 出错时添加一些默认的串口
            default_ports = ["COM1", "COM2", "COM3", "COM4"]
            self.comboBox_2.clear()
            for port in default_ports:
                self.comboBox_2.addItem(port)
            return default_ports

        return ports

    def refreshPorts(self):
        """Refresh available serial ports"""
        # 更新串口列表
        self.update_port_list()

        # 添加波特率选项
        baud_rates = ["9600", "19200", "38400", "57600", "115200"]
        self.comboBox_3.clear()
        for rate in baud_rates:
            self.comboBox_3.addItem(rate)
        # 默认选择115200
        self.comboBox_3.setCurrentText("115200")

        # 添加数据位选项
        data_bits = ["5", "6", "7", "8"]
        self.comboBox_4.clear()
        for bits in data_bits:
            self.comboBox_4.addItem(bits)
        # 默认选8位
        self.comboBox_4.setCurrentText("8")

        # 添加校验位选项
        parity_bits = ["无", "奇校验", "偶校验"]
        self.comboBox_5.clear()
        for parity in parity_bits:
            self.comboBox_5.addItem(parity)
        # 默认选择无校验
        self.comboBox_5.setCurrentIndex(0)

        # 添加停止位选项
        stop_bits = ["1", "1.5", "2"]
        self.comboBox_7.clear()
        for stop in stop_bits:
            self.comboBox_7.addItem(stop)
        # 默认选1位
        self.comboBox_7.setCurrentText("1")

    def searchHistory(self):
        """Search historical detection records"""
        start_date = self.dateEdit_3.date().toString("yyyy-MM-dd")
        end_date = self.dateEdit.date().toString("yyyy-MM-dd")
        defect_type = self.comboBox_6.currentText()

        self.textBrowser.append(f"搜索时间范围: {start_date} 至 {end_date}")
        self.textBrowser.append(f"缺陷类型: {defect_type}")
        self.textBrowser.append("正在搜索历史记录...")

        # 模拟搜索结果
        QtCore.QTimer.singleShot(1000, lambda: self.textBrowser.append("搜索完成，找到3条记录"))


# Main entry point
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DefectDetectionApp()
    window.show()
    sys.exit(app.exec())
