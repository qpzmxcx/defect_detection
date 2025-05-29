from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont
import sys
import os
import numpy as np

# 尝试导入serial模块，如果不存在则设置标志
_has_serial = True
try:
    import serial.tools.list_ports  # 用于获取系统串口列表
except ImportError:
    _has_serial = False

# 缺陷检测detect.py
import time
from pathlib import Path
import csv
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, time_synchronized
from utils.plots import plot_one_box
# 凹坑检测代码
def dent_detect(weights='weights/aoxian&huahen.pt', source='data/val', img_size=640, conf_thres=0.25,
                iou_thres=0.45, device='', classes=1, agnostic_nms=False, augment=False,
                csv_path=None):
    # 初始化
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # 只在CUDA上支持半精度

    # 创建保存检测结果的文件夹，使用递推方式
    save_dir = Path('runs')
    save_dir.mkdir(parents=True, exist_ok=True)

    # 递推查找car文件夹
    exp_num = 0
    for f in save_dir.glob('car*'):
        if f.is_dir():
            try:
                # 提取文件夹名中的数字
                num = int(f.name.replace('car', ''))
                exp_num = max(exp_num, num + 1)
            except ValueError:
                pass

    # 创建新的输出文件夹
    exp_name = f'car{exp_num}'  # 第一个文件夹为car0，然后依次为car1、car2等
    save_dir = save_dir / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 设置CSV文件路径
    if csv_path is None:
        csv_path = save_dir / 'dent_detection_results.csv'
    else:
        # 如果提供了csv_path，但只是文件名而不是完整路径，则放在新建的文件夹中
        if os.path.dirname(csv_path) == '':
            csv_path = save_dir / csv_path

    # 加载模型
    model = attempt_load(weights, map_location=device)  # 加载FP32模型
    stride = int(model.stride.max())  # 模型步长
    img_size = check_img_size(img_size, s=stride)  # 检查图像尺寸

    if half:
        model.half()  # 转为FP16

    # 设置数据加载器（仅图片）
    dataset = LoadImages(source, img_size=img_size, stride=stride)

    # 获取类名
    names = model.module.names if hasattr(model, 'module') else model.names

    # 准备CSV文件
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['image', 'defect_id', 'class', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max'])

    # 全局变量，用于统计缺陷编号
    dent_counter = 0

    # 运行推理
    t0 = time.time()
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # 应用NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # 处理检测结果
        for i, det in enumerate(pred):  # 每张图片的检测结果
            p, im0 = path, im0s

            # 获取图片名称
            img_name = Path(p).name

            if len(det):
                # 将边界框从img_size调整到im0大小
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 写入CSV并绘制边界框
                for *xyxy, conf, cls in reversed(det):
                    # 增加缺陷编号
                    dent_counter += 1

                    # 转换为整数以便于写入CSV
                    x_min, y_min, x_max, y_max = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    class_name = names[int(cls)]
                    confidence = float(conf)

                    # 写入CSV
                    csv_writer.writerow([img_name, dent_counter, class_name, confidence, x_min, y_min, x_max, y_max])

                    # 在图片上绘制边界框，标签中包含缺陷编号
                    label = f'#{dent_counter} {class_name} {confidence:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=(0, 0, 255), line_thickness=2)

            # 保存检测后的图片
            save_path = save_dir / img_name
            cv2.imwrite(str(save_path), im0)

    # 关闭CSV文件
    csv_file.close()
    return dent_counter
# 划痕检测代码
def scratch_detect(weights='weights/aoxian&huahen.pt', source='data', img_size=640, conf_thres=0.25,
                   iou_thres=0.45, device='', classes=0, agnostic_nms=False, augment=False,
                   csv_path=None):
    # 初始化
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # 只在CUDA上支持半精度

    # 创建保存检测结果的文件夹，使用递推方式
    save_dir = Path('runs')
    save_dir.mkdir(parents=True, exist_ok=True)

    # 递推查找car文件夹
    exp_num = 0
    for f in save_dir.glob('car*'):
        if f.is_dir():
            try:
                # 提取文件夹名中的数字
                num = int(f.name.replace('car', ''))
                exp_num = max(exp_num, num + 1)
            except ValueError:
                pass

    # 创建新的输出文件夹
    exp_name = f'car{exp_num}'  # 第一个文件夹为car0，然后依次为car1、car2等
    save_dir = save_dir / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 设置CSV文件路径
    if csv_path is None:
        csv_path = save_dir / 'scratch_detection_results.csv'
    else:
        # 如果提供了csv_path，但只是文件名而不是完整路径，则放在新建的文件夹中
        if os.path.dirname(csv_path) == '':
            csv_path = save_dir / csv_path

    # 加载模型
    model = attempt_load(weights, map_location=device)  # 加载FP32模型
    stride = int(model.stride.max())  # 模型步长
    img_size = check_img_size(img_size, s=stride)  # 检查图像尺寸

    if half:
        model.half()  # 转为FP16

    # 设置数据加载器（仅图片）
    dataset = LoadImages(source, img_size=img_size, stride=stride)

    # 获取类名
    names = model.module.names if hasattr(model, 'module') else model.names

    # 准备CSV文件
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['image', 'defect_id', 'class', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max'])

    # 全局变量，用于统计缺陷编号
    scratch_counter = 0

    # 运行推理
    t0 = time.time()
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # 应用NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # 处理检测结果
        for i, det in enumerate(pred):  # 每张图片的检测结果
            p, im0 = path, im0s

            # 获取图片名称
            img_name = Path(p).name

            if len(det):
                # 将边界框从img_size调整到im0大小
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 写入CSV并绘制边界框
                for *xyxy, conf, cls in reversed(det):
                    # 增加缺陷编号
                    scratch_counter += 1

                    # 转换为整数以便于写入CSV
                    x_min, y_min, x_max, y_max = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    class_name = names[int(cls)]
                    confidence = float(conf)

                    # 写入CSV
                    csv_writer.writerow([img_name, scratch_counter, class_name, confidence, x_min, y_min, x_max, y_max])

                    # 在图片上绘制边界框，标签中包含缺陷编号
                    label = f'#{scratch_counter} {class_name} {confidence:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=(0, 0, 255), line_thickness=2)

            # 保存检测后的图片
            save_path = save_dir / img_name
            cv2.imwrite(str(save_path), im0)

    # 关闭CSV文件
    csv_file.close()
    return scratch_counter
# 均可检测代码
def detect(weights='weights/aoxian&huahen.pt', source='data', img_size=640, conf_thres=0.25,
           iou_thres=0.45, device='', classes=None, agnostic_nms=False, augment=False,
           csv_path=None):
    # 初始化
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # 只在CUDA上支持半精度

    # 创建保存检测结果的文件夹，使用递推方式
    save_dir = Path('runs')
    save_dir.mkdir(parents=True, exist_ok=True)

    # 递推查找car文件夹
    exp_num = 0
    for f in save_dir.glob('car*'):
        if f.is_dir():
            try:
                # 提取文件夹名中的数字
                num = int(f.name.replace('car', ''))
                exp_num = max(exp_num, num + 1)
            except ValueError:
                pass

    # 创建新的输出文件夹
    exp_name = f'car{exp_num}'  # 第一个文件夹为car0，然后依次为car1、car2等
    save_dir = save_dir / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 设置CSV文件路径
    if csv_path is None:
        csv_path = save_dir / 'detection_results.csv'
    else:
        # 如果提供了csv_path，但只是文件名而不是完整路径，则放在新建的文件夹中
        if os.path.dirname(csv_path) == '':
            csv_path = save_dir / csv_path

    # 加载模型
    model = attempt_load(weights, map_location=device)  # 加载FP32模型
    stride = int(model.stride.max())  # 模型步长
    img_size = check_img_size(img_size, s=stride)  # 检查图像尺寸

    if half:
        model.half()  # 转为FP16

    # 设置数据加载器（仅图片）
    dataset = LoadImages(source, img_size=img_size, stride=stride)

    # 获取类名
    names = model.module.names if hasattr(model, 'module') else model.names

    # 准备CSV文件
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['image', 'defect_id', 'class', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max'])

    # 全局变量，用于统计缺陷编号
    detect_counter = 0

    # 运行推理
    t0 = time.time()
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # 应用NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # 处理检测结果
        for i, det in enumerate(pred):  # 每张图片的检测结果
            p, im0 = path, im0s

            # 获取图片名称
            img_name = Path(p).name

            if len(det):
                # 将边界框从img_size调整到im0大小
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 写入CSV并绘制边界框
                for *xyxy, conf, cls in reversed(det):
                    # 增加缺陷编号
                    detect_counter += 1

                    # 转换为整数以便于写入CSV
                    x_min, y_min, x_max, y_max = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    class_name = names[int(cls)]
                    confidence = float(conf)

                    # 写入CSV
                    csv_writer.writerow([img_name, detect_counter, class_name, confidence, x_min, y_min, x_max, y_max])

                    # 在图片上绘制边界框，标签中包含缺陷编号
                    label = f'#{detect_counter} {class_name} {confidence:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=(0, 0, 255), line_thickness=2)

            # 保存检测后的图片
            save_path = save_dir / img_name
            cv2.imwrite(str(save_path), im0)

    # 关闭CSV文件
    csv_file.close()
    return detect_counter


# 颜色识别代码color_detection
def detect_color(image_path):
    image = cv2.imread(image_path)

    image0 = cv2.resize(image, (600, 600))
    image = image0[200:400, 200:400]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "red0": [(0, 50, 50), (10, 255, 255)],
        "red1": [(160, 50, 50), (180, 255, 255)],
        "blue": [(95, 50, 50), (130, 255, 255)],
        "green": [(35, 50, 50), (85, 255, 255)],
        "yellow": [(25, 50, 50), (35, 255, 255)],
        "white0": [(0, 0, 200), (180, 50, 255)],
        "black": [(0, 0, 0), (180, 255, 50)],
        "purple": [(130, 50, 50), (155, 255, 255)],
        "pink": [(150, 50, 50), (170, 255, 255)],
        "orange": [(10, 50, 50), (25, 255, 255)],
        "white1": [(0, 0, 50), (180, 50, 255)],
        "cyan": [(85, 50, 50), (90, 255, 255)],
    }

    # 初始化颜色占比
    color_percentage = {}

    # 遍历颜色范围
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        # 创建掩码
        mask = cv2.inRange(hsv_image, lower, upper)

        # 计算颜色像素占比
        percentage = (cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])) * 100
        color_percentage[color] = percentage

    # 找出占比最高的颜色
    main_color = max(color_percentage, key=color_percentage.get)

    # 合并相同颜色的不同色域
    if main_color == "red0" or main_color == "red1":
        main_color = "red"
    elif main_color == "white0" or main_color == "white1":
        main_color = "white"

    return main_color


# 云同步到服务器端
import subprocess
def run_rclone(command):
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace',
            text=True
        )
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
# 运行时执行 run_rclone(["rclone", "copy", picture_path, myvm_path, "--progress"])


# PyQT界面参数配置
class DefectDetectionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化参数
        self.camera_ids = 2 # 总摄像头数量
        self.Nowcamera_id = 0 # 当前摄像头编号
        self.camera_id = 0  # 摄像头编号
        self.camera_width = 1024  # 视频宽度
        self.camera_height = 576  # 视频高度
        self.conf_thres = 0.5  # 置信度阈值
        self.iou_thres = 0.5  # 交叉比阈值
        self.Leftcarbody_camera_id = 0 # 左侧车身摄像头编号
        self.Rightcarbody_camera_id = 1 # 右侧车身摄像头编号
        self.Roofcarbody_camera_id = 2 # 车顶车身摄像头编号
        self.scratch_detection = True  # 划痕检测
        self.dents_detection = True  # 凹坑检测
        self.dents_picture = True # 凹坑是否查看
        self.scratch_picture = True # 划痕是否查看
        self.file_path = "weights/aoxian&huahen.pt"
        self.port = "COM3"
        self.baud_rate = 9600
        self.data_bits = 8
        self.parity = "N"
        self.stop_bits = 1

        # 初始化摄像头和媒体播放器
        self.camera = None
        self.capture_session = QMediaCaptureSession()

        self.setupUi()

    def closeEvent(self, event):
        """处理应用程序关闭事件"""
        # 关闭摄像头
        if self.camera is not None and self.camera.isActive():
            self.camera.stop()
        event.accept()

    def setVideoWidgetBlack(self):
        """将视频窗口背景设置为黑色"""
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
        self.label_8.setGeometry(QtCore.QRect(20, 50, 320, 180))
        self.label_8.setAutoFillBackground(True)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_9.setGeometry(QtCore.QRect(380, 50, 320, 180))
        self.label_9.setAutoFillBackground(True)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_10.setGeometry(QtCore.QRect(20, 260, 320, 180))
        self.label_10.setAutoFillBackground(True)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_11.setGeometry(QtCore.QRect(380, 260, 320, 180))
        self.label_11.setAutoFillBackground(True)
        self.label_11.setObjectName("label_11")
        self.label_16 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_16.setGeometry(QtCore.QRect(30, 20, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.comboBox_8 = QtWidgets.QComboBox(parent=self.tab_4)
        self.comboBox_8.setGeometry(QtCore.QRect(120, 19, 31, 22))
        self.comboBox_8.setObjectName("comboBox_8")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.pushButton_19 = QtWidgets.QPushButton(parent=self.tab_4)
        self.pushButton_19.setGeometry(QtCore.QRect(190, 20, 81, 21))
        self.pushButton_19.setObjectName("pushButton_19")
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
        self.label_16.setText(_translate("MainWindow", "摄像头编号为"))
        self.comboBox_8.setItemText(0, _translate("MainWindow", "0"))
        self.comboBox_8.setItemText(1, _translate("MainWindow", "1"))
        self.comboBox_8.setItemText(2, _translate("MainWindow", "2"))
        self.comboBox_8.setItemText(3, _translate("MainWindow", "3"))
        self.comboBox_8.setItemText(4, _translate("MainWindow", "4"))
        self.pushButton_19.setText(_translate("MainWindow", "查看"))
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
        """连接信号和槽"""
        # 连接信号和槽
        # 摄像头和检测控制
        self.pushButton.clicked.connect(self.viewLeftcarbody)
        self.pushButton_2.clicked.connect(self.viewRightcarbody)
        self.pushButton_3.clicked.connect(self.viewRoofcarbody)
        self.pushButton_6.clicked.connect(self.openCamera)
        self.pushButton_13.clicked.connect(self.startDetection)
        self.pushButton_14.clicked.connect(self.stopDetection)
        self.pushButton_15.clicked.connect(self.close)
        self.pushButton_18.clicked.connect(self.selectWeightFile)

        # 通信和服务测试
        self.pushButton_4.clicked.connect(self.testCommunication)
        self.pushButton_5.clicked.connect(self.testCloudService)
        self.pushButton_7.clicked.connect(self.update_port_list)

        # 历史记录搜索
        self.pushButton_17.clicked.connect(self.searchHistory)

        # 摄像头选择下拉框
        self.comboBox.currentIndexChanged.connect(self.changeCamera)

        # 串口选择下拉框
        self.comboBox_2.currentIndexChanged.connect(self.choose_port_list)
        self.comboBox_3.currentIndexChanged.connect(self.choose_baud_rate)
        self.comboBox_4.currentIndexChanged.connect(self.choose_data_bits)
        self.comboBox_5.currentIndexChanged.connect(self.choose_parity)
        self.comboBox_7.currentIndexChanged.connect(self.choose_stop_bits)

        # 设置默认值
        self.doubleSpinBox.setValue(self.conf_thres)
        self.doubleSpinBox.setMinimum(0.0)
        self.doubleSpinBox.setMaximum(1.0)
        self.doubleSpinBox.setSingleStep(0.01)
        self.doubleSpinBox.setDecimals(2)
        self.doubleSpinBox.valueChanged.connect(self.change_confidence)
        self.doubleSpinBox_2.setValue(self.iou_thres)
        self.doubleSpinBox_2.setMinimum(0.0)
        self.doubleSpinBox_2.setMaximum(1.0)
        self.doubleSpinBox_2.setSingleStep(0.01)
        self.doubleSpinBox_2.setDecimals(2)
        self.doubleSpinBox_2.valueChanged.connect(self.change_iou)
        self.checkBox.setChecked(self.scratch_detection)
        self.checkBox_2.setChecked(self.dents_detection)
        self.checkBox_5.setChecked(self.dents_picture)
        self.checkBox_6.setChecked(self.scratch_picture)
        if self.dents_picture and self.scratch_picture:
            self.checkBox_7.setChecked(True)

        # 设置日期控件默认值
        current_date = QtCore.QDate.currentDate()
        self.dateEdit.setDate(current_date)
        self.dateEdit_3.setDate(current_date.addDays(-7))  # 默认显示过去一周

        # 初始化串口设置
        self.refreshPorts()

        # 初始化时将视频显示区域设置为黑色
        self.setVideoWidgetBlack()

    # 各种功能方法
    def viewLeftcarbody(self):
        """查看车身左侧"""
        self.textBrowser.append("查看车身左侧")
        # 调用_opencurrentCamera打开左侧摄像头
        self._opencurrentCamera(self.Leftcarbody_camera_id)

    def viewRightcarbody(self):
        """查看车身右侧"""
        self.textBrowser.append("查看车身右侧")
        # 调用_opencurrentCamera打开右侧摄像头
        self._opencurrentCamera(self.Rightcarbody_camera_id)

    def viewRoofcarbody(self):
        """查看车身顶部"""
        self.textBrowser.append("查看车身顶部")
        # 调用_opencurrentCamera打开顶部摄像头
        self._opencurrentCamera(self.Roofcarbody_camera_id)

    def _opencurrentCamera(self, Nowcamera_id):
        """根据指定的摄像头ID打开摄像头（内部方法）"""
        # 如果当前摄像头已经打开，先关闭
        if self.camera is not None and self.camera.isActive():
            # 使用closeCamera函数关闭当前摄像头，但不改变按钮文本
            self.camera.stop()
            self.camera = None
            # 注意这里不调用setVideoWidgetBlack()，因为我们将立即打开新的摄像头

        # 获取可用摄像头列表
        available_cameras = QMediaDevices.videoInputs()

        if not available_cameras:
            self.textBrowser.append("未找到可用摄像头")
            return

        # 确保摄像头ID在有效范围内
        if Nowcamera_id >= len(available_cameras):
            self.textBrowser.append(f"摄像头ID {Nowcamera_id} 超出范围，请重新选择")
            # Nowcamera_id = 0
            return

        # 更新当前摄像头ID
        self.camera_id = Nowcamera_id

        # 确保视频窗口可见并清除黑色背景
        self.videoWidget.show()
        self.videoWidget.setStyleSheet("")

        # 创建摄像头对象
        self.camera = QCamera(available_cameras[Nowcamera_id])

        # 设置摄像头到媒体捕获会话
        self.capture_session.setCamera(self.camera)

        # 启动摄像头
        self.camera.start()
        self.pushButton_6.setText("关闭摄像头")
        self.textBrowser.append(f"已打开摄像头 {Nowcamera_id}")

        # 更新下拉框选中项
        self.comboBox.setCurrentIndex(Nowcamera_id)

    def openCamera(self):
        """打开或关闭摄像头"""
        # 检查摄像头是否已经打开
        if self.camera is not None and self.camera.isActive():
            # 如果摄像头已经打开，则关闭摄像头
            self.closeCamera()
        else:
            # 如果摄像头未打开，则打开摄像头
            self._opencurrentCamera(self.camera_id)

    def closeCamera(self):
        """关闭摄像头并将界面变为黑色"""
        if self.camera is not None and self.camera.isActive():
            # 关闭摄像头
            self.camera.stop()
            self.camera = None

            # 将按钮文本改回“打开摄像头”
            self.pushButton_6.setText("打开摄像头")
            self.textBrowser.append("已关闭摄像头")

            # 将视频显示区域变为黑色
            self.setVideoWidgetBlack()

    def changeCamera(self, index):
        """根据下拉框选择切换摄像头"""
        # 直接调用_opencurrentCamera打开指定编号的摄像头
        self._opencurrentCamera(index)

    def startDetection(self):
        """开始缺陷检测过程"""
        # 更新检测参数
        self.conf_thres = self.doubleSpinBox.value()
        self.iou_thres = self.doubleSpinBox_2.value()
        self.scratch_detection = self.checkBox.isChecked()
        self.dents_detection = self.checkBox_2.isChecked()

        # 检查是否至少启用了一种检测类型
        if not self.scratch_detection and not self.dents_detection:
            self.textBrowser.append("错误: 请至少启用一种检测类型（划痕或凹坑）")
            return

        self.textBrowser.append("开始检测...")
        self.textBrowser.append(f"置信度阈值: {self.conf_thres}")
        self.textBrowser.append(f"交叉比阈值: {self.iou_thres}")
        self.textBrowser.append(f"划痕检测: {'开启' if self.scratch_detection else '关闭'}")
        self.textBrowser.append(f"凹坑检测: {'开启' if self.dents_detection else '关闭'}")

    def stopDetection(self):
        """停止缺陷检测过程"""
        print(self.port)
        print(self.baud_rate)
        print(self.data_bits)
        print(self.parity)
        print(self.stop_bits)

    def selectWeightFile(self):
        """选择检测模型的权重文件"""
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择权重文件", "", "Weight Files (*.pt *.pth);;All Files (*)")
        if file_path:
            self.textBrowser.append(f"已选择权重文件: {file_path}")
        self.file_path = file_path

    def testCommunication(self):
        """测试与外部设备的通信"""
        ser = serial.Serial(port=self.port,
                            baudrate=self.baud_rate,
                            parity=serial.PARITY_NONE,
                            bytesize=serial.EIGHTBITS,
                            stopbits=serial.STOPBITS_ONE,
                            timeout=0)
        if ser.isOpen():
            self.textBrowser.append("串口已打开")
        else:
            self.textBrowser.append("串口未打开")
            return
        ser.write(b'\x01')  # 发送开始信号给摄像头
        time.sleep(1)
        data = ser.read(10)
        if b'\01' in data or b'\00' in data:
            self.textBrowser.append("串口成功建立通信")
            ser.write(b'\x00')  # 发送停止信号
        else:
            self.textBrowser.append("串口通信失败")
            ser.write(b'\x00')  # 发送停止信号
            return
        ser.close()

    def testCloudService(self):
        """测试云服务连接"""
        print(self.conf_thres)
        print(self.iou_thres)

    def update_port_list(self):
        """更新可用串口列表"""
        try:
            # 检查UI控件是否存在
            if not hasattr(self, 'textBrowser') or not hasattr(self, 'comboBox_2'):
                print("UI控件尚未初始化")
                return []

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
                    # 即使没有发现串口，也添加一些默认选项
                    default_ports = ["COM1", "COM2", "COM3", "COM4"]
                    self.comboBox_2.clear()
                    for port in default_ports:
                        self.comboBox_2.addItem(port)
                    return default_ports
                else:
                    # 清空当前串口列表
                    self.comboBox_2.clear()

                    # 添加发现的串口
                    for port in ports_list:
                        try:
                            # 获取串口名称
                            port_name = port.device
                            # 添加到下拉列表
                            self.comboBox_2.addItem(port_name)
                            ports.append(port_name)
                        except Exception as port_error:
                            print(f"处理串口 {port} 时出错: {port_error}")
                            continue

                    self.textBrowser.append(f"已发现{len(ports)}个串口设备")

                    # 如果有串口，选中第一个
                    if ports:
                        self.comboBox_2.setCurrentIndex(0)

            except ImportError as import_error:
                self.textBrowser.append(f"导入串口库失败: {str(import_error)}")
                # 添加默认串口
                default_ports = ["COM1", "COM2", "COM3", "COM4"]
                self.comboBox_2.clear()
                for port in default_ports:
                    self.comboBox_2.addItem(port)
                return default_ports

            except Exception as e:
                self.textBrowser.append(f"获取串口列表时出错: {str(e)}")
                # 出错时添加一些默认的串口
                default_ports = ["COM1", "COM2", "COM3", "COM4"]
                self.comboBox_2.clear()
                for port in default_ports:
                    self.comboBox_2.addItem(port)
                return default_ports

            return ports

        except Exception as outer_error:
            print(f"update_port_list函数发生严重错误: {outer_error}")
            return []

    def refreshPorts(self):
        """刷新可用串口"""
        try:
            # 检查所有必要的UI控件是否存在
            required_widgets = ['comboBox_2', 'comboBox_3', 'comboBox_4', 'comboBox_5', 'comboBox_7', 'textBrowser']
            for widget_name in required_widgets:
                if not hasattr(self, widget_name):
                    print(f"UI控件 {widget_name} 尚未初始化")
                    return

            # 更新串口列表
            self.update_port_list()

            # 添加波特率选项
            try:
                baud_rates = ["9600", "19200", "38400", "57600", "115200"]
                self.comboBox_3.clear()
                for rate in baud_rates:
                    self.comboBox_3.addItem(rate)
                # 默认选择115200
                self.comboBox_3.setCurrentText("9600")
            except Exception as e:
                print(f"设置波特率选项时出错: {e}")

            # 添加数据位选项
            try:
                data_bits = ["8", "7", "6", "5"]
                self.comboBox_4.clear()
                for bits in data_bits:
                    self.comboBox_4.addItem(bits)
                # 默认选8位
                self.comboBox_4.setCurrentText("8")
            except Exception as e:
                print(f"设置数据位选项时出错: {e}")

            # 添加校验位选项
            try:
                parity_bits = ["无", "奇校验", "偶校验"]
                self.comboBox_5.clear()
                for parity in parity_bits:
                    self.comboBox_5.addItem(parity)
                # 默认选择无校验
                self.comboBox_5.setCurrentIndex(0)
            except Exception as e:
                print(f"设置校验位选项时出错: {e}")

            # 添加停止位选项
            try:
                stop_bits = ["1", "1.5", "2"]
                self.comboBox_7.clear()
                for stop in stop_bits:
                    self.comboBox_7.addItem(stop)
                # 默认选1位
                self.comboBox_7.setCurrentText("1")
            except Exception as e:
                print(f"设置停止位选项时出错: {e}")

            if hasattr(self, 'textBrowser'):
                self.textBrowser.append("串口设置已刷新")

        except Exception as e:
            print(f"refreshPorts函数发生错误: {e}")
            if hasattr(self, 'textBrowser'):
                self.textBrowser.append(f"刷新串口时发生错误: {str(e)}")

    def searchHistory(self):
        """搜索历史检测记录"""
        start_date = self.dateEdit_3.date().toString("yyyy-MM-dd")
        end_date = self.dateEdit.date().toString("yyyy-MM-dd")
        defect_type = self.comboBox_6.currentText()

        self.textBrowser.append(f"搜索时间范围: {start_date} 至 {end_date}")
        self.textBrowser.append(f"缺陷类型: {defect_type}")
        self.textBrowser.append("正在搜索历史记录...")

        # 模拟搜索结果
        QtCore.QTimer.singleShot(1000, lambda: self.textBrowser.append("搜索完成，找到3条记录"))

    def updateDetectionSettings(self):
        """更新检测设置（划痕/凹坑检测）"""
        # 获取复选框状态
        self.scratch_detection = self.checkBox.isChecked()
        self.dents_detection = self.checkBox_2.isChecked()

        # 在文本浏览器中显示当前设置
        self.textBrowser.append("检测设置已更新:")
        self.textBrowser.append(f"划痕检测: {'开启' if self.scratch_detection else '关闭'}")
        self.textBrowser.append(f"凹坑检测: {'开启' if self.dents_detection else '关闭'}")

        # 如果两种检测都关闭，显示警告
        if not self.scratch_detection and not self.dents_detection:
            self.textBrowser.append("警告: 所有检测类型已关闭，将无法进行缺陷检测！")

    def change_confidence(self):
        global conf_thres
        conf_thres = self.doubleSpinBox.value()
        self.conf_thres = conf_thres
        # 显示两位小数
        self.textBrowser.append(f"当前置信度为：{conf_thres:.2f}")

    def change_iou(self):
        global iou_thres
        iou_thres = self.doubleSpinBox_2.value()
        self.iou_thres = iou_thres
        # 显示两位小数
        self.textBrowser.append(f"当前交叉比为：{iou_thres:.2f}")

    def choose_port_list(self):
        global port
        port = self.comboBox_2.currentText()
        self.port = port
        self.textBrowser.append(f"当前串口为：{port}")

    def choose_baud_rate(self):
        global baud_rate
        baud_rate = int(self.comboBox_3.currentText())
        self.baud_rate = baud_rate
        self.textBrowser.append(f"当前波特率为：{baud_rate}")

    def choose_data_bits(self):
        global data_bits
        data_bits = int(self.comboBox_4.currentText())
        self.data_bits = data_bits
        self.textBrowser.append(f"当前数据位为：{data_bits}")

    def choose_parity(self):
        global parity
        parity = self.comboBox_5.currentText()
        self.parity = parity
        self.textBrowser.append(f"当前校验位为：{parity}")

    def choose_stop_bits(self):
        global stop_bits
        stop_bits = float(self.comboBox_7.currentText())
        self.stop_bits = stop_bits
        self.textBrowser.append(f"当前停止位为：{stop_bits}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DefectDetectionApp()
    window.show()
    sys.exit(app.exec())
