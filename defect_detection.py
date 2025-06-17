from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont
import sys
import os
import numpy as np
import datetime

# 尝试导入serial模块，如果不存在则设置标志
_has_serial = True
try:
    import serial  # 用于串口通信
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

# 在data文件夹下新建一个csv文件用于存储历史记录，如果存在则不建立
if not os.path.exists('data/detect_history.csv'):
    os.makedirs('data', exist_ok=True)
    with open('data/detect_history.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'detection_time', 'timestamp_folder', 'detection_type', 'car_color',
                        'dent_count', 'scratch_count', 'total_count', 'model_file', 'conf_threshold',
                        'iou_threshold', 'result_path'])


# 凹坑检测代码
def dent_detect(weights='weights/aoxian&huahen.pt', source='data/val', img_size=640, conf_thres=0.25,
                iou_thres=0.45, device='', classes=1, agnostic_nms=False, augment=False,
                csv_path=None, timestamp=None):
    # 初始化
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # 只在CUDA上支持半精度

    # 创建保存检测结果的文件夹
    if timestamp:
        # 使用时间戳创建文件夹
        save_dir = Path('data/car_result') / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        # 使用递推方式创建文件夹（向后兼容）
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
                   csv_path=None, timestamp=None):
    # 初始化
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # 只在CUDA上支持半精度

    # 创建保存检测结果的文件夹
    if timestamp:
        # 使用时间戳创建文件夹
        save_dir = Path('data/car_result') / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        # 使用递推方式创建文件夹（向后兼容）
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
           csv_path=None, timestamp=None):
    # 初始化
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # 只在CUDA上支持半精度

    # 创建保存检测结果的文件夹
    if timestamp:
        # 使用时间戳创建文件夹
        save_dir = Path('data/car_result') / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        # 使用递推方式创建文件夹（向后兼容）
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
    scratch_counter = 0  # 划痕计数器
    dent_counter = 0     # 凹坑计数器

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

                    # 根据类别分别计数
                    class_id = int(cls)
                    if class_id == 0:  # 划痕
                        scratch_counter += 1
                    elif class_id == 1:  # 凹坑
                        dent_counter += 1

                    # 转换为整数以便于写入CSV
                    x_min, y_min, x_max, y_max = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    class_name = names[class_id]
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
    # 返回详细的统计信息：(总数, 划痕数, 凹坑数)
    return detect_counter, scratch_counter, dent_counter


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
    print(main_color)
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


# 通信测试线程类
class CommunicationTestThread(QtCore.QThread):
    # 自定义信号
    update_text = QtCore.pyqtSignal(str)  # 更新文本信号
    finished_signal = QtCore.pyqtSignal()  # 完成信号
    error_signal = QtCore.pyqtSignal(str)  # 错误信号

    def __init__(self, port, baud_rate, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud_rate = baud_rate

    def run(self):
        """在线程中运行通信测试"""
        ser = None
        self.update_text.emit("")
        try:
            # 检查是否安装了 pyserial 库
            if not _has_serial:
                self.error_signal.emit("错误：未安装 pyserial 库，无法进行串口通信测试")
                self.update_text.emit("请安装：pip install pyserial")
                return

            # 检查串口参数是否已设置
            if not self.port:
                self.error_signal.emit("错误：请先选择串口")
                return

            if not self.baud_rate:
                self.error_signal.emit("错误：请先设置波特率")
                return

            self.update_text.emit(f"正在测试串口通信：{self.port}, 波特率：{self.baud_rate}")

            # 尝试打开串口
            ser = serial.Serial(port=self.port,
                                baudrate=self.baud_rate,
                                parity=serial.PARITY_NONE,
                                bytesize=serial.EIGHTBITS,
                                stopbits=serial.STOPBITS_ONE,
                                timeout=1)  # 设置超时时间为1秒

            if ser.isOpen():
                self.update_text.emit("串口已成功打开")
            else:
                self.error_signal.emit("错误：串口未能打开")
                return

            # 发送测试信号
            ser.write(b'\x01')  # 发送开始信号给摄像头
            self.update_text.emit("已发送测试信号")

            # 等待响应
            time.sleep(1)
            data = ser.read(10)

            # 检查响应
            if b'\01' in data or b'\00' in data:
                self.update_text.emit("串口成功建立通信")
                ser.write(b'\x00')  # 发送停止信号
                self.update_text.emit("通信测试完成")
            else:
                self.update_text.emit("串口通信失败：未收到正确结果")
                ser.write(b'\x00')  # 发送停止信号

        except serial.SerialException as e:
            self.error_signal.emit(f"串口错误：{str(e)}")
            self.update_text.emit("请检查串口是否被其他程序占用或串口参数是否正确")

        except ImportError as e:
            self.error_signal.emit(f"导入错误：{str(e)}")
            self.update_text.emit("请确保已安装 pyserial 库")

        except AttributeError as e:
            self.error_signal.emit(f"属性错误：{str(e)}")
            self.update_text.emit("请检查串口参数设置是否完整")

        except Exception as e:
            self.error_signal.emit(f"通信测试发生未知错误：{str(e)}")

        finally:
            # 确保串口被正确关闭
            if ser and ser.isOpen():
                try:
                    ser.close()
                    self.update_text.emit("串口已关闭")
                except Exception as close_error:
                    self.error_signal.emit(f"关闭串口时出错：{str(close_error)}")

            # 发送完成信号
            self.finished_signal.emit()


# 视频录制线程类
class RecordingThread(QtCore.QThread):
    # 自定义信号
    update_text = QtCore.pyqtSignal(str)  # 更新文本信号
    finished_signal = QtCore.pyqtSignal()  # 完成信号
    error_signal = QtCore.pyqtSignal(str)  # 错误信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.running = False

    def run(self):
        """在线程中运行视频录制"""
        ser = None
        self.update_text.emit("")
        try:
            # 检查是否安装了 pyserial 库
            if not _has_serial:
                self.error_signal.emit("错误：未安装 pyserial 库，无法进行串口通信")
                self.update_text.emit("请安装：pip install pyserial")
                return

            # 获取串口参数
            if hasattr(self.parent_app, 'port') and self.parent_app.port:
                port = self.parent_app.port
            else:
                self.error_signal.emit("错误：请先选择串口")
                return

            if hasattr(self.parent_app, 'baud_rate') and self.parent_app.baud_rate:
                baud_rate = self.parent_app.baud_rate
            else:
                self.error_signal.emit("错误：请先设置波特率")
                return

            # 获取其他串口参数
            data_bits = getattr(self.parent_app, 'data_bits', 8)
            parity = getattr(self.parent_app, 'parity', 'N')
            stop_bits = getattr(self.parent_app, 'stop_bits', 1)

            self.update_text.emit(f"正在初始化录制：串口 {port}, 波特率：{baud_rate}")
            self.update_text.emit(f"串口参数：数据位 {data_bits}, 校验位 {parity}, 停止位 {stop_bits}")

            # 获取录制参数（以DefectDetectionApp中的参数为主）
            camera_ids = getattr(self.parent_app, 'camera_ids', 2)
            picture_width = getattr(self.parent_app, 'picture_width', 640)
            picture_height = getattr(self.parent_app, 'picture_height', 480)

            # 创建以检测时间为子文件夹的输出目录结构
            import datetime
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # 视频文件夹（直接在时间戳文件夹中）
            video_base_folder = 'data/car_video'
            video_time_folder = os.path.join(video_base_folder, current_time)
            output_video_folder = video_time_folder

            # 帧图片文件夹
            picture_base_folder = 'data/car_picture'
            picture_time_folder = os.path.join(picture_base_folder, current_time)
            output_frames_folder = picture_time_folder

            # 创建文件夹结构
            if not os.path.exists(output_video_folder):
                os.makedirs(output_video_folder)
            if not os.path.exists(output_frames_folder):
                os.makedirs(output_frames_folder)

            self.update_text.emit(f"创建视频输出目录：{video_time_folder}")
            self.update_text.emit(f"创建图片输出目录：{picture_time_folder}")

            # 保存文件夹路径供后续使用
            self.video_time_folder = video_time_folder
            self.picture_time_folder = picture_time_folder
            self.output_video_folder = output_video_folder
            self.output_frames_folder = output_frames_folder

            # 初始化摄像头列表
            caps = []
            outs = []
            for i in range(camera_ids):
                cap = cv2.VideoCapture(i)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, picture_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, picture_height)
                caps.append(cap)

                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(f'{output_video_folder}/output{i}.avi', fourcc, 30.0,
                                    (picture_width, picture_height))
                outs.append(out)

            self.update_text.emit(f"已初始化 {camera_ids} 个摄像头")

            # 根据界面参数配置串口
            # 校验位映射
            parity_map = {
                'N': serial.PARITY_NONE,
                '无': serial.PARITY_NONE,
                'E': serial.PARITY_EVEN,
                '偶校验': serial.PARITY_EVEN,
                'O': serial.PARITY_ODD,
                '奇校验': serial.PARITY_ODD
            }

            # 停止位映射
            stopbits_map = {
                1: serial.STOPBITS_ONE,
                1.0: serial.STOPBITS_ONE,
                1.5: serial.STOPBITS_ONE_POINT_FIVE,
                2: serial.STOPBITS_TWO,
                2.0: serial.STOPBITS_TWO
            }

            # 获取串口配置
            serial_parity = parity_map.get(parity, serial.PARITY_NONE)
            serial_stopbits = stopbits_map.get(stop_bits, serial.STOPBITS_ONE)

            # 尝试打开串口
            ser = serial.Serial(port=port,
                                baudrate=baud_rate,
                                parity=serial_parity,
                                bytesize=data_bits,
                                stopbits=serial_stopbits,
                                timeout=0)  # 非阻塞模式

            if ser.isOpen():
                self.update_text.emit("串口已成功打开")
            else:
                self.error_signal.emit("错误：串口未能打开")
                return

            # 发送开始信号给摄像头
            ser.write(b'\x01')
            self.update_text.emit("已发送开始信号")
            time.sleep(1)

            # 检查通信
            data = ser.read(10)
            if b'\01' in data or b'\00' in data:
                self.update_text.emit("串口成功建立通信")
            else:
                self.update_text.emit("串口通信检测：未收到响应，继续等待录制信号...")

            # 等待录制信号
            self.update_text.emit("等待录制信号...")
            while not self.isInterruptionRequested():
                data = ser.read(10)
                if data:
                    if b'\x01' in data:
                        self.update_text.emit("收到录制开始信号，开始录像")
                        recording = True

                        # 录制循环
                        while recording and not self.isInterruptionRequested():
                            for i in range(camera_ids):
                                ret, frame = caps[i].read()
                                if ret:
                                    outs[i].write(frame)

                            # 检查停止信号
                            data = ser.read(10)
                            if b'\x00' in data:
                                self.update_text.emit("收到停止信号，停止录像")
                                ser.write(b'\x00')  # 发送停止确认信号
                                recording = False
                                break

                        # 释放摄像头和视频写入器
                        for i in range(camera_ids):
                            caps[i].release()
                            outs[i].release()

                        self.update_text.emit("录像停止，视频已保存")
                        break

                # 短暂休眠避免过度占用CPU
                time.sleep(0.01)

            # 关闭串口
            if ser and ser.isOpen():
                ser.close()
                self.update_text.emit("串口已关闭")

            # 处理视频提取帧
            self.update_text.emit("开始提取视频帧...")
            for camera_id in range(camera_ids):
                video_path = os.path.join(output_video_folder, f'output{camera_id}.avi')
                if not os.path.exists(video_path):
                    self.update_text.emit(f"警告：视频文件 {video_path} 不存在，跳过")
                    continue

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    self.error_signal.emit(f"视频文件 {video_path} 打开失败！")
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if total_frames > 0:
                    # 提取4个关键帧
                    frames_to_extract = [int(total_frames / 4 * j - 1) for j in range(1, 5)]
                    self.extract_frames(camera_id, video_path, frames_to_extract, output_frames_folder)
                    self.update_text.emit(f"成功提取摄像头 {camera_id} 的视频帧")
                else:
                    self.update_text.emit(f"警告：摄像头 {camera_id} 的视频文件为空")

            self.update_text.emit("视频录制和帧提取完成！")
            self.finished_signal.emit()

        except serial.SerialException as e:
            self.error_signal.emit(f"串口错误：{str(e)}")
            self.update_text.emit("请检查串口是否被其他程序占用或串口参数是否正确")

        except ImportError as e:
            self.error_signal.emit(f"导入错误：{str(e)}")
            self.update_text.emit("请确保已安装 pyserial 库")

        except Exception as e:
            self.error_signal.emit(f"录制过程中发生未知错误：{str(e)}")

        finally:
            # 确保串口被正确关闭
            if ser and ser.isOpen():
                try:
                    ser.close()
                    self.update_text.emit("串口已关闭")
                except Exception as close_error:
                    self.error_signal.emit(f"关闭串口时出错：{str(close_error)}")

    def extract_frames(self, camera_id, video_path, frame_numbers, output_frames_folder):
        """从视频中提取指定帧"""
        frame_index = 1
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_number in frame_numbers:
            if frame_number >= total_frames or frame_number < 0:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                self.update_text.emit(f"错误: 无法读取摄像头 {camera_id} 的第 {frame_number} 帧")
                continue

            # 按照摄像头编号_图片编号命名（如：1-1、1-2、1-3、1-4）
            # 摄像头编号从1开始（camera_id + 1）
            frame_filename = os.path.join(output_frames_folder, f'{camera_id + 1}-{frame_index}.jpg')
            cv2.imwrite(frame_filename, frame)
            frame_index += 1

        cap.release()


# PyQT界面参数配置
class DefectDetectionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化参数
        self.picture_width = 640 # 采集图片宽度
        self.picture_height = 480 # 采集图片高度

        self.camera_ids = 2 # 总摄像头数量
        self.Nowcamera_id = 0 # 当前摄像头编号
        self.camera_id = 0  # 摄像头编号
        self.camera_width = 1024  # 视频显示宽度
        self.camera_height = 576  # 视频显示高度
        self.conf_thres = 0.25  # 置信度阈值
        self.iou_thres = 0.45  # 交叉比阈值
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

        # 初始化线程相关变量
        self.communication_test_thread = None
        self.recording_thread = None

        self.setupUi()

    def closeEvent(self, event):
        """处理应用程序关闭事件"""
        # 关闭摄像头
        if self.camera is not None and self.camera.isActive():
            self.camera.stop()

        # 停止通信测试线程
        if self.communication_test_thread and self.communication_test_thread.isRunning():
            self.communication_test_thread.requestInterruption()
            self.communication_test_thread.wait(3000)  # 等待最多3秒
            if self.communication_test_thread.isRunning():
                self.communication_test_thread.terminate()  # 强制终止

        # 停止录制线程
        if self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.requestInterruption()
            self.recording_thread.wait(3000)  # 等待最多3秒
            if self.recording_thread.isRunning():
                self.recording_thread.terminate()  # 强制终止

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
        self.comboBox_8.addItem("0")
        self.comboBox_8.addItem("1")
        self.comboBox_8.addItem("2")
        self.comboBox_8.addItem("3")
        self.comboBox_8.addItem("4")
        self.pushButton_19 = QtWidgets.QPushButton(parent=self.tab_4)
        self.pushButton_19.setGeometry(QtCore.QRect(190, 20, 81, 21))
        self.pushButton_19.setObjectName("pushButton_19")
        self.textBrowser_2 = QtWidgets.QTextBrowser(parent=self.tab_4)
        self.textBrowser_2.setGeometry(QtCore.QRect(730, 450, 400, 200))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.tableWidget = QtWidgets.QTableWidget(parent=self.tab_4)
        self.tableWidget.setGeometry(QtCore.QRect(20, 460, 681, 192))
        self.tableWidget.setObjectName("tableWidget")
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

        # 查看结果界面
        self.pushButton_19.clicked.connect(self.viewDetectionResults)
        self.comboBox_8.currentIndexChanged.connect(self.changeCameraForResults)

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
        # self.setVideoWidgetBlack()

        # 初始化查看结果界面
        self.initializeResultsView()
        self.pushButton_10.clicked.connect(self.viewLeftDetectionResults)
        self.pushButton_9.clicked.connect(self.viewRightDetectionResults)
        self.pushButton_8.clicked.connect(self.viewRoofDetectionResults)

        # 初始化历史记录界面
        self.initializeHistoryView()
        self.loadHistoryRecords()

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
        self.videoWidget.setAspectRatioMode(Qt.AspectRatioMode.IgnoreAspectRatio)

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
        self.textBrowser.append(f"置信度阈值: {self.conf_thres:.2f}")
        self.textBrowser.append(f"交叉比阈值: {self.iou_thres:.2f}")
        self.textBrowser.append(f"划痕检测: {'开启' if self.scratch_detection else '关闭'}")
        self.textBrowser.append(f"凹坑检测: {'开启' if self.dents_detection else '关闭'}")

        self.pushButton_13.setEnabled(False)
        self.textBrowser.append("开始录制和检测过程...")

        # 创建并启动录制线程
        self.recording_thread = RecordingThread(self)
        self.recording_thread.update_text.connect(self.textBrowser.append)
        self.recording_thread.error_signal.connect(self.handle_thread_error)
        self.recording_thread.finished_signal.connect(self.start_detection_process)
        self.recording_thread.start()

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
        """启动通信测试线程"""
        # 检查是否有线程正在运行
        if self.communication_test_thread and self.communication_test_thread.isRunning():
            self.textBrowser.append("通信测试正在进行中，请等待完成...")
            return

        # 检查串口参数是否已设置
        if not hasattr(self, 'port') or not self.port:
            self.textBrowser.append("错误：请先选择串口")
            return

        if not hasattr(self, 'baud_rate') or not self.baud_rate:
            self.textBrowser.append("错误：请先设置波特率")
            return

        # 禁用通信测试按钮，防止重复点击
        self.pushButton_4.setEnabled(False)
        self.textBrowser.append("开始通信测试...")

        # 创建并启动通信测试线程
        self.communication_test_thread = CommunicationTestThread(self.port, self.baud_rate, self)
        self.communication_test_thread.update_text.connect(self.textBrowser.append)
        self.communication_test_thread.error_signal.connect(self.handle_communication_error)
        self.communication_test_thread.finished_signal.connect(self.communication_test_completed)
        self.communication_test_thread.start()

    def handle_communication_error(self, error_msg):
        """处理通信测试中的错误"""
        self.textBrowser.append(f"错误: {error_msg}")
        # 重新启用通信测试按钮
        self.pushButton_4.setEnabled(True)

    def communication_test_completed(self):
        """通信测试完成后的处理"""
        self.textBrowser.append("通信测试流程完成")
        # 重新启用通信测试按钮
        self.pushButton_4.setEnabled(True)

    def handle_thread_error(self, error_msg):
        """处理线程中的错误"""
        self.textBrowser.append(f"错误: {error_msg}")
        # 重新启用开始检测按钮
        self.pushButton_13.setEnabled(True)

    def start_detection_process(self):
        """录制完成后开始检测处理"""
        self.textBrowser.append("录制完成，开始进行缺陷检测...")

        # 获取检测参数
        weights = getattr(self, 'file_path', 'weights/aoxian&huahen.pt')
        conf_thres = getattr(self, 'conf_thres', 0.5)
        iou_thres = getattr(self, 'iou_thres', 0.5)
        scratch_detection = getattr(self, 'scratch_detection', True)
        dents_detection = getattr(self, 'dents_detection', True)

        try:
            # 获取最新的录制文件夹路径和时间戳
            if hasattr(self.recording_thread, 'output_frames_folder'):
                source_folder = self.recording_thread.output_frames_folder
                # 从路径中提取时间戳
                timestamp = os.path.basename(source_folder)
                self.textBrowser.append(f"使用录制文件夹：{source_folder}")
                self.textBrowser.append(f"检测时间戳：{timestamp}")
            else:
                # 回退到默认路径
                timestamp = None

            # 执行检测
            total_defects = 0
            scratch_count = 0
            dent_count = 0

            if scratch_detection and dents_detection:
                # 执行综合检测
                self.textBrowser.append("执行综合缺陷检测（划痕+凹坑）...")
                result = detect(weights=weights,
                              source=source_folder,
                              conf_thres=conf_thres,
                              iou_thres=iou_thres,
                              timestamp=timestamp)
                total_defects, scratch_count, dent_count = result

            elif scratch_detection:
                # 仅执行划痕检测
                self.textBrowser.append("执行划痕检测...")
                scratch_count = scratch_detect(weights=weights,
                                             source=source_folder,
                                             conf_thres=conf_thres,
                                             iou_thres=iou_thres,
                                             timestamp=timestamp)
                total_defects = scratch_count
                dent_count = 0

            elif dents_detection:
                # 仅执行凹坑检测
                self.textBrowser.append("执行凹坑检测...")
                dent_count = dent_detect(weights=weights,
                                       source=source_folder,
                                       conf_thres=conf_thres,
                                       iou_thres=iou_thres,
                                       timestamp=timestamp)
                total_defects = dent_count
                scratch_count = 0

            # 显示检测结果
            self.textBrowser.append(f"检测完成！共发现 {total_defects} 个缺陷")
            if scratch_detection and dents_detection:
                self.textBrowser.append(f"其中：划痕 {scratch_count} 个，凹坑 {dent_count} 个")
            if timestamp:
                self.textBrowser.append(f"检测结果已保存到：data/car_result/{timestamp}/")
            self.textBrowser.append("可以在'查看结果'标签页中查看详细结果")

            # 保存检测历史记录
            self.save_detection_history(timestamp, total_defects, scratch_count, dent_count, scratch_detection, dents_detection, weights, conf_thres, iou_thres)

            # 更新查看结果界面的最新时间戳
            self.updateResultsViewTimestamp(timestamp)

        except Exception as e:
            self.handle_thread_error(f"检测过程中发生错误: {str(e)}")
            return

        # 重新启用开始检测按钮
        self.pushButton_13.setEnabled(True)
        self.textBrowser.append("检测流程完成！")

    def save_detection_history(self, timestamp, total_defects, scratch_count, dent_count, scratch_detection, dents_detection, weights, conf_thres, iou_thres):
        """保存检测历史记录到CSV文件"""
        try:
            import datetime
            import csv

            # 确保data文件夹存在
            data_folder = 'data'
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            history_file = os.path.join(data_folder, 'detect_history.csv')

            # 检查文件是否存在，如果不存在则创建并写入表头
            file_exists = os.path.exists(history_file)

            with open(history_file, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'detection_time', 'timestamp_folder', 'detection_type', 'car_color',
                    'dent_count', 'scratch_count', 'total_count', 'model_file', 'conf_threshold',
                    'iou_threshold', 'result_path'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # 如果文件不存在，写入表头
                if not file_exists:
                    writer.writeheader()

                # 确定检测类型
                if scratch_detection and dents_detection:
                    detection_type = "combined_detection"
                elif scratch_detection:
                    detection_type = "scratch_detection"
                elif dents_detection:
                    detection_type = "dent_detection"
                else:
                    detection_type = "unknown"

                # 获取当前时间
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 生成时间戳（用于timestamp字段）
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                # 构建结果路径
                if timestamp:
                    result_path = f"data/car_result/{timestamp}/"
                    timestamp_folder = timestamp
                else:
                    result_path = "runs/car*/"
                    timestamp_folder = "none"

                # 尝试检测车身颜色（如果有图片的话）
                car_color = "unknown"
                try:
                    if timestamp and os.path.exists(f"data/car_picture/{timestamp}"):
                        # 查找第一张图片进行颜色检测
                        picture_folder = f"data/car_picture/{timestamp}"
                        for file in os.listdir(picture_folder):
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                image_path = os.path.join(picture_folder, file)
                                car_color = detect_color(image_path)
                                break
                except Exception as color_error:
                    print(f"颜色检测失败: {color_error}")
                    car_color = "unknown"

                # 写入记录
                writer.writerow({
                    'timestamp': timestamp_str,
                    'detection_time': current_time,
                    'timestamp_folder': timestamp_folder,
                    'detection_type': detection_type,
                    'car_color': car_color,
                    'dent_count': dent_count,
                    'scratch_count': scratch_count,
                    'total_count': total_defects,
                    'model_file': os.path.basename(weights),
                    'conf_threshold': conf_thres,
                    'iou_threshold': iou_thres,
                    'result_path': result_path
                })

            self.textBrowser.append(f"检测历史保存至: {history_file}")
            self.textBrowser.append(f"检测结果: {total_defects} 个总缺陷 ({scratch_count} 个划痕缺陷, {dent_count} 个凹坑缺陷)")

        except Exception as e:
            self.textBrowser.append(f"保存历史记录失败，原因为: {str(e)}")

    def load_detection_history(self):
        """加载并返回检测历史记录"""
        try:
            import pandas as pd

            history_file = os.path.join('data', 'detect_history.csv')

            if not os.path.exists(history_file):
                return None

            # 读取CSV文件
            df = pd.read_csv(history_file, encoding='utf-8')
            return df

        except Exception as e:
            self.textBrowser.append(f"读取检测历史记录时出错：{str(e)}")
            return None

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
        try:
            start_date = self.dateEdit_3.date().toString("yyyy-MM-dd")
            end_date = self.dateEdit.date().toString("yyyy-MM-dd")
            defect_type = self.comboBox_6.currentText()

            # 清空当前显示的记录
            self.clearHistoryDisplay()

            # 读取历史记录并过滤
            filtered_records = self.filterHistoryRecords(start_date, end_date, defect_type)

            # 显示过滤后的记录
            self.displayHistoryRecords(filtered_records)

            # 显示搜索结果统计
            record_count = len(filtered_records) if filtered_records else 0
            search_info = f"搜索完成：时间范围 {start_date} 至 {end_date}"
            if defect_type != "所有":
                search_info += f"，缺陷类型：{defect_type}"
            search_info += f"，找到 {record_count} 条记录"

            # 在状态栏显示搜索结果
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage(search_info, 5000)  # 显示5秒

        except Exception as e:
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage(f"搜索历史记录时出错：{str(e)}", 5000)

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
        # 禁用检测按钮
        self.pushButton_13.setEnabled(False)

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

    # 查看结果界面相关方法
    def initializeResultsView(self):
        """初始化查看结果界面"""
        try:
            # 获取最新的时间戳文件夹
            self.latest_timestamp = self.getLatestTimestamp()

            # 初始化当前选择的摄像头编号（从0开始）
            self.current_camera_for_results = 0

            # 设置默认的摄像头编号
            self.comboBox_8.setCurrentIndex(0)  # 默认选择第一个摄像头（编号0）

            # 根据最新时间戳的历史记录设置复选框状态
            self.setCheckboxesFromHistory()

            # 在textBrowser_2中显示初始化信息
            if self.latest_timestamp:
                self.textBrowser_2.append(f"已加载最新检测结果：{self.latest_timestamp}")
                self.textBrowser_2.append(f"当前选择摄像头编号：{self.current_camera_for_results}")
                self.textBrowser_2.append("点击'查看'按钮显示对应摄像头的检测结果")
            else:
                self.textBrowser_2.append("未找到检测结果文件夹")
                self.textBrowser_2.append("请先进行缺陷检测")

        except Exception as e:
            self.textBrowser_2.append(f"初始化查看结果界面时出错：{str(e)}")

    def updateResultsViewTimestamp(self, new_timestamp):
        """更新查看结果界面的最新时间戳"""
        try:
            if new_timestamp:
                # 更新最新时间戳
                self.latest_timestamp = new_timestamp

                # 在textBrowser_2中显示更新信息
                self.textBrowser_2.append("=" * 50)
                self.textBrowser_2.append("检测完成，已更新查看结果界面！")
                self.textBrowser_2.append(f"最新检测结果时间戳：{self.latest_timestamp}")
                self.textBrowser_2.append(f"当前选择摄像头编号：{self.current_camera_for_results}")
                self.textBrowser_2.append("可以点击'查看'按钮查看最新的检测结果")

                # 根据新的时间戳更新复选框状态
                self.setCheckboxesFromHistory()

                # 提示用户可以查看最新结果
                self.textBrowser.append("提示：查看结果界面已更新为最新检测结果")

        except Exception as e:
            self.textBrowser_2.append(f"更新查看结果界面时间戳时出错：{str(e)}")

    def refreshResultsViewTimestamp(self):
        """手动刷新查看结果界面的最新时间戳"""
        try:
            # 重新获取最新的时间戳
            new_latest_timestamp = self.getLatestTimestamp()

            if new_latest_timestamp and new_latest_timestamp != self.latest_timestamp:
                # 如果找到了新的时间戳且与当前不同
                old_timestamp = self.latest_timestamp
                self.latest_timestamp = new_latest_timestamp

                self.textBrowser_2.append("=" * 50)
                self.textBrowser_2.append("已刷新查看结果界面！")
                if old_timestamp:
                    self.textBrowser_2.append(f"原时间戳：{old_timestamp}")
                self.textBrowser_2.append(f"最新时间戳：{self.latest_timestamp}")
                self.textBrowser_2.append("可以点击'查看'按钮查看最新的检测结果")

                # 根据新的时间戳更新复选框状态
                self.setCheckboxesFromHistory()

            elif new_latest_timestamp == self.latest_timestamp:
                self.textBrowser_2.append("当前已是最新的检测结果")
            else:
                self.textBrowser_2.append("未找到检测结果文件夹")

        except Exception as e:
            self.textBrowser_2.append(f"刷新查看结果界面时间戳时出错：{str(e)}")

    def setCheckboxesFromHistory(self):
        """根据最新时间戳的历史记录设置复选框状态"""
        try:
            if not self.latest_timestamp:
                # 如果没有最新时间戳，设置默认状态
                self.checkBox_5.setChecked(False)  # 查看凹坑
                self.checkBox_6.setChecked(False)  # 查看划痕
                self.checkBox_7.setChecked(True)   # 所有
                self.textBrowser_2.append("未找到历史记录，设置为默认复选框状态")
                return

            # 读取检测历史记录文件
            history_file = 'data/detect_history.csv'
            if not os.path.exists(history_file):
                self.textBrowser_2.append("未找到检测历史记录文件，设置为默认复选框状态")
                self.checkBox_5.setChecked(False)
                self.checkBox_6.setChecked(False)
                self.checkBox_7.setChecked(True)
                return

            # 查找对应时间戳的记录
            import csv
            detection_type = None
            with open(history_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['timestamp_folder'] == self.latest_timestamp:
                        detection_type = row['detection_type']
                        break

            if detection_type:
                # 根据detection_type设置复选框状态
                if detection_type == "combined_detection":
                    # 综合检测：所有都勾选
                    self.checkBox_5.setChecked(True)   # 查看凹坑
                    self.checkBox_6.setChecked(True)   # 查看划痕
                    self.checkBox_7.setChecked(True)   # 所有
                    self.textBrowser_2.append("检测类型：综合检测 - 已勾选所有复选框")

                elif detection_type == "scratch_detection":
                    # 仅划痕检测
                    self.checkBox_5.setChecked(False)  # 查看凹坑
                    self.checkBox_6.setChecked(True)   # 查看划痕
                    self.checkBox_7.setChecked(False)  # 所有
                    self.textBrowser_2.append("检测类型：划痕检测 - 已勾选查看划痕")

                elif detection_type == "dent_detection":
                    # 仅凹坑检测
                    self.checkBox_5.setChecked(True)   # 查看凹坑
                    self.checkBox_6.setChecked(False)  # 查看划痕
                    self.checkBox_7.setChecked(False)  # 所有
                    self.textBrowser_2.append("检测类型：凹坑检测 - 已勾选查看凹坑")

                else:
                    # 未知检测类型，设置默认状态
                    self.checkBox_5.setChecked(False)
                    self.checkBox_6.setChecked(False)
                    self.checkBox_7.setChecked(True)
                    self.textBrowser_2.append(f"未知检测类型：{detection_type} - 设置为默认状态")
            else:
                # 未找到对应记录，设置默认状态
                self.checkBox_5.setChecked(False)
                self.checkBox_6.setChecked(False)
                self.checkBox_7.setChecked(True)
                self.textBrowser_2.append("未找到对应的历史记录，设置为默认复选框状态")

        except Exception as e:
            # 出错时设置默认状态
            self.checkBox_5.setChecked(False)
            self.checkBox_6.setChecked(False)
            self.checkBox_7.setChecked(True)
            self.textBrowser_2.append(f"设置复选框状态时出错：{str(e)} - 已设置为默认状态")

    def getLatestTimestamp(self):
        """获取最新的时间戳文件夹"""
        try:
            # 检查car_result文件夹
            result_base_path = 'data/car_result'
            if not os.path.exists(result_base_path):
                return None

            # 获取所有时间戳文件夹
            timestamp_folders = []
            for item in os.listdir(result_base_path):
                item_path = os.path.join(result_base_path, item)
                if os.path.isdir(item_path):
                    timestamp_folders.append(item)

            if not timestamp_folders:
                return None

            # 按时间戳排序，获取最新的
            timestamp_folders.sort(reverse=True)
            return timestamp_folders[0]

        except Exception as e:
            print(f"获取最新时间戳时出错：{str(e)}")
            return None

    def changeCameraForResults(self):
        """改变查看结果的摄像头编号"""
        try:
            # 获取选择的摄像头编号（从comboBox_8的索引得到实际编号）
            camera_index = self.comboBox_8.currentIndex()
            self.current_camera_for_results = camera_index

            self.textBrowser_2.append(f"已选择摄像头编号：{self.current_camera_for_results}")
            self.textBrowser_2.append("点击'查看'按钮显示该摄像头的检测结果")

        except Exception as e:
            self.textBrowser_2.append(f"切换摄像头时出错：{str(e)}")

    def viewDetectionResults(self):
        """查看检测结果"""
        try:
            self.textBrowser_2.append("=" * 50)
            self.textBrowser_2.append(f"正在加载摄像头 {self.current_camera_for_results} 的检测结果...")

            # 检查并更新到最新的时间戳
            latest_available_timestamp = self.getLatestTimestamp()

            if not latest_available_timestamp:
                self.textBrowser_2.append("错误：未找到检测结果文件夹")
                self.textBrowser_2.append("请先进行缺陷检测")
                return

            # 如果发现有更新的时间戳，自动更新
            if not hasattr(self, 'latest_timestamp') or not self.latest_timestamp or latest_available_timestamp != self.latest_timestamp:
                if hasattr(self, 'latest_timestamp') and self.latest_timestamp and latest_available_timestamp != self.latest_timestamp:
                    self.textBrowser_2.append(f"发现更新的检测结果，从 {self.latest_timestamp} 更新到 {latest_available_timestamp}")
                self.latest_timestamp = latest_available_timestamp
                self.textBrowser_2.append(f"当前使用时间戳：{self.latest_timestamp}")

            # 显示对应摄像头的图片
            self.displayCameraImages()

            # 显示检测结果信息
            self.displayDetectionInfo()

            self.textBrowser_2.append(f"摄像头 {self.current_camera_for_results} 的检测结果加载完成")

        except Exception as e:
            self.textBrowser_2.append(f"查看检测结果时出错：{str(e)}")

    def displayCameraImages(self):
        """显示对应摄像头的图片在label_8、label_9、label_10、label_11中"""
        try:
            # 构建图片路径
            result_folder = f"data/car_result/{self.latest_timestamp}"

            # 清空之前的图片
            self.label_8.clear()
            self.label_9.clear()
            self.label_10.clear()
            self.label_11.clear()
            self.label_8.setText("图片1")
            self.label_9.setText("图片2")
            self.label_10.setText("图片3")
            self.label_11.setText("图片4")

            # 查找该摄像头的图片文件（使用摄像头编号+1来匹配文件名）
            camera_images = []
            camera_number = self.current_camera_for_results + 1  # 文件名中摄像头编号从1开始
            if os.path.exists(result_folder):
                for filename in os.listdir(result_folder):
                    if filename.startswith(f"{camera_number}-") and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        camera_images.append(filename)

            camera_images.sort()  # 按文件名排序

            # 定义label列表
            labels = [self.label_8, self.label_9, self.label_10, self.label_11]

            if camera_images:
                # 显示最多4张图片
                for i, label in enumerate(labels):
                    if i < len(camera_images):
                        image_path = os.path.join(result_folder, camera_images[i])
                        if os.path.exists(image_path):
                            pixmap = QPixmap(image_path)
                            if not pixmap.isNull():
                                # 缩放图片以适应label大小
                                scaled_pixmap = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                                label.setPixmap(scaled_pixmap)
                                label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                self.textBrowser_2.append(f"已显示摄像头 {self.current_camera_for_results} 的 {min(len(camera_images), 4)} 张检测结果图片")
                self.textBrowser_2.append(f"该摄像头共有 {len(camera_images)} 张图片")
            else:
                self.textBrowser_2.append(f"未找到摄像头 {self.current_camera_for_results} 的图片")

        except Exception as e:
            self.textBrowser_2.append(f"显示图片时出错：{str(e)}")

    def displayDetectionInfo(self):
        """在tableWidget中显示检测结果信息，动态读取CSV文件的列信息并以表格形式显示"""
        try:
            # 检查CSV文件是否存在
            csv_path = f"data/car_result/{self.latest_timestamp}/detection_results.csv"
            if not os.path.exists(csv_path):
                self.textBrowser_2.append("错误：检测结果CSV文件不存在！")
                return

            # 导入必要的模块
            import csv

            # 尝试导入pandas，如果失败则使用备用方法
            try:
                import pandas as pd
                use_pandas = True
            except ImportError:
                self.textBrowser_2.append("提示：未安装pandas库，使用基础CSV读取方法")
                use_pandas = False

            # 读取CSV文件并过滤当前摄像头的数据
            camera_number = self.current_camera_for_results + 1  # 文件名中摄像头编号从1开始

            # 根据是否有pandas选择读取方法
            if use_pandas:
                # 使用pandas读取CSV文件，更好地处理数据
                try:
                    # 读取整个CSV文件
                    df = pd.read_csv(csv_path, encoding='utf-8')
                    csv_name = "detection_results.csv"

                    # 获取列名
                    header = df.columns.tolist()
                    self.textBrowser_2.append(f"CSV文件列信息：{header}")

                    # 过滤当前摄像头的数据
                    if 'image' in df.columns:
                        # 过滤出当前摄像头的数据
                        filtered_df = df[df['image'].str.startswith(f"{camera_number}-", na=False)]
                    else:
                        # 如果没有image列，显示所有数据
                        filtered_df = df
                        self.textBrowser_2.append("警告：CSV文件中没有找到'image'列，显示所有数据")

                    # 设置tableWidget的行数和列数
                    row_count = len(filtered_df)
                    col_count = len(header)
                    self.tableWidget.setRowCount(row_count)
                    self.tableWidget.setColumnCount(col_count)

                    # 创建中文表头映射
                    header_mapping = {
                        'image': '图片名称',
                        'defect_id': '缺陷ID',
                        'class': '缺陷类型',
                        'confidence': '置信度',
                        'x_min': 'X最小值',
                        'y_min': 'Y最小值',
                        'x_max': 'X最大值',
                        'y_max': 'Y最大值'
                    }

                    # 设置中文表头
                    chinese_headers = []
                    for h in header:
                        chinese_headers.append(header_mapping.get(h, h))  # 如果没有映射则使用原名

                    # 设置表头
                    self.tableWidget.setHorizontalHeaderLabels(chinese_headers)

                    # 清空之前的数据
                    self.tableWidget.clearContents()

                    if row_count > 0:
                        # 填充表格数据
                        for row_index in range(row_count):
                            for col_index, column_name in enumerate(header):
                                # 获取单元格数据
                                cell_data = filtered_df.iloc[row_index, col_index]

                                # 创建表格项
                                from PyQt6.QtWidgets import QTableWidgetItem
                                from PyQt6.QtCore import Qt

                                # 对不同类型的数据进行格式化显示
                                if column_name == 'confidence':  # 置信度列
                                    try:
                                        conf_value = float(cell_data)
                                        formatted_data = f"{conf_value:.3f}"
                                    except:
                                        formatted_data = str(cell_data)
                                elif column_name in ['x_min', 'y_min', 'x_max', 'y_max']:  # 坐标列
                                    try:
                                        coord_value = int(float(cell_data))
                                        formatted_data = str(coord_value)
                                    except:
                                        formatted_data = str(cell_data)
                                elif column_name == 'defect_id':  # 缺陷ID列
                                    try:
                                        id_value = int(float(cell_data))
                                        formatted_data = str(id_value)
                                    except:
                                        formatted_data = str(cell_data)
                                else:
                                    # 处理NaN值和其他数据类型
                                    if pd.isna(cell_data):
                                        formatted_data = ""
                                    else:
                                        formatted_data = str(cell_data)

                                item = QTableWidgetItem(formatted_data)

                                # 设置文本居中对齐
                                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                                # 设置为只读
                                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                                self.tableWidget.setItem(row_index, col_index, item)

                        # 调整列宽以适应内容
                        self.tableWidget.resizeColumnsToContents()

                        # 设置表格属性
                        self.tableWidget.setAlternatingRowColors(True)  # 交替行颜色
                        self.tableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)  # 选择整行
                        self.tableWidget.setSortingEnabled(True)  # 启用排序

                        # 设置表格样式
                        self.tableWidget.horizontalHeader().setStretchLastSection(True)  # 最后一列自动拉伸

                    else:
                        # 如果没有数据，显示提示信息
                        self.tableWidget.setRowCount(1)
                        self.tableWidget.setColumnCount(1)
                        self.tableWidget.setHorizontalHeaderLabels(["提示信息"])
                        from PyQt6.QtWidgets import QTableWidgetItem
                        item = QTableWidgetItem("该摄像头未检测到缺陷")
                        self.tableWidget.setItem(0, 0, item)
                        self.tableWidget.resizeColumnsToContents()

                    # 在textBrowser_2中反馈结果
                    self.textBrowser_2.append(f"成功打开{csv_name}，一共 {row_count} 个缺陷记录！")
                    self.textBrowser_2.append(f"摄像头 {self.current_camera_for_results} 的检测结果已在表格中显示")

                    # 显示数据统计信息
                    if row_count > 0:
                        # 统计不同类型的缺陷
                        if 'class' in filtered_df.columns:
                            defect_types = filtered_df['class'].value_counts()
                            self.textBrowser_2.append("缺陷类型统计：")
                            for defect_type, count in defect_types.items():
                                self.textBrowser_2.append(f"  {defect_type}: {count} 个")

                except Exception as e:
                    self.textBrowser_2.append(f"读取CSV文件时出错: {str(e)}")
                    # 如果pandas读取失败，尝试使用原始csv模块
                    try:
                        self.textBrowser_2.append("尝试使用备用方法读取CSV文件...")
                        self._fallback_csv_read(csv_path, camera_number)
                    except Exception as fallback_error:
                        self.textBrowser_2.append(f"备用方法也失败: {str(fallback_error)}")
            else:
                # 如果没有pandas，直接使用备用方法
                try:
                    self._fallback_csv_read(csv_path, camera_number)
                except Exception as e:
                    self.textBrowser_2.append(f"读取CSV文件失败: {str(e)}")

        except Exception as e:
            self.textBrowser_2.append(f"显示检测信息时出错：{str(e)}")

    def _fallback_csv_read(self, csv_path, camera_number):
        """读取csv文件"""
        with open(csv_path, newline='', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)

            # 过滤当前摄像头的数据
            filtered_rows = []
            for row in csv_reader:
                if len(row) > 0 and row[0].startswith(f"{camera_number}-"):
                    filtered_rows.append(row)

            # 创建中文表头映射
            header_mapping = {
                'image': '图片名称',
                'defect_id': '缺陷ID',
                'class': '缺陷类型',
                'confidence': '置信度',
                'x_min': 'X最小值',
                'y_min': 'Y最小值',
                'x_max': 'X最大值',
                'y_max': 'Y最大值'
            }

            # 设置中文表头
            chinese_headers = []
            for h in header:
                chinese_headers.append(header_mapping.get(h, h))

            # 设置表格
            row_count = len(filtered_rows)
            self.tableWidget.setRowCount(row_count)
            self.tableWidget.setColumnCount(len(header))
            self.tableWidget.setHorizontalHeaderLabels(chinese_headers)
            self.tableWidget.clearContents()

            if row_count > 0:
                # 填充数据
                for row_index, row_data in enumerate(filtered_rows):
                    for col_index, cell_data in enumerate(row_data):
                        from PyQt6.QtWidgets import QTableWidgetItem
                        from PyQt6.QtCore import Qt

                        # 对不同类型的数据进行格式化显示
                        column_name = header[col_index]
                        if column_name == 'confidence':  # 置信度列
                            try:
                                conf_value = float(cell_data)
                                formatted_data = f"{conf_value:.3f}"
                            except:
                                formatted_data = str(cell_data)
                        elif column_name in ['x_min', 'y_min', 'x_max', 'y_max']:  # 坐标列
                            try:
                                coord_value = int(float(cell_data))
                                formatted_data = str(coord_value)
                            except:
                                formatted_data = str(cell_data)
                        elif column_name == 'defect_id':  # 缺陷ID列
                            try:
                                id_value = int(float(cell_data))
                                formatted_data = str(id_value)
                            except:
                                formatted_data = str(cell_data)
                        else:
                            formatted_data = str(cell_data)

                        item = QTableWidgetItem(formatted_data)
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        self.tableWidget.setItem(row_index, col_index, item)

                # 设置表格属性
                self.tableWidget.setAlternatingRowColors(True)
                self.tableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
                self.tableWidget.setSortingEnabled(True)
                self.tableWidget.horizontalHeader().setStretchLastSection(True)
            else:
                # 如果没有数据，显示提示信息
                self.tableWidget.setRowCount(1)
                self.tableWidget.setColumnCount(1)
                self.tableWidget.setHorizontalHeaderLabels(["提示信息"])
                from PyQt6.QtWidgets import QTableWidgetItem
                item = QTableWidgetItem("该摄像头未检测到缺陷")
                self.tableWidget.setItem(0, 0, item)

            # 优化列宽度自适应设置
            self.optimizeTableColumnWidths()
            self.textBrowser_2.append(f"成功读取 {row_count} 条缺陷记录")
            self.textBrowser_2.append(f"摄像头 {self.current_camera_for_results} 的检测结果已在表格中显示")

    def optimizeTableColumnWidths(self):
        """优化表格列宽度自适应设置"""
        try:
            # 获取表格的总宽度
            table_width = self.tableWidget.width()
            column_count = self.tableWidget.columnCount()

            if column_count == 0:
                return

            # 首先调整列宽以适应内容
            self.tableWidget.resizeColumnsToContents()

            # 获取各列的当前宽度
            current_widths = []
            total_content_width = 0
            for i in range(column_count):
                width = self.tableWidget.columnWidth(i)
                current_widths.append(width)
                total_content_width += width

            # 记录调试信息（可选）
            # self.textBrowser_2.append(f"表格宽度: {table_width}, 内容宽度: {total_content_width}")

            # 定义各列的最小和最大宽度约束
            column_constraints = {
                '图片名称': {'min': 85, 'max': 160, 'preferred': 125},  # 图片名称需要更多空间
                '缺陷ID': {'min': 55, 'max': 75, 'preferred': 65},     # ID列较窄即可
                '缺陷类型': {'min': 75, 'max': 110, 'preferred': 90},   # 类型名称适中
                '置信度': {'min': 65, 'max': 95, 'preferred': 80},     # 置信度数值列
                'X最小值': {'min': 55, 'max': 85, 'preferred': 70},    # 坐标列紧凑
                'Y最小值': {'min': 55, 'max': 85, 'preferred': 70},
                'X最大值': {'min': 55, 'max': 85, 'preferred': 70},
                'Y最大值': {'min': 55, 'max': 85, 'preferred': 70}
            }

            # 计算可用宽度（减去滚动条和边距）
            available_width = table_width - 30  # 预留滚动条和边距空间

            # 如果内容宽度小于可用宽度，按比例扩展
            if total_content_width < available_width:
                # 获取表头标签
                headers = []
                for i in range(column_count):
                    header_item = self.tableWidget.horizontalHeaderItem(i)
                    if header_item:
                        headers.append(header_item.text())
                    else:
                        headers.append(f"列{i+1}")

                # 计算新的列宽
                new_widths = []
                remaining_width = available_width

                for i, (current_width, header) in enumerate(zip(current_widths, headers)):
                    constraints = column_constraints.get(header, {'min': 50, 'max': 200, 'preferred': 100})

                    if i == column_count - 1:  # 最后一列
                        # 最后一列使用剩余宽度
                        new_width = max(constraints['min'], min(constraints['max'], remaining_width))
                    else:
                        # 其他列使用首选宽度或当前宽度的较大值
                        preferred_width = max(current_width, constraints['preferred'])
                        new_width = max(constraints['min'], min(constraints['max'], preferred_width))
                        remaining_width -= new_width

                    new_widths.append(new_width)

                # 应用新的列宽
                for i, width in enumerate(new_widths):
                    self.tableWidget.setColumnWidth(i, width)

            else:
                # 如果内容宽度大于可用宽度，设置最小宽度并启用水平滚动
                headers = []
                for i in range(column_count):
                    header_item = self.tableWidget.horizontalHeaderItem(i)
                    if header_item:
                        headers.append(header_item.text())
                    else:
                        headers.append(f"列{i+1}")

                for i, (current_width, header) in enumerate(zip(current_widths, headers)):
                    constraints = column_constraints.get(header, {'min': 50, 'max': 200, 'preferred': 100})
                    # 确保不小于最小宽度
                    min_width = constraints['min']
                    if current_width < min_width:
                        self.tableWidget.setColumnWidth(i, min_width)

            # 设置表格的水平滚动策略
            from PyQt6.QtCore import Qt
            self.tableWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

            # 设置表头的拉伸模式
            header = self.tableWidget.horizontalHeader()
            from PyQt6.QtWidgets import QHeaderView

            # 如果总宽度小于可用宽度，最后一列拉伸填充
            if total_content_width < available_width:
                header.setStretchLastSection(True)
            else:
                header.setStretchLastSection(False)

            # 设置表头可以手动调整列宽
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

            # 设置表格的其他显示属性
            self.tableWidget.setShowGrid(True)  # 显示网格线
            self.tableWidget.setGridStyle(Qt.PenStyle.SolidLine)  # 实线网格

            # 设置表格的选择模式
            self.tableWidget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)

        except Exception as e:
            self.textBrowser_2.append(f"优化表格列宽时出错：{str(e)}")
            # 如果出错，回退到基本的自适应方法
            self.tableWidget.resizeColumnsToContents()

    def resizeTableToFitWindow(self):
        """当窗口大小改变时重新调整表格列宽"""
        try:
            if hasattr(self, 'tableWidget') and self.tableWidget.columnCount() > 0:
                # 延迟执行，确保窗口大小已经更新
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(100, self.optimizeTableColumnWidths)
        except Exception as e:
            pass  # 静默处理错误，避免影响主要功能

    def viewLeftDetectionResults(self):
        """查看车身左侧检测结果"""
        # 显示self.Leftcarbody_camera_id对应结果
        self.current_camera_for_results = self.Leftcarbody_camera_id
        self.viewDetectionResults()

    def viewRightDetectionResults(self):
        """查看车身右侧检测结果"""
        # 显示self.Rightcarbody_camera_id对应结果
        self.current_camera_for_results = self.Rightcarbody_camera_id
        self.viewDetectionResults()

    def viewRoofDetectionResults(self):
        """查看车身顶部检测结果"""
        # 显示self.Roofcarbody_camera_id对应结果
        self.current_camera_for_results = self.Roofcarbody_camera_id
        self.viewDetectionResults()

    # 历史记录相关方法
    def initializeHistoryView(self):
        """初始化历史记录界面"""
        try:
            # 创建滚动区域的布局
            self.history_layout = QtWidgets.QVBoxLayout()
            self.history_layout.setSpacing(5)  # 设置间距
            self.history_layout.setContentsMargins(10, 10, 10, 10)  # 设置边距

            # 设置滚动区域内容的布局
            self.scrollAreaWidgetContents.setLayout(self.history_layout)

            # 存储历史记录项的列表
            self.history_record_widgets = []

        except Exception as e:
            print(f"初始化历史记录界面时出错：{str(e)}")

    def loadHistoryRecords(self):
        """加载所有历史记录"""
        try:
            # 读取历史记录文件
            history_file = 'data/detect_history.csv'
            if not os.path.exists(history_file):
                self.displayNoHistoryMessage()
                return

            # 读取CSV文件
            import csv
            records = []
            with open(history_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    records.append(row)

            # 按时间戳倒序排列（最新的在前面）
            records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            # 显示记录
            self.displayHistoryRecords(records)

        except Exception as e:
            print(f"加载历史记录时出错：{str(e)}")
            self.displayNoHistoryMessage()

    def displayNoHistoryMessage(self):
        """显示无历史记录的提示信息"""
        try:
            # 清空现有内容
            self.clearHistoryDisplay()

            # 创建提示标签
            no_data_label = QtWidgets.QLabel("暂无历史检测记录")
            no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_data_label.setStyleSheet("""
                QLabel {
                    color: #666666;
                    font-size: 16px;
                    padding: 50px;
                    background-color: #f5f5f5;
                    border: 2px dashed #cccccc;
                    border-radius: 10px;
                }
            """)

            self.history_layout.addWidget(no_data_label)
            self.history_layout.addStretch()  # 添加弹性空间

        except Exception as e:
            print(f"显示无历史记录提示时出错：{str(e)}")

    def clearHistoryDisplay(self):
        """清空历史记录显示"""
        try:
            # 清空布局中的所有控件
            while self.history_layout.count():
                child = self.history_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            # 清空记录列表
            self.history_record_widgets.clear()

        except Exception as e:
            print(f"清空历史记录显示时出错：{str(e)}")

    def displayHistoryRecords(self, records):
        """显示历史记录列表"""
        try:
            # 清空现有显示
            self.clearHistoryDisplay()

            if not records:
                self.displayNoHistoryMessage()
                return

            # 为每条记录创建显示控件
            for record in records:
                record_widget = self.createHistoryRecordWidget(record)
                if record_widget:
                    self.history_layout.addWidget(record_widget)
                    self.history_record_widgets.append(record_widget)

            # 添加弹性空间，使记录靠上显示
            self.history_layout.addStretch()

        except Exception as e:
            print(f"显示历史记录时出错：{str(e)}")

    def createHistoryRecordWidget(self, record):
        """为单条历史记录创建显示控件"""
        try:
            # 创建主容器
            record_frame = QtWidgets.QFrame()
            record_frame.setFrameStyle(QtWidgets.QFrame.Shape.Box)
            record_frame.setStyleSheet("""
                QFrame {
                    background-color: white;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 10px;
                    margin: 2px;
                }
                QFrame:hover {
                    background-color: #f0f8ff;
                    border-color: #4a90e2;
                    cursor: pointer;
                }
            """)
            record_frame.setFixedHeight(120)  # 设置固定高度

            # 创建布局
            layout = QtWidgets.QHBoxLayout(record_frame)
            layout.setContentsMargins(15, 10, 15, 10)
            layout.setSpacing(20)

            # 左侧：时间和基本信息
            left_layout = QtWidgets.QVBoxLayout()
            left_layout.setSpacing(5)

            # 检测时间
            time_label = QtWidgets.QLabel(f"检测时间：{record.get('timestamp', 'N/A')}")
            time_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
            left_layout.addWidget(time_label)

            # 检测类型和车身颜色
            type_color_text = f"检测类型：{self.getDetectionTypeDisplay(record.get('detection_type', 'N/A'))} | 车身颜色：{record.get('car_color', 'N/A')}"
            type_color_label = QtWidgets.QLabel(type_color_text)
            type_color_label.setStyleSheet("font-size: 12px; color: #34495e;")
            left_layout.addWidget(type_color_label)

            # 结果文件夹
            folder_label = QtWidgets.QLabel(f"结果文件夹：{record.get('timestamp_folder', 'N/A')}")
            folder_label.setStyleSheet("font-size: 11px; color: #7f8c8d;")
            left_layout.addWidget(folder_label)

            layout.addLayout(left_layout, 3)  # 占3份空间

            # 中间：缺陷统计
            middle_layout = QtWidgets.QVBoxLayout()
            middle_layout.setSpacing(5)

            # 缺陷数量标题
            defect_title = QtWidgets.QLabel("缺陷统计")
            defect_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #e74c3c;")
            defect_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            middle_layout.addWidget(defect_title)

            # 缺陷数量信息
            scratch_count = record.get('scratch_count', '0')
            dent_count = record.get('dent_count', '0')
            total_count = record.get('total_count', '0')

            defect_info = f"划痕：{scratch_count} | 凹坑：{dent_count} | 总计：{total_count}"
            defect_label = QtWidgets.QLabel(defect_info)
            defect_label.setStyleSheet("font-size: 11px; color: #e74c3c;")
            defect_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            middle_layout.addWidget(defect_label)

            layout.addLayout(middle_layout, 2)  # 占2份空间

            # 右侧：检测参数
            right_layout = QtWidgets.QVBoxLayout()
            right_layout.setSpacing(5)

            # 检测参数标题
            param_title = QtWidgets.QLabel("检测参数")
            param_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #3498db;")
            param_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right_layout.addWidget(param_title)

            # 置信度和交叉比
            conf_thres = record.get('conf_threshold', 'N/A')
            iou_thres = record.get('iou_threshold', 'N/A')
            param_info = f"置信度：{conf_thres} | 交叉比：{iou_thres}"
            param_label = QtWidgets.QLabel(param_info)
            param_label.setStyleSheet("font-size: 11px; color: #3498db;")
            param_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right_layout.addWidget(param_label)

            # 模型文件
            model_file = record.get('model_file', 'N/A')
            if len(model_file) > 25:  # 如果文件名太长，截断显示
                model_file = "..." + model_file[-22:]
            model_label = QtWidgets.QLabel(f"模型：{model_file}")
            model_label.setStyleSheet("font-size: 10px; color: #95a5a6;")
            model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right_layout.addWidget(model_label)

            layout.addLayout(right_layout, 2)  # 占2份空间

            # 为控件添加点击事件
            record_frame.mousePressEvent = lambda event, r=record: self.onHistoryRecordClicked(r)

            # 存储记录数据到控件属性中
            record_frame.record_data = record

            return record_frame

        except Exception as e:
            print(f"创建历史记录控件时出错：{str(e)}")
            return None

    def getDetectionTypeDisplay(self, detection_type):
        """获取检测类型的中文显示名称"""
        type_mapping = {
            'combined_detection': '综合检测',
            'scratch_detection': '划痕检测',
            'dent_detection': '凹坑检测'
        }
        return type_mapping.get(detection_type, detection_type)

    def filterHistoryRecords(self, start_date, end_date, defect_type):
        """根据条件过滤历史记录"""
        try:
            # 读取历史记录文件
            history_file = 'data/detect_history.csv'
            if not os.path.exists(history_file):
                return []

            import csv
            from datetime import datetime

            filtered_records = []

            with open(history_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # 检查日期范围 - 处理时间戳格式转换
                    timestamp = row.get('timestamp', '')
                    if timestamp:
                        try:
                            # 将时间戳格式 20250603_183657 转换为 2025-06-03 进行比较
                            if len(timestamp) >= 8:
                                date_part = timestamp[:8]  # 取前8位：20250603
                                # 转换为 YYYY-MM-DD 格式
                                formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"

                                if start_date <= formatted_date <= end_date:
                                    # 检查缺陷类型
                                    if defect_type == "所有":
                                        filtered_records.append(row)
                                    elif defect_type == "划痕":
                                        # 检查是否包含划痕检测
                                        detection_type = row.get('detection_type', '')
                                        scratch_count = int(row.get('scratch_count', '0'))
                                        if detection_type in ['scratch_detection', 'combined_detection'] or scratch_count > 0:
                                            filtered_records.append(row)
                                    elif defect_type == "凹坑":
                                        # 检查是否包含凹坑检测
                                        detection_type = row.get('detection_type', '')
                                        dent_count = int(row.get('dent_count', '0'))
                                        if detection_type in ['dent_detection', 'combined_detection'] or dent_count > 0:
                                            filtered_records.append(row)
                        except (ValueError, IndexError) as e:
                            # 如果时间戳格式不正确，跳过这条记录
                            continue

            # 按时间戳倒序排列
            filtered_records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            return filtered_records

        except Exception as e:
            print(f"过滤历史记录时出错：{str(e)}")
            return []

    def onHistoryRecordClicked(self, record):
        """处理历史记录点击事件"""
        try:
            # 获取时间戳文件夹
            timestamp_folder = record.get('timestamp_folder', '')

            if not timestamp_folder:
                if hasattr(self, 'statusbar'):
                    self.statusbar.showMessage("错误：记录中缺少时间戳文件夹信息", 3000)
                return

            # 检查文件夹是否存在
            result_folder = f"data/car_result/{timestamp_folder}"
            if not os.path.exists(result_folder):
                if hasattr(self, 'statusbar'):
                    self.statusbar.showMessage(f"错误：结果文件夹不存在 - {result_folder}", 3000)
                return

            # 更新查看结果界面的时间戳
            self.latest_timestamp = timestamp_folder

            # 切换到查看结果标签页
            self.tabWidget.setCurrentIndex(1)  # 假设查看结果是第二个标签页（索引1）

            # 更新查看结果界面
            self.updateResultsViewFromHistory(record)

            # 显示状态信息
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage(f"已跳转到查看结果界面 - {timestamp_folder}", 3000)

        except Exception as e:
            print(f"处理历史记录点击事件时出错：{str(e)}")
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage(f"跳转失败：{str(e)}", 3000)

    def updateResultsViewFromHistory(self, record):
        """从历史记录更新查看结果界面"""
        try:
            # 在textBrowser_2中显示选择的历史记录信息
            self.textBrowser_2.clear()
            self.textBrowser_2.append("=" * 60)
            self.textBrowser_2.append("从历史记录加载检测结果")
            self.textBrowser_2.append("=" * 60)
            self.textBrowser_2.append(f"检测时间：{record.get('timestamp', 'N/A')}")
            self.textBrowser_2.append(f"检测类型：{self.getDetectionTypeDisplay(record.get('detection_type', 'N/A'))}")
            self.textBrowser_2.append(f"车身颜色：{record.get('car_color', 'N/A')}")
            self.textBrowser_2.append(f"时间戳文件夹：{record.get('timestamp_folder', 'N/A')}")
            self.textBrowser_2.append("")
            self.textBrowser_2.append("缺陷统计：")
            self.textBrowser_2.append(f"  划痕数量：{record.get('scratch_count', '0')}")
            self.textBrowser_2.append(f"  凹坑数量：{record.get('dent_count', '0')}")
            self.textBrowser_2.append(f"  总计数量：{record.get('total_count', '0')}")
            self.textBrowser_2.append("")
            self.textBrowser_2.append("检测参数：")
            self.textBrowser_2.append(f"  置信度阈值：{record.get('conf_threshold', 'N/A')}")
            self.textBrowser_2.append(f"  交叉比阈值：{record.get('iou_threshold', 'N/A')}")
            self.textBrowser_2.append(f"  模型文件：{record.get('model_file', 'N/A')}")
            self.textBrowser_2.append("")
            self.textBrowser_2.append("请选择摄像头编号并点击'查看'按钮查看具体检测结果")

            # 根据检测类型设置复选框状态
            detection_type = record.get('detection_type', '')
            if detection_type == "combined_detection":
                self.checkBox_5.setChecked(True)   # 查看凹坑
                self.checkBox_6.setChecked(True)   # 查看划痕
                self.checkBox_7.setChecked(True)   # 所有
            elif detection_type == "scratch_detection":
                self.checkBox_5.setChecked(False)  # 查看凹坑
                self.checkBox_6.setChecked(True)   # 查看划痕
                self.checkBox_7.setChecked(False)  # 所有
            elif detection_type == "dent_detection":
                self.checkBox_5.setChecked(True)   # 查看凹坑
                self.checkBox_6.setChecked(False)  # 查看划痕
                self.checkBox_7.setChecked(False)  # 所有
            else:
                self.checkBox_5.setChecked(False)
                self.checkBox_6.setChecked(False)
                self.checkBox_7.setChecked(True)

        except Exception as e:
            print(f"从历史记录更新查看结果界面时出错：{str(e)}")
            if hasattr(self, 'textBrowser_2'):
                self.textBrowser_2.append(f"更新界面时出错：{str(e)}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DefectDetectionApp()
    window.show()
    sys.exit(app.exec())
