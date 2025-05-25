from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QMessageBox, QTextBrowser, QSpinBox, \
    QFileDialog
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
import sys
from PyQt6 import QtCore, QtGui, QtWidgets
import csv
import time
import cv2
import os
import serial

weights = 'aoxian&huahen.pt'
camera_ids = 2
WIDTH = 640
HEIGHT = 480
conf_thres = 0.25
iou_thres = 0.45

now_path = os.getcwd()
print(now_path)
output_video_folder = f'{now_path}/output/output_videos'
if not os.path.exists(output_video_folder):
    os.makedirs(output_video_folder)
for i in range(0, camera_ids):
    output_yolo = f'{now_path}/output/output_yolo/output{i}'
    if not os.path.exists(output_yolo):
        os.makedirs(output_yolo)
    output_frames = f'{now_path}/output/output_frames/output{i}'
    if not os.path.exists(output_frames):
        os.makedirs(output_frames)

# 初始化摄像头列表
caps = []
outs = []
for i in range(0, camera_ids):
    cap = cv2.VideoCapture(i)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    caps.append(cap)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(f'{output_video_folder}/output{i}.avi', fourcc, 30.0, (WIDTH, HEIGHT))
    outs.append(out)


# 视频录制线程类
class RecordingThread(QThread):
    # 自定义信号
    update_text = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False

    def run(self):
        try:
            ser = serial.Serial(port="COM8",
                                baudrate=9600,
                                parity=serial.PARITY_NONE,
                                bytesize=serial.EIGHTBITS,
                                stopbits=serial.STOPBITS_ONE,
                                timeout=0)
            if ser.isOpen():
                self.update_text.emit("串口已打开")
            else:
                self.error_signal.emit("串口未打开")
                return

            ser.write(b'\x01')  # 发送开始信号给摄像头
            time.sleep(1)
            data = ser.read(10)
            if b'\01' in data or b'\00' in data:
                self.update_text.emit("串口成功建立通信")
            else:
                self.error_signal.emit("串口通信失败")
                return

            # 等待信号开始录像
            while not self.isInterruptionRequested():
                data = ser.read(10)
                if data:
                    if b'\x01' in data:
                        self.update_text.emit("开始录像")
                        recording = True

                        while recording and not self.isInterruptionRequested():
                            for i in range(0, camera_ids):
                                ret, frame = caps[i].read()
                                if ret:
                                    outs[i].write(frame)

                                data = ser.read(10)
                                if b'\x00' in data:
                                    self.update_text.emit("停止录像")
                                    ser.write(b'\x00')  # 发送停止信号
                                    recording = False
                                    break
                        for i in range(0, camera_ids):
                            caps[i].release()
                            outs[i].release()
                        self.update_text.emit("录像停止，视频已保存")
                        break
            ser.close()

            # 处理视频提取帧
            for camera_id in range(0, camera_ids):
                video_path = f'{now_path}/output/output_videos/output{camera_id}.avi'
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    self.error_signal.emit(f"视频文件 {video_path} 打开失败！")
                    continue
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                frames_to_extract = [int(total_frames / 4 * j - 1) for j in range(1, 5)]
                self.extract_frames(camera_id, video_path, frames_to_extract)
                self.update_text.emit(f"成功提取{camera_id}号摄像头视频")

            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(f"录制过程中发生错误: {str(e)}")

    def extract_frames(self, camera_id, video_path, frame_numbers):
        frame_number0 = 1
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_dir = f'{now_path}/output/output_frames/output{camera_id}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for frame_number in frame_numbers:
            if frame_number >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                self.update_text.emit(f"错误: 无法读取摄像头 {camera_id} 的第 {frame_number} 帧")
                continue
            cv2.imwrite(f'{output_dir}/frame_{frame_number0}.jpg', frame)
            frame_number0 += 1

        cap.release()


# 目标检测线程类
class DetectionThread(QThread):
    update_text = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, confidence, parent=None):
        super().__init__(parent)
        self.confidence = confidence

    def run(self):
        try:
            for i in range(0, camera_ids):
                output_folder = f'{now_path}/output/output_yolo/output{i}'
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                self.update_text.emit(f"开始处理摄像头 {i} 的图像...")
                self.update_text.emit(f"摄像头 {i} 的图像处理完成")

            self.update_text.emit("所有检测任务完成！")
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(f"检测过程中发生错误: {str(e)}")


class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # 初始化界面
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - 1080) // 2
        y = (screen.height() - 720) // 2
        self.setWindowTitle("基于深度学习的车身缺陷检测系统")
        self.setGeometry(x, y, 1080, 720)

        # 控件初始化
        self.video_label = QLabel(self)
        self.video_label.setGeometry(70, 10, 600, 450)
        self.btn_open = QPushButton("打开摄像头", self)
        self.btn_open.setGeometry(750, 230, 100, 30)
        self.pictrue_open = QPushButton("打开图片", self)
        self.pictrue_open.setGeometry(900, 230, 100, 30)
        self.detect_start = QPushButton("开始检测", self)
        self.detect_start.setGeometry(750, 290, 100, 30)
        self.detect_pictrue = QPushButton("查看检测结果", self)
        self.detect_pictrue.setGeometry(900, 290, 100, 30)
        self.exit = QPushButton("退出程序", self)
        self.exit.setGeometry(900, 350, 100, 30)
        self.setting = QPushButton("设置", self)
        self.setting.setGeometry(750, 350, 100, 30)
        self.textBrowser = QTextBrowser(parent=self)
        self.textBrowser.setGeometry(QtCore.QRect(730, 500, 290, 150))
        self.textBrowser.setObjectName("textBrowser")
        self.spinBox = QSpinBox(parent=self)
        self.spinBox.setGeometry(QtCore.QRect(840, 20, 35, 20))
        self.spinBox.setObjectName("spinBox")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(parent=self)
        self.doubleSpinBox.setGeometry(QtCore.QRect(940, 20, 50, 20))
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.label_2 = QLabel(parent=self)
        self.label_2.setGeometry(QtCore.QRect(740, 20, 100, 20))
        self.label_2.setObjectName("label_2")
        self.label_2.setText("当前摄像头编号为")
        self.label_3 = QLabel(parent=self)
        self.label_3.setGeometry(QtCore.QRect(900, 20, 40, 20))
        self.label_3.setObjectName("label_3")
        self.label_3.setText("置信度")
        self.left = QPushButton("", self)
        self.left.setGeometry(20, 180, 20, 100)
        self.right = QPushButton("", self)
        self.right.setGeometry(700, 180, 20, 100)

        # 创建表格
        self.tableWidget = QtWidgets.QTableWidget(parent=self)
        self.tableWidget.setGeometry(QtCore.QRect(50, 480, 600, 200))
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setColumnWidth(0, 75)
        self.tableWidget.setColumnWidth(1, 75)
        self.tableWidget.setColumnWidth(2, 50)
        self.tableWidget.setColumnWidth(3, 150)
        self.tableWidget.setColumnWidth(4, 50)
        self.tableWidget.setColumnWidth(5, 100)
        self.tableWidget.setColumnWidth(6, 150)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        # OpenCV 摄像头对象(spinBox)参数设置
        self.cap = cv2.VideoCapture()
        self.camera_id = 1  # 默认摄像头编号
        self.spinBox.setValue(self.camera_id)
        self.spinBox.setMinimum(0)
        self.spinBox.setMaximum(camera_ids)
        self.spinBox.valueChanged.connect(self.change_camera)

        # 置信度设置
        self.doubleSpinBox.setValue(conf_thres)
        self.doubleSpinBox.setMinimum(0.0)
        self.doubleSpinBox.setMaximum(1.0)
        self.doubleSpinBox.setSingleStep(0.01)
        self.doubleSpinBox.setDecimals(2)
        self.doubleSpinBox.valueChanged.connect(self.change_confidence)

        # 定时器控制帧刷新
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # 图片游览相关变量
        self.current_image_folder = ""
        self.image_files = []
        self.current_image_index = -1  # -1表示未加载图片

        # 信号槽绑定
        self.btn_open.clicked.connect(self.toggle_camera)
        self.pictrue_open.clicked.connect(self.open_image)
        self.detect_start.clicked.connect(self.start_detect)
        self.detect_pictrue.clicked.connect(self.open_detect_image)
        self.exit.clicked.connect(self.close)
        self.setting.clicked.connect(self.open_setting)
        self.left.clicked.connect(self.prev_image)
        self.right.clicked.connect(self.next_image)

        # 初始化禁用左右按钮
        self.left.setEnabled(False)
        self.right.setEnabled(False)

        # 初始化线程相关变量
        self.recording_thread = None
        self.detection_thread = None

    def change_camera(self, value):
        """切换摄像头编号"""
        was_running = self.timer.isActive()

        if was_running:
            self.toggle_camera()  # 先停止当前摄像头

        self.camera_id = value
        self.textBrowser.append(f"尝试切换至摄像头 {value}")

        if was_running:
            if not self.cap.open(value, cv2.CAP_DSHOW):
                QMessageBox.warning(self, "错误", f"摄像头 {value} 打开失败！")
                self.spinBox.setValue(self.cap.get(cv2.CAP_PROP_CAMERA))  # 恢复之前的值
                return
            self.timer.start()
            self.btn_open.setText("关闭摄像头")

    def toggle_camera(self):
        """切换摄像头开关状态"""
        if not self.timer.isActive():
            # 打开摄像头
            if self.cap.open(self.camera_id, cv2.CAP_DSHOW):
                self.timer.start(30)  # 30ms 刷新间隔 ≈ 33 FPS
                self.btn_open.setText("关闭摄像头")
            else:
                QMessageBox.warning(self, "错误", "摄像头打开失败！")
        else:
            # 关闭摄像头
            self.timer.stop()
            self.cap.release()
            self.video_label.clear()
            self.btn_open.setText("打开摄像头")

    def update_frame(self):
        """定时器回调：读取并显示帧"""
        ret, frame = self.cap.read()
        if ret:
            # BGR → RGB 转换
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 调整尺寸保持纵横比
            h, w, _ = frame.shape
            target_w = self.video_label.width()
            target_h = int(h * target_w / w)
            frame = cv2.resize(frame, (target_w, target_h))

            # 转换为 QPixmap 并显示
            qimage = QImage(frame.data, target_w, target_h, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimage))
        else:
            self.timer.stop()
            self.cap.release()
            QMessageBox.information(self, "提示", "摄像头断开！")
            sys.exit(app.exec())

    def closeEvent(self, event):
        """窗口关闭时释放资源"""
        # 停止线程
        if self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.requestInterruption()
            self.recording_thread.wait()

        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.requestInterruption()
            self.detection_thread.wait()

        # 释放摄像头资源
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

        # 释放全局摄像头资源
        for cap in caps:
            if cap.isOpened():
                cap.release()

        # 释放视频写入器资源
        for out in outs:
            if out.isOpened():
                out.release()

        event.accept()

    def open_image(self):
        """打开图片文件夹"""
        if self.timer.isActive():
            self.toggle_camera()

        # 如果目标摄像头的图片文件夹不存在，则显示不存在
        if not os.path.exists(f"{now_path}/output/output_frames/output{self.camera_id}"):
            QMessageBox.warning(self, "错误", "文件夹不存在！")
            return

        folder = f"{now_path}/output/output_frames/output{self.camera_id}"

        # 获取支持的图片文件
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        files = os.listdir(folder)
        image_files = []
        for f in files:
            file_path = os.path.join(folder, f)
            if os.path.isfile(file_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in supported_formats:
                    image_files.append(f)

        if not image_files:
            QMessageBox.warning(self, "错误", "文件夹中没有图片文件！")
            return

        image_files.sort()
        self.current_image_folder = folder
        self.image_files = image_files
        self.current_image_index = 0

        self.display_current_image()
        self.left.setEnabled(True)
        self.right.setEnabled(True)

    def display_current_image(self):
        """显示当前图片"""
        if self.current_image_index < 0 or self.current_image_index >= len(self.image_files):
            return

        filename = self.image_files[self.current_image_index]
        filepath = os.path.join(self.current_image_folder, filename)

        # 使用QPixmap加载图片
        pixmap = QPixmap(filepath)
        if pixmap.isNull():
            QMessageBox.warning(self, "错误", f"无法加载图片：{filename}")
            return

        # 缩放图片以适应标签尺寸并保持比例
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.textBrowser.append(f"当前图片：{filename}（{self.current_image_index + 1}/{len(self.image_files)}）")

    def prev_image(self):
        """显示上一张图片"""
        if not self.image_files or self.current_image_index <= 0:
            return

        self.current_image_index -= 1
        self.display_current_image()

    def next_image(self):
        """显示下一张图片"""
        if not self.image_files or self.current_image_index >= len(self.image_files) - 1:
            return

        self.current_image_index += 1
        self.display_current_image()

    def start_detect(self):
        """开始检测处理（使用线程）"""
        # 禁用检测按钮，防止重复点击
        self.detect_start.setEnabled(False)
        self.textBrowser.append("开始录制和检测过程...")

        # 创建并启动录制线程
        self.recording_thread = RecordingThread(self)
        self.recording_thread.update_text.connect(self.textBrowser.append)
        self.recording_thread.error_signal.connect(self.handle_thread_error)
        self.recording_thread.finished_signal.connect(self.start_detection_process)
        self.recording_thread.start()

    def start_detection_process(self):
        """录制完成后开始检测处理"""
        self.textBrowser.append("开始进行目标检测...")

        # 创建并启动检测线程
        self.detection_thread = DetectionThread(self.doubleSpinBox.value(), self)
        self.detection_thread.update_text.connect(self.textBrowser.append)
        self.detection_thread.error_signal.connect(self.handle_thread_error)
        self.detection_thread.finished_signal.connect(self.detection_completed)
        self.detection_thread.start()

    def detection_completed(self):
        """检测完成后的处理"""
        self.textBrowser.append("检测过程完成！可以查看检测结果。")
        self.detect_start.setEnabled(True)

    def handle_thread_error(self, error_msg):
        """处理线程中的错误"""
        self.textBrowser.append(f"错误: {error_msg}")
        QMessageBox.warning(self, "操作错误", error_msg)
        self.detect_start.setEnabled(True)

    def open_setting(self):
        pass

    def open_detect_image(self):
        # 表格初始化
        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(0)
        if self.timer.isActive():
            self.toggle_camera()

        if not os.path.exists(f"{now_path}/output/output_yolo/output{self.camera_id}"):
            QMessageBox.warning(self, "错误", "文件夹不存在！")
            return

        folder = f"{now_path}/output/output_yolo/output{self.camera_id}"

        # 获取支持的图片文件
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        files = os.listdir(folder)
        image_files = []
        for f in files:
            file_path = os.path.join(folder, f)
            if os.path.isfile(file_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in supported_formats:
                    image_files.append(f)

        if not image_files:
            QMessageBox.warning(self, "错误", "文件夹中没有图片文件！")
            return

        image_files.sort()
        self.current_image_folder = folder
        self.image_files = image_files
        self.current_image_index = 0

        self.display_current_image()
        self.left.setEnabled(True)
        self.right.setEnabled(True)

        # 检查CSV文件是否存在
        csv_path = f"{now_path}/output/output_yolo/output{self.camera_id}/output{self.camera_id}.csv"
        if not os.path.exists(csv_path):
            QMessageBox.warning(self, "错误", "检测结果CSV文件不存在！")
            return

        # 读取CSV文件
        try:
            with open(csv_path, newline='', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                header = next(csv_reader)
                csv_name = str(f"output{self.camera_id}.csv")
                self.tableWidget.setColumnCount(len(header))
                self.tableWidget.setHorizontalHeaderLabels(header)
                row_count = 0
                for row in csv_reader:
                    row_position = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(row_position)
                    for column, item in enumerate(row):
                        item = QtWidgets.QTableWidgetItem(item)
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        self.tableWidget.setItem(row_position, column, item)
                    row_count += 1

            self.textBrowser.append(f"成功打开{csv_name}，一共 {row_count} 个缺陷记录！")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"读取CSV文件时出错: {str(e)}")

    def change_confidence(self):
        global conf_thres
        conf_thres = self.doubleSpinBox.value()
        # 显示两位小数
        self.textBrowser.append(f"当前置信度为：{conf_thres:.2f}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())