import sys
import os
import csv
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QLabel, QLineEdit,
                             QComboBox, QPushButton, QScrollArea, QFrame,
                             QDateEdit, QGroupBox)
from PyQt6.QtCore import Qt, QDate
from PyQt6.QtGui import QFont, QIcon, QPixmap


class DefectDetectionViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("车辆缺陷检测系统")
        self.setGeometry(100, 100, 1200, 800)

        # 创建主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 创建主布局
        self.main_layout = QVBoxLayout(self.central_widget)

        # 创建顶部搜索区域
        self.create_search_section()

        # 创建滚动区域显示检测历史
        self.create_history_section()

        # 加载CSV数据
        self.load_csv_data("data/detect_history.csv")

    def create_search_section(self):
        """创建顶部搜索区域"""
        search_frame = QFrame()
        search_frame.setFrameShape(QFrame.Shape.StyledPanel)
        search_frame.setStyleSheet("background-color: #f0f0f0;")

        search_layout = QGridLayout(search_frame)

        # 扫描日期区间
        search_layout.addWidget(QLabel("Scanning Date:"), 0, 0)

        date_layout = QHBoxLayout()

        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate())
        self.start_date.setCalendarPopup(True)
        date_layout.addWidget(self.start_date)

        date_layout.addWidget(QLabel("→"))

        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        date_layout.addWidget(self.end_date)

        search_layout.addLayout(date_layout, 0, 1)

        # 缺陷类型
        search_layout.addWidget(QLabel("Defect:"), 0, 2)
        self.defect_combo = QComboBox()
        self.defect_combo.addItems(["Select Defect State", "Dent", "Scratch", "Both", "None"])
        self.defect_combo.setMinimumWidth(150)
        search_layout.addWidget(self.defect_combo, 0, 3)

        # 扫描状态
        search_layout.addWidget(QLabel("Scan Status:"), 0, 4)
        self.status_combo = QComboBox()
        self.status_combo.addItems(["Scan Succeed", "Scan Failed"])
        self.status_combo.setMinimumWidth(150)
        search_layout.addWidget(self.status_combo, 0, 5)

        # VIN码
        search_layout.addWidget(QLabel("VIN:"), 0, 6)
        self.vin_edit = QLineEdit()
        self.vin_edit.setMinimumWidth(150)
        search_layout.addWidget(self.vin_edit, 0, 7)

        # 搜索按钮
        search_button = QPushButton("Search")
        search_button.setStyleSheet("background-color: #4CAF50; color: white;")
        search_layout.addWidget(search_button, 0, 8)

        # 添加到主布局
        self.main_layout.addWidget(search_frame)

    def create_history_section(self):
        """创建历史记录滚动区域"""
        # 创建滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: white;")

        # 创建滚动区域的内容窗口
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setSpacing(10)

        # 设置滚动区域的内容窗口
        self.scroll_area.setWidget(self.scroll_content)

        # 添加到主布局
        self.main_layout.addWidget(self.scroll_area)

    def load_csv_data(self, csv_file_path):
        """从CSV文件加载数据"""
        try:
            with open(csv_file_path, 'r') as file:
                # 跳过标题行
                reader = csv.DictReader(file)

                # 清除现有内容
                while self.scroll_layout.count():
                    item = self.scroll_layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()

                # 为每一行添加一个记录部件
                for row in reader:
                    self.add_record_widget(row)

                # 添加一个弹性空间在最后
                self.scroll_layout.addStretch()

        except Exception as e:
            print(f"Error loading CSV data: {e}")

    def add_record_widget(self, data):
        """添加一个检测记录部件"""
        # 创建记录框架
        record_frame = QFrame()
        record_frame.setFrameShape(QFrame.Shape.StyledPanel)
        record_frame.setStyleSheet("background-color: #f9f9f9; border: 1px solid #ddd;")

        # 设置记录内容的布局
        record_layout = QHBoxLayout(record_frame)

        # 添加车辆图片
        image_label = QLabel()
        placeholder_image = QPixmap("images/car_placeholder.png")  # 替换为实际的图片路径或使用占位图
        if not placeholder_image.isNull():
            image_label.setPixmap(placeholder_image.scaled(150, 100, Qt.AspectRatioMode.KeepAspectRatio))
        else:
            # 如果没有图片，显示一个占位符
            image_label.setText("Car Image")
            image_label.setStyleSheet("background-color: #ddd; min-width: 150px; min-height: 100px;")

        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        record_layout.addWidget(image_label)

        # 添加损伤信息
        damage_info = QGroupBox("Damages Found")
        damage_layout = QVBoxLayout(damage_info)

        # 订单号
        order_layout = QHBoxLayout()
        order_layout.addWidget(QLabel("Order No."))
        order_number = QLabel(data['timestamp'])
        order_number.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        order_layout.addWidget(order_number)
        damage_layout.addLayout(order_layout)

        # VIN号
        vin_layout = QHBoxLayout()
        vin_layout.addWidget(QLabel("VIN"))
        vin_number = QLabel("-")
        vin_layout.addWidget(vin_number)
        damage_layout.addLayout(vin_layout)

        record_layout.addWidget(damage_info)

        # 添加扫描时间信息
        time_info = QGroupBox("Scanning Time")
        time_layout = QVBoxLayout(time_info)

        scan_time = QLabel(data['detection_time'])
        scan_time.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        time_layout.addWidget(scan_time)

        # 添加工作模式
        scratch_mode = QLabel("Scratch Working Mode")
        time_layout.addWidget(scratch_mode)

        scratch_state = QLabel("● Off" if data['scratch_count'] == '0' else "● On")
        scratch_state.setStyleSheet("color: red;" if data['scratch_count'] == '0' else "color: green;")
        time_layout.addWidget(scratch_state)

        record_layout.addWidget(time_info)

        # 添加车牌信息
        plate_info = QGroupBox("License Plate")
        plate_layout = QVBoxLayout(plate_info)

        plate_number = QLabel(f"Unknown-{data['timestamp'][-4:]}")  # 使用时间戳的最后4位作为示例
        plate_number.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        plate_layout.addWidget(plate_number)

        # 添加工作模式
        dent_mode = QLabel("Dent Working Mode")
        plate_layout.addWidget(dent_mode)

        dent_state = QLabel("● Standard")
        dent_state.setStyleSheet("color: blue;")
        plate_layout.addWidget(dent_state)

        record_layout.addWidget(plate_info)

        # 添加车型信息
        type_info = QGroupBox("Car Type")
        type_layout = QVBoxLayout(type_info)

        car_type = QLabel("Sedan")  # 假设都是轿车
        car_type.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        type_layout.addWidget(car_type)

        record_layout.addWidget(type_info)

        # 添加颜色信息
        color_info = QGroupBox("Car Color")
        color_layout = QVBoxLayout(color_info)

        car_color = QLabel(data['car_color'].capitalize())
        car_color.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        color_layout.addWidget(car_color)

        record_layout.addWidget(color_info)

        # 调整各部件的大小策略
        record_layout.setStretch(0, 2)  # 图片
        record_layout.setStretch(1, 2)  # 损伤信息
        record_layout.setStretch(2, 2)  # 扫描时间
        record_layout.setStretch(3, 2)  # 车牌
        record_layout.setStretch(4, 1)  # 车型
        record_layout.setStretch(5, 1)  # 颜色

        # 添加到滚动区域
        self.scroll_layout.addWidget(record_frame)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle("Fusion")

    window = DefectDetectionViewer()
    window.show()

    sys.exit(app.exec())