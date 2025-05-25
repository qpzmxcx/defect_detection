import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices
from PyQt6.QtMultimediaWidgets import QVideoWidget

app = QApplication(sys.argv)

# 1. 创建并显示视频输出控件
video_widget = QVideoWidget()
video_widget.resize(640, 480)
video_widget.show()

# 2. 选择系统默认摄像头并启动
camera = QCamera(QMediaDevices.defaultVideoInput())
camera.start()

# 3. 建立媒体会话并绑定摄像头与输出
capture_session = QMediaCaptureSession()
capture_session.setCamera(camera)
capture_session.setVideoOutput(video_widget)

sys.exit(app.exec())
