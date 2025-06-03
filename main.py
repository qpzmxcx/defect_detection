from defect_detection import DefectDetectionApp
from PyQt6 import QtWidgets
import sys

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DefectDetectionApp()
    window.show()
    sys.exit(app.exec())
