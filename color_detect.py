import cv2
import numpy as np


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


if __name__ == "__main__":
    image_path = "data/frame_1.jpg"
    color = detect_color(image_path)
    print(f"The main color detected is: {color}")