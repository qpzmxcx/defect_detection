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

if __name__ == '__main__':
    with torch.no_grad():
        dent_num=dent_detect()
    print(dent_num)
    # with torch.no_grad():
    #     scratch_num=scratch_detect()
    # print(scratch_num)
    # with torch.no_grad():
    #     detect_num=detect()
    # print(detect_num)