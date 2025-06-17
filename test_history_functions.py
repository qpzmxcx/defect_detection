#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史记录功能测试脚本
测试 defect_detection.py 中新增的历史记录相关功能
"""

import os
import csv
from datetime import datetime, timedelta


def test_read_history_csv():
    """测试读取历史记录CSV文件"""
    print("=" * 60)
    print("测试读取历史记录CSV文件")
    print("=" * 60)
    
    history_file = 'data/detect_history.csv'
    if not os.path.exists(history_file):
        print("错误：历史记录文件不存在")
        return []
    
    records = []
    try:
        with open(history_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                records.append(row)
        
        print(f"成功读取 {len(records)} 条历史记录")
        
        # 显示记录详情
        for i, record in enumerate(records, 1):
            print(f"\n记录 {i}:")
            print(f"  检测时间: {record.get('timestamp', 'N/A')}")
            print(f"  检测类型: {get_detection_type_display(record.get('detection_type', 'N/A'))}")
            print(f"  车身颜色: {record.get('car_color', 'N/A')}")
            print(f"  划痕数量: {record.get('scratch_count', '0')}")
            print(f"  凹坑数量: {record.get('dent_count', '0')}")
            print(f"  总计数量: {record.get('total_count', '0')}")
            print(f"  置信度阈值: {record.get('conf_threshold', 'N/A')}")
            print(f"  交叉比阈值: {record.get('iou_threshold', 'N/A')}")
            print(f"  结果文件夹: {record.get('timestamp_folder', 'N/A')}")
        
        return records
        
    except Exception as e:
        print(f"读取历史记录时出错：{str(e)}")
        return []


def get_detection_type_display(detection_type):
    """获取检测类型的中文显示名称"""
    type_mapping = {
        'combined_detection': '综合检测',
        'scratch_detection': '划痕检测',
        'dent_detection': '凹坑检测'
    }
    return type_mapping.get(detection_type, detection_type)


def test_filter_history_records(records, start_date, end_date, defect_type):
    """测试过滤历史记录功能"""
    print("\n" + "=" * 60)
    print(f"测试过滤功能：{start_date} 至 {end_date}，缺陷类型：{defect_type}")
    print("=" * 60)
    
    filtered_records = []
    
    try:
        for row in records:
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
                except (ValueError, IndexError):
                    # 如果时间戳格式不正确，跳过这条记录
                    continue
        
        # 按时间戳倒序排列
        filtered_records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        print(f"过滤结果：找到 {len(filtered_records)} 条符合条件的记录")
        
        # 显示过滤后的记录
        for i, record in enumerate(filtered_records, 1):
            print(f"\n过滤结果 {i}:")
            print(f"  检测时间: {record.get('timestamp', 'N/A')}")
            print(f"  检测类型: {get_detection_type_display(record.get('detection_type', 'N/A'))}")
            print(f"  车身颜色: {record.get('car_color', 'N/A')}")
            print(f"  缺陷统计: 划痕{record.get('scratch_count', '0')} | 凹坑{record.get('dent_count', '0')} | 总计{record.get('total_count', '0')}")
            print(f"  结果文件夹: {record.get('timestamp_folder', 'N/A')}")
        
        return filtered_records
        
    except Exception as e:
        print(f"过滤历史记录时出错：{str(e)}")
        return []


def test_check_result_folders(records):
    """测试检查结果文件夹是否存在"""
    print("\n" + "=" * 60)
    print("测试检查结果文件夹")
    print("=" * 60)
    
    for record in records:
        timestamp_folder = record.get('timestamp_folder', '')
        if timestamp_folder:
            result_folder = f"data/car_result/{timestamp_folder}"
            exists = os.path.exists(result_folder)
            status = "✓ 存在" if exists else "✗ 不存在"
            print(f"  {timestamp_folder}: {status}")
            
            if exists:
                # 检查文件夹内容
                try:
                    files = os.listdir(result_folder)
                    csv_files = [f for f in files if f.endswith('.csv')]
                    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    print(f"    - CSV文件: {len(csv_files)} 个")
                    print(f"    - 图片文件: {len(image_files)} 个")
                except Exception as e:
                    print(f"    - 读取文件夹内容时出错: {str(e)}")


def simulate_history_record_click(record):
    """模拟点击历史记录的功能"""
    print("\n" + "=" * 60)
    print("模拟点击历史记录功能")
    print("=" * 60)
    
    timestamp_folder = record.get('timestamp_folder', '')
    
    if not timestamp_folder:
        print("错误：记录中缺少时间戳文件夹信息")
        return False
    
    # 检查文件夹是否存在
    result_folder = f"data/car_result/{timestamp_folder}"
    if not os.path.exists(result_folder):
        print(f"错误：结果文件夹不存在 - {result_folder}")
        return False
    
    print(f"✓ 成功定位到结果文件夹: {result_folder}")
    print("模拟跳转到查看结果界面...")
    print("显示历史记录详细信息:")
    print(f"  检测时间：{record.get('timestamp', 'N/A')}")
    print(f"  检测类型：{get_detection_type_display(record.get('detection_type', 'N/A'))}")
    print(f"  车身颜色：{record.get('car_color', 'N/A')}")
    print(f"  时间戳文件夹：{record.get('timestamp_folder', 'N/A')}")
    print(f"  缺陷统计：")
    print(f"    划痕数量：{record.get('scratch_count', '0')}")
    print(f"    凹坑数量：{record.get('dent_count', '0')}")
    print(f"    总计数量：{record.get('total_count', '0')}")
    print(f"  检测参数：")
    print(f"    置信度阈值：{record.get('conf_threshold', 'N/A')}")
    print(f"    交叉比阈值：{record.get('iou_threshold', 'N/A')}")
    print(f"    模型文件：{record.get('model_file', 'N/A')}")
    
    return True


def main():
    """主测试函数"""
    print("车身缺陷检测系统 - 历史记录功能测试")
    print("=" * 60)
    
    # 1. 测试读取历史记录
    records = test_read_history_csv()
    
    if not records:
        print("没有历史记录可供测试")
        return
    
    # 2. 测试过滤功能
    # 测试不同的过滤条件
    test_cases = [
        ("2025-06-01", "2025-06-30", "所有"),
        ("2025-06-01", "2025-06-30", "划痕"),
        ("2025-06-01", "2025-06-30", "凹坑"),
        ("2025-06-03", "2025-06-03", "所有"),
    ]
    
    for start_date, end_date, defect_type in test_cases:
        filtered_records = test_filter_history_records(records, start_date, end_date, defect_type)
    
    # 3. 测试检查结果文件夹
    test_check_result_folders(records)
    
    # 4. 模拟点击历史记录
    if records:
        print("\n" + "=" * 60)
        print("选择第一条记录进行点击测试")
        simulate_history_record_click(records[0])
    
    print("\n" + "=" * 60)
    print("历史记录功能测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
