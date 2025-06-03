# 缺陷检测历史记录修改说明

## 修改概述

本次修改将 `defect_detection.py` 中的检测历史记录系统从中文改为英文，并增加了对划痕和凹坑缺陷的分别计数功能。

## 主要修改内容

### 1. CSV 文件结构修改

**原始中文字段：**
```
检测时间,时间戳文件夹,总缺陷数,检测类型,模型文件,置信度阈值,IOU阈值,结果路径
```

**新的英文字段：**
```
timestamp,detection_time,timestamp_folder,detection_type,car_color,dent_count,scratch_count,total_count,model_file,conf_threshold,iou_threshold,result_path
```

### 2. 字段说明

| 英文字段名 | 中文含义 | 数据类型 | 说明 |
|-----------|---------|---------|------|
| timestamp | 时间戳 | 字符串 | 格式：YYYYMMDD_HHMMSS |
| detection_time | 检测时间 | 字符串 | 格式：YYYY-MM-DD HH:MM:SS |
| timestamp_folder | 时间戳文件夹 | 字符串 | 检测结果文件夹名称 |
| detection_type | 检测类型 | 字符串 | combined_detection/scratch_detection/dent_detection |
| car_color | 车身颜色 | 字符串 | 自动检测的车身颜色 |
| dent_count | 凹坑数量 | 整数 | 检测到的凹坑缺陷数量 |
| scratch_count | 划痕数量 | 整数 | 检测到的划痕缺陷数量 |
| total_count | 总缺陷数 | 整数 | 总缺陷数量 |
| model_file | 模型文件 | 字符串 | 使用的AI模型文件名 |
| conf_threshold | 置信度阈值 | 浮点数 | 检测置信度阈值 |
| iou_threshold | IOU阈值 | 浮点数 | 交叉比阈值 |
| result_path | 结果路径 | 字符串 | 检测结果存储路径 |

### 3. 检测函数修改

#### 3.1 `detect()` 函数
- **原返回值：** `detect_counter` (总数)
- **新返回值：** `(detect_counter, scratch_counter, dent_counter)` (总数, 划痕数, 凹坑数)
- **功能增强：** 在检测过程中分别统计划痕（class_id=0）和凹坑（class_id=1）

#### 3.2 `scratch_detect()` 和 `dent_detect()` 函数
- **返回值：** 保持不变，分别返回划痕数和凹坑数
- **功能：** 专门检测特定类型的缺陷

### 4. 历史记录保存功能修改

#### 4.1 `save_detection_history()` 函数
- **参数增加：** 新增 `scratch_count` 和 `dent_count` 参数
- **颜色检测：** 自动检测车身颜色并记录
- **英文输出：** 所有输出信息改为英文

#### 4.2 `show_detection_history()` 函数
- **显示格式：** 改为英文格式
- **详细信息：** 显示划痕和凹坑的分别计数

### 5. 主检测流程修改

在 `startDetection()` 方法中：
- 处理 `detect()` 函数的新返回值格式
- 分别显示划痕和凹坑的检测结果
- 传递正确的参数给历史记录保存函数

## 示例数据

### CSV 文件示例
```csv
timestamp,detection_time,timestamp_folder,detection_type,car_color,dent_count,scratch_count,total_count,model_file,conf_threshold,iou_threshold,result_path
20250603_182812,2025-06-03 18:28:12,test_20250603_120000,combined_detection,red,2,3,5,aoxian&huahen.pt,0.5,0.45,data/car_result/test_20250603_120000/
20250603_182812,2025-06-03 18:28:12,test_20250603_120100,scratch_detection,blue,0,4,4,aoxian&huahen.pt,0.25,0.45,data/car_result/test_20250603_120100/
20250603_182812,2025-06-03 18:28:12,test_20250603_120200,dent_detection,white,1,0,1,aoxian&huahen.pt,0.6,0.5,data/car_result/test_20250603_120200/
```

### 检测结果显示示例
```
检测完成！共发现 5 个缺陷
其中：划痕 3 个，凹坑 2 个
Detection history saved to: data/detect_history.csv
Record: 5 total defects (3 scratches, 2 dents)
```

## 兼容性说明

1. **向后兼容：** 旧的CSV文件会被自动替换为新格式
2. **数据迁移：** 建议备份现有的检测历史记录
3. **界面显示：** 历史记录查看功能已更新为英文显示

## 测试验证

已通过 `test_csv_only.py` 脚本验证：
- ✓ CSV文件创建和表头正确性
- ✓ 历史记录保存功能
- ✓ 数据完整性检查
- ✓ 分别计数功能

## 使用说明

1. **自动创建：** 首次运行时会自动创建新格式的CSV文件
2. **分别计数：** 系统会自动统计划痕和凹坑的数量
3. **颜色检测：** 自动检测并记录车身颜色
4. **英文记录：** 所有记录都使用英文字段名和值

## 注意事项

1. 确保 `data` 文件夹有写入权限
2. 颜色检测需要有效的图片文件
3. 检测类型会根据用户选择自动设置
4. 所有数值字段都会进行有效性验证
