"""
将 openpecha_line_segmentation parquet 数据集转换为 PaddleOCR 检测训练格式

使用方法:
    python convert_openpecha_to_paddleocr_det.py --input_dir ./data --output_dir ./paddleocr_data --source_images_dir ./source_images --visualize_samples 5

参数说明:
    --input_dir: parquet 文件所在目录 (默认: ./data)
    --output_dir: 输出目录 (默认: ./paddleocr_data)
    --source_images_dir: 源图片目录 (默认: ./source_images)
    --train_ratio: 训练集比例 (默认: 0.9)
    --visualize_samples: 要可视化的样本数量，用于验证转换结果 (默认: 0，不可视化)

输出:
    - images/ 目录: 复制的图片文件
    - train.txt: 训练集标注文件
    - val.txt: 验证集标注文件
    - visualizations/ 目录: 可视化结果 (如果指定了--visualize_samples)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import json
from shapely.geometry import Polygon
import shutil
import random


def visualize_detections(image_path, annotations, output_path):
    """
    在图片上可视化检测结果

    Args:
        image_path: 图片路径
        annotations: 标注列表
        output_path: 输出图片路径
    """
    # 打开图片
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # 为每个标注绘制四边形
    for annotation in annotations:
        points = annotation['points']
        if len(points) == 4:
            # 将点连接成多边形
            polygon_points = []
            for point in points:
                polygon_points.extend(point)

            # 绘制多边形
            draw.polygon(polygon_points, outline='red', width=2)

            # 添加文本标签
            transcription = annotation.get('transcription', '')
            if transcription and transcription != '###':
                # 在第一个点附近添加文本
                text_x, text_y = points[0]
                draw.text((text_x, text_y-10), transcription, fill='red')

    # 保存结果
    img.save(output_path)
    print(f"可视化结果已保存: {output_path}")


def polygon_to_bbox(coordinates):
    """
    将多边形坐标转换为边界框

    Args:
        coordinates: list of [x, y] - 多边形顶点坐标

    Returns:
        list: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] - 四边形四个顶点坐标
    """
    coords = np.array(coordinates)

    if len(coords) == 4:
        # 正好是四边形，直接返回
        return coords.tolist()
    elif len(coords) < 4:
        # 点数太少，无法形成四边形
        return []
    else:
        # 对于文本行，使用轴对齐的边界框而不是最小外接矩形
        # 这样可以避免过长的框
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)

        # 返回轴对齐的四边形
        bbox = [
            [int(min_x), int(min_y)],
            [int(max_x), int(min_y)],
            [int(max_x), int(max_y)],
            [int(min_x), int(max_y)]
        ]
        return bbox


def convert_openpecha_to_paddleocr_det(input_dir, output_dir, source_images_dir, train_ratio=0.9, visualize_samples=0):
    """
    将 openpecha parquet 文件转换为 PaddleOCR 检测训练格式

    Args:
        input_dir: parquet 文件所在目录
        output_dir: 输出目录
        source_images_dir: 源图片目录
        train_ratio: 训练集比例
        visualize_samples: 要可视化的样本数量 (0表示不可视化)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有 parquet 文件
    parquet_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.parquet'):
            parquet_files.append(os.path.join(input_dir, file))

    parquet_files.sort()  # 确保顺序一致

    print(f"找到 {len(parquet_files)} 个 parquet 文件")

    # 读取所有 parquet 文件
    all_data = []

    for parquet_file in parquet_files:
        print(f"正在读取: {parquet_file}")
        df = pd.read_parquet(parquet_file)
        print(f"  包含 {len(df)} 条数据")

        # 检查列名
        print(f"  列名: {df.columns.tolist()}")

        # 根据parquet文件名确定图片子目录
        if 'without_superscript_subscript' in parquet_file:
            image_subdir = 'without_superscript_subscript'
        elif 'with_superscript_subscript' in parquet_file:
            image_subdir = 'with_superscript_subscript'
        else:
            image_subdir = ''

        # 处理每一行数据
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {os.path.basename(parquet_file)}"):
            try:
                # 获取数据
                line_id = row['line_id']
                line_coordinates = row['line_coordinates']
                source_image = row['source_image']
                image_size = row['image_size']

                # 解析图片尺寸
                if 'x' in image_size:
                    width, height = map(int, image_size.split('x'))
                else:
                    print(f"警告: 无法解析图片尺寸: {image_size}")
                    continue

                # 构建完整的图片路径
                full_image_path = os.path.join(source_images_dir, image_subdir, source_image)

                # 检查源图片是否存在
                if not os.path.exists(full_image_path):
                    print(f"警告: 源图片不存在: {full_image_path}")
                    continue

                # 处理坐标数据
                # line_coordinates 是一个数组的序列，需要转换为numpy数组
                coords_list = []
                for coord in line_coordinates:
                    if hasattr(coord, '__iter__') and len(coord) == 2:
                        coords_list.append([float(coord[0]), float(coord[1])])

                if len(coords_list) < 3:
                    print(f"警告: 坐标点数太少: {len(coords_list)}")
                    continue

                # 转换为四边形坐标
                bbox_points = polygon_to_bbox(coords_list)
                if not bbox_points:
                    continue

                # 创建标注对象
                annotation = {
                    "transcription": "###",  # 检测任务中通常用###表示文本区域
                    "points": bbox_points
                }

                # 构建相对路径（相对于source_images_dir）
                relative_image_path = os.path.join(image_subdir, source_image) if image_subdir else source_image

                # 检查是否已存在该图片的标注
                existing_entry = None
                for entry in all_data:
                    if entry['image_path'] == relative_image_path:
                        existing_entry = entry
                        break

                if existing_entry:
                    # 添加到现有图片的标注列表
                    existing_entry['annotations'].append(annotation)
                else:
                    # 创建新条目
                    all_data.append({
                        'image_path': relative_image_path,
                        'full_image_path': full_image_path,  # 保存完整路径用于可视化
                        'annotations': [annotation],
                        'image_size': (width, height)
                    })

            except Exception as e:
                print(f"处理第 {idx} 行时出错: {e}")
                continue

    print(f"\n总共处理了 {len(all_data)} 张图片")

    # 分割训练集和验证集
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    print(f"训练集: {len(train_data)} 张图片")
    print(f"验证集: {len(val_data)} 张图片")

    # 写入训练集文件
    train_file = os.path.join(output_dir, "train.txt")
    with open(train_file, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(train_data, 1):
            # PaddleOCR 检测格式: 图像路径\t[标注JSON数组]
            annotations_json = json.dumps(item['annotations'], ensure_ascii=False)
            f.write(f"{item['image_path']}\t{annotations_json}\n")

    # 写入验证集文件
    val_file = os.path.join(output_dir, "val.txt")
    with open(val_file, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(val_data, 1):
            annotations_json = json.dumps(item['annotations'], ensure_ascii=False)
            f.write(f"{item['image_path']}\t{annotations_json}\n")

    # 可视化样本
    if visualize_samples > 0:
        print(f"\n开始可视化 {visualize_samples} 个样本...")

        # 创建可视化目录
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # 从训练集中随机选择样本进行可视化
        samples_to_visualize = random.sample(train_data, min(visualize_samples, len(train_data)))

        for i, sample in enumerate(samples_to_visualize):
            image_path = sample['full_image_path']  # 使用完整路径
            annotations = sample['annotations']
            output_vis_path = os.path.join(vis_dir, f"sample_{i+1}_{os.path.basename(sample['image_path'])}")

            try:
                visualize_detections(image_path, annotations, output_vis_path)
            except Exception as e:
                print(f"可视化样本 {i+1} 失败: {e}")

        print(f"可视化完成! 结果保存在: {vis_dir}")

    print(f"\n转换完成!")
    print(f"训练集文件: {train_file}")
    print(f"验证集文件: {val_file}")
    print(f"图片目录: {source_images_dir}")
    if visualize_samples > 0:
        print(f"可视化目录: {os.path.join(output_dir, 'visualizations')}")
    print(f"\n在配置文件中使用:")
    print(f"  data_dir: {os.path.abspath(source_images_dir)}")
    print(f"  label_file_list:")
    print(f"    - {os.path.abspath(train_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将 openpecha_line_segmentation parquet 数据集转换为 PaddleOCR 检测格式')
    parser.add_argument('--input_dir', type=str, default='./data',
                        help='parquet 文件所在目录')
    parser.add_argument('--output_dir', type=str, default='./paddleocr_data',
                        help='输出目录')
    parser.add_argument('--source_images_dir', type=str, default='./source_images',
                        help='源图片目录')
    parser.add_argument('--train_ratio', type=float, default=0.95,
                        help='训练集比例 (默认: 0.95)')
    parser.add_argument('--visualize_samples', type=int, default=0,
                        help='要可视化的样本数量 (默认: 0, 表示不可视化)')

    args = parser.parse_args()

    convert_openpecha_to_paddleocr_det(args.input_dir, args.output_dir, args.source_images_dir, args.train_ratio, args.visualize_samples)
