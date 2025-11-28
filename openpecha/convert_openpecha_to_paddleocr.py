"""
将 openpecha parquet 数据集转换为 PaddleOCR 训练格式

使用方法:
    python convert_openpecha_to_paddleocr.py --input_dir ./data --images_dirs ./ocr_benchmark_images ./benchmark_images_2 --output_dir ./paddleocr_data
"""

import os
import sys
import argparse
import pandas as pd
from PIL import Image
import shutil
from tqdm import tqdm
import glob
import random

# 数据集名称映射：parquet 文件名 -> 图像目录名
DATASET_NAME_MAPPING = {
    'Norbuketaka': 'OCR-Norbuketaka',
    'Lithang_Kanjur': 'OCR-Lithangkanjur',
    'Lhasa_Kanjur': 'OCR-Lhasakanjur',
    'Betsug': 'OCR-Betsug',
    'Google_Books': 'OCR-Google_Books',
    'Karmapa8': 'OCR-Karmapa8',
    'Drutsa': 'OCR-Drutsa',
    'KhyentseWangpo': 'OCR-KhyentseWangpo',
    'Derge_Tenjur': 'OCR-Dergetenjur',
    'NorbukatekaNumbers': 'OCR-NorbuketakaNumbers',
}

def find_image_file(image_dirs, image_subdir, filename):
    """
    在多个图像根目录中查找图像文件（支持 .jpg, .tif, .png 等格式）
    
    Args:
        image_dirs: 图像根目录列表（例如：['./ocr_benchmark_images', './benchmark_images_2']）
        image_subdir: 图像子目录名（例如：'OCR-Norbuketaka'）
        filename: 文件名（不含扩展名）
    
    Returns:
        找到的图像文件路径，如果未找到返回 None
    """
    # 支持的图像扩展名
    extensions = ['.jpg', '.jpeg', '.tif', '.tiff', '.png']
    
    # 在所有图像根目录中搜索
    for image_root_dir in image_dirs:
        image_dir = os.path.join(image_root_dir, image_subdir)
        
        if not os.path.exists(image_dir):
            continue
        
        # 先尝试精确匹配
        for ext in extensions:
            img_path = os.path.join(image_dir, f"{filename}{ext}")
            if os.path.exists(img_path):
                return img_path
        
        # 如果精确匹配失败，尝试模糊匹配
        for ext in extensions:
            pattern = os.path.join(image_dir, f"{filename}*{ext}")
            matches = glob.glob(pattern, recursive=False)
            if matches:
                return matches[0]
    
    return None

def get_dataset_name_from_parquet(parquet_file):
    """
    从 parquet 文件名提取数据集名称
    
    Args:
        parquet_file: parquet 文件路径
    
    Returns:
        数据集名称
    """
    basename = os.path.basename(parquet_file)
    # 例如: Norbuketaka-00000-of-00001.parquet -> Norbuketaka
    dataset_name = basename.split('-')[0]
    return dataset_name

def convert_openpecha_to_paddleocr(input_dir, images_dirs, output_dir, train_ratio=0.9):
    """
    将 openpecha parquet 文件转换为 PaddleOCR 训练格式
    
    Args:
        input_dir: parquet 文件所在目录
        images_dirs: 图像文件所在根目录列表（例如：['./ocr_benchmark_images', './benchmark_images_2']）
        output_dir: 输出目录
        train_ratio: 训练集比例
    """
    # 确保 images_dirs 是列表
    if isinstance(images_dirs, str):
        images_dirs = [images_dirs]
    
    print(f"使用图像目录: {images_dirs}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    
    # 查找所有 parquet 文件
    parquet_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.parquet'):
            parquet_files.append(os.path.join(input_dir, file))
    
    parquet_files.sort()  # 确保顺序一致
    
    print(f"找到 {len(parquet_files)} 个 parquet 文件")
    
    # 读取所有 parquet 文件
    all_data = []
    total_processed = 0
    total_skipped = 0
    
    for parquet_file in parquet_files:
        print(f"\n正在处理: {os.path.basename(parquet_file)}")
        df = pd.read_parquet(parquet_file)
        print(f"  包含 {len(df)} 条数据")
        
        # 获取数据集名称并映射到图像目录
        dataset_name = get_dataset_name_from_parquet(parquet_file)
        image_subdir = DATASET_NAME_MAPPING.get(dataset_name)
        
        if image_subdir is None:
            print(f"  警告: 未找到数据集 '{dataset_name}' 的图像目录映射，跳过")
            continue
        
        # 检查至少一个图像目录存在
        found_image_dir = None
        for img_root_dir in images_dirs:
            image_dir = os.path.join(img_root_dir, image_subdir)
            if os.path.exists(image_dir):
                found_image_dir = image_dir
                break
        
        if found_image_dir is None:
            print(f"  警告: 在所有图像目录中都未找到子目录 '{image_subdir}'，跳过")
            continue
        
        print(f"  使用图像目录: {found_image_dir}")
        
        # 处理每一行数据
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {dataset_name}"):
            try:
                # 获取文件名和标签
                filename = str(row['filename'])
                label = str(row['label'])
                
                # 在所有图像目录中查找图像文件
                img_path = find_image_file(images_dirs, image_subdir, filename)
                
                if img_path is None:
                    total_skipped += 1
                    if total_skipped <= 10:  # 只打印前10个警告
                        print(f"\n警告: 未找到图像文件: {filename} (在 {image_subdir})")
                    continue
                
                # 生成输出图像文件名（使用数据集名和原始文件名避免冲突）
                output_img_filename = f"{dataset_name}_{filename}.png"
                output_img_path = os.path.join(output_images_dir, output_img_filename)
                
                # 如果文件已存在，跳过（避免重复处理）
                if os.path.exists(output_img_path):
                    # 检查是否已经在 all_data 中
                    existing_id = f"{dataset_name}_{filename}"
                    if not any(item['id'] == existing_id for item in all_data):
                        all_data.append({
                            'image_path': f"images/{output_img_filename}",
                            'transcription': label,
                            'id': existing_id
                        })
                        total_processed += 1
                    continue
                
                # 复制并转换图像为 PNG 格式
                try:
                    img = Image.open(img_path)
                    # 如果是 RGBA 模式，转换为 RGB
                    if img.mode == 'RGBA':
                        # 创建白色背景
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[3])  # 使用 alpha 通道作为 mask
                        img = rgb_img
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img.save(output_img_path, 'PNG')
                except Exception as e:
                    print(f"\n警告: 无法处理图像 {img_path}: {e}")
                    total_skipped += 1
                    continue
                
                # 保存数据信息
                all_data.append({
                    'image_path': f"images/{output_img_filename}",
                    'transcription': label,
                    'id': f"{dataset_name}_{filename}"
                })
                
                total_processed += 1
                
            except Exception as e:
                print(f"\n处理第 {idx} 行时出错: {e}")
                total_skipped += 1
                continue
    
    print(f"\n总共处理了 {total_processed} 条数据")
    print(f"跳过了 {total_skipped} 条数据")
    
    if len(all_data) == 0:
        print("错误: 没有成功处理任何数据!")
        return
    
    # 随机打乱数据，确保训练集和验证集分布均匀
    print(f"\n正在打乱数据...")
    random.seed(42)  # 设置随机种子，确保结果可复现
    random.shuffle(all_data)
    
    # 分割训练集和验证集
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"\n训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    
    # 写入训练集文件
    train_file = os.path.join(output_dir, "train.txt")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            # PaddleOCR 格式: 图像路径\t标签
            f.write(f"{item['image_path']}\t{item['transcription']}\n")
    
    # 写入验证集文件
    val_file = os.path.join(output_dir, "val.txt")
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(f"{item['image_path']}\t{item['transcription']}\n")
    
    print(f"\n转换完成!")
    print(f"训练集文件: {train_file}")
    print(f"验证集文件: {val_file}")
    print(f"图像目录: {output_images_dir}")
    print(f"\n在配置文件中使用:")
    print(f"  data_dir: {os.path.abspath(output_dir)}")
    print(f"  label_file_list:")
    print(f"    - {os.path.abspath(train_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将 openpecha parquet 数据集转换为 PaddleOCR 格式')
    parser.add_argument('--input_dir', type=str, default='./data',
                        help='parquet 文件所在目录')
    parser.add_argument('--images_dirs', type=str, nargs='+', 
                        default=['./ocr_benchmark_images', './benchmark_images_2'],
                        help='图像文件所在根目录列表（可指定多个）')
    parser.add_argument('--output_dir', type=str, default='./paddleocr_data',
                        help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='训练集比例 (默认: 0.9)')
    
    args = parser.parse_args()
    
    convert_openpecha_to_paddleocr(args.input_dir, args.images_dirs, args.output_dir, args.train_ratio)

