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
from multiprocessing import Pool, cpu_count
from functools import partial

# Windows 兼容性：设置多进程启动方法
if sys.platform == 'win32':
    import multiprocessing
    multiprocessing.freeze_support()

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

def process_single_image(task_info):
    """
    处理单个图像转换任务（用于多进程）
    
    Args:
        task_info: 包含任务信息的字典
            - images_dirs: 图像根目录列表
            - image_subdir: 图像子目录名
            - filename: 文件名
            - label: 文本标签
            - dataset_name: 数据集名称
            - output_images_dir: 输出图像目录
    
    Returns:
        成功时返回数据字典，失败时返回 None
    """
    images_dirs = task_info['images_dirs']
    image_subdir = task_info['image_subdir']
    filename = task_info['filename']
    label = task_info['label']
    dataset_name = task_info['dataset_name']
    output_images_dir = task_info['output_images_dir']
    
    try:
        # 在所有图像目录中查找图像文件
        img_path = find_image_file(images_dirs, image_subdir, filename)
        
        if img_path is None:
            return None
        
        # 生成输出图像文件名（使用数据集名和原始文件名避免冲突）
        output_img_filename = f"{dataset_name}_{filename}.png"
        output_img_path = os.path.join(output_images_dir, output_img_filename)
        
        # 如果文件已存在，直接返回数据信息（避免重复处理）
        if os.path.exists(output_img_path):
            return {
                'image_path': f"images/{output_img_filename}",
                'transcription': label,
                'id': f"{dataset_name}_{filename}"
            }
        
        # 复制并转换图像为 PNG 格式
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
        
        # 返回数据信息
        return {
            'image_path': f"images/{output_img_filename}",
            'transcription': label,
            'id': f"{dataset_name}_{filename}"
        }
    except Exception as e:
        # 静默处理错误，返回 None
        return None

def convert_openpecha_to_paddleocr(input_dir, images_dirs, output_dir, train_ratio=0.9, num_workers=None):
    """
    将 openpecha parquet 文件转换为 PaddleOCR 训练格式（使用多进程加速）
    
    Args:
        input_dir: parquet 文件所在目录
        images_dirs: 图像文件所在根目录列表（例如：['./ocr_benchmark_images', './benchmark_images_2']）
        output_dir: 输出目录
        train_ratio: 训练集比例
        num_workers: 进程数，默认为 CPU 核心数
    """
    # 确保 images_dirs 是列表
    if isinstance(images_dirs, str):
        images_dirs = [images_dirs]
    
    # 设置进程数
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"使用图像目录: {images_dirs}")
    print(f"使用 {num_workers} 个进程进行并行处理")
    
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
    
    # 第一步：读取所有 parquet 文件，收集所有任务
    print("\n第一步：读取所有 parquet 文件，收集任务...")
    all_tasks = []
    
    for parquet_file in parquet_files:
        print(f"  读取: {os.path.basename(parquet_file)}")
        df = pd.read_parquet(parquet_file)
        
        # 获取数据集名称并映射到图像目录
        dataset_name = get_dataset_name_from_parquet(parquet_file)
        image_subdir = DATASET_NAME_MAPPING.get(dataset_name)
        
        if image_subdir is None:
            print(f"    警告: 未找到数据集 '{dataset_name}' 的图像目录映射，跳过")
            continue
        
        # 检查至少一个图像目录存在
        found_image_dir = None
        for img_root_dir in images_dirs:
            image_dir = os.path.join(img_root_dir, image_subdir)
            if os.path.exists(image_dir):
                found_image_dir = image_dir
                break
        
        if found_image_dir is None:
            print(f"    警告: 在所有图像目录中都未找到子目录 '{image_subdir}'，跳过")
            continue
        
        print(f"    包含 {len(df)} 条数据，使用图像目录: {found_image_dir}")
        
        # 收集所有任务
        for idx, row in df.iterrows():
            filename = str(row['filename'])
            label = str(row['label'])
            
            task_info = {
                'images_dirs': images_dirs,
                'image_subdir': image_subdir,
                'filename': filename,
                'label': label,
                'dataset_name': dataset_name,
                'output_images_dir': output_images_dir
            }
            all_tasks.append(task_info)
    
    print(f"\n总共收集了 {len(all_tasks)} 个任务")
    
    # 第二步：使用多进程并行处理所有任务
    print(f"\n第二步：使用 {num_workers} 个进程并行处理图像转换...")
    all_data = []
    total_skipped = 0
    
    with Pool(processes=num_workers) as pool:
        # 使用 tqdm 显示进度
        results = list(tqdm(
            pool.imap(process_single_image, all_tasks),
            total=len(all_tasks),
            desc="处理图像"
        ))
    
    # 收集成功的结果
    for result in results:
        if result is not None:
            all_data.append(result)
        else:
            total_skipped += 1
    
    print(f"\n总共处理了 {len(all_data)} 条数据")
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
    parser.add_argument('--train_ratio', type=float, default=0.95,
                        help='训练集比例 (默认: 0.9)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='并行处理的进程数 (默认: CPU核心数)')
    
    args = parser.parse_args()
    
    convert_openpecha_to_paddleocr(args.input_dir, args.images_dirs, args.output_dir, args.train_ratio, args.num_workers)

