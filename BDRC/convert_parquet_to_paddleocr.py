"""
将 BDRC parquet 数据集转换为 PaddleOCR 训练格式

使用方法:
    python convert_parquet_to_paddleocr.py --input_dir ./data --output_dir ./paddleocr_data
"""

import os
import sys
import argparse
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm

def convert_parquet_to_paddleocr(input_dir, output_dir, train_ratio=0.9):
    """
    将 parquet 文件转换为 PaddleOCR 训练格式
    
    Args:
        input_dir: parquet 文件所在目录
        output_dir: 输出目录
        train_ratio: 训练集比例
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
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
        
        # 处理每一行数据
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {os.path.basename(parquet_file)}"):
            try:
                # 获取图像和文本
                image = row['line']  # 根据 README，图像列名为 'line'
                transcription = row['transcription']
                data_id = row.get('id', f"{os.path.basename(parquet_file)}_{idx}")
                
                # 处理图像数据
                img = None
                
                # 如果 image 是字典格式（包含 bytes 和 path）
                if isinstance(image, dict):
                    if 'bytes' in image:
                        img_bytes = image['bytes']
                        img = Image.open(io.BytesIO(img_bytes))
                    elif 'path' in image:
                        # 如果只有路径，尝试读取文件
                        img_path = image['path']
                        if os.path.exists(img_path):
                            img = Image.open(img_path)
                        else:
                            print(f"警告: 图像路径不存在: {img_path}")
                            continue
                    else:
                        print(f"警告: 字典格式的图像数据缺少 'bytes' 或 'path' 键")
                        continue
                # 如果 image 是 PIL Image 对象
                elif hasattr(image, 'save'):
                    img = image
                # 如果 image 是字节数据
                elif isinstance(image, bytes):
                    img = Image.open(io.BytesIO(image))
                # 如果 image 是 numpy 数组
                elif hasattr(image, 'shape'):
                    img = Image.fromarray(image)
                else:
                    print(f"警告: 无法处理图像类型 {type(image)}, 跳过")
                    continue
                
                # 保存图像
                img_filename = f"{data_id}.jpg"
                img_path = os.path.join(images_dir, img_filename)
                img.save(img_path, 'JPEG')
                
                # 保存数据信息
                all_data.append({
                    'image_path': f"images/{img_filename}",
                    'transcription': transcription,
                    'id': data_id
                })
            except Exception as e:
                print(f"处理第 {idx} 行时出错: {e}")
                continue
    
    print(f"\n总共处理了 {len(all_data)} 条数据")
    
    # 分割训练集和验证集
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"训练集: {len(train_data)} 条")
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
    print(f"图像目录: {images_dir}")
    print(f"\n在配置文件中使用:")
    print(f"  data_dir: {os.path.abspath(output_dir)}")
    print(f"  label_file_list:")
    print(f"    - {os.path.abspath(train_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将 BDRC parquet 数据集转换为 PaddleOCR 格式')
    parser.add_argument('--input_dir', type=str, default='./data',
                        help='parquet 文件所在目录')
    parser.add_argument('--output_dir', type=str, default='./paddleocr_data',
                        help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='训练集比例 (默认: 0.9)')
    
    args = parser.parse_args()
    
    convert_parquet_to_paddleocr(args.input_dir, args.output_dir, args.train_ratio)

