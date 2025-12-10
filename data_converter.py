import os
import cv2
import json
import numpy as np
import math
from pathlib import Path
import random
from multiprocessing import Pool, cpu_count

# 配置路径
SRC_ROOT = Path(r"pdf_ocr/pdfs")
DST_ROOT = Path(r"ocr_rec_dataset_output")
LANGUAGES = ['en', 'zh']
VAL_RATIO = 0.05  # 5% 用于验证
MAX_IMAGES_PER_BOOK = None  # Set to None for full processing

def get_rotate_crop_image(img, points):
    # 使用外接矩形裁剪，不进行透视变换和旋转
    img_height, img_width = img.shape[0:2]
    
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    
    # 边界检查，防止越界
    left = max(0, left)
    top = max(0, top)
    right = min(img_width, right)
    bottom = min(img_height, bottom)
    
    img_crop = img[top:bottom, left:right, :].copy()
    return img_crop

def process_book(args):
    """
    处理单本书籍的函数，用于多进程调用
    """
    book_dir, lang, images_dir, val_ratio, max_images_per_book = args
    
    local_train_lines = []
    local_val_lines = []
    local_chars = set()
    local_crops = 0
    
    label_file = book_dir / "Label.txt"
    if not label_file.exists():
        print(f"Warning: No Label.txt in {book_dir}")
        return local_train_lines, local_val_lines, local_chars, local_crops

    # 简单的打印，避免多进程打印混乱，可以考虑去掉或保留
    print(f"Processing {book_dir.name}...")
    
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {label_file}: {e}")
        return local_train_lines, local_val_lines, local_chars, local_crops

    count = 0
    for line in lines:
        if max_images_per_book is not None and count >= max_images_per_book:
            break
        
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        
        rel_path = parts[0] # e.g. BookName/0001.jpg
        json_str = parts[1]
        
        img_name = Path(rel_path).name
        img_path = book_dir / img_name
        
        if not img_path.exists():
            # 尝试查找不同后缀，或静默跳过
            continue

        try:
            # 使用 cv2.imdecode 读取中文路径图片
            img_np = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            if img is None:
                continue
                
            data = json.loads(json_str)
            
            for idx, item in enumerate(data):
                transcription = item['transcription']
                points = np.array(item['points'], dtype=np.float32)
                
                # 忽略非法字符
                if transcription == "###":
                    continue
                
                # 裁剪图片
                try:
                    crop_img = get_rotate_crop_image(img, points)
                except Exception:
                    continue
                    
                # 保存裁剪图片
                # 构造文件名: lang_book_imgname_idx.jpg
                # 移除书名中的空格等特殊字符以避免问题
                safe_book_name = book_dir.name.replace(' ', '_')
                save_name = f"{lang}_{safe_book_name}_{img_path.stem}_{idx}.jpg"
                save_path = images_dir / save_name
                
                # cv2.imwrite 不支持中文路径，使用 cv2.imencode + tofile
                success, encoded_img = cv2.imencode('.jpg', crop_img)
                if success:
                    encoded_img.tofile(str(save_path))
                    
                    # 记录标签
                    # 使用相对路径，兼容 PaddleOCR
                    rec_line = f"images/{save_name}\t{transcription}"
                    
                    if random.random() < val_ratio:
                        local_val_lines.append(rec_line)
                    else:
                        local_train_lines.append(rec_line)
                    
                    # 收集字符用于字典
                    for char in transcription:
                        local_chars.add(char)
                    
                    local_crops += 1
            
            count += 1
                
        except Exception as e:
            # 忽略单个图片错误，避免刷屏
            continue
            
    return local_train_lines, local_val_lines, local_chars, local_crops

def main():
    if not DST_ROOT.exists():
        DST_ROOT.mkdir(parents=True)
    
    images_dir = DST_ROOT / "images"
    if not images_dir.exists():
        images_dir.mkdir()

    print(f"Start processing from {SRC_ROOT} to {DST_ROOT}")

    # 收集任务
    tasks = []
    for lang in LANGUAGES:
        lang_dir = SRC_ROOT / lang
        if not lang_dir.exists():
            print(f"Skipping {lang_dir}, not found.")
            continue
        
        # 遍历语言目录下的书名目录
        for book_dir in lang_dir.iterdir():
            if not book_dir.is_dir():
                continue
            # 将任务参数打包
            tasks.append((book_dir, lang, images_dir, VAL_RATIO, MAX_IMAGES_PER_BOOK))

    print(f"Found {len(tasks)} books to process.")
    
    # 根据CPU核数决定进程数，预留一些资源
    num_processes = max(1, int(cpu_count() * 0.2))
    print(f"Using {num_processes} processes for acceleration...")

    train_lines = []
    val_lines = []
    all_chars = set()
    total_crops = 0

    # 启动多进程处理
    with Pool(processes=num_processes) as pool:
        # imap_unordered 可以让结果无序返回，稍微快一点，处理完一个就返回一个
        results = pool.imap_unordered(process_book, tasks)
        
        for i, (t_lines, v_lines, chars, crops) in enumerate(results):
            train_lines.extend(t_lines)
            val_lines.extend(v_lines)
            all_chars.update(chars)
            total_crops += crops
            
            # 简单的进度展示
            print(f"[{i+1}/{len(tasks)}] Processed book. Total crops: {total_crops}", end='\r')

    print("\nProcessing complete. Writing annotation files...")

    # 写入 train.txt
    with open(DST_ROOT / "train.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
        
    # 写入 val.txt
    with open(DST_ROOT / "val.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))
        
    # 写入 dict.txt
    with open(DST_ROOT / "dict.txt", 'w', encoding='utf-8') as f:
        chars_list = sorted(list(all_chars))
        f.write('\n'.join(chars_list))

    print(f"Done. Processed {total_crops} crops.")
    print(f"Train samples: {len(train_lines)}")
    print(f"Val samples: {len(val_lines)}")
    print(f"Dict size: {len(all_chars)}")

if __name__ == "__main__":
    main()
