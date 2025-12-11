#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并多个dict.txt文件，按顺序去重
"""

import argparse
from pathlib import Path
from collections import OrderedDict


def merge_dict_files(dict_files, output_file):
    """
    按顺序合并多个dict.txt文件，去重

    Args:
        dict_files: 字典文件路径列表
        output_file: 输出文件路径
    """
    # 使用OrderedDict保持插入顺序
    merged_chars = OrderedDict()

    total_files = len(dict_files)
    print(f"开始合并 {total_files} 个字典文件...")

    for i, dict_file in enumerate(dict_files, 1):
        dict_path = Path(dict_file)
        if not dict_path.exists():
            print(f"警告: 文件不存在 {dict_file}")
            continue

        print(f"[{i}/{total_files}] 正在处理: {dict_path.name}")

        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            char_count = 0
            for line in lines:
                char = line.strip()
                # 跳过空行
                if not char:
                    continue
                # 添加到有序字典（自动去重并保持顺序）
                if char not in merged_chars:
                    merged_chars[char] = True
                    char_count += 1

            print(f"  从 {dict_path.name} 添加了 {char_count} 个字符")

        except Exception as e:
            print(f"错误: 处理文件 {dict_file} 时出错: {e}")
            continue

    # 写入输出文件
    print(f"\n正在写入输出文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        # 先写入一个空行（按照PaddleOCR dict.txt格式）
        f.write('\n')
        # 写入所有字符，每行一个
        for char in merged_chars.keys():
            f.write(f"{char}\n")

    print("合并完成！")
    print(f"总字符数: {len(merged_chars)}")
    print(f"输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="按顺序合并多个dict.txt文件并去重")
    parser.add_argument(
        "dict_files",
        nargs="+",
        help="要合并的dict.txt文件路径（按顺序）"
    )
    parser.add_argument(
        "-o", "--output",
        default="merged_dict.txt",
        help="输出文件路径 (默认: merged_dict.txt)"
    )

    args = parser.parse_args()

    merge_dict_files(args.dict_files, args.output)


if __name__ == "__main__":
    main()
