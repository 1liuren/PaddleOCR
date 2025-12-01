"""
检查训练和验证数据中的字符是否都在词典中
用于验证数据集的字符覆盖情况
"""

import argparse
import sys
from collections import Counter


def load_dict(dict_file):
    """
    加载词典文件，返回字符集合
    
    Args:
        dict_file: 词典文件路径，每行一个字符
        
    Returns:
        set: 词典中的字符集合
    """
    char_set = set()
    try:
        with open(dict_file, 'r', encoding='utf-8') as f:
            for line in f:
                char = line.strip()
                if char:  # 忽略空行
                    char_set.add(char)
        print(f"✅ 成功加载词典: {dict_file}")
        print(f"   词典包含 {len(char_set)} 个字符")
        return char_set
    except FileNotFoundError:
        print(f"❌ 错误: 词典文件不存在: {dict_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: 读取词典文件失败: {e}")
        sys.exit(1)


def extract_chars_from_label_file(label_file):
    """
    从标签文件中提取所有字符
    
    Args:
        label_file: 标签文件路径，格式为: 图像路径\t文本标签
        
    Returns:
        set: 标签文件中出现的所有字符集合
        Counter: 字符出现次数统计
    """
    char_set = set()
    char_counter = Counter()
    line_count = 0
    
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 解析标签文件格式: 图像路径\t文本标签
                parts = line.split('\t', 1)
                if len(parts) < 2:
                    print(f"⚠️  警告: 第 {line_count + 1} 行格式不正确，跳过: {line[:50]}...")
                    continue
                
                label_text = parts[1]
                line_count += 1
                
                # 提取所有字符
                for char in label_text:
                    char_set.add(char)
                    char_counter[char] += 1
        
        print(f"✅ 成功读取标签文件: {label_file}")
        print(f"   总行数: {line_count}")
        print(f"   唯一字符数: {len(char_set)}")
        return char_set, char_counter
    except FileNotFoundError:
        print(f"❌ 错误: 标签文件不存在: {label_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: 读取标签文件失败: {e}")
        sys.exit(1)


def check_char_coverage(dict_chars, label_chars, label_counter, label_name):
    """
    检查标签文件中的字符是否都在词典中
    
    Args:
        dict_chars: 词典中的字符集合
        label_chars: 标签文件中的字符集合
        label_counter: 标签文件中字符出现次数统计
        label_name: 标签文件名称（用于显示）
        
    Returns:
        tuple: (是否全部覆盖, 缺失字符集合)
    """
    missing_chars = label_chars - dict_chars
    
    print(f"\n{'='*60}")
    print(f"检查 {label_name} 的字符覆盖情况")
    print(f"{'='*60}")
    print(f"标签文件中的唯一字符数: {len(label_chars)}")
    print(f"词典中的字符数: {len(dict_chars)}")
    print(f"缺失的字符数: {len(missing_chars)}")
    
    if missing_chars:
        print(f"\n❌ 发现 {len(missing_chars)} 个字符不在词典中:")
        print(f"{'字符':<20} {'出现次数':<15} {'Unicode编码':<20}")
        print("-" * 60)
        
        # 按出现次数排序
        missing_with_count = [(char, label_counter[char]) for char in missing_chars]
        missing_with_count.sort(key=lambda x: x[1], reverse=True)
        
        for char, count in missing_with_count:
            unicode_code = f"U+{ord(char):04X}"
            print(f"{char!r:<20} {count:<15} {unicode_code:<20}")
        
        return False, missing_chars
    else:
        print(f"\n✅ 所有字符都在词典中!")
        return True, set()


def filter_whitespace_chars(chars):
    """
    过滤掉空白字符（空格、制表符等）
    
    Args:
        chars: 字符集合
        
    Returns:
        set: 过滤后的字符集合
    """
    # 过滤掉空格、制表符、换行符等空白字符
    whitespace_chars = {' ', '\t', '\n', '\r', '\v', '\f'}
    filtered = {char for char in chars if char not in whitespace_chars and not char.isspace()}
    return filtered


def generate_new_dict(original_dict_file, missing_chars, output_file):
    """
    生成新的字典文件，将缺失的字符追加到原字典后面
    
    Args:
        original_dict_file: 原始字典文件路径
        missing_chars: 缺失的字符集合
        output_file: 输出字典文件路径
        
    Returns:
        int: 添加的字符数量
    """
    # 过滤掉空白字符
    filtered_missing = filter_whitespace_chars(missing_chars)
    
    if not filtered_missing:
        print(f"\n⚠️  没有需要添加的字符（已过滤空白字符）")
        return 0
    
    # 读取原始字典文件的所有行（保持原有顺序）
    original_lines = []
    original_chars_set = set()
    
    try:
        with open(original_dict_file, 'r', encoding='utf-8') as f:
            for line in f:
                char = line.strip()
                if char:
                    original_lines.append(char)
                    original_chars_set.add(char)
    except Exception as e:
        print(f"❌ 错误: 读取原始字典文件失败: {e}")
        return 0
    
    # 过滤掉已经在字典中的字符
    new_chars = filtered_missing - original_chars_set
    
    if not new_chars:
        print(f"\n⚠️  所有缺失字符都已存在于字典中")
        return 0
    
    # 将新字符按Unicode编码排序
    sorted_new_chars = sorted(new_chars, key=lambda x: ord(x))
    
    # 写入新字典文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # 先写入原始字典内容
            for char in original_lines:
                f.write(f"{char}\n")
            
            # 再写入新字符
            for char in sorted_new_chars:
                f.write(f"{char}\n")
        
        print(f"\n✅ 成功生成新字典文件: {output_file}")
        print(f"   原始字典字符数: {len(original_chars_set)}")
        print(f"   新增字符数: {len(sorted_new_chars)}")
        print(f"   新字典总字符数: {len(original_chars_set) + len(sorted_new_chars)}")
        
        # 显示新增的字符
        print(f"\n新增的字符列表:")
        print("-" * 60)
        for char in sorted_new_chars:
            unicode_code = f"U+{ord(char):04X}"
            print(f"  {char!r} ({unicode_code})")
        
        return len(sorted_new_chars)
    except Exception as e:
        print(f"❌ 错误: 写入新字典文件失败: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='检查训练和验证数据中的字符是否都在词典中',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python check_dict_coverage.py --train train.txt --val val.txt --dict dict.txt
  python check_dict_coverage.py -t BDRC/paddleocr_data/train.txt -v BDRC/paddleocr_data/val.txt -d BDRC/paddleocr_data/paddleocr_Tibetan_dict_complete.txt --output dict_new.txt
        """
    )
    
    parser.add_argument(
        '--train', '-t',
        type=str,
        required=True,
        help='训练数据标签文件路径 (train.txt)'
    )
    
    parser.add_argument(
        '--val', '-v',
        type=str,
        required=True,
        help='验证数据标签文件路径 (val.txt)'
    )
    
    parser.add_argument(
        '--dict', '-d',
        type=str,
        required=True,
        help='词典文件路径 (dict.txt)，每行一个字符'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出新字典文件路径（如果指定，会将缺失字符追加到原字典后生成新文件）'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("字符覆盖检查工具")
    print("="*60)
    
    # 加载词典
    dict_chars = load_dict(args.dict)
    
    # 检查训练数据
    train_chars, train_counter = extract_chars_from_label_file(args.train)
    train_ok, train_missing = check_char_coverage(dict_chars, train_chars, train_counter, "训练数据 (train.txt)")
    
    # 检查验证数据
    val_chars, val_counter = extract_chars_from_label_file(args.val)
    val_ok, val_missing = check_char_coverage(dict_chars, val_chars, val_counter, "验证数据 (val.txt)")
    
    # 检查所有字符（训练+验证）
    all_label_chars = train_chars | val_chars
    all_label_counter = train_counter + val_counter
    all_ok, all_missing = check_char_coverage(dict_chars, all_label_chars, all_label_counter, "所有数据 (train.txt + val.txt)")
    
    # 总结
    print(f"\n{'='*60}")
    print("检查总结")
    print(f"{'='*60}")
    print(f"训练数据: {'✅ 通过' if train_ok else '❌ 失败'}")
    print(f"验证数据: {'✅ 通过' if val_ok else '❌ 失败'}")
    print(f"所有数据: {'✅ 通过' if all_ok else '❌ 失败'}")
    
    # 如果指定了输出文件，生成新字典
    if args.output:
        print(f"\n{'='*60}")
        print("生成新字典文件")
        print(f"{'='*60}")
        added_count = generate_new_dict(args.dict, all_missing, args.output)
        if added_count > 0:
            print(f"\n✅ 已生成包含缺失字符的新字典文件!")
    
    if train_ok and val_ok and all_ok:
        print("\n🎉 所有检查通过!")
        return 0
    else:
        if args.output:
            print("\n⚠️  发现缺失字符，已生成包含缺失字符的新字典文件!")
        else:
            print("\n⚠️  发现缺失字符，请使用 --output 参数生成包含缺失字符的新字典文件!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

