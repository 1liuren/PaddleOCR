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
        bool: 如果所有字符都在词典中返回True，否则返回False
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
        
        return False
    else:
        print(f"\n✅ 所有字符都在词典中!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='检查训练和验证数据中的字符是否都在词典中',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python check_dict_coverage.py --train train.txt --val val.txt --dict dict.txt
  python check_dict_coverage.py -t BDRC/paddleocr_data/train.txt -v BDRC/paddleocr_data/val.txt -d BDRC/paddleocr_data/paddleocr_Tibetan_dict_complete.txt
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
    
    args = parser.parse_args()
    
    print("="*60)
    print("字符覆盖检查工具")
    print("="*60)
    
    # 加载词典
    dict_chars = load_dict(args.dict)
    
    # 检查训练数据
    train_chars, train_counter = extract_chars_from_label_file(args.train)
    train_ok = check_char_coverage(dict_chars, train_chars, train_counter, "训练数据 (train.txt)")
    
    # 检查验证数据
    val_chars, val_counter = extract_chars_from_label_file(args.val)
    val_ok = check_char_coverage(dict_chars, val_chars, val_counter, "验证数据 (val.txt)")
    
    # 检查所有字符（训练+验证）
    all_label_chars = train_chars | val_chars
    all_label_counter = train_counter + val_counter
    all_ok = check_char_coverage(dict_chars, all_label_chars, all_label_counter, "所有数据 (train.txt + val.txt)")
    
    # 总结
    print(f"\n{'='*60}")
    print("检查总结")
    print(f"{'='*60}")
    print(f"训练数据: {'✅ 通过' if train_ok else '❌ 失败'}")
    print(f"验证数据: {'✅ 通过' if val_ok else '❌ 失败'}")
    print(f"所有数据: {'✅ 通过' if all_ok else '❌ 失败'}")
    
    if train_ok and val_ok and all_ok:
        print("\n🎉 所有检查通过!")
        return 0
    else:
        print("\n⚠️  发现缺失字符，请更新词典文件!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

