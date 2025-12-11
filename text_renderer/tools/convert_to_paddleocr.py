import json
import os
import argparse
import random

def convert_labels(label_file, train_file, val_file, train_ratio=0.9):
    """
    Convert text_renderer labels.json to PaddleOCR recognition dataset format.
    Format: image_path\tlabel_text
    Splits data into train and validation sets based on train_ratio.
    """
    if not os.path.exists(label_file):
        print(f"Error: Label file not found at {label_file}")
        return

    with open(label_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    labels = data.get('labels', {})

    # Get label file directory
    label_dir = os.path.dirname(label_file)
    image_dir = os.path.join(label_dir, "images")

    # Prepare train and val lines
    all_lines = []
    for img_id, label in labels.items():
        # PaddleOCR expects tab-separated format
        # image path should be relative to the dataset root or absolute
        image_name = f"{img_id}.png"
        image_path = os.path.join("images", image_name)

        # Remove newlines or tabs from label if any, as they break the format
        clean_label = label.strip().replace('\t', ' ').replace('\n', '')

        line = f"{image_path}\t{clean_label}"
        all_lines.append(line)

    # Shuffle and split
    random.seed(42)  # For reproducibility
    random.shuffle(all_lines)

    split_idx = int(len(all_lines) * train_ratio)
    train_lines = all_lines[:split_idx]
    val_lines = all_lines[split_idx:]

    # Write train file
    with open(train_file, 'w', encoding='utf-8') as f_train:
        f_train.write('\n'.join(train_lines) + '\n')

    # Write val file
    with open(val_file, 'w', encoding='utf-8') as f_val:
        f_val.write('\n'.join(val_lines) + '\n')

    print(f"Successfully converted {len(labels)} labels:")
    print(f"  Train set: {len(train_lines)} samples -> {train_file}")
    print(f"  Val set: {len(val_lines)} samples -> {val_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert labels.json to PaddleOCR train.txt and val.txt")
    parser.add_argument("--label_file", required=True, help="Path to input labels.json")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of training data (default: 0.9)")

    args = parser.parse_args()

    # Generate output file paths in the same directory as label_file
    label_dir = os.path.dirname(args.label_file)
    train_file = os.path.join(label_dir, "train.txt")
    val_file = os.path.join(label_dir, "val.txt")

    convert_labels(args.label_file, train_file, val_file, args.train_ratio)

