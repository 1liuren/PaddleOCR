import json
import os
import argparse

def convert_labels(label_file, output_file, image_dir="images"):
    """
    Convert text_renderer labels.json to PaddleOCR recognition dataset format.
    Format: image_path\tlabel_text
    """
    if not os.path.exists(label_file):
        print(f"Error: Label file not found at {label_file}")
        return

    with open(label_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    labels = data.get('labels', {})
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for img_id, label in labels.items():
            # PaddleOCR expects tab-separated format
            # image path should be relative to the dataset root or absolute
            image_name = f"{img_id}.jpg"
            image_path = os.path.join(image_dir, image_name)
            
            # Remove newlines or tabs from label if any, as they break the format
            clean_label = label.strip().replace('\t', ' ').replace('\n', '')
            
            line = f"{image_path}\t{clean_label}\n"
            f_out.write(line)
            
    print(f"Successfully converted {len(labels)} labels to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert labels.json to PaddleOCR rec_gt.txt")
    parser.add_argument("--label_file", default="text_renderer/example_data/output/mixed_line_data/labels.json", help="Path to input labels.json")
    parser.add_argument("--output_file", default="text_renderer/example_data/output/mixed_line_data/rec_gt.txt", help="Path to output txt file")
    parser.add_argument("--image_sub_dir", default="images", help="Subdirectory where images are stored relative to the label file location context")
    
    args = parser.parse_args()
    convert_labels(args.label_file, args.output_file, args.image_sub_dir)

