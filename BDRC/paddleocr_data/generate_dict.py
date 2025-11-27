import os

def generate_dict(file_list, output_path):
    chars = set()
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                label = parts[1]
                # Add each character in the label to the set
                for char in label:
                    chars.add(char)
    
    # Sort characters to ensure deterministic order
    sorted_chars = sorted(list(chars))
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for char in sorted_chars:
            f.write(char + '\n')
    
    print(f"Dictionary generated with {len(sorted_chars)} characters at {output_path}")
    print(f"Sample characters: {sorted_chars[:10]}")

if __name__ == "__main__":
    train_file = "BDRC/paddleocr_data/train.txt"
    # We should also include val characters if possible, but train is critical
    # Assuming val.txt exists in the same dir
    val_file = "BDRC/paddleocr_data/val.txt"
    
    output_dict = "BDRC/paddleocr_data/tibetan_char_dict_generated.txt"
    
    generate_dict([train_file, val_file], output_dict)

