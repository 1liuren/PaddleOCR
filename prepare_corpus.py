import os
from pathlib import Path

TRAIN_FILE = Path("ocr_rec_dataset_output/train.txt")
CHN_OUT = Path("text_renderer/example_data/text/chn_rich.txt")
ENG_OUT = Path("text_renderer/example_data/text/eng_rich.txt")

def is_contains_chinese(string):
    for ch in string:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def main():
    if not TRAIN_FILE.exists():
        print(f"Error: {TRAIN_FILE} not found.")
        return

    print(f"Reading from {TRAIN_FILE}...")
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    chn_data = []
    eng_data = []

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        text = parts[1]
        
        # Check if text contains Chinese
        if is_contains_chinese(text):
            chn_data.append(text)
        else:
            # Assume it's English/ASCII if not Chinese (simplification)
            # You might want to filter out other scripts if necessary
            eng_data.append(text)

    print(f"Extracted {len(chn_data)} Chinese lines.")
    print(f"Extracted {len(eng_data)} English/Other lines.")

    # Write to files
    # chn_rich.txt: Just write lines. TextRenderer reads file and extracts chars.
    with open(CHN_OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(chn_data))

    # eng_rich.txt: Write lines. TextRenderer WordCorpus splits by space.
    with open(ENG_OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(eng_data))

    print(f"Written to {CHN_OUT} and {ENG_OUT}")

if __name__ == "__main__":
    main()

