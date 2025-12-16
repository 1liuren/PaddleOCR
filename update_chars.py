import os
from pathlib import Path

CHN_TEXT = Path("text_renderer/example_data/text/chn_rich.txt")
ENG_TEXT = Path("text_renderer/example_data/text/eng_rich.txt")

CHN_CHARS = Path("text_renderer/example_data/char/chn_rich_chars.txt")
ENG_CHARS = Path("text_renderer/example_data/char/eng_rich_chars.txt")

def generate_char_file(text_file, char_file):
    if not text_file.exists():
        print(f"{text_file} does not exist.")
        return

    print(f"Processing {text_file}...")
    with open(text_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Filter out newline and other invisible chars if necessary, but keep punctuation
    # TextRenderer usually expects printable chars.
    unique_chars = sorted(list(set(content)))
    
    # Remove newline characters if present in the set
    unique_chars = [c for c in unique_chars if c not in ['\n', '\r', '\t', '\u200b', '\u3000']]
    
    # For English corpus (WordCorpus), we NEED spaces to split words.
    # For Chinese corpus (CharCorpus), we generally don't want random spaces unless they are part of the charset.
    # But strictly speaking, if we use the char in text, we should include it.
    # TextRenderer's load_chars_file strips \n \r but preserves space if the line is " "
    
    # We will exclude space only if it's empty string AFTER strip, UNLESS we specifically want to keep space.
    # But to be safe for WordCorpus, we must include space if it exists in text.
    
    # Simplified logic: Keep char if it is printable or space.
    # unique_chars = [c for c in unique_chars if c.strip()] <- This removed space.
    
    # Let's filter out empty strings if any (shouldn't be with set(content))
    # But we want to keep " " (space).
    final_chars = []
    for c in unique_chars:
        if c == ' ':
            final_chars.append(c)
        elif c.strip():
            final_chars.append(c)
            
    unique_chars = sorted(final_chars)

    print(f"Found {len(unique_chars)} unique characters.")
    
    with open(char_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(unique_chars))
    print(f"Written to {char_file}")

def main():
    generate_char_file(CHN_TEXT, CHN_CHARS)
    generate_char_file(ENG_TEXT, ENG_CHARS)

if __name__ == "__main__":
    main()

