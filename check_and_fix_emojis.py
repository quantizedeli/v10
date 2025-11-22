#!/usr/bin/env python3
"""
Comprehensive Emoji Checker and Fixer
Tüm emoji karakterlerini bulur ve ASCII'ye çevirir
"""
import re
from pathlib import Path
import sys

# KAPSAMLI emoji mapping - tüm olası karakterler
EMOJI_MAP = {
    # Checkmarks and crosses
    '[SUCCESS]': '[SUCCESS]',
    '[OK]': '[OK]',
    '[ERROR]': '[ERROR]',
    '[FAIL]': '[FAIL]',
    '[WARNING]': '[WARNING]',
    '[WARNING]': '[WARNING]',

    # Circles and shapes
    '[INFO]': '[INFO]',
    '[OK]': '[OK]',
    '[ERROR]': '[ERROR]',
    '[EMPTY]': '[EMPTY]',
    '[FULL]': '[FULL]',
    '[ALERT]': '[ALERT]',
    '[INFO]': '[INFO]',

    # Folders and files
    '[FOLDER]': '[FOLDER]',
    '[OPEN]': '[OPEN]',
    '[FILE]': '[FILE]',
    '[DOC]': '[DOC]',
    '[LIST]': '[LIST]',
    '[NOTE]': '[NOTE]',
    '[SAVE]': '[SAVE]',

    # Science and tools
    '[CALC]': '[CALC]',
    '[TEST]': '[TEST]',
    '[TOOL]': '[TOOL]',
    '[SETTING]': '[SETTING]',
    '[SETTING]': '[SETTING]',
    '[REPAIR]': '[REPAIR]',
    '[REPAIR]': '[REPAIR]',

    # Search and view
    '[SEARCH]': '[SEARCH]',
    '[FIND]': '[FIND]',
    '[VIEW]': '[VIEW]',
    '[VIEW]': '[VIEW]',
    '[WATCH]': '[WATCH]',

    # Package and box
    '[PACKAGE]': '[PACKAGE]',
    '[POST]': '[POST]',
    '[GIFT]': '[GIFT]',
    '[ARCHIVE]': '[ARCHIVE]',
    '[ARCHIVE]': '[ARCHIVE]',

    # Charts and graphs
    '[REPORT]': '[REPORT]',
    '[CHART]': '[CHART]',
    '[DECLINE]': '[DECLINE]',
    '[MEASURE]': '[MEASURE]',
    '[RULER]': '[RULER]',

    # Time and progress
    '[WAIT]': '[WAIT]',
    '[ALARM]': '[ALARM]',
    '[TIMER]': '[TIMER]',
    '[TIMER]': '[TIMER]',
    '[CLOCK]': '[CLOCK]',
    '[CLOCK]': '[CLOCK]',

    # Actions
    '[RUN]': '[RUN]',
    '[REPEAT]': '[REPEAT]',
    '[LOOP]': '[LOOP]',
    '[SKIP]': '[SKIP]',
    '[SKIP]': '[SKIP]',
    '[PAUSE]': '[PAUSE]',
    '[PAUSE]': '[PAUSE]',
    '[STOP]': '[STOP]',
    '[STOP]': '[STOP]',
    '[RECORD]': '[RECORD]',
    '[RECORD]': '[RECORD]',

    # Status
    '[UNKNOWN]': '[UNKNOWN]',
    '[QUESTION]': '[QUESTION]',
    '[IMPORTANT]': '[IMPORTANT]',
    '[ALERT]': '[ALERT]',
    '[TIP]': '[TIP]',
    '[NOTIFY]': '[NOTIFY]',

    # Emotions and reactions
    '[START]': '[START]',
    '[INTERACTIVE]': '[INTERACTIVE]',
    '[EXIT]': '[EXIT]',
    '[COMPLETE]': '[COMPLETE]',
    '[PARTY]': '[PARTY]',
    '[BALLOON]': '[BALLOON]',

    # Awards and achievements
    '[BEST]': '[BEST]',
    '[GOLD]': '[GOLD]',
    '[SILVER]': '[SILVER]',
    '[BRONZE]': '[BRONZE]',
    '[STAR]': '[STAR]',
    '[SPARKLE]': '[SPARKLE]',
    '[FEATURE]': '[FEATURE]',

    # Targets and goals
    '[TARGET]': '[TARGET]',
    '[POOL]': '[POOL]',
    '[DICE]': '[DICE]',

    # Strength and power
    '[STRONG]': '[STRONG]',
    '[HOT]': '[HOT]',
    '[FAST]': '[FAST]',
    '[BOOM]': '[BOOM]',

    # Graphics and design
    '[DESIGN]': '[DESIGN]',
    '[PICTURE]': '[PICTURE]',
    '[PICTURE]': '[PICTURE]',
    '[BRUSH]': '[BRUSH]',
    '[BRUSH]': '[BRUSH]',

    # Tags and labels
    '[TAG]': '[TAG]',
    '[TAG]': '[TAG]',
    '[PIN]': '[PIN]',
    '[LOCATION]': '[LOCATION]',

    # New and updated
    '[NEW]': '[NEW]',
    '[UP]': '[UP]',
    '[TOP]': '[TOP]',

    # Arrows
    '->': '->',
    '->': '->',
    '<-': '<-',
    '<-': '<-',
    '^': '^',
    '^': '^',
    'v': 'v',
    'v': 'v',
    '->': '->',
    '->': '->',
    '->': '->',
    '->': '->',
    '<-': '<-',
    '<-': '<-',
    '<-': '<-',
    '<-': '<-',
    '->': '->',
    '<-': '<-',
    '^': '^',
    'v': 'v',

    # Computer and tech
    '[PC]': '[PC]',
    '[DESKTOP]': '[DESKTOP]',
    '[DESKTOP]': '[DESKTOP]',
    '[KEYBOARD]': '[KEYBOARD]',
    '[KEYBOARD]': '[KEYBOARD]',
    '[MOUSE]': '[MOUSE]',
    '[MOUSE]': '[MOUSE]',
    '[PRINTER]': '[PRINTER]',
    '[PRINTER]': '[PRINTER]',

    # Other common emojis
    '[ROBOT]': '[ROBOT]',
    '[BRAIN]': '[BRAIN]',
    '[USER]': '[USER]',
    '[USERS]': '[USERS]',
    '[GLOBAL]': '[GLOBAL]',
    '[WORLD]': '[WORLD]',
    '[LINK]': '[LINK]',
    '[MOBILE]': '[MOBILE]',
}

def has_emoji(text):
    """Check if text contains any emoji"""
    # Check our known emojis
    for emoji in EMOJI_MAP.keys():
        if emoji in text:
            return True

    # Check for any Unicode emoji range
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U00002700-\U000027BF"  # Dingbats
        "]+",
        flags=re.UNICODE
    )
    return bool(emoji_pattern.search(text))

def fix_emojis_in_file(filepath):
    """Dosyadaki tüm emojileri değiştir"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # First check if file has emojis
        if not has_emoji(content):
            return False

        # Replace each emoji
        for emoji, replacement in EMOJI_MAP.items():
            if emoji in content:
                content = content.replace(emoji, replacement)
                print(f"      Replacing '{emoji}' -> '{replacement}'")

        # Write back if changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")
        return False

def scan_and_report():
    """Scan all Python files and report emoji usage"""
    base_dir = Path('/home/user/nucdatav1')

    # Find all Python files
    py_files = list(base_dir.rglob('*.py'))
    py_files = [f for f in py_files if '.git' not in str(f)]

    print(f"[INFO] Scanning {len(py_files)} Python files...")
    print("=" * 80)

    files_with_emojis = []
    emoji_counts = {}

    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for emojis
            found_emojis = []
            for emoji in EMOJI_MAP.keys():
                count = content.count(emoji)
                if count > 0:
                    found_emojis.append(f"{emoji}({count})")
                    emoji_counts[emoji] = emoji_counts.get(emoji, 0) + count

            if found_emojis:
                files_with_emojis.append((py_file, found_emojis))

        except Exception as e:
            print(f"[ERROR] Skipping {py_file}: {e}")

    # Print report
    if files_with_emojis:
        print(f"\n[FOUND] {len(files_with_emojis)} files with emojis:\n")
        for filepath, emojis in files_with_emojis:
            rel_path = filepath.relative_to(base_dir)
            print(f"  {rel_path}")
            print(f"    Emojis: {', '.join(emojis)}")

        print(f"\n[SUMMARY] Total emoji occurrences:")
        for emoji, count in sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emoji} -> {EMOJI_MAP[emoji]}: {count} times")

        return files_with_emojis
    else:
        print(f"\n[SUCCESS] No emojis found! All {len(py_files)} files are clean.")
        return []

def fix_all():
    """Fix all emojis in all files"""
    base_dir = Path('/home/user/nucdatav1')

    # Find all Python files
    py_files = list(base_dir.rglob('*.py'))
    py_files = [f for f in py_files if '.git' not in str(f)]

    print(f"[INFO] {len(py_files)} Python files found")
    print("[INFO] Fixing emojis...")
    print("=" * 80)

    fixed_count = 0
    for py_file in py_files:
        if fix_emojis_in_file(py_file):
            rel_path = py_file.relative_to(base_dir)
            print(f"[FIXED] {rel_path}")
            fixed_count += 1

    print("=" * 80)
    print(f"\n[SUCCESS] {fixed_count} files fixed!")
    print(f"[INFO] {len(py_files) - fixed_count} files were already clean")

def main():
    """Ana fonksiyon"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║     Comprehensive Emoji Checker & Fixer for Windows           ║
║     Tüm emoji karakterlerini bulur ve düzeltir                ║
╚════════════════════════════════════════════════════════════════╝
""")

    if len(sys.argv) > 1 and sys.argv[1] == 'scan':
        # Scan mode - just report
        files_with_emojis = scan_and_report()
        if files_with_emojis:
            response = input("\nFix all emojis? (E/h): ").lower()
            if response == 'e' or response == '':
                fix_all()
    else:
        # Fix mode - scan and fix
        files_with_emojis = scan_and_report()
        if files_with_emojis:
            print(f"\n[ACTION] Fixing emojis in {len(files_with_emojis)} files...\n")
            fix_all()

if __name__ == '__main__':
    main()
