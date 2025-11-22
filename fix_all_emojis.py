#!/usr/bin/env python3
"""
Windows Emoji Fix Script
Tüm Python dosyalarındaki emojileri ASCII karakterlere çevirir
"""
import re
from pathlib import Path

# Emoji replacement mapping
EMOJI_MAP = {
    '[SUCCESS]': '[SUCCESS]',
    '[ERROR]': '[ERROR]',
    '[WARNING]': '[WARNING]',
    '[WARNING]': '[WARNING]',
    '[INFO]': '[INFO]',
    '[FOLDER]': '[FOLDER]',
    '[CALC]': '[CALC]',
    '[SEARCH]': '[SEARCH]',
    '[PACKAGE]': '[PACKAGE]',
    '[REPORT]': '[REPORT]',
    '[WAIT]': '[WAIT]',
    '[RUN]': '[RUN]',
    '[SKIP]': '[SKIP]',
    '[SKIP]': '[SKIP]',
    '[UNKNOWN]': '[UNKNOWN]',
    '[START]': '[START]',
    '[TIP]': '[TIP]',
    '[INTERACTIVE]': '[INTERACTIVE]',
    '[STOP]': '[STOP]',
    '[STOP]': '[STOP]',
    '[EXIT]': '[EXIT]',
    '[COMPLETE]': '[COMPLETE]',
    '[CHART]': '[CHART]',
    '[DECLINE]': '[DECLINE]',
    '[BEST]': '[BEST]',
    '[STAR]': '[STAR]',
    '[SPARKLE]': '[SPARKLE]',
    '[TARGET]': '[TARGET]',
    '[STRONG]': '[STRONG]',
    '[HOT]': '[HOT]',
    '[SAVE]': '[SAVE]',
    '[NOTE]': '[NOTE]',
    '[TEST]': '[TEST]',
    '[TOOL]': '[TOOL]',
    '[SETTING]': '[SETTING]',
    '[SETTING]': '[SETTING]',
    '[PIN]': '[PIN]',
    '[DESIGN]': '[DESIGN]',
    '[FEATURE]': '[FEATURE]',
    '[NEW]': '[NEW]',
    '->': '->',
    '->': '->',
    '<-': '<-',
    '<-': '<-',
    '->': '->',
    '->': '->',
    '->': '->',
    '<-': '<-',
}

def fix_emojis_in_file(filepath):
    """Dosyadaki tüm emojileri değiştir"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Replace each emoji
        for emoji, replacement in EMOJI_MAP.items():
            content = content.replace(emoji, replacement)

        # Write back if changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")
        return False

def main():
    """Ana fonksiyon"""
    base_dir = Path('/home/user/nucdatav1')

    # Find all Python files
    py_files = list(base_dir.rglob('*.py'))
    py_files = [f for f in py_files if '.git' not in str(f)]

    print(f"[INFO] {len(py_files)} Python dosyası bulundu")
    print("[INFO] Emojiler temizleniyor...")

    fixed_count = 0
    for py_file in py_files:
        if fix_emojis_in_file(py_file):
            print(f"[FIXED] {py_file.relative_to(base_dir)}")
            fixed_count += 1

    print(f"\n[SUCCESS] {fixed_count} dosya düzeltildi!")
    print(f"[INFO] {len(py_files) - fixed_count} dosyada emoji yoktu")

if __name__ == '__main__':
    main()
