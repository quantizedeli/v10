"""
Log Parser for Nuclear Physics AI Project
==========================================
Bu script log dosyalarını tarayarak ERROR ve WARNING mesajlarını
ayrı ayrı dosyalara kaydeder.

Usage:
    python log_parser.py [log_file_or_directory]

    Argüman verilmezse 'logs' klasöründeki tüm log dosyalarını tarar.
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class LogParser:
    """Log dosyalarını parse eden ve hataları/uyarıları ayıklayan sınıf"""

    def __init__(self, output_dir='log_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)

    def parse_log_file(self, log_file):
        """Tek bir log dosyasını parse et"""
        print(f"\n[INFO] Parse ediliyor: {log_file}")

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            current_entry = []
            current_level = None

            for line in lines:
                # Yeni bir log satırı mı kontrol et
                # Format: YYYY-MM-DD HH:MM:SS,mmm - name - LEVEL - message
                log_pattern = r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+-\s+[\w\.]+\s+-\s+(ERROR|WARNING|INFO|DEBUG|CRITICAL)'
                match = re.match(log_pattern, line)

                if match:
                    # Önceki entry'yi kaydet
                    if current_entry and current_level:
                        self._save_entry(current_entry, current_level)

                    # Yeni entry başlat
                    current_level = match.group(1)
                    current_entry = [line]
                    self.stats[current_level] += 1

                elif current_entry:
                    # Çok satırlı log mesajının devamı
                    current_entry.append(line)

            # Son entry'yi kaydet
            if current_entry and current_level:
                self._save_entry(current_entry, current_level)

            print(f"[SUCCESS] {log_file.name} parse edildi")

        except Exception as e:
            print(f"[ERROR] {log_file} parse edilirken hata: {e}")

    def _save_entry(self, entry, level):
        """Log entry'sini uygun listeye kaydet"""
        entry_text = ''.join(entry)

        if level == 'ERROR' or level == 'CRITICAL':
            self.errors.append(entry_text)
        elif level == 'WARNING':
            self.warnings.append(entry_text)

    def parse_directory(self, log_dir):
        """Bir dizindeki tüm log dosyalarını parse et"""
        log_dir = Path(log_dir)

        if not log_dir.exists():
            print(f"[ERROR] Dizin bulunamadı: {log_dir}")
            return

        log_files = list(log_dir.glob('*.log'))

        if not log_files:
            print(f"[WARNING] {log_dir} dizininde log dosyası bulunamadı")
            return

        print(f"\n[INFO] {len(log_files)} log dosyası bulundu")

        for log_file in sorted(log_files):
            self.parse_log_file(log_file)

    def save_results(self):
        """Parse sonuçlarını dosyalara kaydet"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # ERROR dosyasını kaydet
        if self.errors:
            error_file = self.output_dir / f'ERRORS_{timestamp}.log'
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("ERROR ve CRITICAL Log Kayıtları\n")
                f.write("="*80 + "\n")
                f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Toplam Hata Sayısı: {len(self.errors)}\n")
                f.write("="*80 + "\n\n")

                for i, error in enumerate(self.errors, 1):
                    f.write(f"\n{'='*80}\n")
                    f.write(f"ERROR #{i}\n")
                    f.write(f"{'='*80}\n")
                    f.write(error)
                    if not error.endswith('\n'):
                        f.write('\n')

            print(f"\n[SUCCESS] {len(self.errors)} hata kaydedildi: {error_file}")
        else:
            print("\n[INFO] Hiç ERROR kaydı bulunamadı")

        # WARNING dosyasını kaydet
        if self.warnings:
            warning_file = self.output_dir / f'WARNINGS_{timestamp}.log'
            with open(warning_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("WARNING Log Kayıtları\n")
                f.write("="*80 + "\n")
                f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Toplam Uyarı Sayısı: {len(self.warnings)}\n")
                f.write("="*80 + "\n\n")

                for i, warning in enumerate(self.warnings, 1):
                    f.write(f"\n{'='*80}\n")
                    f.write(f"WARNING #{i}\n")
                    f.write(f"{'='*80}\n")
                    f.write(warning)
                    if not warning.endswith('\n'):
                        f.write('\n')

            print(f"[SUCCESS] {len(self.warnings)} uyarı kaydedildi: {warning_file}")
        else:
            print("\n[INFO] Hiç WARNING kaydı bulunamadı")

        # İstatistik dosyasını kaydet
        stats_file = self.output_dir / f'STATS_{timestamp}.txt'
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Log Analiz İstatistikleri\n")
            f.write("="*80 + "\n")
            f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Log Seviyeleri:\n")
            f.write("-" * 40 + "\n")
            for level in ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']:
                count = self.stats.get(level, 0)
                f.write(f"  {level:12s}: {count:6d}\n")

            f.write("\n" + "="*80 + "\n")
            f.write(f"Toplam ERROR/CRITICAL: {len(self.errors)}\n")
            f.write(f"Toplam WARNING      : {len(self.warnings)}\n")
            f.write(f"Toplam Tüm Loglar   : {sum(self.stats.values())}\n")
            f.write("="*80 + "\n")

        print(f"[SUCCESS] İstatistikler kaydedildi: {stats_file}")

        # Özet rapor
        print("\n" + "="*80)
        print("ÖZET RAPOR")
        print("="*80)
        print(f"ERROR/CRITICAL: {len(self.errors)}")
        print(f"WARNING       : {len(self.warnings)}")
        print(f"Toplam Log    : {sum(self.stats.values())}")
        print("="*80)


def main():
    """Ana fonksiyon"""
    print("="*80)
    print("Log Parser - Nuclear Physics AI Project")
    print("="*80)

    # Argüman kontrolü
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = 'logs'

    target_path = Path(target)

    # Parser oluştur
    parser = LogParser()

    # Dosya mı dizin mi?
    if target_path.is_file():
        parser.parse_log_file(target_path)
    elif target_path.is_dir():
        parser.parse_directory(target_path)
    else:
        print(f"[ERROR] Geçersiz hedef: {target}")
        print("\nKullanım:")
        print("  python log_parser.py [log_file_or_directory]")
        print("\nÖrnekler:")
        print("  python log_parser.py                    # logs/ dizinini tara")
        print("  python log_parser.py logs/main_20231122.log  # Tek dosya")
        print("  python log_parser.py /path/to/logs      # Özel dizin")
        sys.exit(1)

    # Sonuçları kaydet
    parser.save_results()

    print("\n[SUCCESS] Log parse işlemi tamamlandı!")


if __name__ == "__main__":
    main()
