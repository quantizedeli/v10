"""
File I/O Utilities for Nuclear Physics AI Project
==================================================

Esnek dosya okuma ve yazma fonksiyonları.
Tüm PFAZ modüllerinde kullanılabilir.

Author: Nuclear Physics AI Research Team
Version: 1.0.0
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def read_nuclear_data(filepath, **kwargs):
    """
    Esnek nükleer veri okuma fonksiyonu

    .csv, .xlsx, .xls, .tsv, .txt formatlarını destekler
    .txt dosyaları için otomatik delimiter tespiti yapar

    Args:
        filepath: Dosya yolu (str veya Path)
        **kwargs: pandas read_* fonksiyonlarına iletilecek ek parametreler

    Returns:
        pd.DataFrame: Yüklenen veri

    Raises:
        FileNotFoundError: Dosya bulunamazsa
        ValueError: Desteklenmeyen dosya formatı

    Examples:
        >>> df = read_nuclear_data('aaa2.txt')
        >>> df = read_nuclear_data('data.csv', encoding='utf-8')
        >>> df = read_nuclear_data('data.xlsx', sheet_name='Sheet1')
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Reading nuclear data from: {filepath}")

    file_suffix = filepath.suffix.lower()

    try:
        # CSV files
        if file_suffix == '.csv':
            df = pd.read_csv(filepath, **kwargs)

        # Excel files
        elif file_suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, **kwargs)

        # TSV files
        elif file_suffix == '.tsv':
            df = pd.read_csv(filepath, sep='\t', **kwargs)

        # TXT files - auto-detect delimiter
        elif file_suffix == '.txt':
            # Try different delimiters in order of likelihood
            delimiters = ['\t', ',', r'\s+']
            df = None

            for delimiter in delimiters:
                try:
                    if delimiter == r'\s+':
                        # whitespace-delimited
                        df = pd.read_csv(filepath, delim_whitespace=True, **kwargs)
                    else:
                        df = pd.read_csv(filepath, sep=delimiter, **kwargs)

                    # Validate: check if we got reasonable columns
                    if len(df.columns) > 1 and len(df) > 0:
                        logger.info(f"  Successfully read with delimiter: {repr(delimiter)}")
                        break
                except Exception:
                    continue

            if df is None or len(df.columns) <= 1:
                # Fallback to tab-delimited (most common for .txt)
                logger.warning("Auto-detection failed, using tab-delimited as fallback")
                df = pd.read_csv(filepath, sep='\t', **kwargs)

        else:
            raise ValueError(f"Unsupported file format: {file_suffix}")

        logger.info(f"  [SUCCESS] Loaded: {len(df)} rows, {len(df.columns)} columns")

        return df

    except Exception as e:
        logger.error(f"  [ERROR] Failed to read {filepath}: {str(e)}")
        raise


def save_nuclear_data(df, filepath, **kwargs):
    """
    Nükleer veriyi dosyaya kaydet

    Format otomatik olarak uzantıdan belirlenir

    Args:
        df: Kaydedilecek DataFrame
        filepath: Dosya yolu (str veya Path)
        **kwargs: pandas to_* fonksiyonlarına iletilecek ek parametreler

    Examples:
        >>> save_nuclear_data(df, 'output.csv')
        >>> save_nuclear_data(df, 'output.xlsx', index=False)
        >>> save_nuclear_data(df, 'output.txt', sep='\t')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    file_suffix = filepath.suffix.lower()

    try:
        if file_suffix == '.csv':
            df.to_csv(filepath, **kwargs)
        elif file_suffix in ['.xlsx', '.xls']:
            df.to_excel(filepath, **kwargs)
        elif file_suffix == '.tsv':
            df.to_csv(filepath, sep='\t', **kwargs)
        elif file_suffix == '.txt':
            # Default to tab-delimited for .txt
            sep = kwargs.pop('sep', '\t')
            df.to_csv(filepath, sep=sep, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_suffix}")

        logger.info(f"  [SUCCESS] Saved: {filepath}")

    except Exception as e:
        logger.error(f"  [ERROR] Failed to save {filepath}: {str(e)}")
        raise


if __name__ == "__main__":
    # Test
    import numpy as np

    print("Testing file_io_utils...")

    # Create test data
    test_df = pd.DataFrame({
        'A': [10, 20, 30],
        'Z': [5, 10, 15],
        'N': [5, 10, 15],
        'MM': [1.2, -0.5, 2.1]
    })

    # Test saving
    test_path = Path('test_output.csv')
    save_nuclear_data(test_df, test_path, index=False)

    # Test loading
    loaded_df = read_nuclear_data(test_path)

    print(f"Original shape: {test_df.shape}")
    print(f"Loaded shape: {loaded_df.shape}")
    print(f"Data match: {test_df.equals(loaded_df)}")

    # Cleanup
    test_path.unlink()

    print("[OK] file_io_utils test completed")
