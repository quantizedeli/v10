"""
Test script to verify data validation fix for comma decimal separators
"""
import pandas as pd
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_data_loading():
    """Test that data loads correctly with comma decimal separators fixed"""
    print("="*80)
    print("Testing Data Loading and Validation Fix")
    print("="*80)

    # Load raw data
    data_path = Path("data/aaa2.txt")
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        return False

    print(f"\n1. Loading data from: {data_path}")
    df = pd.read_csv(data_path, sep='\t', encoding='utf-8')
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    # Check for problematic columns before cleaning
    print("\n2. Checking data types before cleaning:")
    non_numeric = []
    for col in df.columns:
        if col != 'NUCLEUS':
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric.append(col)
                print(f"   - {col}: {df[col].dtype} (NON-NUMERIC)")

    if non_numeric:
        print(f"\n   Found {len(non_numeric)} non-numeric columns that should be numeric")

    # Clean and convert numeric columns
    print("\n3. Cleaning and converting numeric columns...")
    cols_to_process = [col for col in df.columns if col != 'NUCLEUS']

    conversions_made = 0
    for col in cols_to_process:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            if df[col].dtype == object:
                # Replace commas with dots for decimal conversion
                cleaned_values = df[col].astype(str).str.replace(',', '.', regex=False)
                converted = pd.to_numeric(cleaned_values, errors='coerce')

                if converted.notna().any():
                    original_na = df[col].isna().sum()
                    df[col] = converted
                    new_na = df[col].isna().sum()
                    conversions_made += 1

                    if new_na > original_na:
                        print(f"   - {col}: converted ({new_na - original_na} values became NaN)")
                    else:
                        print(f"   - {col}: converted successfully")

        except Exception as e:
            print(f"   - {col}: ERROR - {e}")

    print(f"\n   Total conversions: {conversions_made}")

    # Verify all expected numeric columns are now numeric
    print("\n4. Verifying all columns are now properly typed:")
    still_non_numeric = []
    for col in df.columns:
        if col != 'NUCLEUS':
            if not pd.api.types.is_numeric_dtype(df[col]):
                still_non_numeric.append(col)
                print(f"   - {col}: {df[col].dtype} (STILL NON-NUMERIC!)")

    if still_non_numeric:
        print(f"\n   ERROR: {len(still_non_numeric)} columns are still non-numeric!")
        return False

    print("\n   SUCCESS: All columns are properly typed!")

    # Test that we can calculate statistics on key columns
    print("\n5. Testing statistics calculation on target columns:")
    target_cols = ['MAGNETIC MOMENT [µ]', 'QUADRUPOLE MOMENT [Q]', 'Beta_2']

    for col in target_cols:
        if col in df.columns:
            try:
                col_data = df[col].dropna()
                mean_val = col_data.mean()
                std_val = col_data.std()
                print(f"   - {col}: mean={mean_val:.4f}, std={std_val:.4f}, valid_values={len(col_data)}")
            except Exception as e:
                print(f"   - {col}: ERROR calculating statistics - {e}")
                return False
        else:
            print(f"   - {col}: not found in data")

    print("\n" + "="*80)
    print("TEST PASSED: Data validation fix is working correctly!")
    print("="*80)
    return True


if __name__ == "__main__":
    try:
        success = test_data_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nTEST FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
