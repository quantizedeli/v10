# Nuclear Physics Dataset Guide
## Comprehensive Documentation for Dataset Generation and Usage

**Date**: 2025-11-21
**Version**: 2.0
**Project**: Nuclear Physics AI - Dataset Generation (PFAZ 01)

---

## Table of Contents

1. [Overview](#overview)
2. [Training Scenarios](#training-scenarios)
3. [Output Formats](#output-formats)
4. [Nuclei Distribution Reports](#nuclei-distribution-reports)
5. [MATLAB Integration](#matlab-integration)
6. [Excel Files](#excel-files)
7. [Nuclei Catalog](#nuclei-catalog)
8. [Dataset Configuration](#dataset-configuration)
9. [Usage Examples](#usage-examples)
10. [Quality Assurance](#quality-assurance)

---

## Overview

This project generates comprehensive nuclear physics datasets for machine learning and ANFIS training. The datasets include:

- **267 nuclei** from the aaa2.txt database
- **Multiple training scenarios** (60/20/20, 70/15/15, 80/10/10)
- **4 target variables** (MM, Q, MM_QM, Beta_2)
- **Multiple formats** (CSV, Excel, MATLAB)
- **Detailed distribution analysis** for each dataset

### Key Features

✅ **Multiple Training Scenarios**: Choose between 60/20/20, 70/15/15, or 80/10/10 splits
✅ **MATLAB-Compatible**: Direct .mat file export for ANFIS training
✅ **Excel Reports**: Comprehensive Excel files with distribution analysis
✅ **Nuclei Lists**: Detailed lists of which nuclei are in each dataset
✅ **Distribution Analysis**: Z, N, A distributions, magic number proximity, isotope diversity
✅ **Scalable**: Works with 75, 100, 150, 200 nuclei, or ALL available

---

## Training Scenarios

### S60: Balanced Validation/Test (60/20/20)

**Best for**: When you need substantial validation and test sets

- **Training**: 60% of data
- **Validation (Check)**: 20% of data
- **Testing**: 20% of data

**Use cases**:
- Small datasets where you want to maximize validation/test data
- Rigorous model evaluation
- Cross-validation experiments

### S70: Standard Split (70/15/15)

**Best for**: Most general-purpose applications

- **Training**: 70% of data
- **Validation (Check)**: 15% of data
- **Testing**: 15% of data

**Use cases**:
- Standard machine learning workflows
- Balanced training/validation/testing
- Default recommended split

### S80: High Training Ratio (80/10/10)

**Best for**: When you want maximum training data

- **Training**: 80% of data
- **Validation (Check)**: 10% of data
- **Testing**: 10% of data

**Use cases**:
- Large datasets where 10% is still substantial
- Models that benefit from more training data
- Production deployments

---

## Output Formats

Each dataset is saved in **three formats** for maximum compatibility:

### 1. CSV Format (.csv)
- `train.csv`, `check.csv`, `test.csv`
- Standard comma-separated values
- Compatible with Python, R, Excel, and most tools
- Human-readable

### 2. Excel Format (.xlsx)
- `train.xlsx`, `check.xlsx`, `test.xlsx`
- Microsoft Excel compatible
- Easy to inspect and visualize
- Professional reporting

### 3. MATLAB Format (.mat)
- `train.mat`, `check.mat`, `test.mat`
- Native MATLAB format
- Optimized for ANFIS training
- Includes variable names and metadata

**File Structure Example**:
```
MM_75_S70_anomalisiz_AZN_standard_stratified/
├── train.csv, train.xlsx, train.mat
├── check.csv, check.xlsx, check.mat
├── test.csv, test.xlsx, test.mat
├── metadata.json
├── nucleus_selection.xlsx
├── nuclei_distribution_report.xlsx
└── scaler.pkl (if scaling applied)
```

---

## Nuclei Distribution Reports

Each dataset includes a detailed `nuclei_distribution_report.xlsx` with multiple sheets:

### Sheet 1: Summary
- Total number of nuclei
- Z, N, A ranges
- Unique counts
- Double magic nuclei count

### Sheet 2: Z_Distribution
- Proton number distribution
- Count for each Z value
- Histogram-ready data

### Sheet 3: N_Distribution
- Neutron number distribution
- Count for each N value

### Sheet 4: A_Distribution
- Mass number distribution
- Count for each A value

### Sheet 5: Mass_Groups
Distribution by mass regions:
- **Light** (A < 40)
- **Medium-Light** (40 ≤ A < 90)
- **Medium** (90 ≤ A < 140)
- **Medium-Heavy** (140 ≤ A < 200)
- **Heavy** (A ≥ 200)

### Sheet 6: Nucleus_Types
Distribution by parity:
- **Even-Even**: Z and N both even (most stable)
- **Odd-Odd**: Z and N both odd (least stable)
- **Even-Odd**: Z even, N odd
- **Odd-Even**: Z odd, N even

Includes both counts and percentages.

### Sheet 7: Isotope_Diversity
- Number of isotopes per element (same Z)
- Which elements have most isotopes
- Average isotopes per element

### Sheet 8: Magic_Numbers
Analysis of proximity to magic numbers:
- **Exact Magic Z**: Nuclei with Z = 2, 8, 20, 28, 50, 82, 114, 126
- **Exact Magic N**: Nuclei with N = 2, 8, 20, 28, 50, 82, 126, 184
- **Near Magic**: Within ±2 of magic numbers
- **Double Magic**: Both Z and N are magic (e.g., He-4, O-16, Ca-48)

### Sheet 9-10: Deformation (for Beta_2 target)
- Spherical vs deformed nuclei
- Beta_2 statistics (min, max, mean, std)
- Deformation regions distribution

---

## MATLAB Integration

### Loading Data in MATLAB

```matlab
% Load training data
train_data = load('train.mat');

% Access inputs and outputs
X_train = train_data.train_input;      % Features matrix
y_train = train_data.train_output;     % Target values
features = train_data.feature_names;   % Feature names
targets = train_data.target_names;     % Target names

% Check dimensions
fprintf('Training samples: %d\n', size(X_train, 1));
fprintf('Number of features: %d\n', size(X_train, 2));
fprintf('Number of targets: %d\n', size(y_train, 2));
```

### ANFIS Training Example

```matlab
% Load data
train_data = load('train.mat');
check_data = load('check.mat');
test_data = load('test.mat');

% Prepare training data
trainData = [train_data.train_input, train_data.train_output];
checkData = [check_data.check_input, check_data.check_output];

% Configure ANFIS options
opt = anfisOptions;
opt.InitialFIS = 2;  % Number of membership functions
opt.EpochNumber = 100;
opt.DisplayANFISInformation = 1;
opt.DisplayErrorValues = 1;
opt.ValidationData = checkData;

% Train ANFIS
[fis, trainError, stepSize, checkFIS, checkError] = anfis(trainData, opt);

% Evaluate on test set
predictions = evalfis(fis, test_data.test_input);
actual = test_data.test_output;

% Calculate metrics
mse = mean((predictions - actual).^2);
rmse = sqrt(mse);
mae = mean(abs(predictions - actual));

fprintf('Test RMSE: %.4f\n', rmse);
fprintf('Test MAE: %.4f\n', mae);
```

### Grid Partition (genfis1)

```matlab
% Generate initial FIS using grid partition
numMFs = [2 2 2];  % 2 MFs for each of 3 inputs
mfType = 'gaussmf';

fis = genfis1(trainData, numMFs, mfType);

% Train with anfis
opt = anfisOptions;
opt.EpochNumber = 50;
trainedFIS = anfis(trainData, fis, opt);
```

### Subtractive Clustering (genfis2/3)

```matlab
% Generate FIS using subtractive clustering
radii = 0.5;
fis = genfis2(train_data.train_input, train_data.train_output, radii);

% Train
opt = anfisOptions;
opt.EpochNumber = 50;
trainedFIS = anfis(trainData, fis, opt);
```

---

## Excel Files

### nucleus_selection.xlsx

Lists all nuclei used in this specific dataset:

| NUCLEUS | A | Z | N | SPIN | PARITY | Beta_2 | MM | Q | p_factor |
|---------|---|---|---|------|--------|--------|-----|-----|----------|
| H2      | 2 | 1 | 1 | 1    | 1      | 0.0    | 0.857 | 0.003 | 0.5 |
| He4     | 4 | 2 | 2 | 0    | 1      | 0.0    | 0.0   | 0.0   | 0.0 |
| ...     | ... | ... | ... | ...  | ...    | ...    | ...   | ...   | ... |

**Purpose**: Identify exactly which nuclei are in train/validation/test sets

### Dataset_Catalog.xlsx

Master catalog of ALL generated datasets:

**Sheet: All_Datasets**
- dataset_name
- target
- nucleus_count
- scenario
- split_ratios
- n_train, n_check, n_test
- total_samples
- feature_set
- scaling method
- sampling method

**Sheets by Target**: Separate sheets for MM, QM, MM_QM, Beta_2

**Sheet: Summary**
- Total datasets generated
- Datasets per target
- Sample size statistics

### Master_Nuclei_Catalog.xlsx

Complete catalog of all nuclei in the database:

**Sheet: All_Nuclei**
- All 267 nuclei with complete information
- Sorted by Z, then N

**Sheets Z1, Z2, ..., Z92**
- Individual sheets for each element
- All isotopes of that element
- Easy to find specific elements

---

## Nuclei Catalog

### What Nuclei Are Included?

The dataset uses data from **aaa2.txt**, which contains 267 nuclei with measured:
- Magnetic moments (MM)
- Quadrupole moments (Q)
- Beta_2 deformation parameters
- Spin and parity
- Other nuclear properties

### Coverage

- **Z range**: 1 (Hydrogen) to 92 (Uranium)
- **A range**: 2 to 262
- **Isotopes**: Multiple isotopes for most elements
- **Variety**: Even-even, odd-odd, even-odd, odd-even nuclei

### Special Nuclei

**Magic Nuclei** (Z or N = magic number):
- He-4 (Z=2, N=2) - Double magic
- O-16 (Z=8, N=8) - Double magic
- Ca-40 (Z=20, N=20) - Double magic
- Ca-48 (Z=20, N=28) - Double magic
- Pb-208 (Z=82, N=126) - Double magic

**Deformed Nuclei** (|Beta_2| > 0.2):
- Rare earth nuclei (Z=57-71)
- Actinide nuclei (Z=89-92)
- Some medium-mass nuclei

---

## Dataset Configuration

### Nucleus Counts

Choose from:
- **75 nuclei**: Small, fast training
- **100 nuclei**: Moderate size
- **150 nuclei**: Larger dataset
- **200 nuclei**: Very large
- **ALL**: Use all available nuclei (~267)

### Target Variables

1. **MM (Magnetic Moment)**
   - Single output: Magnetic moment in nuclear magnetons (μN)
   - 200+ nuclei with measured values

2. **QM (Quadrupole Moment)**
   - Single output: Quadrupole moment in barns
   - 150+ nuclei with measured values

3. **MM_QM (Combined)**
   - Two outputs: Both MM and Q
   - Only nuclei with both measurements
   - ~120+ nuclei

4. **Beta_2 (Deformation)**
   - Single output: Quadrupole deformation parameter
   - 100+ nuclei with measured values

### Feature Sets

Multiple feature combinations available:

**Basic**:
- AZN: Just A, Z, N
- AZNS: A, Z, N, Spin
- AZNP: A, Z, N, Parity
- AZNSP: A, Z, N, Spin, Parity

**With Physics**:
- AZN_beta: A, Z, N, Beta_2
- AZN_p: A, Z, N, p_factor
- AZN_beta_p: A, Z, N, Beta_2, p_factor

**Advanced**:
- ADVANCED: A, Z, N, Spin, Parity, Beta_2, p_factor, BE_per_A, magic distances

**Beta_2 Specific** (when predicting Beta_2):
- Beta2_AZ, Beta2_AN, Beta2_ZN (2 inputs)
- Beta2_AZN, Beta2_AZP, etc. (3-4 inputs)
- Beta2_Physics (8 inputs with full physics)

### Scaling Methods

1. **none**: No scaling (raw values)
2. **standard**: StandardScaler (mean=0, std=1)
3. **robust**: RobustScaler (robust to outliers)

### Sampling Methods

1. **random**: Pure random sampling
2. **stratified**: Maintains distribution of nucleus types, mass groups, etc.

### Anomaly Modes

1. **anomalisiz**: Anomalies filtered out (cleaner data)
2. **anomalili**: Includes anomalies (complete data)

---

## Usage Examples

### Python: Load and Analyze

```python
import pandas as pd
import numpy as np

# Load training data
train_df = pd.read_csv('train.csv')
check_df = pd.read_csv('check.csv')
test_df = pd.read_csv('test.csv')

# Load metadata
import json
with open('metadata.json', 'r') as f:
    meta = json.load(f)

print(f"Dataset: {meta['dataset_name']}")
print(f"Target: {meta['target']}")
print(f"Features: {meta['features']}")
print(f"Train samples: {meta['n_train']}")
print(f"Check samples: {meta['n_check']}")
print(f"Test samples: {meta['n_test']}")

# Load nuclei list
nuclei = pd.read_excel('nucleus_selection.xlsx')
print(f"\nNuclei used: {len(nuclei)}")
print(f"Z range: {nuclei['Z'].min()}-{nuclei['Z'].max()}")
print(f"A range: {nuclei['A'].min()}-{nuclei['A'].max()}")

# Load distribution report
dist_summary = pd.read_excel('nuclei_distribution_report.xlsx',
                             sheet_name='Summary')
print("\nDistribution Summary:")
print(dist_summary)
```

### Python: Train a Model

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Separate features and targets
feature_cols = ['A', 'Z', 'N', 'SPIN', 'PARITY']
target_col = 'MM'

X_train = train[feature_cols]
y_train = train[target_col]
X_test = test[feature_cols]
y_test = test[target_col]

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")

# Save model
joblib.dump(model, 'trained_model.pkl')
```

### R: Load and Analyze

```r
# Load training data
train <- read.csv('train.csv')
check <- read.csv('check.csv')
test <- read.csv('test.csv')

# Load nuclei list
library(readxl)
nuclei <- read_excel('nucleus_selection.xlsx')

# Summary statistics
summary(train)
summary(nuclei)

# Distribution analysis
dist_report <- read_excel('nuclei_distribution_report.xlsx',
                          sheet = 'Summary')
print(dist_report)

# Z distribution
z_dist <- read_excel('nuclei_distribution_report.xlsx',
                     sheet = 'Z_Distribution')
barplot(z_dist$Count, names.arg = z_dist$Z,
        xlab = 'Proton Number (Z)', ylab = 'Count',
        main = 'Z Distribution')
```

---

## Quality Assurance

### Data Validation

✅ **NaN Handling**: Automatic removal of NaN values
✅ **Consistency Checks**: Verify A = Z + N
✅ **Range Validation**: Physical limits on values
✅ **Anomaly Detection**: Statistical outlier detection

### Stratified Sampling

Ensures representative samples by maintaining:
- Mass group distribution
- Nucleus type distribution (even-even, etc.)
- Isotope diversity
- Magic number proximity

### Reproducibility

- **Fixed random seeds** (random_state=42)
- **Saved scalers** for consistent preprocessing
- **Metadata logging** for full traceability
- **Version control** of configuration

### Quality Reports

Each dataset generation creates:
1. Distribution analysis (verify representativeness)
2. Nuclei lists (verify no data leakage)
3. Metadata (verify configuration)
4. Catalog (track all datasets)

---

## Summary

This dataset generation system provides:

✅ **3 Training Scenarios**: 60/20/20, 70/15/15, 80/10/10
✅ **3 Output Formats**: CSV, Excel, MATLAB
✅ **Detailed Reports**: Distribution analysis for every dataset
✅ **MATLAB Ready**: Direct .mat files for ANFIS training
✅ **Excel Reports**: Professional-quality documentation
✅ **Nuclei Lists**: Complete transparency of data splits
✅ **Quality Assured**: Validated, reproducible, well-documented

### Quick Reference

| Feature | Available Options |
|---------|------------------|
| **Scenarios** | S60 (60/20/20), S70 (70/15/15), S80 (80/10/10) |
| **Nuclei Counts** | 75, 100, 150, 200, ALL |
| **Targets** | MM, QM, MM_QM, Beta_2 |
| **Formats** | CSV, XLSX, MAT |
| **Scaling** | none, standard, robust |
| **Sampling** | random, stratified |
| **Reports** | distribution_report.xlsx, nucleus_selection.xlsx |

---

**For more details, see**:
- `pfaz_modules/pfaz01_dataset_generation/README.md` - Module documentation
- `core_modules/constants.py` - Configuration constants
- `MASTER_PROJECT_CHECKLIST.md` - Overall project status

**Questions or issues?**
- Check the logs in the output directory
- Review the metadata.json files
- Consult the distribution reports

---

**Last Updated**: 2025-11-21
**Version**: 2.0
**Author**: Nuclear Physics AI Project
