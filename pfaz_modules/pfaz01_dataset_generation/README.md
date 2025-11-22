# PFAZ 01: Dataset Generation

## Description

Dataset Generation phase - Comprehensive nuclear physics dataset creation with theoretical calculations, quality control, and detailed distribution analysis.

## Features

- **Multiple Training Scenarios**: 70/15/15, 80/10/10 (train/validation/test splits)
- **Nuclei Counts**: 75, 100, 150, 200, and ALL available nuclei
- **Target Variables**: Magnetic Moment (MM), Quadrupole Moment (Q), Combined (MM_QM), Beta_2 deformation
- **Advanced Sampling**: Stratified and random sampling methods
- **Quality Control**: Anomaly detection and filtering
- **Comprehensive Reporting**: Nuclei distributions, statistics, and catalogs

## Training Scenarios

The module supports two different data split scenarios:

1. **S70 (70/15/15)**: Standard split
   - 70% Training
   - 15% Validation (Check)
   - 15% Testing

2. **S80 (80/10/10)**: High training ratio
   - 80% Training
   - 10% Validation (Check)
   - 10% Testing

## Output Formats

Each generated dataset includes:

1. **Data Files**:
   - `train.csv`, `check.csv`, `test.csv` - CSV format
   - `train.xlsx`, `check.xlsx`, `test.xlsx` - Excel format
   - `train.mat`, `check.mat`, `test.mat` - MATLAB format

2. **Documentation**:
   - `metadata.json` - Dataset configuration and statistics
   - `nucleus_selection.xlsx` - List of nuclei used in this dataset
   - `nuclei_distribution_report.xlsx` - Detailed distribution analysis
   - `scaler.pkl` - Fitted scaler (if scaling applied)

3. **Master Catalog**:
   - `Master_Nuclei_Catalog.xlsx` - Complete catalog of all nuclei used across all datasets
   - `Dataset_Catalog.xlsx` - Summary of all generated datasets

## Distribution Reports

The `nuclei_distribution_report.xlsx` includes:

- **Summary**: Total nuclei, ranges, unique counts
- **Z Distribution**: Proton number distribution
- **N Distribution**: Neutron number distribution
- **A Distribution**: Mass number distribution
- **Mass Groups**: Light, medium, heavy nuclei counts
- **Nucleus Types**: even-even, odd-odd, even-odd, odd-even distributions
- **Isotope Diversity**: Number of isotopes per element
- **Magic Numbers**: Proximity to magic numbers analysis
- **Deformation**: Beta_2 deformation statistics (when applicable)

## Modules

- `data_loader.py` - Load nuclear physics data from aaa2.txt
- `dataset_generator.py` - Generate training/test datasets with multiple scenarios
- `nuclei_distribution_analyzer.py` - Analyze and report nuclei distributions
- `data_quality_modules.py` - Quality control and validation
- `qm_filter_manager.py` - Quantum mechanics filtering
- `control_group_generator.py` - Control group generation
- `data_enricher.py` - Feature enrichment and engineering

## Usage Example

```python
from pfaz_modules.pfaz01_dataset_generation import DatasetGenerator
from pfaz_modules.pfaz01_dataset_generation import data_loader

# Load enriched data
df = data_loader.load_and_enrich_data()

# Create dataset generator
generator = DatasetGenerator(base_path='ANFIS_Datasets')

# Generate all dataset combinations
# This will create datasets for all scenarios (S70, S80)
generator.generate_all_datasets(df)

# Output will include:
# - Individual datasets in ANFIS_Datasets/{target}/{scenario}/{dataset_name}/
# - Master_Nuclei_Catalog.xlsx - All nuclei used
# - Dataset_Catalog.xlsx - Summary of all datasets
# - distribution_reports/ - Detailed distribution analyses
```

## Generated Files Structure

```
ANFIS_Datasets/
├── Master_Nuclei_Catalog.xlsx
├── Dataset_Catalog.xlsx
├── distribution_reports/
│   └── [Various distribution analysis reports]
├── MM/
│   ├── S70/
│   │   └── MM_75_S70_anomalisiz_AZN_standard_stratified/
│   │       ├── train.csv, train.xlsx, train.mat
│   │       ├── check.csv, check.xlsx, check.mat
│   │       ├── test.csv, test.xlsx, test.mat
│   │       ├── metadata.json
│   │       ├── nucleus_selection.xlsx
│   │       ├── nuclei_distribution_report.xlsx
│   │       └── scaler.pkl
│   └── S80/
│       └── [Similar structure]
├── QM/
│   └── [Similar structure]
├── MM_QM/
│   └── [Similar structure]
└── Beta_2/
    └── [Similar structure]
```

## MATLAB Format Details

The `.mat` files are compatible with MATLAB and include:

**train.mat**:
- `train_input`: Input features matrix (n_samples × n_features)
- `train_output`: Target values matrix (n_samples × n_targets)
- `feature_names`: Cell array of feature names
- `target_names`: Cell array of target variable names

**check.mat** and **test.mat**:
- `check_input` / `test_input`: Input features
- `check_output` / `test_output`: Target values

### MATLAB Usage Example

```matlab
% Load training data
data = load('train.mat');

% Access data
X_train = data.train_input;
y_train = data.train_output;
features = data.feature_names;
targets = data.target_names;

% Train ANFIS model
fis = anfis([X_train y_train]);

% Load test data and evaluate
test_data = load('test.mat');
predictions = evalfis(fis, test_data.test_input);
```

## Nuclei Selection

The `nucleus_selection.xlsx` file contains:
- NUCLEUS: Nuclear symbol (e.g., "He4", "O16")
- A: Mass number
- Z: Proton number
- N: Neutron number
- SPIN: Nuclear spin (if available)
- PARITY: Parity (+1 or -1)
- Beta_2: Quadrupole deformation parameter
- MM: Magnetic moment
- Q: Quadrupole moment
- p_factor: Pairing factor

This allows you to identify exactly which nuclei were used in training vs validation vs test sets.

## Configuration

Key parameters (defined in `core_modules/constants.py`):

```python
NUCLEUS_COUNTS = [75, 100, 150, 200, 'ALL']
SCENARIOS = {
    'S70': (0.70, 0.15, 0.15),
    'S80': (0.80, 0.10, 0.10)
}
TARGETS = {
    'MM': ['MM'],
    'QM': ['Q'],
    'MM_QM': ['MM', 'Q'],
    'Beta_2': ['Beta_2']
}
SCALING_METHODS = ['none', 'standard', 'robust']
SAMPLING_METHODS = ['random', 'stratified']
```

## Quality Assurance

- Automatic NaN handling
- Anomaly detection and filtering options
- Stratified sampling for representative splits
- Data validation and consistency checks
- Detailed logging of all operations
