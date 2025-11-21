# PFAZ 01: Dataset Generation

## Description

Dataset Generation phase - Processing 267 nuclei data with feature engineering, quality control, and control group generation.

## Modules

- `data_loader.py` - Load nuclear physics data from aaa2.txt
- `dataset_generator.py` - Generate training/test datasets
- `data_quality_modules.py` - Quality control and validation
- `qm_filter_manager.py` - Quantum mechanics filtering
- `control_group_generator.py` - Control group generation
- `data_enricher.py` - Feature enrichment and engineering

## Usage

```python
from pfaz_modules.pfaz01_dataset_generation import data_loader, dataset_generator

# Load data
data = data_loader.load_nuclear_data()

# Generate datasets
datasets = dataset_generator.create_datasets(data)
```
