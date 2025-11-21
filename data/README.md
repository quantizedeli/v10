# Data Directory

## Description

Contains nuclear physics datasets and data files.

## Files

- `aaa2.txt` - Main dataset with 267 nuclei data (Z, N, A, Binding Energy, etc.)

## Data Format

The aaa2.txt file contains nuclear data with columns:
- Z: Proton number
- N: Neutron number
- A: Mass number (A = Z + N)
- BE: Binding energy (MeV)
- Additional nuclear properties

## Usage

```python
from pfaz_modules.pfaz01_dataset_generation import data_loader

# Load data
data = data_loader.load_nuclear_data('data/aaa2.txt')
```
