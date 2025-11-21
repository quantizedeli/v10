# PFAZ 04: Unknown Nuclei Predictions

## Description

Unknown Nuclei Predictions - Extrapolation to unknown nuclei with uncertainty quantification and generalization analysis.

## Modules

- `unknown_nuclei_predictor.py` - Main prediction module for unknown nuclei
- `unknown_nuclei_splitter.py` - Dataset splitting for unknown nuclei
- `all_nuclei_predictor.py` - Predictions for all nuclei in dataset
- `generalization_analyzer.py` - Generalization capability analysis

## Usage

```python
from pfaz_modules.pfaz04_unknown_predictions import unknown_nuclei_predictor

# Predict unknown nuclei
predictions = unknown_nuclei_predictor.predict_unknown(models, unknown_data)
```
