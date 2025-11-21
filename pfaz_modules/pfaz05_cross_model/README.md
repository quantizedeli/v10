# PFAZ 05: Cross-Model Analysis

## Description

Cross-Model Analysis - Model comparison, agreement analysis, and best model selection.

## Modules

- `cross_model_evaluator.py` - Main cross-model evaluation
- `faz5_complete_cross_model.py` - Complete cross-model pipeline
- `best_model_selector.py` - Best model selection criteria
- `faz5_cross_model_analysis.py` - Detailed cross-model analysis

## Usage

```python
from pfaz_modules.pfaz05_cross_model import cross_model_evaluator

# Evaluate models
comparison = cross_model_evaluator.compare_all_models(ai_models, anfis_models)
```
