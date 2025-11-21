# PFAZ 13: AutoML Integration

## Description

AutoML Integration - Automated hyperparameter optimization, feature engineering, and model selection.

## Modules

- `automl_anfis_optimizer.py` - ANFIS AutoML optimization
- `automl_hyperparameter_optimizer.py` - Hyperparameter AutoML
- `automl_feature_engineer.py` - Automated feature engineering
- `automl_visualizer.py` - AutoML visualization
- `automl_logging_reporting_system.py` - AutoML logging and reporting
- `automl_optimizer.py` - General AutoML optimizer

## Usage

```python
from pfaz_modules.pfaz13_automl import automl_hyperparameter_optimizer

# Run AutoML
best_model = automl_hyperparameter_optimizer.optimize(X_train, y_train)
```
