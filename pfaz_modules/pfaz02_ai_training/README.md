# PFAZ 02: AI Model Training

## Description

AI Model Training phase - Training multiple models including Random Forest, XGBoost, DNN, BNN, and PINN with hyperparameter optimization.

## Modules

- `model_trainer.py` - Main training orchestrator
- `hyperparameter_tuner.py` - Hyperparameter optimization
- `model_validator.py` - Model validation and cross-validation
- `parallel_ai_trainer.py` - Parallel training for multiple models
- `advanced_models.py` - Advanced architectures (DNN, BNN, PINN)
- `overfitting_detector.py` - Overfitting detection and prevention

## Usage

```python
from pfaz_modules.pfaz02_ai_training import model_trainer

# Train models
results = model_trainer.train_all_models(X_train, y_train, X_test, y_test)
```
