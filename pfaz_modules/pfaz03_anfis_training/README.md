# PFAZ 03: ANFIS Training

## Description

ANFIS (Adaptive Neuro-Fuzzy Inference System) Training phase - 8 different configurations with MATLAB integration.

## Modules

- `matlab_anfis_trainer.py` - MATLAB ANFIS integration
- `anfis_parallel_trainer_v2.py` - Parallel ANFIS training
- `anfis_config_manager.py` - Configuration management
- `anfis_adaptive_strategy.py` - Adaptive training strategies
- `anfis_dataset_selector.py` - Dataset selection for ANFIS
- `anfis_visualizer.py` - ANFIS visualization
- `anfis_performance_analyzer.py` - Performance analysis
- `anfis_robustness_tester.py` - Robustness testing
- `anfis_all_nuclei_predictor.py` - Predictions for all nuclei
- `anfis_model_saver.py` - Model persistence

## Usage

```python
from pfaz_modules.pfaz03_anfis_training import matlab_anfis_trainer

# Train ANFIS models
anfis_models = matlab_anfis_trainer.train_all_configs(X_train, y_train)
```
