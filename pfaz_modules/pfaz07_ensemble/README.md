# PFAZ 07: Ensemble Methods

## Description

Ensemble Methods - Stacking, voting, bagging, and meta-learning for improved predictions.

## Modules

- `ensemble_model_builder.py` - Ensemble model construction
- `stacking_meta_learner.py` - Stacking and meta-learning
- `ensemble_evaluator.py` - Ensemble performance evaluation
- `faz7_ensemble_pipeline.py` - Ensemble pipeline
- `pfaz7_complete_ensemble_pipeline.py` - Complete ensemble system
- `pfaz7_ensemble.py` - Main ensemble module

## Usage

```python
from pfaz_modules.pfaz07_ensemble import ensemble_model_builder

# Build ensemble
ensemble = ensemble_model_builder.create_ensemble(base_models)
```
