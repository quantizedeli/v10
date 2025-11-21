# Physics Modules

## Description

Physics calculation modules for nuclear physics models including SEMF, Woods-Saxon potential, and Nilsson model.

## Modules

- `semf_calculator.py` - Semi-Empirical Mass Formula calculations
- `woods_saxon.py` - Woods-Saxon potential calculations
- `nilsson_model.py` - Nilsson model implementation
- `theoretical_calculations_manager.py` - Manager for all theoretical calculations

## Usage

```python
from physics_modules import semf_calculator

# Calculate binding energy
BE = semf_calculator.calculate_binding_energy(Z, N, A)
```
