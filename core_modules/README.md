# Core Modules

## Description

Core modules containing constants, progress tracking, and anomaly detection used across all PFAZ phases.

## Modules

- `constants_v1.1.0.py` - Project constants and configuration (v1.1.0)
- `constants.py` - Base constants
- `progress_tracker.py` - Progress tracking and logging
- `anomaly_detector.py` - Anomaly detection utilities

## Usage

```python
from core_modules import constants_v1_1_0 as constants
from core_modules import progress_tracker

# Use constants
Z_MIN = constants.Z_MIN

# Track progress
tracker = progress_tracker.ProgressTracker()
```
