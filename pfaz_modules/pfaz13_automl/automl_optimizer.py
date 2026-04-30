"""
AutoML Hyperparameter Optimization with Optuna

Find optimal hyperparameters automatically using:
- Bayesian optimization (TPE sampler)
- Multi-objective optimization
- Pruning (early stopping)
- Parallel trials

Supported model types: rf, xgb, gbm, lgb, cb, svr, dnn
Targets: MM, QM, Beta_2, MM_QM
"""

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    MedianPruner = None
    TPESampler = None
    OPTUNA_AVAILABLE = False

import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
import json
from pathlib import Path
import time

logger = logging.getLogger(__name__)

ALL_TARGETS = ['MM', 'QM', 'Beta_2', 'MM_QM']


class AutoMLOptimizer:
    """
    AutoML Hyperparameter Optimizer

    Usage:
        optimizer = AutoMLOptimizer(X_train, y_train, X_val, y_val, model_type='xgb')
        study = optimizer.optimize(n_trials=100)
        best_params = study.best_params
    """

    SUPPORTED_MODELS = ['rf', 'xgb', 'gbm', 'lgb', 'cb', 'svr', 'dnn']

    def __init__(self,
                 X_train, y_train,
                 X_val, y_val,
                 model_type='rf',
                 gpu_enabled: bool = False):
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.X_val   = np.asarray(X_val)
        self.y_val   = np.asarray(y_val)
        self.model_type = model_type.lower()
        self.gpu_enabled = gpu_enabled

        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Supported: {self.SUPPORTED_MODELS}")

        logger.info(f"[OK] AutoML Optimizer initialized: {model_type.upper()}, gpu={gpu_enabled}")
        logger.info(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples")

    def objective(self, trial) -> float:
        from sklearn.metrics import r2_score

        if self.model_type == 'rf':
            params = self._suggest_rf_params(trial)
            model = self._train_rf(params)
        elif self.model_type == 'xgb':
            params = self._suggest_xgb_params(trial)
            model = self._train_xgb(params)
        elif self.model_type == 'gbm':
            params = self._suggest_gbm_params(trial)
            model = self._train_gbm(params)
        elif self.model_type == 'lgb':
            params = self._suggest_lgb_params(trial)
            model = self._train_lgb(params)
        elif self.model_type == 'cb':
            params = self._suggest_cb_params(trial)
            model = self._train_cb(params)
        elif self.model_type == 'svr':
            params = self._suggest_svr_params(trial)
            model = self._train_svr(params)
        elif self.model_type == 'dnn':
            params = self._suggest_dnn_params(trial)
            model = self._train_dnn(params, trial)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Handle multi-output (MM_QM)
        y_pred = model.predict(self.X_val)
        if self.y_val.ndim > 1 and self.y_val.shape[1] > 1:
            from sklearn.metrics import r2_score
            r2 = float(np.mean([r2_score(self.y_val[:, i], y_pred[:, i])
                                 for i in range(self.y_val.shape[1])]))
        else:
            r2 = float(r2_score(self.y_val.ravel(), y_pred.ravel()))

        # Divergence guard: heavily penalize crashed runs
        if np.isnan(r2) or r2 < -2.0:
            return -2.0

        trial.report(r2, step=0)
        if OPTUNA_AVAILABLE and trial.should_prune():
            raise optuna.TrialPruned()

        return r2

    # -------------------------------------------------------------------------
    # Hyperparameter spaces
    # -------------------------------------------------------------------------

    def _suggest_rf_params(self, trial) -> Dict:
        return {
            'n_estimators':      trial.suggest_int('n_estimators', 50, 500),
            'max_depth':         trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42, 'n_jobs': 1,
        }

    def _suggest_xgb_params(self, trial) -> Dict:
        return {
            'n_estimators':    trial.suggest_int('n_estimators', 50, 500),
            'max_depth':       trial.suggest_int('max_depth', 3, 12),
            'learning_rate':   trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample':       trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':       trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda':      trial.suggest_float('reg_lambda', 0.0, 5.0),
            'random_state': 42, 'n_jobs': 1, 'verbosity': 0,
        }

    def _suggest_gbm_params(self, trial) -> Dict:
        return {
            'n_estimators':  trial.suggest_int('n_estimators', 50, 500),
            'max_depth':     trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample':     trial.suggest_float('subsample', 0.5, 1.0),
            'random_state': 42,
        }

    def _suggest_lgb_params(self, trial) -> Dict:
        return {
            'n_estimators':    trial.suggest_int('n_estimators', 50, 500),
            'num_leaves':      trial.suggest_int('num_leaves', 20, 200),
            'learning_rate':   trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample':       trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':       trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda':      trial.suggest_float('reg_lambda', 0.0, 5.0),
            'min_child_samples':trial.suggest_int('min_child_samples', 5, 50),
            'random_state': 42, 'n_jobs': 1, 'verbose': -1,
        }

    def _suggest_cb_params(self, trial) -> Dict:
        return {
            'iterations':     trial.suggest_int('iterations', 50, 500),
            'depth':          trial.suggest_int('depth', 3, 10),
            'learning_rate':  trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'l2_leaf_reg':    trial.suggest_float('l2_leaf_reg', 0.5, 10.0),
            'random_seed': 42, 'verbose': 0,
        }

    def _suggest_svr_params(self, trial) -> Dict:
        return {
            'C':       trial.suggest_float('C', 0.01, 100.0, log=True),
            'epsilon': trial.suggest_float('epsilon', 1e-4, 1.0, log=True),
            'kernel':  trial.suggest_categorical('kernel', ['rbf', 'linear']),
            'gamma':   trial.suggest_categorical('gamma', ['scale', 'auto']),
        }

    def _suggest_dnn_params(self, trial) -> Dict:
        n_layers = trial.suggest_int('n_layers', 2, 4)
        return {
            'hidden_layers': [trial.suggest_int(f'units_{i}', 32, 256) for i in range(n_layers)],
            'dropout':       trial.suggest_float('dropout', 0.0, 0.4),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size':    trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'epochs': 80,
        }

    # -------------------------------------------------------------------------
    # Trainers
    # -------------------------------------------------------------------------

    def _wrap_multi(self, base_model):
        """Wrap in MultiOutputRegressor for multi-output targets."""
        if self.y_train.ndim > 1 and self.y_train.shape[1] > 1:
            from sklearn.multioutput import MultiOutputRegressor
            return MultiOutputRegressor(base_model, n_jobs=1)
        return base_model

    def _train_rf(self, params):
        from sklearn.ensemble import RandomForestRegressor
        model = self._wrap_multi(RandomForestRegressor(**params))
        model.fit(self.X_train, self.y_train.ravel() if self.y_train.ndim == 1 else self.y_train)
        return model

    def _train_xgb(self, params):
        try:
            from xgboost import XGBRegressor
            xgb_params = dict(params)
            if self.gpu_enabled:
                from utils.gpu_manager import get_gpu_manager
                _gpu_params = get_gpu_manager().get_xgb_params()
                if _gpu_params:
                    xgb_params.update(_gpu_params)
                    xgb_params.pop('n_jobs', None)
            model = self._wrap_multi(XGBRegressor(**xgb_params))
        except ImportError:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        model.fit(self.X_train, self.y_train.ravel() if self.y_train.ndim == 1 else self.y_train)
        return model

    def _train_gbm(self, params):
        from sklearn.ensemble import GradientBoostingRegressor
        model = self._wrap_multi(GradientBoostingRegressor(**params))
        model.fit(self.X_train, self.y_train.ravel() if self.y_train.ndim == 1 else self.y_train)
        return model

    def _train_lgb(self, params):
        try:
            from lightgbm import LGBMRegressor
            model = self._wrap_multi(LGBMRegressor(**params))
        except ImportError:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        model.fit(self.X_train, self.y_train.ravel() if self.y_train.ndim == 1 else self.y_train)
        return model

    def _train_cb(self, params):
        try:
            from catboost import CatBoostRegressor
            model = self._wrap_multi(CatBoostRegressor(**params))
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train.ravel() if self.y_train.ndim == 1 else self.y_train)
        return model

    def _train_svr(self, params):
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([('scaler', StandardScaler()), ('svr', SVR(**params))])
        model = self._wrap_multi(pipe)
        model.fit(self.X_train, self.y_train.ravel() if self.y_train.ndim == 1 else self.y_train)
        return model

    def _train_dnn(self, params, trial):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.callbacks import EarlyStopping
            from sklearn.preprocessing import StandardScaler

            # Scale targets to prevent DNN divergence (same as PFAZ2 fix)
            y_scaler = StandardScaler()
            y_tr = self.y_train.reshape(-1, 1) if self.y_train.ndim == 1 else self.y_train
            y_vl = self.y_val.reshape(-1, 1) if self.y_val.ndim == 1 else self.y_val
            y_tr_sc = y_scaler.fit_transform(y_tr)
            y_vl_sc = y_scaler.transform(y_vl)

            n_out = y_tr_sc.shape[1]
            model = Sequential()
            model.add(Dense(params['hidden_layers'][0], activation='relu',
                            input_shape=(self.X_train.shape[1],)))
            model.add(BatchNormalization())
            for units in params['hidden_layers'][1:]:
                model.add(Dense(units, activation='relu'))
                if params['dropout'] > 0:
                    model.add(Dropout(params['dropout']))
            model.add(Dense(n_out))

            model.compile(
                optimizer=tf.keras.optimizers.Adam(params['learning_rate'],
                                                   clipnorm=1.0),
                loss='huber'
            )
            model.fit(
                self.X_train, y_tr_sc,
                validation_data=(self.X_val, y_vl_sc),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                verbose=0,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
            )

            # Wrap so predict returns original scale
            class _WrappedDNN:
                def __init__(self, m, sc):
                    self._m, self._sc = m, sc
                def predict(self, X):
                    p = self._m.predict(X, verbose=0)
                    return self._sc.inverse_transform(p).ravel()

            return _WrappedDNN(model, y_scaler)

        except Exception as e:
            logger.warning(f"DNN training failed: {e}, falling back to RF")
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
            m.fit(self.X_train, self.y_train.ravel())
            return m

    # -------------------------------------------------------------------------
    # Build a fitted model from explicit params (for test-set prediction)
    # -------------------------------------------------------------------------

    def _build_model(self, params: Dict):
        """
        Verilen parametrelerle model olusturur ve X_train/y_train uzerinde egitir.
        optimize() sonrasi test seti tahmini icin kullanilir.
        """
        mt = self.model_type
        if mt == 'rf':
            return self._train_rf(params)
        elif mt == 'xgb':
            return self._train_xgb(params)
        elif mt == 'gbm':
            return self._train_gbm(params)
        elif mt == 'lgb':
            return self._train_lgb(params)
        elif mt == 'cb':
            return self._train_cb(params)
        elif mt == 'svr':
            return self._train_svr(params)
        elif mt == 'dnn':
            # DNN icin trial-less params — dummy trial
            class _DummyTrial:
                def suggest_int(self, n, lo, hi): return params.get(n, lo)
                def suggest_float(self, n, lo, hi, **kw): return params.get(n, lo)
                def suggest_categorical(self, n, choices): return params.get(n, choices[0])
                def report(self, *a): pass
                def should_prune(self): return False
            return self._train_dnn(params, _DummyTrial())
        else:
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
            m.fit(self.X_train, self.y_train.ravel() if self.y_train.ndim == 1 else self.y_train)
            return m

    # -------------------------------------------------------------------------
    # Run optimization
    # -------------------------------------------------------------------------

    def optimize(self, n_trials=50, n_jobs=1, timeout=None):
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("optuna is not installed. Run: pip install optuna")

        logger.info(f"\n{'='*60}")
        logger.info(f"AUTOML: {self.model_type.upper()} | {n_trials} trials")
        logger.info(f"{'='*60}")

        study = optuna.create_study(
            study_name=f'automl_{self.model_type}',
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs,
                       timeout=timeout, show_progress_bar=False)

        logger.info(f"Best R^2: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        logger.info(f"Trials: {len(study.trials)} total")
        return study

    def save_results(self, study, output_file: str):
        results = {
            'model_type': self.model_type,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'trials_data': [
                {'trial_id': t.number, 'params': t.params,
                 'value': t.value, 'state': t.state.name}
                for t in study.trials
            ]
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"[SAVE] {output_file}")


# =============================================================================
# VISUALIZATION (requires optuna[visualization])
# =============================================================================

def visualize_optimization(study, output_dir='automl_plots'):
    if not OPTUNA_AVAILABLE:
        return
    try:
        import optuna.visualization as vis
    except ImportError:
        logger.warning("optuna[visualization] not available. Run: pip install optuna[visualization]")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for name, fn in [
        ('optimization_history', vis.plot_optimization_history),
        ('param_importances',    vis.plot_param_importances),
        ('parallel_coordinate',  vis.plot_parallel_coordinate),
    ]:
        try:
            fig = fn(study)
            fig.write_html(str(output_dir / f'{name}.html'))
            logger.info(f"  [OK] {name}.html")
        except Exception as e:
            logger.warning(f"  [SKIP] {name}: {e}")


# =============================================================================
# MULTI-TARGET OPTIMIZATION (MM, QM, Beta_2, MM_QM)
# =============================================================================

def optimize_all_targets(X_train, y_train_dict, X_val, y_val_dict,
                         model_type='rf', n_trials=50,
                         output_dir='automl_results'):
    """
    Optimize hyperparameters for all targets.

    Args:
        X_train, X_val: Feature matrices
        y_train_dict: {'MM': ..., 'QM': ..., 'Beta_2': ..., 'MM_QM': ...}
        y_val_dict:   {'MM': ..., 'QM': ..., 'Beta_2': ..., 'MM_QM': ...}
        model_type: One of AutoMLOptimizer.SUPPORTED_MODELS
        n_trials: Number of trials per target
        output_dir: Output directory

    Returns:
        results_dict: {target: optuna.Study}
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("[SKIP] optuna not installed -- AutoML skipped")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    results_dict = {}

    for target in ALL_TARGETS:
        if target not in y_train_dict:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"AUTOML TARGET: {target}")
        logger.info(f"{'='*60}")

        try:
            optimizer = AutoMLOptimizer(
                X_train, y_train_dict[target],
                X_val,   y_val_dict[target],
                model_type=model_type
            )
            study = optimizer.optimize(n_trials=n_trials)
            optimizer.save_results(study, str(output_dir / f'{model_type}_{target}_results.json'))
            visualize_optimization(study, output_dir=str(output_dir / f'{model_type}_{target}_plots'))
            results_dict[target] = study
        except Exception as e:
            logger.error(f"[ERROR] AutoML {target}: {e}")

    return results_dict
