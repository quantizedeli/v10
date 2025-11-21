"""
AutoML Hyperparameter Optimization with Optuna

Find optimal hyperparameters automatically using:
- Bayesian optimization (TPE sampler)
- Multi-objective optimization
- Pruning (early stopping)
- Parallel trials
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
import json
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class AutoMLOptimizer:
    """
    AutoML Hyperparameter Optimizer

    Usage:
        optimizer = AutoMLOptimizer(X_train, y_train, X_val, y_val, model_type='xgb')
        study = optimizer.optimize(n_trials=100)
        best_params = study.best_params
    """

    def __init__(self,
                 X_train, y_train,
                 X_val, y_val,
                 model_type='rf'):
        """
        Initialize AutoML optimizer

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_type: 'rf', 'xgb', 'dnn', 'gbm'
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_type = model_type.lower()

        logger.info(f"✓ AutoML Optimizer initialized: {model_type.upper()}")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Val: {len(X_val)} samples")

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna

        This function is called for each trial.
        It suggests hyperparameters, trains model, evaluates, returns score.

        Args:
            trial: Optuna trial object

        Returns:
            score: float (to maximize)
        """

        from sklearn.metrics import r2_score

        # Suggest hyperparameters based on model type
        if self.model_type == 'rf':
            params = self._suggest_rf_params(trial)
            model = self._train_rf(params)

        elif self.model_type == 'xgb':
            params = self._suggest_xgb_params(trial)
            model = self._train_xgb(params)

        elif self.model_type == 'gbm':
            params = self._suggest_gbm_params(trial)
            model = self._train_gbm(params)

        elif self.model_type == 'dnn':
            params = self._suggest_dnn_params(trial)
            model = self._train_dnn(params, trial)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Evaluate
        y_pred = model.predict(self.X_val)
        r2 = r2_score(self.y_val, y_pred)

        # Report for pruning
        trial.report(r2, step=0)

        # Check if should prune
        if trial.should_prune():
            raise optuna.TrialPruned()

        return r2

    def _suggest_rf_params(self, trial) -> Dict:
        """
        Suggest Random Forest hyperparameters
        """
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }

    def _suggest_xgb_params(self, trial) -> Dict:
        """
        Suggest XGBoost hyperparameters
        """
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': 42,
            'n_jobs': -1
        }

    def _suggest_gbm_params(self, trial) -> Dict:
        """Suggest Gradient Boosting hyperparameters"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'random_state': 42
        }

    def _suggest_dnn_params(self, trial) -> Dict:
        """
        Suggest DNN hyperparameters
        """
        n_layers = trial.suggest_int('n_layers', 2, 5)

        hidden_layers = []
        for i in range(n_layers):
            units = trial.suggest_int(f'layer_{i}_units', 32, 512)
            hidden_layers.append(units)

        return {
            'hidden_layers': hidden_layers,
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),
            'epochs': 100
        }

    def _train_rf(self, params):
        """Train Random Forest"""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(**params)
        model.fit(self.X_train, self.y_train)

        return model

    def _train_xgb(self, params):
        """Train XGBoost"""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            logger.warning("XGBoost not installed, falling back to RandomForest")
            from sklearn.ensemble import RandomForestRegressor
            # Convert params to RF-compatible
            rf_params = {
                'n_estimators': params['n_estimators'],
                'max_depth': params['max_depth'],
                'random_state': params.get('random_state', 42),
                'n_jobs': params.get('n_jobs', -1)
            }
            model = RandomForestRegressor(**rf_params)
            model.fit(self.X_train, self.y_train)
            return model

        # Check for GPU optimization
        try:
            from gpu_optimization import GPUOptimizer
            gpu_opt = GPUOptimizer()
            if gpu_opt.gpu_available:
                params = gpu_opt.optimize_xgboost(params)
        except ImportError:
            pass

        model = XGBRegressor(**params)
        model.fit(self.X_train, self.y_train, verbose=False)

        return model

    def _train_gbm(self, params):
        """Train Gradient Boosting"""
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(**params)
        model.fit(self.X_train, self.y_train)

        return model

    def _train_dnn(self, params, trial):
        """Train DNN"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            logger.warning("TensorFlow not installed, falling back to RandomForest")
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(self.X_train, self.y_train)
            return model

        # Build model
        model = Sequential()

        # Input layer
        model.add(Dense(params['hidden_layers'][0], activation='relu',
                       input_shape=(self.X_train.shape[1],)))

        # Hidden layers
        for units in params['hidden_layers'][1:]:
            model.add(Dense(units, activation='relu'))
            if params['dropout'] > 0:
                model.add(Dropout(params['dropout']))

        # Output
        model.add(Dense(1))

        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(params['learning_rate']),
            loss='mse'
        )

        # Train
        try:
            model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                verbose=0,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True),
                    # Optuna pruning callback
                    optuna.integration.TFKerasPruningCallback(trial, 'val_loss')
                ]
            )
        except Exception as e:
            logger.warning(f"DNN training failed: {e}, using simple model")
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(self.X_train, self.y_train)

        return model

    def optimize(self,
                n_trials=100,
                n_jobs=1,
                timeout=None) -> optuna.Study:
        """
        Run optimization

        Args:
            n_trials: Number of trials
            n_jobs: Parallel workers (1 = sequential)
            timeout: Max time in seconds

        Returns:
            study: Optuna study with results
        """

        logger.info(f"\n{'='*70}")
        logger.info(f"AUTOML OPTIMIZATION: {self.model_type.upper()}")
        logger.info(f"{'='*70}")
        logger.info(f"Trials: {n_trials}")
        logger.info(f"Parallel: {n_jobs} worker(s)")
        if timeout:
            logger.info(f"Timeout: {timeout}s ({timeout/3600:.1f}h)")

        # Create study
        study = optuna.create_study(
            study_name=f'automl_{self.model_type}',
            direction='maximize',  # Maximize R²
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )

        # Optimize
        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=True
        )

        # Results
        logger.info(f"\n{'='*70}")
        logger.info(f"OPTIMIZATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Best R²: {study.best_value:.4f}")
        logger.info(f"Best params:")
        for k, v in study.best_params.items():
            logger.info(f"  {k}: {v}")
        logger.info(f"Total trials: {len(study.trials)}")
        logger.info(f"Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        logger.info(f"Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

        return study

    def save_results(self, study: optuna.Study, output_file: str):
        """
        Save optimization results
        """

        results = {
            'model_type': self.model_type,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'trials_data': []
        }

        for trial in study.trials:
            results['trials_data'].append({
                'trial_id': trial.number,
                'params': trial.params,
                'value': trial.value,
                'state': trial.state.name
            })

        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"💾 Results saved: {output_path}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_optimization(study: optuna.Study, output_dir='automl_plots'):
    """
    Create optimization visualizations
    """

    try:
        import optuna.visualization as vis
    except ImportError:
        logger.warning("Optuna visualization not available")
        return

    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info(f"\n📊 Generating visualizations...")

    try:
        # 1. Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(output_dir / 'optimization_history.html'))
        logger.info(f"  ✓ optimization_history.html")
    except Exception as e:
        logger.warning(f"  ✗ optimization_history.html: {e}")

    try:
        # 2. Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(str(output_dir / 'param_importances.html'))
        logger.info(f"  ✓ param_importances.html")
    except Exception as e:
        logger.warning(f"  ✗ param_importances.html: {e}")

    try:
        # 3. Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(str(output_dir / 'parallel_coordinate.html'))
        logger.info(f"  ✓ parallel_coordinate.html")
    except Exception as e:
        logger.warning(f"  ✗ parallel_coordinate.html: {e}")

    try:
        # 4. Contour plot
        fig = vis.plot_contour(study)
        fig.write_html(str(output_dir / 'contour.html'))
        logger.info(f"  ✓ contour.html")
    except Exception as e:
        logger.warning(f"  ✗ contour.html: {e}")

    logger.info(f"✓ Visualizations saved to {output_dir}/")


# ============================================================================
# MULTI-TARGET OPTIMIZATION
# ============================================================================

def optimize_all_targets(X_train, y_train_dict, X_val, y_val_dict,
                        model_type='rf', n_trials=100,
                        output_dir='automl_results'):
    """
    Optimize hyperparameters for all targets

    Args:
        X_train, X_val: Feature matrices
        y_train_dict: {'MM': y_mm_train, 'QM': y_qm_train, 'Beta_2': y_beta_train}
        y_val_dict: {'MM': y_mm_val, 'QM': y_qm_val, 'Beta_2': y_beta_val}
        model_type: Model type
        n_trials: Number of trials per target
        output_dir: Output directory for results

    Returns:
        results_dict: {'MM': study, 'QM': study, 'Beta_2': study}
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    results_dict = {}

    for target in ['MM', 'QM', 'Beta_2']:
        logger.info(f"\n{'='*70}")
        logger.info(f"OPTIMIZING FOR TARGET: {target}")
        logger.info(f"{'='*70}")

        # Create optimizer
        optimizer = AutoMLOptimizer(
            X_train, y_train_dict[target],
            X_val, y_val_dict[target],
            model_type=model_type
        )

        # Optimize
        study = optimizer.optimize(n_trials=n_trials)

        # Save results
        optimizer.save_results(study, str(output_dir / f'{model_type}_{target}_results.json'))

        # Visualize
        visualize_optimization(study, output_dir=str(output_dir / f'{model_type}_{target}_plots'))

        results_dict[target] = study

    return results_dict


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def cli_automl():
    """CLI for AutoML optimization"""

    import argparse

    parser = argparse.ArgumentParser(description="AutoML Optimizer CLI")
    parser.add_argument('--model', type=str, required=True,
                       choices=['rf', 'xgb', 'gbm', 'dnn'],
                       help='Model type')
    parser.add_argument('--target', type=str, required=True,
                       choices=['MM', 'QM', 'Beta_2'],
                       help='Target property')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of trials')
    parser.add_argument('--jobs', type=int, default=1,
                       help='Parallel workers')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds')
    parser.add_argument('--data', type=str, default='aaa2.txt',
                       help='Data file path')

    args = parser.parse_args()

    # Load data
    logger.info("Loading data...")
    try:
        from data_loader import load_and_prepare_data
        X_train, X_val, y_train, y_val = load_and_prepare_data(
            args.data,
            target=args.target
        )
    except ImportError:
        logger.error("data_loader module not found")
        logger.info("Please ensure data_loader.py is available or provide custom data loading")
        return

    # Optimize
    optimizer = AutoMLOptimizer(X_train, y_train, X_val, y_val,
                               model_type=args.model)
    study = optimizer.optimize(n_trials=args.trials,
                              n_jobs=args.jobs,
                              timeout=args.timeout)

    # Save results
    output_file = f'automl_{args.model}_{args.target}_results.json'
    optimizer.save_results(study, output_file)

    # Visualize
    visualize_optimization(study, output_dir=f'automl_{args.model}_plots')

    print(f"\n✓ Optimization complete!")
    print(f"  Best R²: {study.best_value:.4f}")
    print(f"  Results: {output_file}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    cli_automl()
