# -*- coding: utf-8 -*-
"""
FAZ 5: Cross-Model Analysis Pipeline v2.0
==========================================

Loads trained AI (PFAZ2) and ANFIS (PFAZ3) models, predicts on test.csv
of each dataset, then runs cross-model agreement analysis.

For each target (MM, QM, Beta_2, MM_QM), evaluates:
  - Per-nucleus agreement across all models
  - Good / Medium / Poor nucleus classification
  - Model correlation matrix
  - Comprehensive Excel reports

Author: Nuclear Physics AI Project
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from .cross_model_evaluator import CrossModelEvaluator
except ImportError:
    from cross_model_evaluator import CrossModelEvaluator
    CrossModelEvaluator = None
    CrossModelEvaluator = None

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    JOBLIB_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    matplotlib = None
    plt = None
    MATPLOTLIB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Actual CSV column names for targets
TARGET_COLS = {
    'MM':     'MAGNETIC MOMENT [µ]',
    'QM':     'QUADRUPOLE MOMENT [Q]',
    'Beta_2': 'Beta_2',
}
NON_FEATURE_COLS = {'NUCLEUS', 'MAGNETIC MOMENT [µ]', 'QUADRUPOLE MOMENT [Q]', 'Beta_2'}


class CrossModelAnalysisPipeline:
    """
    PFAZ5 Cross-Model Analysis Pipeline v2.0

    Collects predictions from all trained models (AI + ANFIS) on test data,
    then performs cross-model agreement analysis per target type.

    Directory structure expected:
      ai_models_dir/  {dataset}/{model_type}/{config}/model_{model_type}_{config}.pkl
      anfis_models_dir/ {dataset}/{config}/model_{config}.pkl
      datasets_dir/   {dataset}/test.csv
    """

    def __init__(self,
                 trained_models_dir: str,
                 output_dir: str = 'cross_model_analysis',
                 anfis_models_dir: Optional[str] = None,
                 datasets_dir: Optional[str] = None):

        self.ai_models_dir = Path(trained_models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-derive sibling dirs if not provided
        base = self.ai_models_dir.parent
        self.anfis_models_dir = Path(anfis_models_dir) if anfis_models_dir else base / 'anfis_models'
        self.datasets_dir = Path(datasets_dir) if datasets_dir else base / 'generated_datasets'

        # {target_key: {model_label: df[nucleus, experimental, predicted]}}
        self.all_predictions: Dict[str, Dict[str, pd.DataFrame]] = {
            'MM': {}, 'QM': {}, 'Beta_2': {}
        }

        logger.info("=" * 80)
        logger.info("PFAZ 5: CROSS-MODEL ANALYSIS PIPELINE v2.0")
        logger.info("=" * 80)
        logger.info(f"AI models dir   : {self.ai_models_dir}")
        logger.info(f"ANFIS models dir: {self.anfis_models_dir}")
        logger.info(f"Datasets dir    : {self.datasets_dir}")
        logger.info(f"Output dir      : {self.output_dir}")
        logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _identify_target_key(self, df: pd.DataFrame) -> Optional[str]:
        """Return 'MM', 'QM', or 'Beta_2' based on CSV columns."""
        if 'Beta_2' in df.columns:
            return 'Beta_2'
        if 'MAGNETIC MOMENT [µ]' in df.columns and 'QUADRUPOLE MOMENT [Q]' not in df.columns:
            return 'MM'
        if 'QUADRUPOLE MOMENT [Q]' in df.columns and 'MAGNETIC MOMENT [µ]' not in df.columns:
            return 'QM'
        # MM_QM dataset: return both (handled separately)
        if 'MAGNETIC MOMENT [µ]' in df.columns and 'QUADRUPOLE MOMENT [Q]' in df.columns:
            return 'MM_QM'
        return None

    def _get_feature_cols(self, df: pd.DataFrame, target_cols_set) -> List[str]:
        return [c for c in df.columns if c not in NON_FEATURE_COLS and c != 'NUCLEUS']

    def _predict_model(self, model, X: np.ndarray) -> Optional[np.ndarray]:
        try:
            if isinstance(model, list):
                return np.column_stack([m.predict(X) for m in model])
            return model.predict(X)
        except Exception as e:
            logger.debug(f"Predict error: {e}")
            return None

    def _load_ai_model(self, pkl_path: Path):
        if not JOBLIB_AVAILABLE:
            return None
        try:
            return joblib.load(pkl_path)
        except Exception:
            return None

    def _load_anfis_model(self, pkl_path: Path):
        if not JOBLIB_AVAILABLE:
            return None
        try:
            return joblib.load(pkl_path)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Prediction collection
    # ------------------------------------------------------------------

    def _collect_all_predictions(self):
        """Walk datasets, load models, predict on test.csv, store per-target."""
        logger.info("\n--- Collecting predictions from all models ---")

        dataset_dirs = sorted([
            d for d in self.datasets_dir.iterdir()
            if d.is_dir() and (d / 'test.csv').exists()
        ])
        logger.info(f"Found {len(dataset_dirs)} datasets with test.csv")

        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            try:
                test_df = pd.read_csv(dataset_dir / 'test.csv')
            except Exception:
                continue

            target_key = self._identify_target_key(test_df)
            if target_key is None:
                continue

            feature_cols = self._get_feature_cols(test_df, NON_FEATURE_COLS)
            if not feature_cols:
                continue

            X_test = test_df[feature_cols].values
            nuclei = test_df['NUCLEUS'].tolist() if 'NUCLEUS' in test_df.columns else [f'N{i}' for i in range(len(X_test))]

            self._collect_ai_predictions(dataset_name, dataset_dir, target_key, X_test, nuclei, test_df)
            self._collect_anfis_predictions(dataset_name, target_key, X_test, nuclei, test_df)

        # Summary
        for tkey, models in self.all_predictions.items():
            logger.info(f"  {tkey}: {len(models)} model predictions collected")

    def _collect_ai_predictions(self, dataset_name, dataset_dir, target_key,
                                 X_test, nuclei, test_df):
        """Load AI models for dataset and collect predictions."""
        ai_dataset_dir = self.ai_models_dir / dataset_name
        if not ai_dataset_dir.exists():
            return

        for pkl in ai_dataset_dir.glob('**/model_*.pkl'):
            config_id = pkl.parent.name      # e.g. DNN_036
            model_type = pkl.parent.parent.name  # e.g. DNN

            model = self._load_ai_model(pkl)
            if model is None:
                continue

            y_pred = self._predict_model(model, X_test)
            if y_pred is None:
                continue

            y_pred = np.asarray(y_pred)

            model_label = f"{model_type}_{config_id}"

            if target_key == 'MM_QM':
                # Handle both MM and QM predictions
                if y_pred.ndim == 2 and y_pred.shape[1] >= 2:
                    self._store_prediction('MM', f"{model_label}_{dataset_name}",
                                           nuclei, test_df['MAGNETIC MOMENT [µ]'].values, y_pred[:, 0])
                    self._store_prediction('QM', f"{model_label}_{dataset_name}",
                                           nuclei, test_df['QUADRUPOLE MOMENT [Q]'].values, y_pred[:, 1])
                elif y_pred.ndim == 1 and 'MAGNETIC MOMENT [µ]' in test_df.columns:
                    self._store_prediction('MM', f"{model_label}_{dataset_name}",
                                           nuclei, test_df['MAGNETIC MOMENT [µ]'].values, y_pred)
            else:
                tcol = TARGET_COLS.get(target_key)
                if tcol and tcol in test_df.columns:
                    y_true = test_df[tcol].values
                    y_out = y_pred.ravel() if y_pred.ndim > 1 else y_pred
                    if len(y_out) == len(y_true):
                        self._store_prediction(target_key, f"{model_label}_{dataset_name}",
                                               nuclei, y_true, y_out)

    def _collect_anfis_predictions(self, dataset_name, target_key,
                                    X_test, nuclei, test_df):
        """Load ANFIS models for dataset and collect predictions."""
        anfis_dataset_dir = self.anfis_models_dir / dataset_name
        if not anfis_dataset_dir.exists():
            return

        for pkl in anfis_dataset_dir.glob('*/model_*.pkl'):
            config_id = pkl.parent.name  # e.g. CFG_Grid_2MF_Gauss

            model = self._load_anfis_model(pkl)
            if model is None:
                continue

            y_pred = self._predict_model(model, X_test)
            if y_pred is None:
                continue

            y_pred = np.asarray(y_pred)
            model_label = f"ANFIS_{config_id}"

            if target_key == 'MM_QM':
                if isinstance(model, list) and y_pred.ndim == 2:
                    if y_pred.shape[1] >= 2:
                        self._store_prediction('MM', f"{model_label}_{dataset_name}",
                                               nuclei, test_df['MAGNETIC MOMENT [µ]'].values, y_pred[:, 0])
                        self._store_prediction('QM', f"{model_label}_{dataset_name}",
                                               nuclei, test_df['QUADRUPOLE MOMENT [Q]'].values, y_pred[:, 1])
            else:
                tcol = TARGET_COLS.get(target_key)
                if tcol and tcol in test_df.columns:
                    y_true = test_df[tcol].values
                    y_out = y_pred.ravel() if y_pred.ndim > 1 else y_pred
                    if len(y_out) == len(y_true):
                        self._store_prediction(target_key, f"{model_label}_{dataset_name}",
                                               nuclei, y_true, y_out)

    def _store_prediction(self, target_key: str, model_label: str,
                          nuclei: List[str], y_true: np.ndarray, y_pred: np.ndarray):
        """Store one model's predictions for one target key."""
        if target_key not in self.all_predictions:
            self.all_predictions[target_key] = {}
        if model_label in self.all_predictions[target_key]:
            return  # already stored

        valid = np.isfinite(y_pred) & np.isfinite(y_true)
        if valid.sum() < 2:
            return

        df = pd.DataFrame({
            'nucleus': [nuclei[i] for i in range(len(nuclei)) if valid[i]],
            'experimental': y_true[valid],
            'predicted': y_pred[valid],
        })
        self.all_predictions[target_key][model_label] = df

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_complete_analysis(self) -> Dict:
        """Collect predictions, run cross-model evaluation, generate reports."""
        logger.info("\n" + "=" * 80)
        logger.info("FAZ 5: CROSS-MODEL ANALYSIS STARTING")
        logger.info("=" * 80)
        start_time = datetime.now()

        # 1. Collect predictions
        logger.info("\n[1/4] Collecting model predictions...")
        self._collect_all_predictions()

        # 2. Per-target cross-model evaluation
        logger.info("\n[2/4] Running cross-model evaluation per target...")
        results = {}
        for target_key in ['MM', 'QM']:
            models_dict = self.all_predictions.get(target_key, {})
            if len(models_dict) < 2:
                logger.warning(f"  {target_key}: <2 models, skipping cross-model analysis")
                continue

            logger.info(f"\n  Target: {target_key} ({len(models_dict)} models)")
            target_out = self.output_dir / target_key
            target_out.mkdir(exist_ok=True)

            evaluator = CrossModelEvaluator(output_dir=str(target_out), use_best_model_selector=False)
            for model_label, df in models_dict.items():
                try:
                    evaluator.add_predictions(
                        model_name=model_label,
                        predictions_df=df,
                        target_col='experimental',
                        prediction_col='predicted',
                        nucleus_col='nucleus',
                    )
                except Exception as e:
                    logger.debug(f"    Skip {model_label}: {e}")

            if len(evaluator.predictions) < 2:
                continue

            try:
                target_results = evaluator.evaluate_common_performance(
                    target_name=target_key, top_n=50
                )
                evaluator.save_cross_model_report(f'{target_key}_cross_model_report.xlsx')
                if MATPLOTLIB_AVAILABLE:
                    try:
                        evaluator.visualize_results(target_key)
                    except Exception:
                        pass
                results[target_key] = target_results
                logger.info(f"  [OK] {target_key} analysis complete")
            except Exception as e:
                logger.error(f"  [FAIL] {target_key}: {e}")

        # 3. Master Excel report
        logger.info("\n[3/4] Building master Excel report...")
        self._create_master_report(results)

        # 4. Summary JSON
        logger.info("\n[4/4] Saving summary JSON...")
        duration = (datetime.now() - start_time).total_seconds()
        self._save_summary_json(results, duration)

        logger.info("\n" + "=" * 80)
        logger.info("FAZ 5: CROSS-MODEL ANALYSIS COMPLETED")
        logger.info(f"  Duration       : {duration:.1f}s ({duration/60:.1f} min)")
        logger.info(f"  Targets done   : {list(results.keys())}")
        logger.info(f"  Output dir     : {self.output_dir}")
        logger.info("=" * 80)

        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _create_master_report(self, results: Dict):
        """Master Excel with overall summary + per-target Good/Medium/Poor sheets."""
        master_file = self.output_dir / 'MASTER_CROSS_MODEL_REPORT.xlsx'
        logger.info(f"  Writing: {master_file}")

        try:
            with pd.ExcelWriter(master_file, engine='openpyxl') as writer:
                # Sheet 1: Overall summary
                self._write_overall_summary(writer, results)

                # Sheets 2+: per-target Good/Medium/Poor
                for target_key, target_results in results.items():
                    self._write_target_sheets(writer, target_key, target_results)

                # Model stats
                self._write_model_stats(writer)

                # Pairwise R² agreement matrix
                self._write_agreement_overview(writer, results)

            logger.info(f"  [OK] Master report saved: {master_file}")
        except Exception as e:
            logger.error(f"  [FAIL] Master report: {e}")

    def _write_overall_summary(self, writer, results: Dict):
        rows = []
        for target_key, res in results.items():
            rows.append({
                'Target': target_key,
                'N_Models': len(self.all_predictions.get(target_key, {})),
                'Good_Count': len(res.get('good_nuclei', [])),
                'Medium_Count': len(res.get('medium_nuclei', [])),
                'Poor_Count': len(res.get('poor_nuclei', [])),
                'Good_Mean_Error': res.get('good_stats', {}).get('mean_error', np.nan),
                'Good_Mean_R2': res.get('good_stats', {}).get('mean_r2', np.nan),
                'Medium_Mean_Error': res.get('medium_stats', {}).get('mean_error', np.nan),
                'Medium_Mean_R2': res.get('medium_stats', {}).get('mean_r2', np.nan),
                'Poor_Mean_Error': res.get('poor_stats', {}).get('mean_error', np.nan),
                'Poor_Mean_R2': res.get('poor_stats', {}).get('mean_r2', np.nan),
                'Overall_Agreement': res.get('model_agreement', {}).get('overall_agreement', np.nan),
            })
        pd.DataFrame(rows).to_excel(writer, sheet_name='Overall_Summary', index=False)

    def _write_target_sheets(self, writer, target_key: str, results: Dict):
        """Good/Medium/Poor sheets for this target."""
        models_dict = self.all_predictions.get(target_key, {})

        for category in ('good', 'medium', 'poor'):
            nuclei_list = results.get(f'{category}_nuclei', [])
            if not nuclei_list:
                continue

            rows = []
            for nucleus in nuclei_list:
                row = {'Nucleus': nucleus}
                exp_val = None
                for model_label, df in models_dict.items():
                    nd = df[df['nucleus'] == nucleus]
                    if len(nd) > 0:
                        if exp_val is None:
                            exp_val = float(nd['experimental'].iloc[0])
                            row['Experimental'] = exp_val
                        row[f'{model_label}_Pred'] = float(nd['predicted'].iloc[0])
                        row[f'{model_label}_Error'] = float(abs(nd['predicted'].iloc[0] - nd['experimental'].iloc[0]))
                rows.append(row)

            df_out = pd.DataFrame(rows)
            sheet_name = f'{target_key}_{category.capitalize()}'[:31]
            df_out.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"    [OK] Sheet {sheet_name} ({len(nuclei_list)} nuclei)")

    def _write_model_stats(self, writer):
        """Per-model error statistics across all targets."""
        rows = []
        for target_key, models_dict in self.all_predictions.items():
            for model_label, df in models_dict.items():
                err = (df['predicted'] - df['experimental']).abs()
                rows.append({
                    'Target': target_key,
                    'Model': model_label,
                    'N': len(df),
                    'Mean_Error': float(err.mean()),
                    'Std_Error': float(err.std()),
                    'Median_Error': float(err.median()),
                    'Max_Error': float(err.max()),
                })
        if rows:
            pd.DataFrame(rows).to_excel(writer, sheet_name='Model_Statistics', index=False)

    def _write_agreement_overview(self, writer, results: Dict):
        """Agreement scores per target."""
        rows = []
        for target_key, res in results.items():
            ag = res.get('model_agreement', {})
            rows.append({
                'Target': target_key,
                'Overall_Agreement': ag.get('overall_agreement', np.nan),
                'Good_Nuclei_Agreement': ag.get('good_nuclei_agreement', np.nan),
                'Medium_Nuclei_Agreement': ag.get('medium_nuclei_agreement', np.nan),
                'Poor_Nuclei_Agreement': ag.get('poor_nuclei_agreement', np.nan),
            })
        if rows:
            pd.DataFrame(rows).to_excel(writer, sheet_name='Agreement_Overview', index=False)

    def _save_summary_json(self, results: Dict, duration: float):
        summary = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': round(duration, 2),
            'targets_analyzed': list(results.keys()),
            'total_models_per_target': {
                t: len(self.all_predictions.get(t, {})) for t in self.all_predictions
            },
            'results_summary': {
                target: {
                    'n_models': len(self.all_predictions.get(target, {})),
                    'good_count': len(res.get('good_nuclei', [])),
                    'medium_count': len(res.get('medium_nuclei', [])),
                    'poor_count': len(res.get('poor_nuclei', [])),
                    'overall_agreement': res.get('model_agreement', {}).get('overall_agreement', 0),
                }
                for target, res in results.items()
            }
        }
        json_file = self.output_dir / 'cross_model_analysis_summary.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"  [OK] Summary JSON saved: {json_file}")


def main():
    """Standalone test."""
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent.parent / 'outputs'

    pipeline = CrossModelAnalysisPipeline(
        trained_models_dir=str(base / 'trained_models'),
        output_dir=str(base / 'cross_model_analysis'),
        anfis_models_dir=str(base / 'anfis_models'),
        datasets_dir=str(base / 'generated_datasets'),
    )
    results = pipeline.run_complete_analysis()
    print(f"\n[SUCCESS] Targets analysed: {list(results.keys())}")


if __name__ == '__main__':
    main()
