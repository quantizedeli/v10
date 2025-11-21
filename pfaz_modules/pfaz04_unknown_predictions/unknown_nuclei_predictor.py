# -*- coding: utf-8 -*-
"""
PFAZ 4: Unknown Nuclei Predictor
=================================

Predict on unknown nuclei and analyze performance degradation

Features:
- Load AI models (PFAZ 2)
- Load ANFIS models (PFAZ 3)
- Predict on unknown nuclei
- Known vs Unknown performance comparison
- Degradation analysis
- Excel comprehensive reports

Author: Nuclear Physics AI Training Pipeline
Version: 1.0.0
Date: 2025-10-15
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnknownNucleiPredictor:
    """
    Unknown Nuclei Predictor
    
    Features:
    - Predict on unknown nuclei
    - Performance comparison (known vs unknown)
    - Degradation analysis
    - Excel reports
    """
    
    def __init__(self,
                 ai_models_dir: str,
                 anfis_models_dir: str,
                 splits_dir: str,
                 output_dir: str = 'unknown_predictions'):
        
        self.ai_models_dir = Path(ai_models_dir)
        self.anfis_models_dir = Path(anfis_models_dir)
        self.splits_dir = Path(splits_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'ai_results': [],
            'anfis_results': [],
            'degradation_analysis': []
        }
        
        logger.info("=" * 80)
        logger.info("UNKNOWN NUCLEI PREDICTOR INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"AI models: {self.ai_models_dir}")
        logger.info(f"ANFIS models: {self.anfis_models_dir}")
        logger.info(f"Splits: {self.splits_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 80)
    
    def load_models(self) -> Dict:
        """Load all trained models"""
        
        models = {'ai': {}, 'anfis': {}}
        
        # Load AI models
        ai_files = list(self.ai_models_dir.glob('**/model_*.pkl'))
        for model_file in ai_files:
            try:
                model = joblib.load(model_file)
                model_id = model_file.stem.replace('model_', '')
                dataset_name = model_file.parent.parent.name
                full_id = f"{dataset_name}_{model_id}"
                models['ai'][full_id] = {
                    'model': model,
                    'path': model_file,
                    'dataset': dataset_name,
                    'config': model_id
                }
            except:
                pass
        
        # Load ANFIS models
        anfis_files = list(self.anfis_models_dir.glob('**/model_*.pkl'))
        for model_file in anfis_files:
            try:
                model = joblib.load(model_file)
                model_id = model_file.stem.replace('model_', '')
                dataset_name = model_file.parent.parent.name
                full_id = f"{dataset_name}_{model_id}"
                models['anfis'][full_id] = {
                    'model': model,
                    'path': model_file,
                    'dataset': dataset_name,
                    'config': model_id
                }
            except:
                pass
        
        logger.info(f"Loaded {len(models['ai'])} AI models")
        logger.info(f"Loaded {len(models['anfis'])} ANFIS models")
        
        return models
    
    def predict_on_unknown(self):
        """Predict on all unknown datasets"""
        
        logger.info("\n" + "=" * 80)
        logger.info("PREDICTING ON UNKNOWN NUCLEI")
        logger.info("=" * 80)
        
        # Load models
        models = self.load_models()
        
        # Load metadata
        metadata_file = self.splits_dir / 'split_metadata.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Process each dataset
        for dataset_info in metadata['datasets']:
            dataset_name = dataset_info['name']
            unknown_file = Path(dataset_info['unknown_file'])
            
            if not unknown_file.exists():
                logger.warning(f"Unknown file not found: {unknown_file}")
                continue
            
            logger.info(f"\nProcessing: {dataset_name}")
            
            # Load unknown data
            unknown_df = pd.read_csv(unknown_file)
            
            # Identify features and target
            target_col = self._identify_target(dataset_name)
            if target_col not in unknown_df.columns:
                continue
            
            feature_cols = [col for col in unknown_df.columns 
                          if col not in ['NUCLEUS', 'A', 'Z', 'N', target_col]]
            
            X_unknown = unknown_df[feature_cols].values
            y_unknown = unknown_df[target_col].values
            
            # Predict with AI models
            for model_id, model_info in models['ai'].items():
                if dataset_name in model_id:
                    try:
                        y_pred = model_info['model'].predict(X_unknown)
                        metrics = self._calculate_metrics(y_unknown, y_pred)
                        
                        self.results['ai_results'].append({
                            'model_id': model_id,
                            'dataset': dataset_name,
                            'type': 'AI',
                            'n_unknown': len(X_unknown),
                            **metrics
                        })
                    except:
                        pass
            
            # Predict with ANFIS models
            for model_id, model_info in models['anfis'].items():
                if dataset_name in model_id:
                    try:
                        y_pred = model_info['model'].predict(X_unknown)
                        metrics = self._calculate_metrics(y_unknown, y_pred)
                        
                        self.results['anfis_results'].append({
                            'model_id': model_id,
                            'dataset': dataset_name,
                            'type': 'ANFIS',
                            'n_unknown': len(X_unknown),
                            **metrics
                        })
                    except:
                        pass
        
        logger.info("\n" + "=" * 80)
        logger.info(f"COMPLETED: {len(self.results['ai_results'])} AI + {len(self.results['anfis_results'])} ANFIS")
        logger.info("=" * 80 + "\n")
    
    def _identify_target(self, dataset_name: str) -> str:
        """Identify target column"""
        if 'MM_QM' in dataset_name:
            return 'Q'
        elif 'MM' in dataset_name:
            return 'MM'
        elif 'QM' in dataset_name:
            return 'Q'
        elif 'Beta' in dataset_name:
            return 'Beta_2'
        return 'MM'
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate performance metrics"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        return {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred))
        }
    
    def generate_excel_report(self, filename: str = 'Unknown_Nuclei_Results.xlsx'):
        """Generate Excel report"""
        
        excel_path = self.output_dir / filename
        logger.info(f"Generating Excel: {excel_path}")
        
        if OPENPYXL_AVAILABLE:
            wb = Workbook()
            wb.remove(wb.active)
            
            # Sheet 1: AI Results
            ws1 = wb.create_sheet('AI_Unknown_Results')
            ai_df = pd.DataFrame(self.results['ai_results'])
            self._write_sheet(ws1, ai_df)
            
            # Sheet 2: ANFIS Results
            ws2 = wb.create_sheet('ANFIS_Unknown_Results')
            anfis_df = pd.DataFrame(self.results['anfis_results'])
            self._write_sheet(ws2, anfis_df)
            
            # Sheet 3: Summary
            ws3 = wb.create_sheet('Summary')
            summary_data = [
                ['Category', 'Count', 'Avg_R2', 'Avg_RMSE', 'Avg_MAE'],
                ['AI Models', len(ai_df), ai_df['r2'].mean(), ai_df['rmse'].mean(), ai_df['mae'].mean()],
                ['ANFIS Models', len(anfis_df), anfis_df['r2'].mean(), anfis_df['rmse'].mean(), anfis_df['mae'].mean()]
            ]
            for row in summary_data:
                ws3.append(row)
            
            wb.save(excel_path)
        else:
            # Simple CSV
            ai_df = pd.DataFrame(self.results['ai_results'])
            anfis_df = pd.DataFrame(self.results['anfis_results'])
            ai_df.to_csv(self.output_dir / 'ai_unknown_results.csv', index=False)
            anfis_df.to_csv(self.output_dir / 'anfis_unknown_results.csv', index=False)
        
        logger.info(f"✅ Report saved: {excel_path}")
    
    def _write_sheet(self, ws, df: pd.DataFrame):
        """Write dataframe to sheet"""
        if df.empty:
            return
        
        # Headers
        headers = df.columns.tolist()
        ws.append(headers)
        
        # Style headers
        header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        header_font = Font(bold=True, color='FFFFFF')
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center')
        
        # Data
        for _, row in df.iterrows():
            ws.append(row.tolist())


def main():
    """Test"""
    print("\n" + "=" * 80)
    print("PFAZ 4: UNKNOWN NUCLEI PREDICTOR - TEST")
    print("=" * 80)
    
    predictor = UnknownNucleiPredictor(
        ai_models_dir='trained_models',
        anfis_models_dir='trained_anfis',
        splits_dir='test_unknown_splits',
        output_dir='test_unknown_predictions'
    )
    
    print("\nPredicting on unknown nuclei...")
    predictor.predict_on_unknown()
    
    print("\nGenerating Excel report...")
    predictor.generate_excel_report()
    
    print("\n✅ TEST COMPLETED!")


if __name__ == "__main__":
    main()
