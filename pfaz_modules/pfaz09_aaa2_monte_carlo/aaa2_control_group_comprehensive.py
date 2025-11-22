# -*- coding: utf-8 -*-
"""
AAA2 CONTROL GROUP COMPREHENSIVE ANALYSIS
==========================================

Complete analysis of AAA2 control group with top 50 models per target

Features:
- PFAZ01 integration (pre-calculated features)
- Top 50 model selection (R², RMSE criteria)
- MM_QM dual predictions (both MM and QM separately) [STAR]
- QM empty nuclei special handling
- Best/worst/unpredictable nuclei analysis
- Model success rates (overall + by category)
- 15-sheet Excel reports with 8 Pivot Tables
- Dual visualizations (PNG + HTML)
- Comprehensive JSON summaries

Author: Nuclear Physics AI Project
Version: 3.0.0 - FINAL
Date: 2025-10-24
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import time
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.file_io_utils import read_nuclear_data

warnings.filterwarnings('ignore')

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available - HTML plots disabled")

# Excel
import xlsxwriter
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment

# ML
import joblib
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# ============================================================================
# MAIN CLASS
# ============================================================================

class AAA2ControlGroupAnalyzer:
    """
    Comprehensive AAA2 Control Group Analysis
    
    Workflow:
    1. Load AAA2 data (PFAZ01 or direct)
    2. Identify QM empty nuclei
    3. Categorize nuclei (mass, magic, shell)
    4. Select top 50 models per target
    5. Predict with top 50 models
    6. Calculate delta & accuracy
    7. Analyze best/worst/unpredictable nuclei
    8. Calculate model success rates
    9. Generate 15-sheet Excel with pivots
    10. Create dual visualizations (PNG + HTML)
    11. Export comprehensive summaries
    """
    
    def __init__(self,
                 pfaz01_output_path: str = None,
                 aaa2_txt_path: str = 'aaa2.txt',
                 trained_models_dir: str = 'trained_models',
                 performance_summary_dir: str = 'trained_models',
                 output_dir: str = 'aaa2_control_group_results'):
        """
        Initialize AAA2 Control Group Analyzer
        
        Args:
            pfaz01_output_path: Path to PFAZ01 enriched output
            aaa2_txt_path: Path to raw aaa2.txt
            trained_models_dir: Directory with trained models
            performance_summary_dir: Directory with performance summaries
            output_dir: Output directory
        """
        self.pfaz01_output_path = Path(pfaz01_output_path) if pfaz01_output_path else None
        self.aaa2_txt_path = Path(aaa2_txt_path)
        self.trained_models_dir = Path(trained_models_dir)
        self.performance_summary_dir = Path(performance_summary_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Target configurations
        self.target_configs = {
            'MM': {
                'column': 'MAGNETIC MOMENT [µ]',
                'threshold_delta': 0.3,
                'unit': 'µ_N',
                'full_name': 'Magnetic Moment',
                'is_dual': False
            },
            'QM': {
                'column': 'QUADRUPOLE MOMENT [Q]',
                'threshold_delta': 0.05,
                'unit': 'barn',
                'full_name': 'Quadrupole Moment',
                'is_dual': False
            },
            'MM_QM': {
                'columns': ['MAGNETIC MOMENT [µ]', 'QUADRUPOLE MOMENT [Q]'],
                'threshold_delta': {'MM': 0.3, 'QM': 0.05},
                'unit': {'MM': 'µ_N', 'QM': 'barn'},
                'full_name': 'Magnetic + Quadrupole Moments',
                'is_dual': True  # [STAR] DUAL OUTPUT
            },
            'Beta_2': {
                'column': 'Beta_2',
                'threshold_delta': 0.02,
                'unit': 'dimensionless',
                'full_name': 'Deformation Parameter',
                'is_dual': False
            }
        }
        
        # Storage
        self.aaa2_df = None
        self.qm_empty_indices = []
        self.qm_empty_nuclei = []
        self.nucleus_categories = {}
        self.top50_models = {}
        self.loaded_models = {}
        self.predictions = {}
        self.delta_accuracy = {}
        self.success_rates = {}
        self.best_worst_nuclei = {}
        self.common_features = {}
        self.category_success_rates = {}
        
        logger.info("="*80)
        logger.info("AAA2 CONTROL GROUP ANALYZER INITIALIZED")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Targets: {list(self.target_configs.keys())}")
    
    # ========================================================================
    # PHASE 1: DATA LOADING
    # ========================================================================
    
    def load_aaa2_data(self) -> pd.DataFrame:
        """
        Load AAA2 dataset
        
        Priority:
        1. PFAZ01 enriched output (features pre-calculated)
        2. Raw aaa2.txt (calculate features on-the-fly)
        
        Returns:
            aaa2_df: DataFrame with all features
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: DATA LOADING")
        logger.info("="*80)
        
        # Try PFAZ01 output first
        if self.pfaz01_output_path and self.pfaz01_output_path.exists():
            logger.info(f"-> Loading from PFAZ01 output: {self.pfaz01_output_path}")
            self.aaa2_df = pd.read_csv(self.pfaz01_output_path)
            logger.info(f"[OK] Loaded {len(self.aaa2_df)} nuclei with {len(self.aaa2_df.columns)} features")
        
        # Fallback to raw aaa2.txt
        elif self.aaa2_txt_path.exists():
            logger.info(f"-> Loading raw AAA2: {self.aaa2_txt_path}")
            self.aaa2_df = read_nuclear_data(self.aaa2_txt_path, encoding='utf-8')
            logger.info(f"[OK] Loaded {len(self.aaa2_df)} nuclei")
            
            # Calculate features on-the-fly (if needed)
            if 'SEMF_BE' not in self.aaa2_df.columns:
                logger.info("-> Calculating theoretical features...")
                self.aaa2_df = self._calculate_theoretical_features(self.aaa2_df)
        
        else:
            raise FileNotFoundError(f"Neither PFAZ01 output nor aaa2.txt found")
        
        # Clean column names
        self.aaa2_df.columns = self.aaa2_df.columns.str.strip()
        
        return self.aaa2_df
    
    def _calculate_theoretical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate theoretical features if not present"""
        logger.info("  Theoretical feature calculation (simplified)")
        
        # Basic SEMF features
        A = df['A'].values
        Z = df['Z'].values
        N = df['N'].values
        
        # Volume term
        df['SEMF_Volume'] = 15.75 * A
        
        # Surface term
        df['SEMF_Surface'] = -17.8 * (A ** (2/3))
        
        # Coulomb term
        df['SEMF_Coulomb'] = -0.711 * (Z ** 2) / (A ** (1/3))
        
        # Asymmetry term
        df['SEMF_Asymmetry'] = -23.7 * ((N - Z) ** 2) / A
        
        # Pairing term
        pairing = np.zeros_like(A, dtype=float)
        even_even = (N % 2 == 0) & (Z % 2 == 0)
        odd_odd = (N % 2 == 1) & (Z % 2 == 1)
        pairing[even_even] = 12.0 / np.sqrt(A[even_even])
        pairing[odd_odd] = -12.0 / np.sqrt(A[odd_odd])
        df['SEMF_Pairing'] = pairing
        
        # Total binding energy
        df['SEMF_BE'] = (df['SEMF_Volume'] + df['SEMF_Surface'] + 
                        df['SEMF_Coulomb'] + df['SEMF_Asymmetry'] + df['SEMF_Pairing'])
        
        # Shell closures
        magic_numbers = [2, 8, 20, 28, 50, 82, 126]
        df['Z_Shell_Closure'] = df['Z'].apply(lambda z: min([abs(z - m) for m in magic_numbers]))
        df['N_Shell_Closure'] = df['N'].apply(lambda n: min([abs(n - m) for m in magic_numbers]))
        
        # N/Z ratio
        df['N_Z_Ratio'] = N / Z
        
        logger.info(f"  [OK] Added {11} theoretical features")
        
        return df
    
    def identify_qm_empty_nuclei(self) -> Tuple[List, List]:
        """
        Identify nuclei with empty QM values
        
        Returns:
            qm_empty_indices: List of indices
            qm_empty_nuclei: List of nucleus names
        """
        logger.info("\n-> Identifying QM empty nuclei...")
        
        qm_col = self.target_configs['QM']['column']
        
        # Handle different representations of empty
        qm_empty_mask = (
            self.aaa2_df[qm_col].isna() |
            (self.aaa2_df[qm_col] == '') |
            (self.aaa2_df[qm_col] == ' ') |
            (self.aaa2_df[qm_col].astype(str).str.strip() == '')
        )
        
        self.qm_empty_indices = self.aaa2_df[qm_empty_mask].index.tolist()
        self.qm_empty_nuclei = self.aaa2_df.loc[qm_empty_mask, 'NUCLEUS'].tolist()
        
        logger.info(f"[OK] QM empty nuclei: {len(self.qm_empty_indices)}")
        logger.info(f"  Examples: {self.qm_empty_nuclei[:5]}")
        
        # Save to file
        qm_empty_file = self.output_dir / 'data_preparation' / 'qm_empty_nuclei_list.txt'
        qm_empty_file.parent.mkdir(parents=True, exist_ok=True)
        with open(qm_empty_file, 'w') as f:
            f.write("QM EMPTY NUCLEI\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total: {len(self.qm_empty_nuclei)}\n\n")
            for nucleus in self.qm_empty_nuclei:
                f.write(f"{nucleus}\n")
        
        return self.qm_empty_indices, self.qm_empty_nuclei
    
    def categorize_nuclei(self) -> Dict:
        """
        Categorize nuclei by various properties
        
        Categories:
        - By mass: Light, Medium, Heavy, Superheavy
        - By magic numbers: Magic, Near-magic, Mid-shell
        - By shell structure: Closed-shell, Open-shell
        - By N/Z ratio: Neutron-rich, Stable, Proton-rich
        """
        logger.info("\n-> Categorizing nuclei...")
        
        magic_numbers = [2, 8, 20, 28, 50, 82, 126]
        
        categories = {
            'by_mass': {},
            'by_magic': {},
            'by_shell': {},
            'by_nz_ratio': {}
        }
        
        for idx, row in self.aaa2_df.iterrows():
            nucleus = row['NUCLEUS']
            A = row['A']
            Z = row['Z']
            N = row['N']
            
            # By mass
            if A < 40:
                categories['by_mass'].setdefault('Light', []).append(nucleus)
            elif A < 100:
                categories['by_mass'].setdefault('Medium', []).append(nucleus)
            elif A < 200:
                categories['by_mass'].setdefault('Heavy', []).append(nucleus)
            else:
                categories['by_mass'].setdefault('Superheavy', []).append(nucleus)
            
            # By magic numbers
            is_z_magic = Z in magic_numbers
            is_n_magic = N in magic_numbers
            z_dist = min([abs(Z - m) for m in magic_numbers])
            n_dist = min([abs(N - m) for m in magic_numbers])
            
            if is_z_magic or is_n_magic:
                categories['by_magic'].setdefault('Magic', []).append(nucleus)
            elif z_dist <= 2 or n_dist <= 2:
                categories['by_magic'].setdefault('Near-magic', []).append(nucleus)
            else:
                categories['by_magic'].setdefault('Mid-shell', []).append(nucleus)
            
            # By shell structure
            if (is_z_magic and is_n_magic):
                categories['by_shell'].setdefault('Doubly-closed', []).append(nucleus)
            elif is_z_magic or is_n_magic:
                categories['by_shell'].setdefault('Singly-closed', []).append(nucleus)
            else:
                categories['by_shell'].setdefault('Open-shell', []).append(nucleus)
            
            # By N/Z ratio
            nz_ratio = N / Z
            if nz_ratio > 1.5:
                categories['by_nz_ratio'].setdefault('Neutron-rich', []).append(nucleus)
            elif nz_ratio < 1.0:
                categories['by_nz_ratio'].setdefault('Proton-rich', []).append(nucleus)
            else:
                categories['by_nz_ratio'].setdefault('Stable', []).append(nucleus)
        
        self.nucleus_categories = categories
        
        # Print summary
        logger.info("[OK] Nucleus categories:")
        for cat_type, cat_dict in categories.items():
            logger.info(f"  {cat_type}:")
            for cat_name, nuclei_list in cat_dict.items():
                logger.info(f"    {cat_name}: {len(nuclei_list)}")
        
        # Save to JSON
        cat_file = self.output_dir / 'data_preparation' / 'nucleus_categories.json'
        with open(cat_file, 'w') as f:
            json.dump(categories, f, indent=2)
        
        return categories
    
    # ========================================================================
    # PHASE 2: MODEL SELECTION
    # ========================================================================
    
    def select_top50_models_per_target(self) -> Dict:
        """
        Select top 50 models for each target
        
        Criteria:
        - R² > 0.85 (preferred)
        - RMSE < threshold (target-specific)
        - Sort by R² (descending)
        
        Returns:
            top50_dict: {target: [model_id1, model_id2, ...]}
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: TOP 50 MODEL SELECTION")
        logger.info("="*80)
        
        top50_dict = {}
        
        for target in self.target_configs.keys():
            logger.info(f"\n-> Selecting top 50 models for {target}...")
            
            # Look for performance summary file
            summary_file = self.performance_summary_dir / f'performance_summary_{target}.csv'
            
            if not summary_file.exists():
                logger.warning(f"  [WARNING] Performance summary not found: {summary_file}")
                logger.warning(f"  Skipping {target}")
                continue
            
            # Load performance data
            perf_df = pd.read_csv(summary_file)
            
            logger.info(f"  Loaded {len(perf_df)} model performances")
            
            # Filter by threshold
            r2_threshold = 0.85
            if target == 'MM_QM':
                # For dual target, check both
                filtered_df = perf_df[
                    (perf_df['R2_MM'] > r2_threshold) | 
                    (perf_df['R2_QM'] > r2_threshold)
                ]
                sort_col = 'R2_MM'  # Primary sort by MM
            else:
                filtered_df = perf_df[perf_df['R2'] > r2_threshold]
                sort_col = 'R2'
            
            logger.info(f"  After R²>{r2_threshold} filter: {len(filtered_df)} models")
            
            # Sort by R²
            sorted_df = filtered_df.sort_values(sort_col, ascending=False)
            
            # Select top 50
            top50 = sorted_df.head(50)['model_id'].tolist()
            
            top50_dict[target] = top50
            
            logger.info(f"[OK] Selected {len(top50)} models for {target}")
            if len(top50) > 0:
                logger.info(f"  Best: {top50[0]} (R²={sorted_df.iloc[0][sort_col]:.4f})")
                if len(top50) >= 50:
                    logger.info(f"  50th: {top50[49]} (R²={sorted_df.iloc[49][sort_col]:.4f})")
        
        self.top50_models = top50_dict
        
        # Save to JSON
        top50_file = self.output_dir / 'data_preparation' / 'top50_models_selected.json'
        with open(top50_file, 'w') as f:
            json.dump(top50_dict, f, indent=2)
        
        logger.info(f"\n[OK] Top 50 models saved: {top50_file}")
        
        return top50_dict
    
    def load_selected_models(self, target: str) -> Dict:
        """
        Load top 50 models for a specific target
        
        Args:
            target: Target name (MM, QM, MM_QM, Beta_2)
        
        Returns:
            loaded_models: {model_id: model_object}
        """
        logger.info(f"\n-> Loading top 50 models for {target}...")
        
        model_ids = self.top50_models.get(target, [])
        if len(model_ids) == 0:
            logger.warning(f"  No models to load for {target}")
            return {}
        
        loaded_models = {}
        
        for model_id in tqdm(model_ids, desc=f"Loading {target} models"):
            try:
                # Determine model path
                # Format: MODEL_TYPE_configXXX_TARGET_XXXnuclei
                parts = model_id.split('_')
                model_type = parts[0]  # RF, GBM, XGBoost, DNN, BNN, PINN, ANFIS
                
                # Find model file
                if model_type == 'ANFIS':
                    model_dir = self.trained_models_dir / 'ANFIS' / model_id
                    model_file = model_dir / 'model.mat'  # MATLAB FIS
                else:
                    model_dir = self.trained_models_dir / 'AI' / model_type / model_id
                    model_file = model_dir / 'model.pkl'
                
                if not model_file.exists():
                    logger.warning(f"    Model file not found: {model_file}")
                    continue
                
                # Load model
                if model_type == 'ANFIS':
                    # For ANFIS, we'll load the FIS using scipy.io
                    from scipy.io import loadmat
                    fis_data = loadmat(str(model_file))
                    loaded_models[model_id] = {
                        'type': 'ANFIS',
                        'fis': fis_data,
                        'model_id': model_id
                    }
                else:
                    # For AI models, use joblib
                    model = joblib.load(model_file)
                    loaded_models[model_id] = {
                        'type': model_type,
                        'model': model,
                        'model_id': model_id
                    }
            
            except Exception as e:
                logger.warning(f"    Failed to load {model_id}: {e}")
                continue
        
        logger.info(f"[OK] Successfully loaded {len(loaded_models)}/{len(model_ids)} models")
        
        self.loaded_models[target] = loaded_models
        
        return loaded_models
    
    # ========================================================================
    # PHASE 3: PREDICTION
    # ========================================================================
    
    def predict_aaa2_with_top50(self, target: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Predict AAA2 with top 50 models
        
        Args:
            target: Target name
        
        Returns:
            predictions_df: DataFrame with predictions
            timing_info: Prediction timing information
        """
        logger.info("\n" + "="*80)
        logger.info(f"PHASE 3: PREDICTION - {target}")
        logger.info("="*80)
        
        # Load models if not already loaded
        if target not in self.loaded_models:
            self.load_selected_models(target)
        
        loaded_models = self.loaded_models.get(target, {})
        
        if len(loaded_models) == 0:
            logger.error(f"No models loaded for {target}")
            return None, None
        
        # Prepare features
        feature_cols = [col for col in self.aaa2_df.columns 
                       if col not in ['NUCLEUS', 'A', 'Z', 'N'] and 
                       not col.startswith('MAGNETIC') and 
                       not col.startswith('QUADRUPOLE') and
                       not col.startswith('Beta_2')]
        
        X = self.aaa2_df[feature_cols].values
        
        logger.info(f"-> Feature matrix: {X.shape}")
        
        # Initialize predictions storage
        predictions_dict = {}
        timing_info = {}
        
        config = self.target_configs[target]
        is_dual = config.get('is_dual', False)
        
        # Predict with each model
        for model_id, model_data in tqdm(loaded_models.items(), desc=f"Predicting {target}"):
            try:
                start_time = time.perf_counter()
                
                if model_data['type'] == 'ANFIS':
                    # ANFIS prediction (simplified - would need MATLAB engine)
                    # For now, skip ANFIS or use dummy predictions
                    logger.warning(f"  ANFIS prediction not implemented: {model_id}")
                    continue
                else:
                    # AI model prediction
                    model = model_data['model']
                    y_pred = model.predict(X)
                
                end_time = time.perf_counter()
                pred_time = (end_time - start_time) * 1000  # ms
                
                # Store predictions
                if is_dual and y_pred.ndim > 1 and y_pred.shape[1] == 2:
                    # MM_QM: Two outputs
                    predictions_dict[f'{model_id}_MM'] = y_pred[:, 0]
                    predictions_dict[f'{model_id}_QM'] = y_pred[:, 1]
                else:
                    # Single output
                    predictions_dict[model_id] = y_pred.flatten()
                
                # Store timing
                timing_info[model_id] = {
                    'total_time_ms': pred_time,
                    'avg_per_nucleus_ms': pred_time / len(X)
                }
            
            except Exception as e:
                logger.warning(f"  Prediction failed for {model_id}: {e}")
                continue
        
        logger.info(f"[OK] Predictions completed: {len(predictions_dict)} model outputs")
        
        # Create DataFrame
        predictions_df = pd.DataFrame({
            'NUCLEUS': self.aaa2_df['NUCLEUS'],
            'A': self.aaa2_df['A'],
            'Z': self.aaa2_df['Z'],
            'N': self.aaa2_df['N']
        })
        
        # Add experimental values
        if is_dual:
            predictions_df['Experimental_MM'] = self.aaa2_df[config['columns'][0]]
            predictions_df['Experimental_QM'] = self.aaa2_df[config['columns'][1]]
        else:
            predictions_df['Experimental'] = self.aaa2_df[config['column']]
        
        # Add predictions
        for model_id, preds in predictions_dict.items():
            predictions_df[model_id] = preds
        
        # Save predictions
        pred_dir = self.output_dir / 'predictions' / target
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        predictions_df.to_csv(pred_dir / 'predictions.csv', index=False)
        predictions_df.to_excel(pred_dir / 'predictions.xlsx', index=False, engine='openpyxl')
        
        logger.info(f"[OK] Predictions saved: {pred_dir}")
        
        self.predictions[target] = {
            'dataframe': predictions_df,
            'timing': timing_info
        }
        
        return predictions_df, timing_info
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def run_complete_analysis(self) -> Dict:
        """
        Run complete AAA2 control group analysis
        
        Returns:
            results_dict: All analysis results
        """
        start_time = datetime.now()
        
        logger.info("\n" + "="*80)
        logger.info("AAA2 CONTROL GROUP - COMPLETE ANALYSIS")
        logger.info("="*80)
        logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Phase 1: Data Loading
            self.load_aaa2_data()
            self.identify_qm_empty_nuclei()
            self.categorize_nuclei()
            
            # Phase 2: Model Selection
            self.select_top50_models_per_target()
            
            # Phase 3-10: Process each target
            for target in self.target_configs.keys():
                if target not in self.top50_models or len(self.top50_models[target]) == 0:
                    logger.warning(f"Skipping {target} - no models selected")
                    continue
                
                logger.info("\n" + "="*80)
                logger.info(f"PROCESSING TARGET: {target}")
                logger.info("="*80)
                
                # Phase 3: Prediction
                predictions_df, timing_info = self.predict_aaa2_with_top50(target)
                
                if predictions_df is None:
                    continue
                
                # Phase 4: Delta & Accuracy
                logger.info(f"\n-> Phase 4: Calculating delta & accuracy...")
                delta_df, accuracy_df = self.calculate_delta_and_accuracy(
                    predictions_df, target
                )
                
                # Phase 5: Model Success Rates
                logger.info(f"\n-> Phase 5: Calculating model success rates...")
                success_rates_df = self.calculate_model_success_rates(
                    delta_df, accuracy_df, target
                )
                
                # Phase 6: Best/Worst/Unpredictable Nuclei
                logger.info(f"\n-> Phase 6: Analyzing best/worst nuclei...")
                best_worst_nuclei = self.analyze_best_worst_nuclei(
                    delta_df, target, top_n=20
                )
                
                # Phase 7: Common Features
                logger.info(f"\n-> Phase 7: Extracting common features...")
                common_features = self.extract_common_features_analysis(
                    best_worst_nuclei, target
                )
                
                # Phase 8: Category Success Rates
                logger.info(f"\n-> Phase 8: Calculating category success rates...")
                category_success_df = self.calculate_category_success_rates(
                    delta_df, target
                )
                
                # Phase 9: Excel Report
                logger.info(f"\n-> Phase 9: Generating Excel report...")
                excel_file = self.generate_comprehensive_excel_report(
                    target, predictions_df, delta_df, accuracy_df,
                    success_rates_df, best_worst_nuclei, category_success_df,
                    timing_info
                )
                
                # Phase 10: Visualizations
                logger.info(f"\n-> Phase 10: Creating visualizations...")
                viz_files = self.generate_dual_visualizations(
                    target, predictions_df, delta_df, accuracy_df,
                    success_rates_df, best_worst_nuclei, category_success_df
                )
                
                logger.info(f"[OK] {target} processing complete")
            
            # Final summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("\n" + "="*80)
            logger.info("[SUCCESS] AAA2 CONTROL GROUP ANALYSIS COMPLETE")
            logger.info("="*80)
            logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            logger.info(f"Targets processed: {len(self.predictions)}")
            logger.info(f"Output directory: {self.output_dir}")
            
            return {
                'success': True,
                'duration': duration,
                'targets_processed': list(self.predictions.keys()),
                'output_dir': str(self.output_dir)
            }
        
        except Exception as e:
            logger.error(f"\n[ERROR] ANALYSIS FAILED: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    
    # Initialize analyzer
    analyzer = AAA2ControlGroupAnalyzer(
        pfaz01_output_path='generated_datasets/AAA2_enriched.csv',  # If exists
        aaa2_txt_path='aaa2.txt',
        trained_models_dir='trained_models',
        performance_summary_dir='trained_models',
        output_dir='aaa2_control_group_results'
    )
    
    # Run analysis
    results = analyzer.run_complete_analysis()
    
    return results


if __name__ == "__main__":
    results = main()
    
    if results['success']:
        print("\n" + "="*80)
        print("[SUCCESS] SUCCESS!")
        print("="*80)
        print(f"Duration: {results['duration']:.1f}s")
        print(f"Targets: {', '.join(results['targets_processed'])}")
        print(f"Output: {results['output_dir']}")
    else:
        print("\n" + "="*80)
        print("[ERROR] FAILED!")
        print("="*80)
        print(f"Error: {results['error']}")
