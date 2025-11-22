"""
FAZ 7: Ensemble & Meta-Model Pipeline
======================================

Tüm ensemble yöntemlerini test eden ve en iyi yöntemi seçen ana pipeline

Modules:
- EnsembleModelBuilder: Voting ensembles
- StackingMetaLearner: Meta-model ensembles  
- EnsembleEvaluator: Comparison & evaluation

Author: AI Dataset Training Pipeline
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
sys.path.insert(0, '/home/claude')

from ensemble_model_builder import EnsembleModelBuilder
from stacking_meta_learner import StackingMetaLearner
from ensemble_evaluator import EnsembleEvaluator

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def faz7_ensemble_pipeline():
    """
    FAZ 7 - Complete Ensemble Pipeline
    
    1. Load base models
    2. Create voting ensembles (simple, weighted)
    3. Create stacking ensembles (ridge, lasso, rf, gbm)
    4. Evaluate all ensembles
    5. Select best ensemble
    """
    
    logger.info("="*80)
    logger.info("FAZ 7: ENSEMBLE & META-MODEL PIPELINE")
    logger.info("="*80)
    
    # =========================================================================
    # STEP 0: Prepare Mock Data & Models
    # =========================================================================
    logger.info("\n[REPORT] STEP 0: PREPARING DATA & MODELS")
    logger.info("-"*80)
    
    # Mock data
    np.random.seed(42)
    n_train = 300
    n_val = 100
    n_test = 100
    
    # Features (A, Z, N, SPIN, PARITY)
    X_train = np.random.randn(n_train, 5)
    X_val = np.random.randn(n_val, 5)
    X_test = np.random.randn(n_test, 5)
    
    # Target (MM = f(A, Z, N))
    def target_function(X):
        return 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3]
    
    y_train = target_function(X_train) + np.random.randn(n_train) * 0.5
    y_val = target_function(X_val) + np.random.randn(n_val) * 0.5
    y_test = target_function(X_test) + np.random.randn(n_test) * 0.5
    
    logger.info(f"[OK] Data prepared:")
    logger.info(f"  Train: {n_train} samples")
    logger.info(f"  Val: {n_val} samples")
    logger.info(f"  Test: {n_test} samples")
    
    # Mock base models (sklearn models)
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    
    base_models = {
        'Ridge': Ridge(alpha=1.0).fit(X_train, y_train),
        'Lasso': Lasso(alpha=0.1).fit(X_train, y_train),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train, y_train),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X_train, y_train),
        'GBM': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_train, y_train),
        'MLP': MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42).fit(X_train, y_train)
    }
    
    logger.info(f"[OK] Base models trained: {len(base_models)}")
    
    # =========================================================================
    # STEP 1: Voting Ensembles
    # =========================================================================
    logger.info("\n[TARGET] STEP 1: VOTING ENSEMBLES")
    logger.info("-"*80)
    
    # Initialize ensemble builder
    voting_builder = EnsembleModelBuilder('output/voting_ensembles')
    
    # Add models
    for model_id, model in base_models.items():
        voting_builder.add_model(
            model_id=model_id,
            model=model,
            metadata={'model_type': model_id, 'features': list(range(5))},
            X_val=X_val,
            y_val=y_val
        )
    
    # 1.1. Simple Voting
    logger.info("\n-> Creating Simple Voting Ensemble...")
    result_simple = voting_builder.create_simple_voting(
        model_ids=list(base_models.keys()),
        X_test=X_test,
        y_test=y_test
    )
    voting_builder.save_ensemble(result_simple, 'SimpleVoting')
    
    # 1.2. Weighted Voting (R²)
    logger.info("\n-> Creating Weighted Voting Ensemble (R² optimization)...")
    result_weighted_r2 = voting_builder.create_weighted_voting(
        model_ids=list(base_models.keys()),
        X_test=X_test,
        y_test=y_test,
        optimization_method='r2'
    )
    voting_builder.save_ensemble(result_weighted_r2, 'WeightedVoting_R2')
    
    # 1.3. Weighted Voting (RMSE)
    logger.info("\n-> Creating Weighted Voting Ensemble (RMSE optimization)...")
    result_weighted_rmse = voting_builder.create_weighted_voting(
        model_ids=list(base_models.keys()),
        X_test=X_test,
        y_test=y_test,
        optimization_method='rmse'
    )
    voting_builder.save_ensemble(result_weighted_rmse, 'WeightedVoting_RMSE')
    
    # 1.4. Weighted Voting (Inverse Error)
    logger.info("\n-> Creating Weighted Voting Ensemble (Inverse Error)...")
    result_weighted_inv = voting_builder.create_weighted_voting(
        model_ids=list(base_models.keys()),
        X_test=X_test,
        y_test=y_test,
        optimization_method='inverse_error'
    )
    voting_builder.save_ensemble(result_weighted_inv, 'WeightedVoting_InvError')
    
    logger.info("\n[SUCCESS] VOTING ENSEMBLES COMPLETED")
    
    # =========================================================================
    # STEP 2: Stacking Ensembles
    # =========================================================================
    logger.info("\n🏗️ STEP 2: STACKING ENSEMBLES")
    logger.info("-"*80)
    
    stacking_results = []
    
    # 2.1. Stacking with Ridge meta-model
    logger.info("\n-> Creating Stacking Ensemble (Ridge meta-model)...")
    stacker_ridge = StackingMetaLearner(meta_model_type='ridge', cv_folds=5)
    
    for model_id, model in base_models.items():
        stacker_ridge.add_base_model(model_id, model, {'model_type': model_id, 'features': list(range(5))})
    
    stacker_ridge.generate_oof_predictions(X_train, y_train)
    stacker_ridge.train_meta_model(y_train)
    result_stacking_ridge = stacker_ridge.evaluate(X_test, y_test)
    stacker_ridge.save_stacking_model('Stacking_Ridge')
    stacking_results.append(result_stacking_ridge)
    
    # 2.2. Stacking with Lasso meta-model
    logger.info("\n-> Creating Stacking Ensemble (Lasso meta-model)...")
    stacker_lasso = StackingMetaLearner(meta_model_type='lasso', cv_folds=5)
    
    for model_id, model in base_models.items():
        stacker_lasso.add_base_model(model_id, model, {'model_type': model_id, 'features': list(range(5))})
    
    stacker_lasso.generate_oof_predictions(X_train, y_train)
    stacker_lasso.train_meta_model(y_train)
    result_stacking_lasso = stacker_lasso.evaluate(X_test, y_test)
    stacker_lasso.save_stacking_model('Stacking_Lasso')
    stacking_results.append(result_stacking_lasso)
    
    # 2.3. Stacking with RF meta-model
    logger.info("\n-> Creating Stacking Ensemble (RF meta-model)...")
    stacker_rf = StackingMetaLearner(meta_model_type='rf', cv_folds=5)
    
    for model_id, model in base_models.items():
        stacker_rf.add_base_model(model_id, model, {'model_type': model_id, 'features': list(range(5))})
    
    stacker_rf.generate_oof_predictions(X_train, y_train)
    stacker_rf.train_meta_model(y_train)
    result_stacking_rf = stacker_rf.evaluate(X_test, y_test)
    stacker_rf.save_stacking_model('Stacking_RF')
    stacking_results.append(result_stacking_rf)
    
    # 2.4. Stacking with GBM meta-model
    logger.info("\n-> Creating Stacking Ensemble (GBM meta-model)...")
    stacker_gbm = StackingMetaLearner(meta_model_type='gbm', cv_folds=5)
    
    for model_id, model in base_models.items():
        stacker_gbm.add_base_model(model_id, model, {'model_type': model_id, 'features': list(range(5))})
    
    stacker_gbm.generate_oof_predictions(X_train, y_train)
    stacker_gbm.train_meta_model(y_train)
    result_stacking_gbm = stacker_gbm.evaluate(X_test, y_test)
    stacker_gbm.save_stacking_model('Stacking_GBM')
    stacking_results.append(result_stacking_gbm)
    
    logger.info("\n[SUCCESS] STACKING ENSEMBLES COMPLETED")
    
    # =========================================================================
    # STEP 3: Evaluate All Ensembles
    # =========================================================================
    logger.info("\n[REPORT] STEP 3: COMPREHENSIVE EVALUATION")
    logger.info("-"*80)
    
    evaluator = EnsembleEvaluator('output/final_ensemble_evaluation')
    
    # Add voting ensembles
    evaluator.add_ensemble_result('SimpleVoting', result_simple, result_simple['predictions'], y_test)
    evaluator.add_ensemble_result('WeightedVoting_R2', result_weighted_r2, result_weighted_r2['predictions'], y_test)
    evaluator.add_ensemble_result('WeightedVoting_RMSE', result_weighted_rmse, result_weighted_rmse['predictions'], y_test)
    evaluator.add_ensemble_result('WeightedVoting_InvError', result_weighted_inv, result_weighted_inv['predictions'], y_test)
    
    # Add stacking ensembles
    evaluator.add_ensemble_result('Stacking_Ridge', result_stacking_ridge, result_stacking_ridge['predictions'], y_test)
    evaluator.add_ensemble_result('Stacking_Lasso', result_stacking_lasso, result_stacking_lasso['predictions'], y_test)
    evaluator.add_ensemble_result('Stacking_RF', result_stacking_rf, result_stacking_rf['predictions'], y_test)
    evaluator.add_ensemble_result('Stacking_GBM', result_stacking_gbm, result_stacking_gbm['predictions'], y_test)
    
    # Compare all
    comparison_df = evaluator.compare_all_ensembles()
    
    # Select best
    best_ensemble = evaluator.select_best_ensemble('r2')
    
    # Generate final report
    final_report = evaluator.generate_final_report()
    
    logger.info("\n[SUCCESS] EVALUATION COMPLETED")
    
    # =========================================================================
    # STEP 4: Summary & Recommendations
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("FAZ 7: FINAL SUMMARY & RECOMMENDATIONS")
    logger.info("="*80)
    
    logger.info(f"\n[LIST] ENSEMBLE METHODS TESTED: {len(final_report['all_results'])}")
    logger.info(f"\n[SUCCESS] BEST ENSEMBLE:")
    logger.info(f"   Name: {best_ensemble['name']}")
    logger.info(f"   Method: {best_ensemble['method']}")
    logger.info(f"   R² = {best_ensemble['r2']:.4f}")
    logger.info(f"   RMSE = {best_ensemble['rmse']:.4f}")
    logger.info(f"   MAE = {best_ensemble['mae']:.4f}")
    
    logger.info(f"\n[REPORT] TOP 3 ENSEMBLES:")
    top3 = sorted(final_report['all_results'], key=lambda x: x['r2'], reverse=True)[:3]
    for i, result in enumerate(top3):
        logger.info(f"   {i+1}. {result['name']}: R²={result['r2']:.4f}, RMSE={result['rmse']:.4f}")
    
    logger.info(f"\n[TIP] RECOMMENDATIONS:")
    if best_ensemble['method'] == 'stacking':
        logger.info(f"   -> Stacking with {best_ensemble['meta_model_type']} meta-model performed best")
        logger.info(f"   -> Use this for production predictions")
        logger.info(f"   -> Meta-model successfully learned from base model predictions")
    else:
        logger.info(f"   -> {best_ensemble['method']} performed best")
        logger.info(f"   -> Simpler ensemble methods may be sufficient")
    
    logger.info(f"\n[FOLDER] OUTPUT FILES:")
    logger.info(f"   -> output/voting_ensembles/")
    logger.info(f"   -> output/stacking_models/")
    logger.info(f"   -> output/final_ensemble_evaluation/")
    
    logger.info("\n" + "="*80)
    logger.info("[COMPLETE] FAZ 7 COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    
    return {
        'best_ensemble': best_ensemble,
        'final_report': final_report,
        'comparison_df': comparison_df
    }


def main():
    """Run FAZ 7 pipeline"""
    try:
        results = faz7_ensemble_pipeline()
        return results
    except Exception as e:
        logger.error(f"\n[ERROR] ERROR in FAZ 7 pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()
