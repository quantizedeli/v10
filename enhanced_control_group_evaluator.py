"""
Enhanced Control Group Evaluator
Detailed nucleus-level predictions and accuracy tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedControlGroupEvaluator:
    """
    Detailed control group evaluation with nucleus-level tracking
    Shows which nuclei were predicted correctly and which failed
    """
    
    def __init__(self):
        self.evaluation_results = []
        self.nucleus_predictions = []
        
    def evaluate_model_on_control_group(self,
                                       model,
                                       X_test: np.ndarray,
                                       y_test: np.ndarray,
                                       nucleus_ids: List[str] = None,
                                       model_name: str = "Model") -> Dict:
        """
        Comprehensive control group evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            nucleus_ids: List of nucleus identifiers (A, Z, N, etc.)
            model_name: Name of model being evaluated
            
        Returns:
            Detailed evaluation results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"CONTROL GROUP EVALUATION: {model_name}")
        logger.info(f"{'='*60}")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Overall metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        
        # Residuals
        residuals = y_test - y_pred
        abs_residuals = np.abs(residuals)
        
        # Classification by accuracy
        excellent = np.sum(abs_residuals < 0.1)  # Error < 0.1
        good = np.sum((abs_residuals >= 0.1) & (abs_residuals < 0.5))  # 0.1 <= Error < 0.5
        acceptable = np.sum((abs_residuals >= 0.5) & (abs_residuals < 1.0))  # 0.5 <= Error < 1.0
        poor = np.sum(abs_residuals >= 1.0)  # Error >= 1.0
        
        # Nucleus-level details
        nucleus_details = []
        for i in range(len(y_test)):
            nucleus_id = nucleus_ids[i] if nucleus_ids else f"Nucleus_{i}"
            
            error = abs_residuals[i]
            
            if error < 0.1:
                accuracy_class = "Excellent"
                emoji = "🎯"
            elif error < 0.5:
                accuracy_class = "Good"
                emoji = "✅"
            elif error < 1.0:
                accuracy_class = "Acceptable"
                emoji = "⚠️"
            else:
                accuracy_class = "Poor"
                emoji = "❌"
            
            nucleus_details.append({
                'nucleus_id': nucleus_id,
                'actual': float(y_test[i]),
                'predicted': float(y_pred[i]),
                'error': float(error),
                'relative_error_%': float(error / (abs(y_test[i]) + 1e-10) * 100),
                'accuracy_class': accuracy_class,
                'emoji': emoji
            })
        
        # Sort by error (worst first)
        nucleus_details.sort(key=lambda x: x['error'], reverse=True)
        
        results = {
            'model_name': model_name,
            'n_samples': len(y_test),
            
            # Overall metrics
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            
            # Accuracy distribution
            'excellent_predictions': int(excellent),
            'good_predictions': int(good),
            'acceptable_predictions': int(acceptable),
            'poor_predictions': int(poor),
            
            'excellent_percent': float(excellent / len(y_test) * 100),
            'good_percent': float(good / len(y_test) * 100),
            'acceptable_percent': float(acceptable / len(y_test) * 100),
            'poor_percent': float(poor / len(y_test) * 100),
            
            # Residual statistics
            'mean_residual': float(np.mean(residuals)),
            'std_residual': float(np.std(residuals)),
            'max_error': float(np.max(abs_residuals)),
            'min_error': float(np.min(abs_residuals)),
            
            # Nucleus-level
            'nucleus_predictions': nucleus_details
        }
        
        self.evaluation_results.append(results)
        self.nucleus_predictions.extend(nucleus_details)
        
        # Print summary
        logger.info(f"\n📊 Overall Metrics:")
        logger.info(f"   MAE:  {mae:.4f}")
        logger.info(f"   RMSE: {rmse:.4f}")
        logger.info(f"   R²:   {r2:.4f}")
        logger.info(f"   MAPE: {mape:.2f}%")
        
        logger.info(f"\n🎯 Accuracy Distribution:")
        logger.info(f"   🎯 Excellent (<0.1 error): {excellent} ({excellent/len(y_test)*100:.1f}%)")
        logger.info(f"   ✅ Good (0.1-0.5 error):   {good} ({good/len(y_test)*100:.1f}%)")
        logger.info(f"   ⚠️  Acceptable (0.5-1.0):   {acceptable} ({acceptable/len(y_test)*100:.1f}%)")
        logger.info(f"   ❌ Poor (>1.0 error):      {poor} ({poor/len(y_test)*100:.1f}%)")
        
        logger.info(f"\n🔍 Worst 5 Predictions:")
        for i, nucleus in enumerate(nucleus_details[:5], 1):
            logger.info(f"   {i}. {nucleus['emoji']} {nucleus['nucleus_id']}: "
                       f"Actual={nucleus['actual']:.4f}, "
                       f"Predicted={nucleus['predicted']:.4f}, "
                       f"Error={nucleus['error']:.4f}")
        
        logger.info(f"\n✅ Best 5 Predictions:")
        for i, nucleus in enumerate(nucleus_details[-5:][::-1], 1):
            logger.info(f"   {i}. {nucleus['emoji']} {nucleus['nucleus_id']}: "
                       f"Actual={nucleus['actual']:.4f}, "
                       f"Predicted={nucleus['predicted']:.4f}, "
                       f"Error={nucleus['error']:.4f}")
        
        return results
    
    def generate_nucleus_report(self, output_path: str = "nucleus_predictions.csv"):
        """Generate detailed nucleus-level prediction report"""
        if not self.nucleus_predictions:
            logger.warning("No nucleus predictions to report")
            return
        
        df = pd.DataFrame(self.nucleus_predictions)
        df.to_csv(output_path, index=False)
        
        logger.info(f"\n📄 Nucleus-level predictions saved to: {output_path}")
        
        return df
    
    def compare_models(self, models: List[Dict]) -> pd.DataFrame:
        """
        Compare multiple models on control group
        
        Args:
            models: List of {'model': model_obj, 'name': str, 'X_test': ..., 'y_test': ...}
        """
        comparison_results = []
        
        for model_info in models:
            results = self.evaluate_model_on_control_group(
                model=model_info['model'],
                X_test=model_info['X_test'],
                y_test=model_info['y_test'],
                nucleus_ids=model_info.get('nucleus_ids'),
                model_name=model_info['name']
            )
            
            comparison_results.append({
                'Model': results['model_name'],
                'R²': results['r2'],
                'MAE': results['mae'],
                'RMSE': results['rmse'],
                'MAPE%': results['mape'],
                'Excellent%': results['excellent_percent'],
                'Good%': results['good_percent'],
                'Poor%': results['poor_percent']
            })
        
        df = pd.DataFrame(comparison_results)
        df = df.sort_values('R²', ascending=False)
        
        logger.info(f"\n{'='*80}")
        logger.info("MODEL COMPARISON ON CONTROL GROUP")
        logger.info(f"{'='*80}")
        print(df.to_string(index=False))
        
        return df
    
    def identify_problematic_nuclei(self, error_threshold: float = 1.0) -> pd.DataFrame:
        """
        Identify nuclei that multiple models struggle with
        
        Args:
            error_threshold: Minimum error to be considered problematic
        """
        if not self.nucleus_predictions:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.nucleus_predictions)
        
        # Group by nucleus
        problematic = df[df['error'] > error_threshold].groupby('nucleus_id').agg({
            'error': ['count', 'mean', 'std'],
            'actual': 'first',
            'accuracy_class': lambda x: x.mode()[0] if len(x) > 0 else 'Poor'
        }).round(4)
        
        problematic.columns = ['frequency', 'avg_error', 'std_error', 'actual_value', 'typical_class']
        problematic = problematic.sort_values('frequency', ascending=False)
        
        logger.info(f"\n⚠️  PROBLEMATIC NUCLEI (Error > {error_threshold}):")
        logger.info(f"   Total: {len(problematic)}")
        print(problematic.head(10))
        
        return problematic


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    
    X_test = np.random.randn(50, 5)
    y_test = np.sum(X_test, axis=1) + np.random.randn(50) * 0.3
    
    nucleus_ids = [f"Nucleus_{i}_Z{26+i}_N{30+i}" for i in range(50)]
    
    # Mock model
    class MockModel:
        def predict(self, X):
            return np.sum(X, axis=1) + np.random.randn(len(X)) * 0.5
    
    evaluator = EnhancedControlGroupEvaluator()
    
    # Generate reports
    nucleus_report = evaluator.generate_nucleus_report()
    print("\n=== Nucleus Report Sample ===")
    print(nucleus_report.head(10))

    