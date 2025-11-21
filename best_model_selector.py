"""
Enhanced Best Model Selection Module for PYV5_5
Multi-criteria model evaluation and intelligent selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Container for model performance metrics"""
    model_name: str
    model_type: str
    mae: float
    rmse: float
    r2: float
    mape: float
    training_time: float
    prediction_time: float
    model_size: float  # MB
    complexity_score: float
    stability_score: float
    generalization_score: float
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'mae': self.mae,
            'rmse': self.rmse,
            'r2': self.r2,
            'mape': self.mape,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'model_size': self.model_size,
            'complexity_score': self.complexity_score,
            'stability_score': self.stability_score,
            'generalization_score': self.generalization_score
        }


class BestModelSelector:
    """Advanced model selection with multi-criteria analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.model_performances: List[ModelPerformance] = []
        self.selection_history: List[Dict] = []
        
    def _default_config(self) -> Dict:
        return {
            'selection_criteria': {
                'accuracy': 0.35,      # R2, MAE, RMSE
                'speed': 0.20,         # Training + prediction time
                'efficiency': 0.15,    # Model size, complexity
                'stability': 0.15,     # Cross-validation variance
                'generalization': 0.15 # Test vs train performance
            },
            'min_r2_threshold': 0.70,
            'max_training_time': 3600,  # seconds
            'max_model_size': 500,      # MB
            'enable_ensemble': True,
            'top_k_models': 3
        }
    
    def add_model_performance(self, performance: ModelPerformance):
        """Add model performance metrics"""
        self.model_performances.append(performance)
        logger.info(f"Added performance metrics for {performance.model_name}")
    
    def calculate_composite_score(self, performance: ModelPerformance) -> float:
        """
        Calculate composite score based on multiple criteria
        
        Score components:
        1. Accuracy (35%): R2, MAE, RMSE
        2. Speed (20%): Training + prediction time
        3. Efficiency (15%): Model size, complexity
        4. Stability (15%): Cross-validation consistency
        5. Generalization (15%): Test vs train gap
        """
        weights = self.config['selection_criteria']
        
        # Normalize metrics to 0-1 scale
        all_performances = self.model_performances
        
        # 1. Accuracy Score
        r2_scores = [p.r2 for p in all_performances]
        mae_scores = [p.mae for p in all_performances]
        rmse_scores = [p.rmse for p in all_performances]
        
        r2_norm = self._normalize(performance.r2, r2_scores, higher_is_better=True)
        mae_norm = self._normalize(performance.mae, mae_scores, higher_is_better=False)
        rmse_norm = self._normalize(performance.rmse, rmse_scores, higher_is_better=False)
        
        accuracy_score = (r2_norm * 0.5 + mae_norm * 0.25 + rmse_norm * 0.25)
        
        # 2. Speed Score
        train_times = [p.training_time for p in all_performances]
        pred_times = [p.prediction_time for p in all_performances]
        
        train_norm = self._normalize(performance.training_time, train_times, higher_is_better=False)
        pred_norm = self._normalize(performance.prediction_time, pred_times, higher_is_better=False)
        
        speed_score = (train_norm * 0.7 + pred_norm * 0.3)
        
        # 3. Efficiency Score
        sizes = [p.model_size for p in all_performances]
        complexities = [p.complexity_score for p in all_performances]
        
        size_norm = self._normalize(performance.model_size, sizes, higher_is_better=False)
        complexity_norm = self._normalize(performance.complexity_score, complexities, higher_is_better=False)
        
        efficiency_score = (size_norm * 0.5 + complexity_norm * 0.5)
        
        # 4. Stability Score (already normalized 0-1)
        stability_score = performance.stability_score
        
        # 5. Generalization Score (already normalized 0-1)
        generalization_score = performance.generalization_score
        
        # Composite score
        composite = (
            accuracy_score * weights['accuracy'] +
            speed_score * weights['speed'] +
            efficiency_score * weights['efficiency'] +
            stability_score * weights['stability'] +
            generalization_score * weights['generalization']
        )
        
        return composite
    
    def _normalize(self, value: float, all_values: List[float], higher_is_better: bool = True) -> float:
        """Normalize value to 0-1 scale"""
        min_val = min(all_values)
        max_val = max(all_values)
        
        if max_val == min_val:
            return 1.0
        
        normalized = (value - min_val) / (max_val - min_val)
        
        if not higher_is_better:
            normalized = 1.0 - normalized
        
        return normalized
    
    def select_best_model(self, 
                         criterion: str = 'composite',
                         return_top_k: Optional[int] = None) -> List[Tuple[ModelPerformance, float]]:
        """
        Select best model(s) based on criterion
        
        Args:
            criterion: 'composite', 'accuracy', 'speed', 'efficiency', 'balanced'
            return_top_k: Return top K models instead of just best
            
        Returns:
            List of (ModelPerformance, score) tuples
        """
        if not self.model_performances:
            logger.error("No model performances recorded")
            return []
        
        # Calculate scores for all models
        scored_models = []
        
        for perf in self.model_performances:
            # Apply hard constraints
            if perf.r2 < self.config['min_r2_threshold']:
                logger.info(f"Skipping {perf.model_name}: R2 below threshold")
                continue
            
            if perf.training_time > self.config['max_training_time']:
                logger.info(f"Skipping {perf.model_name}: Training time exceeds limit")
                continue
            
            if perf.model_size > self.config['max_model_size']:
                logger.info(f"Skipping {perf.model_name}: Model size exceeds limit")
                continue
            
            # Calculate score based on criterion
            if criterion == 'composite':
                score = self.calculate_composite_score(perf)
            elif criterion == 'accuracy':
                score = perf.r2
            elif criterion == 'speed':
                score = 1.0 / (perf.training_time + perf.prediction_time)
            elif criterion == 'efficiency':
                score = 1.0 / (perf.model_size * perf.complexity_score)
            elif criterion == 'balanced':
                score = (perf.r2 + perf.stability_score + perf.generalization_score) / 3.0
            else:
                score = self.calculate_composite_score(perf)
            
            scored_models.append((perf, score))
        
        # Sort by score (descending)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K or all
        k = return_top_k or self.config['top_k_models']
        top_models = scored_models[:k]
        
        # Log selection
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL SELECTION RESULTS (Criterion: {criterion})")
        logger.info(f"{'='*60}")
        for i, (perf, score) in enumerate(top_models, 1):
            logger.info(f"{i}. {perf.model_name} (Type: {perf.model_type})")
            logger.info(f"   Score: {score:.4f}")
            logger.info(f"   R²: {perf.r2:.4f} | MAE: {perf.mae:.4f} | RMSE: {perf.rmse:.4f}")
            logger.info(f"   Training: {perf.training_time:.2f}s | Size: {perf.model_size:.2f}MB")
            logger.info(f"   Stability: {perf.stability_score:.4f} | Generalization: {perf.generalization_score:.4f}")
            logger.info("")
        
        # Save selection history
        self.selection_history.append({
            'criterion': criterion,
            'timestamp': pd.Timestamp.now().isoformat(),
            'selected_models': [perf.model_name for perf, _ in top_models],
            'scores': [float(score) for _, score in top_models]
        })
        
        return top_models
    
    def recommend_ensemble(self, top_k: int = 3) -> List[Tuple[ModelPerformance, float]]:
        """
        Recommend models for ensemble based on diversity and performance
        
        Returns:
            List of (ModelPerformance, weight) tuples
        """
        if not self.config['enable_ensemble']:
            return self.select_best_model(return_top_k=1)
        
        # Get top performing models
        top_models = self.select_best_model(criterion='composite', return_top_k=top_k*2)
        
        # Select diverse models
        selected_models = []
        model_types_used = set()
        
        for perf, score in top_models:
            if perf.model_type not in model_types_used or len(selected_models) < top_k:
                selected_models.append((perf, score))
                model_types_used.add(perf.model_type)
            
            if len(selected_models) >= top_k:
                break
        
        # Calculate ensemble weights (normalized scores)
        total_score = sum(score for _, score in selected_models)
        ensemble_weights = [(perf, score / total_score) for perf, score in selected_models]
        
        logger.info(f"\n{'='*60}")
        logger.info("ENSEMBLE RECOMMENDATION")
        logger.info(f"{'='*60}")
        for perf, weight in ensemble_weights:
            logger.info(f"{perf.model_name} ({perf.model_type}): Weight = {weight:.3f}")
        
        return ensemble_weights
    
    def generate_selection_report(self, output_path: str = "model_selection_report.json"):
        """Generate comprehensive selection report"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_models_evaluated': len(self.model_performances),
            'config': self.config,
            'all_performances': [p.to_dict() for p in self.model_performances],
            'selection_history': self.selection_history
        }
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Selection report saved to {output_path}")
        
        return report
    
    def compare_models(self, model_names: List[str]) -> pd.DataFrame:
        """Generate comparison table for specific models"""
        selected_perfs = [p for p in self.model_performances if p.model_name in model_names]
        
        data = []
        for perf in selected_perfs:
            composite_score = self.calculate_composite_score(perf)
            data.append({
                'Model': perf.model_name,
                'Type': perf.model_type,
                'R²': f"{perf.r2:.4f}",
                'MAE': f"{perf.mae:.4f}",
                'RMSE': f"{perf.rmse:.4f}",
                'MAPE': f"{perf.mape:.2f}%",
                'Training (s)': f"{perf.training_time:.2f}",
                'Prediction (ms)': f"{perf.prediction_time*1000:.2f}",
                'Size (MB)': f"{perf.model_size:.2f}",
                'Stability': f"{perf.stability_score:.4f}",
                'Generalization': f"{perf.generalization_score:.4f}",
                'Composite Score': f"{composite_score:.4f}"
            })
        
        df = pd.DataFrame(data)
        return df
    
    def get_best_for_task(self, task_type: str) -> ModelPerformance:
        """
        Get best model for specific task
        
        Args:
            task_type: 'production', 'research', 'real_time', 'high_accuracy'
        """
        if task_type == 'production':
            # Balanced: speed + accuracy + reliability
            criterion = 'balanced'
        elif task_type == 'research':
            # Best accuracy regardless of speed
            criterion = 'accuracy'
        elif task_type == 'real_time':
            # Fastest model with acceptable accuracy
            criterion = 'speed'
        elif task_type == 'high_accuracy':
            # Best composite score
            criterion = 'composite'
        else:
            criterion = 'composite'
        
        best_models = self.select_best_model(criterion=criterion, return_top_k=1)
        
        if best_models:
            return best_models[0][0]
        return None


if __name__ == "__main__":
    # Test the selector
    selector = BestModelSelector()
    
    # Add sample performances
    models = [
        ModelPerformance("ANFIS_V1", "ANFIS", 0.04, 0.065, 0.92, 5.2, 45.3, 0.002, 12.5, 0.7, 0.85, 0.88),
        ModelPerformance("BNN_V1", "Bayesian", 0.038, 0.061, 0.94, 4.8, 120.5, 0.005, 45.2, 0.8, 0.90, 0.87),
        ModelPerformance("PINN_V1", "Physics", 0.042, 0.068, 0.91, 5.5, 95.2, 0.003, 38.5, 0.75, 0.86, 0.89),
        ModelPerformance("Ensemble_V1", "Ensemble", 0.035, 0.058, 0.95, 4.2, 180.0, 0.008, 95.0, 0.85, 0.92, 0.91),
        ModelPerformance("RF_V1", "RandomForest", 0.045, 0.072, 0.89, 5.8, 25.5, 0.001, 8.5, 0.65, 0.82, 0.84)
    ]
    
    for model in models:
        selector.add_model_performance(model)
    
    # Test different selection criteria
    print("\n=== COMPOSITE SCORE SELECTION ===")
    best_composite = selector.select_best_model('composite', return_top_k=3)
    
    print("\n=== ACCURACY FOCUSED SELECTION ===")
    best_accuracy = selector.select_best_model('accuracy', return_top_k=3)
    
    print("\n=== SPEED FOCUSED SELECTION ===")
    best_speed = selector.select_best_model('speed', return_top_k=3)
    
    print("\n=== ENSEMBLE RECOMMENDATION ===")
    ensemble = selector.recommend_ensemble(top_k=3)
    
    print("\n=== COMPARISON TABLE ===")
    comparison = selector.compare_models([m.model_name for m in models])
    print(comparison.to_string(index=False))
    
    # Generate report
    selector.generate_selection_report()