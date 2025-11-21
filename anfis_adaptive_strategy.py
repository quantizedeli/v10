"""
Adaptive Learning Strategy - COMPLETE
3-Stage Intelligent Training Strategy

Stages:
1. Exploration (1-20): Test all configurations
2. Validation (21-50): Filter poor performers
3. Confirmation (51+): Focus on best configurations

Features:
- Multi-metric evaluation
- Pattern-based pruning
- Configuration tracking
- Resource optimization
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# MULTI-METRIC EVALUATOR
# ============================================================================

class MultiMetricEvaluator:
    """
    Multi-metric evaluation for model performance
    
    Metrics:
    - R² (primary)
    - RMSE
    - MAE
    - Training time
    - Composite score
    """
    
    def __init__(self):
        self.weights = {
            'R2': 0.50,
            'RMSE': 0.25,
            'MAE': 0.15,
            'Time': 0.10
        }
    
    def evaluate(self, metrics):
        """
        Calculate composite score
        
        Args:
            metrics: Dict with R2, RMSE, MAE, training_time
        
        Returns:
            float: Composite score [0, 1]
        """
        
        r2_score = metrics.get('R2', 0.0)
        rmse = metrics.get('RMSE', 1.0)
        mae = metrics.get('MAE', 1.0)
        time = metrics.get('training_time', 100.0)
        
        # Normalize RMSE and MAE (lower is better)
        # Assume typical ranges
        rmse_norm = max(0, 1 - rmse / 0.5)  # Assume max RMSE ~0.5
        mae_norm = max(0, 1 - mae / 0.3)    # Assume max MAE ~0.3
        
        # Normalize time (lower is better)
        time_norm = max(0, 1 - time / 300)  # Assume max time 300s
        
        # Composite
        composite = (
            self.weights['R2'] * r2_score +
            self.weights['RMSE'] * rmse_norm +
            self.weights['MAE'] * mae_norm +
            self.weights['Time'] * time_norm
        )
        
        return composite
    
    def is_acceptable(self, metrics, min_r2=0.5):
        """Check if metrics meet minimum requirements"""
        return metrics.get('R2', 0.0) >= min_r2


# ============================================================================
# ADAPTIVE LEARNING STRATEGY
# ============================================================================

class AdaptiveLearningStrategy:
    """
    3-Stage Adaptive Learning Strategy
    
    Stage 1 (Exploration): Train all (1-20)
    Stage 2 (Validation): Filter poor configs (21-50)
    Stage 3 (Confirmation): Focus on best (51+)
    """
    
    def __init__(self, output_dir='adaptive_reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluator = MultiMetricEvaluator()
        
        # Stage thresholds
        self.stage1_limit = 20
        self.stage2_limit = 50
        
        # Performance thresholds
        self.excellent_threshold = 0.85
        self.good_threshold = 0.70
        self.acceptable_threshold = 0.50
        
        # Tracking
        self.config_stats = defaultdict(lambda: {
            'n_trained': 0,
            'metrics': [],
            'composite_scores': [],
            'stage': 1
        })
        
        self.iteration = 0
        self.skipped_count = 0
        
        logger.info("Adaptive Learning Strategy initialized")
    
    def should_train_config(self, config_key):
        """
        Decide whether to train this configuration
        
        Args:
            config_key: Configuration identifier
        
        Returns:
            (bool, str): (should_train, reason)
        """
        
        self.iteration += 1
        current_stage = self._get_current_stage()
        
        stats = self.config_stats[config_key]
        
        # Stage 1: Exploration (train all)
        if current_stage == 1:
            return True, "Stage 1: Exploration"
        
        # Stage 2: Validation
        if current_stage == 2:
            # If trained 3+ times with consistently poor results, skip
            if stats['n_trained'] >= 3:
                avg_r2 = np.mean([m.get('R2', 0) for m in stats['metrics']])
                
                if avg_r2 < self.acceptable_threshold:
                    self.skipped_count += 1
                    return False, f"Poor performance: Avg R²={avg_r2:.3f} < {self.acceptable_threshold}"
            
            return True, "Stage 2: Validation"
        
        # Stage 3: Confirmation
        if current_stage == 3:
            # Only train if showing good performance
            if stats['n_trained'] >= 5:
                avg_r2 = np.mean([m.get('R2', 0) for m in stats['metrics']])
                
                if avg_r2 >= self.good_threshold:
                    return True, f"Good performance: Avg R²={avg_r2:.3f}"
                else:
                    self.skipped_count += 1
                    return False, f"Below threshold: Avg R²={avg_r2:.3f} < {self.good_threshold}"
            
            return True, "Stage 3: Initial training"
        
        return True, "Default: train"
    
    def update_stats(self, config_key, metrics):
        """
        Update configuration statistics
        
        Args:
            config_key: Configuration identifier
            metrics: Dict with performance metrics
        """
        
        stats = self.config_stats[config_key]
        
        stats['n_trained'] += 1
        stats['metrics'].append(metrics)
        
        # Calculate composite score
        composite = self.evaluator.evaluate(metrics)
        stats['composite_scores'].append(composite)
        
        # Update stage
        stats['stage'] = self._get_current_stage()
        
        # Log
        avg_r2 = np.mean([m.get('R2', 0) for m in stats['metrics']])
        logger.info(f"  Config {config_key}: n={stats['n_trained']}, avg_R²={avg_r2:.4f}, composite={composite:.4f}")
    
    def _get_current_stage(self):
        """Determine current stage based on iteration"""
        if self.iteration <= self.stage1_limit:
            return 1
        elif self.iteration <= self.stage2_limit:
            return 2
        else:
            return 3
    
    def get_statistics(self):
        """Get overall statistics"""
        
        stats = {
            'total_iterations': self.iteration,
            'total_skipped': self.skipped_count,
            'savings_percentage': (self.skipped_count / max(1, self.iteration)) * 100,
            'current_stage': self._get_current_stage(),
            'configs': {}
        }
        
        for config_key, config_stats in self.config_stats.items():
            if config_stats['n_trained'] > 0:
                metrics_list = config_stats['metrics']
                
                stats['configs'][config_key] = {
                    'n_trained': config_stats['n_trained'],
                    'avg_r2': float(np.mean([m.get('R2', 0) for m in metrics_list])),
                    'std_r2': float(np.std([m.get('R2', 0) for m in metrics_list])),
                    'avg_composite': float(np.mean(config_stats['composite_scores'])),
                    'best_r2': float(max([m.get('R2', 0) for m in metrics_list])),
                    'stage': config_stats['stage']
                }
        
        return stats
    
    def generate_report(self):
        """Generate adaptive learning report"""
        
        stats = self.get_statistics()
        
        # Save JSON
        report_file = self.output_dir / 'adaptive_learning_report.json'
        with open(report_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✓ Adaptive learning report: {report_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("ADAPTIVE LEARNING STRATEGY SUMMARY")
        print("="*80)
        print(f"Total iterations: {stats['total_iterations']}")
        print(f"Total skipped: {stats['total_skipped']}")
        print(f"Resource savings: {stats['savings_percentage']:.1f}%")
        print(f"Current stage: {stats['current_stage']}")
        print(f"Unique configs: {len(stats['configs'])}")
        print("="*80)
        
        # Top performers
        if stats['configs']:
            print("\nTOP 5 CONFIGURATIONS:")
            print("-"*80)
            sorted_configs = sorted(
                stats['configs'].items(),
                key=lambda x: x[1]['avg_r2'],
                reverse=True
            )
            
            for i, (config, config_stats) in enumerate(sorted_configs[:5], 1):
                print(f"{i}. {config}")
                print(f"   Avg R²: {config_stats['avg_r2']:.4f} (±{config_stats['std_r2']:.4f})")
                print(f"   Best R²: {config_stats['best_r2']:.4f}")
                print(f"   Trained: {config_stats['n_trained']} times")
        
        return stats


# ============================================================================
# PATTERN TRACKER
# ============================================================================

class PatternTracker:
    """
    Track patterns in configuration performance
    
    Identifies:
    - Best feature combinations
    - Best preprocessing settings
    - Best model types per target
    """
    
    def __init__(self):
        self.patterns = defaultdict(list)
    
    def record_result(self, config_info, metrics):
        """Record configuration result"""
        
        # Extract config components
        model = config_info.get('model', 'Unknown')
        target = config_info.get('target', 'Unknown')
        features = config_info.get('features', 'Unknown')
        scaling = config_info.get('scaling', 'Unknown')
        
        r2 = metrics.get('R2', 0.0)
        
        # Record patterns
        self.patterns['model'].append({'name': model, 'r2': r2})
        self.patterns['target'].append({'name': target, 'r2': r2})
        self.patterns['features'].append({'name': features, 'r2': r2})
        self.patterns['scaling'].append({'name': scaling, 'r2': r2})
    
    def analyze_patterns(self):
        """Analyze recorded patterns"""
        
        analysis = {}
        
        for category, records in self.patterns.items():
            if not records:
                continue
            
            # Group by name
            grouped = defaultdict(list)
            for record in records:
                grouped[record['name']].append(record['r2'])
            
            # Calculate statistics
            stats = {}
            for name, r2_list in grouped.items():
                stats[name] = {
                    'mean': np.mean(r2_list),
                    'std': np.std(r2_list),
                    'count': len(r2_list)
                }
            
            # Sort by mean
            sorted_stats = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)
            
            analysis[category] = sorted_stats
        
        return analysis
    
    def print_analysis(self):
        """Print pattern analysis"""
        
        analysis = self.analyze_patterns()
        
        print("\n" + "="*80)
        print("PATTERN ANALYSIS")
        print("="*80)
        
        for category, sorted_stats in analysis.items():
            print(f"\n{category.upper()} PERFORMANCE:")
            print("-"*80)
            
            for name, stats in sorted_stats[:5]:
                print(f"{name}: Mean R²={stats['mean']:.4f} (±{stats['std']:.4f}), n={stats['count']}")


# ============================================================================
# TEST
# ============================================================================

def test_adaptive_strategy():
    """Test adaptive learning strategy"""
    
    print("\n" + "="*80)
    print("ADAPTIVE LEARNING STRATEGY TEST")
    print("="*80)
    
    strategy = AdaptiveLearningStrategy()
    
    # Simulate 100 training iterations
    configs = ['Config_A', 'Config_B', 'Config_C', 'Config_D']
    
    for i in range(100):
        for config in configs:
            # Check if should train
            should_train, reason = strategy.should_train_config(config)
            
            if should_train:
                # Simulate training
                # Config_A: excellent, Config_B: good, Config_C: poor, Config_D: variable
                if config == 'Config_A':
                    r2 = np.random.uniform(0.85, 0.95)
                elif config == 'Config_B':
                    r2 = np.random.uniform(0.70, 0.80)
                elif config == 'Config_C':
                    r2 = np.random.uniform(0.30, 0.50)
                else:
                    r2 = np.random.uniform(0.50, 0.85)
                
                metrics = {
                    'R2': r2,
                    'RMSE': np.random.uniform(0.1, 0.3),
                    'MAE': np.random.uniform(0.05, 0.2),
                    'training_time': np.random.uniform(10, 100)
                }
                
                strategy.update_stats(config, metrics)
    
    # Generate report
    strategy.generate_report()
    
    print("\n✓ Test completed!")


if __name__ == "__main__":
    test_adaptive_strategy()
    print("\n✅ Adaptive Strategy - COMPLETE")
    print("Location: adaptive_learning/adaptive_strategy.py")

# ==================== EKLEME BAŞI ====================
class ANFISAdaptiveStrategy:
    """Wrapper for AdaptiveLearningStrategy"""
    def __init__(self, config=None):
        self.strategy = AdaptiveLearningStrategy(config)
    
    def analyze_data_complexity(self, X, y):
        return self.strategy.analyze_data_complexity(X, y)
    
    def recommend_clusters(self, complexity_analysis):
        return self.strategy.recommend_clusters(complexity_analysis)
    
    def recommend_epochs(self, complexity_analysis):
        return self.strategy.recommend_epochs(complexity_analysis)
    
    def recommend_training_params(self, X, y):
        return self.strategy.recommend_training_params(X, y)
    
    def should_retry(self, current_r2, previous_attempts=None):
        return self.strategy.should_retry(current_r2, previous_attempts)
    
    def adjust_params_for_retry(self, current_params, current_r2):
        return self.strategy.adjust_params_for_retry(current_params, current_r2)
# ==================== EKLEME SON ====================