#!/usr/bin/env python3
"""
Generate Sample Data for PFAZ 10 Testing
=========================================

Creates realistic sample data for testing the thesis generation system.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def generate_sample_model_reports(output_dir: Path):
    """Generate sample model performance reports."""
    models = ['rf', 'xgb', 'dnn', 'svr', 'anfis']
    targets = ['MM', 'QM', 'Beta_2']

    reports = []

    for model_type in models:
        for target in targets:
            for trial in range(3):  # 3 trials per model-target combination
                # Generate realistic metrics
                if model_type == 'xgb':
                    base_r2 = 0.92
                elif model_type == 'rf':
                    base_r2 = 0.89
                elif model_type == 'dnn':
                    base_r2 = 0.88
                elif model_type == 'anfis':
                    base_r2 = 0.87
                else:  # svr
                    base_r2 = 0.85

                # Add some variation
                r2 = base_r2 + np.random.uniform(-0.05, 0.05)
                r2 = min(0.99, max(0.70, r2))

                rmse = (1 - r2) * 0.3 + np.random.uniform(0, 0.05)
                mae = rmse * 0.7 + np.random.uniform(0, 0.02)

                report = {
                    'model_id': f'{model_type}_{target}_{trial:03d}',
                    'model_type': model_type,
                    'target': target,
                    'trial': trial,
                    'metrics': {
                        'r2': float(r2),
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'mse': float(rmse ** 2)
                    },
                    'hyperparameters': generate_hyperparameters(model_type),
                    'training_time': float(np.random.uniform(30, 300)),
                    'n_features': 44,
                    'n_samples_train': 213,
                    'n_samples_test': 54
                }

                reports.append(report)

                # Save individual report
                filename = f'{model_type}_{target}_trial{trial}.json'
                with open(output_dir / filename, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2)

    print(f"Generated {len(reports)} model reports")
    return reports


def generate_hyperparameters(model_type: str) -> dict:
    """Generate realistic hyperparameters."""
    if model_type == 'rf':
        max_depth_val = np.random.choice([10, 20, 30])
        return {
            'n_estimators': int(np.random.choice([100, 200, 500])),
            'max_depth': int(max_depth_val) if max_depth_val is not None else None,
            'min_samples_split': int(np.random.choice([2, 5, 10])),
            'min_samples_leaf': int(np.random.choice([1, 2, 4]))
        }
    elif model_type == 'xgb':
        return {
            'n_estimators': int(np.random.choice([100, 200, 500])),
            'learning_rate': float(np.random.choice([0.01, 0.05, 0.1, 0.3])),
            'max_depth': int(np.random.choice([3, 5, 7, 9])),
            'subsample': float(np.random.choice([0.7, 0.8, 0.9, 1.0]))
        }
    elif model_type == 'dnn':
        return {
            'layers': [64, 32, 16],
            'activation': 'relu',
            'optimizer': 'adam',
            'learning_rate': float(np.random.choice([0.001, 0.0001])),
            'batch_size': int(np.random.choice([16, 32, 64])),
            'epochs': 100,
            'dropout': float(np.random.choice([0.2, 0.3, 0.5]))
        }
    elif model_type == 'svr':
        return {
            'kernel': 'rbf',
            'C': float(np.random.choice([0.1, 1, 10, 100])),
            'gamma': str(np.random.choice(['scale', 'auto'])),
            'epsilon': float(np.random.choice([0.01, 0.1, 0.2]))
        }
    else:  # anfis
        return {
            'n_mf': int(np.random.choice([2, 3, 5])),
            'mf_type': 'gaussian',
            'epochs': 100,
            'learning_algorithm': 'hybrid'
        }


def generate_ensemble_report(output_dir: Path):
    """Generate ensemble model report."""
    ensemble = {
        'method': 'Weighted Voting',
        'n_models': 5,
        'weights': [0.25, 0.25, 0.20, 0.15, 0.15],
        'base_models': ['rf', 'xgb', 'dnn', 'svr', 'anfis'],
        'metrics': {
            'r2': 0.95,
            'rmse': 0.072,
            'mae': 0.055,
            'mse': 0.0052
        },
        'improvement_over_best': 0.03
    }

    with open(output_dir / 'ensemble_results.json', 'w', encoding='utf-8') as f:
        json.dump(ensemble, f, indent=2)

    print("Generated ensemble report")


def generate_cross_model_report(output_dir: Path):
    """Generate cross-model analysis report."""
    cross_model = {
        'analysis_type': 'Cross-Model Comparison',
        'models_compared': ['rf', 'xgb', 'dnn', 'svr', 'anfis'],
        'targets': ['MM', 'QM', 'Beta_2'],
        'best_by_target': {
            'MM': 'xgb',
            'QM': 'xgb',
            'Beta_2': 'rf'
        },
        'correlation_matrix': {
            'rf_xgb': 0.85,
            'rf_dnn': 0.78,
            'xgb_dnn': 0.82,
            'svr_anfis': 0.73
        },
        'diversity_score': 0.68
    }

    with open(output_dir / 'cross_model_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(cross_model, f, indent=2)

    print("Generated cross-model analysis report")


def generate_excel_reports(output_dir: Path):
    """Generate sample Excel reports."""
    # Performance summary
    models = ['RF', 'XGBoost', 'DNN', 'SVR', 'ANFIS']
    targets = ['MM', 'QM', 'Beta_2']

    data = []
    for model in models:
        for target in targets:
            r2 = 0.85 + np.random.uniform(0, 0.10)
            data.append({
                'Model': model,
                'Target': target,
                'R²': r2,
                'RMSE': (1 - r2) * 0.3,
                'MAE': (1 - r2) * 0.2,
                'Training Time (s)': np.random.uniform(30, 300)
            })

    df = pd.DataFrame(data)

    with pd.ExcelWriter(output_dir / 'performance_summary.xlsx') as writer:
        df.to_excel(writer, sheet_name='Overall Performance', index=False)

        # Best configurations per model
        for model in models:
            model_data = df[df['Model'] == model].sort_values('R²', ascending=False).head(5)
            model_data.to_excel(writer, sheet_name=f'{model} Best', index=False)

    print("Generated Excel performance report")


def generate_sample_visualizations(output_dir: Path):
    """Generate sample visualization plots."""
    output_dir.mkdir(exist_ok=True)

    # Training convergence plot
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = np.arange(1, 101)
    for model in ['RF', 'XGBoost', 'DNN']:
        loss = 1.0 / (1 + 0.05 * epochs) + np.random.normal(0, 0.02, 100)
        ax.plot(epochs, loss, label=model, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_convergence.png', dpi=150)
    plt.close()

    # Performance comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['RF', 'XGBoost', 'DNN', 'SVR', 'ANFIS']
    r2_scores = [0.89, 0.92, 0.88, 0.85, 0.87]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(models, r2_scores, color=colors)
    ax.set_ylabel('R² Score')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0.8, 1.0)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=150)
    plt.close()

    # Ensemble performance
    fig, ax = plt.subplots(figsize=(10, 6))
    models_all = ['RF', 'XGBoost', 'DNN', 'SVR', 'ANFIS', 'Ensemble']
    r2_all = [0.89, 0.92, 0.88, 0.85, 0.87, 0.95]
    colors_all = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2']
    bars = ax.bar(models_all, r2_all, color=colors_all)
    ax.set_ylabel('R² Score')
    ax.set_title('Ensemble vs Individual Models')
    ax.set_ylim(0.8, 1.0)
    ax.grid(True, axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold' if bar.get_x() > 4.5 else 'normal')

    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_performance.png', dpi=150)
    plt.close()

    print(f"Generated 3 sample visualizations")


def main():
    """Generate all sample data."""
    print("\n" + "="*80)
    print("GENERATING SAMPLE DATA FOR PFAZ 10")
    print("="*80 + "\n")

    # Create directories
    reports_dir = Path('reports')
    viz_dir = Path('output/visualizations')

    reports_dir.mkdir(exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    print("1. Generating model reports...")
    generate_sample_model_reports(reports_dir)

    print("\n2. Generating ensemble report...")
    generate_ensemble_report(reports_dir)

    print("\n3. Generating cross-model analysis...")
    generate_cross_model_report(reports_dir)

    print("\n4. Generating Excel reports...")
    generate_excel_reports(reports_dir)

    print("\n5. Generating sample visualizations...")
    generate_sample_visualizations(viz_dir)

    print("\n" + "="*80)
    print("[OK] SAMPLE DATA GENERATION COMPLETE!")
    print("="*80)
    print(f"\nReports directory: {reports_dir} ({len(list(reports_dir.glob('*')))} files)")
    print(f"Visualizations: {viz_dir} ({len(list(viz_dir.glob('*.png')))} files)")
    print("\nYou can now run: python pfaz10_complete_package.py --quick")


if __name__ == '__main__':
    main()
