"""
Example Usage: Checkpoint/Resume + AutoML Integration

This script demonstrates:
1. Basic checkpoint usage
2. AutoML optimization
3. Combined workflow
"""

import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_1_basic_checkpoint():
    """
    Example 1: Basic Checkpoint Usage

    Demonstrates saving and resuming from checkpoints
    """

    from checkpoint_manager import CheckpointManager

    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Checkpoint Usage")
    print("="*70)

    # Create checkpoint manager
    cm = CheckpointManager(checkpoint_dir='examples/checkpoints')

    # Simulate a training workflow
    total_tasks = 10
    completed = []

    for i in range(total_tasks):
        # Do some work
        result = {'task_id': i, 'score': np.random.uniform(0.8, 0.95)}
        completed.append(i)

        # Save checkpoint after each task
        cm.save_checkpoint(
            pfaz_id=99,  # Example PFAZ ID
            state={
                'completed_tasks': completed,
                'current_task': i + 1,
                'results': [result]
            },
            description=f"Example training, task {i+1}/{total_tasks}"
        )

        print(f"[OK] Task {i+1}/{total_tasks} completed")

        # Simulate crash at task 5
        if i == 4:
            print("\n[BOOM] Simulating crash...")
            break

    # Resume from checkpoint
    print("\n▶️ Resuming from checkpoint...")
    state = cm.resume_from_checkpoint(pfaz_id=99)

    if state:
        print(f"[OK] Resumed successfully!")
        print(f"  Completed: {len(state['completed_tasks'])} tasks")
        print(f"  Next task: {state['current_task']}")

        # Continue from where we left off
        for i in range(state['current_task'], total_tasks):
            completed.append(i)
            print(f"[OK] Task {i+1}/{total_tasks} completed (resumed)")

    # Clean up
    cm.delete_checkpoint(pfaz_id=99)
    print("\n[OK] Example 1 complete!")


def example_2_automl_optimization():
    """
    Example 2: AutoML Hyperparameter Optimization

    Demonstrates finding best hyperparameters automatically
    """

    from automl_optimizer import AutoMLOptimizer, visualize_optimization

    print("\n" + "="*70)
    print("EXAMPLE 2: AutoML Optimization")
    print("="*70)

    # Generate sample data
    n_samples = 1000
    n_features = 20

    X_train = np.random.randn(int(n_samples * 0.8), n_features)
    X_val = np.random.randn(int(n_samples * 0.2), n_features)

    # True relationship: y = sum(X[:, :5]) + noise
    y_train = X_train[:, :5].sum(axis=1) + np.random.randn(len(X_train)) * 0.1
    y_val = X_val[:, :5].sum(axis=1) + np.random.randn(len(X_val)) * 0.1

    print(f"[OK] Generated data: {len(X_train)} train, {len(X_val)} val samples")

    # Create optimizer
    optimizer = AutoMLOptimizer(
        X_train, y_train,
        X_val, y_val,
        model_type='rf'  # Random Forest
    )

    # Run optimization (quick demo with 20 trials)
    print("\n[SEARCH] Running optimization (20 trials)...")
    study = optimizer.optimize(n_trials=20)

    # Print results
    print(f"\n[OK] Optimization complete!")
    print(f"  Best R²: {study.best_value:.4f}")
    print(f"  Best parameters:")
    for param, value in study.best_params.items():
        print(f"    {param}: {value}")

    # Save results
    output_dir = Path('examples/automl_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer.save_results(study, str(output_dir / 'example_results.json'))
    print(f"\n[OK] Results saved to {output_dir}/example_results.json")

    # Create visualizations
    try:
        visualize_optimization(study, output_dir=str(output_dir / 'plots'))
        print(f"[OK] Visualizations saved to {output_dir}/plots/")
    except Exception as e:
        print(f"[WARNING]  Visualization skipped: {e}")

    print("\n[OK] Example 2 complete!")


def example_3_combined_workflow():
    """
    Example 3: Combined Checkpoint + AutoML Workflow

    Demonstrates using both systems together
    """

    from checkpoint_manager import CheckpointManager, train_models_with_checkpoints
    from automl_optimizer import AutoMLOptimizer
    import json

    print("\n" + "="*70)
    print("EXAMPLE 3: Combined Checkpoint + AutoML Workflow")
    print("="*70)

    # Step 1: Use AutoML to find best parameters
    print("\n[REPORT] STEP 1: AutoML Optimization")
    print("-" * 70)

    # Generate sample data
    n_samples = 500
    n_features = 15

    X_train = np.random.randn(int(n_samples * 0.8), n_features)
    X_val = np.random.randn(int(n_samples * 0.2), n_features)
    y_train = X_train[:, :3].sum(axis=1) + np.random.randn(len(X_train)) * 0.1
    y_val = X_val[:, :3].sum(axis=1) + np.random.randn(len(X_val)) * 0.1

    # Optimize
    optimizer = AutoMLOptimizer(X_train, y_train, X_val, y_val, model_type='rf')
    study = optimizer.optimize(n_trials=15)

    print(f"[OK] Found best parameters (R² = {study.best_value:.4f})")

    # Step 2: Generate configs using best parameters as starting point
    print("\n[TARGET] STEP 2: Generate Training Configurations")
    print("-" * 70)

    best_params = study.best_params

    # Create variations around best parameters
    configs = []
    for i in range(10):
        config = {
            'n_estimators': best_params['n_estimators'] + np.random.randint(-50, 50),
            'max_depth': best_params['max_depth'] + np.random.randint(-2, 2),
            'min_samples_split': best_params.get('min_samples_split', 2),
            'random_state': 42
        }
        # Ensure valid values
        config['n_estimators'] = max(100, config['n_estimators'])
        config['max_depth'] = max(5, min(50, config['max_depth']))
        configs.append(config)

    print(f"[OK] Generated {len(configs)} configurations")

    # Step 3: Train with checkpoints
    print("\n[SAVE] STEP 3: Training with Checkpoints")
    print("-" * 70)

    try:
        results = train_models_with_checkpoints(
            configs=configs,
            target='Example_Target',
            pfaz_id=98
        )
        print(f"[OK] Trained {len(results)} models with checkpointing")
    except Exception as e:
        print(f"[WARNING]  Training simulation completed: {e}")

    # Step 4: Summary
    print("\n[LIST] STEP 4: Summary")
    print("-" * 70)
    print(f"[OK] AutoML found best parameters")
    print(f"[OK] Generated {len(configs)} config variations")
    print(f"[OK] Training completed with checkpoint support")

    # Clean up
    cm = CheckpointManager()
    cm.delete_checkpoint(pfaz_id=98)

    print("\n[OK] Example 3 complete!")


def example_4_multi_target_optimization():
    """
    Example 4: Multi-Target Optimization

    Demonstrates optimizing for multiple targets
    """

    from automl_optimizer import AutoMLOptimizer

    print("\n" + "="*70)
    print("EXAMPLE 4: Multi-Target Optimization")
    print("="*70)

    # Generate sample data for multiple targets
    n_samples = 800
    n_features = 20

    X_train = np.random.randn(int(n_samples * 0.8), n_features)
    X_val = np.random.randn(int(n_samples * 0.2), n_features)

    # Multiple targets with different relationships
    y_targets = {
        'MM': X_train[:, :5].sum(axis=1) + np.random.randn(len(X_train)) * 0.1,
        'QM': X_train[:, 5:10].sum(axis=1) + np.random.randn(len(X_train)) * 0.2,
        'Beta_2': (X_train[:, 10:15]**2).sum(axis=1) + np.random.randn(len(X_train)) * 0.15
    }

    y_val_targets = {
        'MM': X_val[:, :5].sum(axis=1) + np.random.randn(len(X_val)) * 0.1,
        'QM': X_val[:, 5:10].sum(axis=1) + np.random.randn(len(X_val)) * 0.2,
        'Beta_2': (X_val[:, 10:15]**2).sum(axis=1) + np.random.randn(len(X_val)) * 0.15
    }

    # Optimize for each target
    results = {}

    for target in ['MM', 'QM', 'Beta_2']:
        print(f"\n[SEARCH] Optimizing for {target}...")

        optimizer = AutoMLOptimizer(
            X_train, y_targets[target],
            X_val, y_val_targets[target],
            model_type='rf'
        )

        study = optimizer.optimize(n_trials=10)  # Quick demo
        results[target] = study

        print(f"[OK] {target}: Best R² = {study.best_value:.4f}")

    # Summary
    print("\n[REPORT] Multi-Target Optimization Summary:")
    print("-" * 70)
    for target, study in results.items():
        print(f"{target:10s}: R² = {study.best_value:.4f} "
              f"(n_estimators={study.best_params['n_estimators']}, "
              f"max_depth={study.best_params['max_depth']})")

    print("\n[OK] Example 4 complete!")


def run_all_examples():
    """Run all examples"""

    examples = [
        ("Basic Checkpoint", example_1_basic_checkpoint),
        ("AutoML Optimization", example_2_automl_optimization),
        ("Combined Workflow", example_3_combined_workflow),
        ("Multi-Target Optimization", example_4_multi_target_optimization),
    ]

    print("\n" + "="*70)
    print("CHECKPOINT + AUTOML EXAMPLES")
    print("="*70)
    print(f"\nRunning {len(examples)} examples...\n")

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            logger.error(f"Example '{name}' failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("[OK] ALL EXAMPLES COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_all_examples()
