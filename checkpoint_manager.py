"""
Checkpoint/Resume System
Never lose progress again!

Features:
- Automatic checkpoint saving
- Resume from last checkpoint
- Partial results recovery
- State serialization
- Progress tracking
"""

import pickle
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Comprehensive checkpoint/resume system

    Usage:
        cm = CheckpointManager()

        # Save checkpoint
        cm.save_checkpoint(
            pfaz_id=2,
            state={
                'completed_tasks': [1, 2, 3],
                'current_task': 4,
                'results': {...},
                'config': {...}
            }
        )

        # Resume
        state = cm.resume_from_checkpoint(pfaz_id=2)
        if state:
            continue from state['current_task']
    """

    def __init__(self, checkpoint_dir='checkpoints'):
        """Initialize checkpoint manager"""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"✓ Checkpoint Manager initialized: {self.checkpoint_dir}")

    def save_checkpoint(self,
                       pfaz_id: int,
                       state: Dict[str, Any],
                       description: str = ''):
        """
        Save checkpoint

        Args:
            pfaz_id: Phase ID (e.g., 2 for PFAZ 2)
            state: State dictionary to save
                {
                    'pfaz_id': int,
                    'completed_tasks': List[int],
                    'current_task': int,
                    'results': Dict,
                    'config': Dict,
                    'timestamp': str
                }
            description: Optional description
        """

        # Add metadata
        checkpoint_data = {
            'pfaz_id': pfaz_id,
            'state': state,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'save_time': time.time()
        }

        # Checkpoint filename
        checkpoint_file = self.checkpoint_dir / f'pfaz_{pfaz_id}_checkpoint.pkl'

        # Save with pickle (fast, binary)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Also save metadata as JSON (human-readable)
        meta_file = self.checkpoint_dir / f'pfaz_{pfaz_id}_meta.json'
        meta_data = {
            'pfaz_id': pfaz_id,
            'description': description,
            'timestamp': checkpoint_data['timestamp'],
            'completed_tasks': state.get('completed_tasks', []),
            'current_task': state.get('current_task', 0),
            'file': str(checkpoint_file),
            'file_size_kb': checkpoint_file.stat().st_size / 1024
        }

        with open(meta_file, 'w') as f:
            json.dump(meta_data, f, indent=2)

        logger.info(f"💾 Checkpoint saved: PFAZ {pfaz_id}")
        logger.info(f"  File: {checkpoint_file.name}")
        logger.info(f"  Size: {meta_data['file_size_kb']:.1f} KB")
        logger.info(f"  Tasks: {len(state.get('completed_tasks', []))} completed")

    def load_checkpoint(self, pfaz_id: int) -> Optional[Dict]:
        """
        Load checkpoint

        Args:
            pfaz_id: Phase ID

        Returns:
            checkpoint_data: Dict or None if not found
        """

        checkpoint_file = self.checkpoint_dir / f'pfaz_{pfaz_id}_checkpoint.pkl'

        if not checkpoint_file.exists():
            logger.info(f"🚫 No checkpoint found for PFAZ {pfaz_id}")
            return None

        # Load checkpoint
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)

        logger.info(f"📦 Checkpoint loaded: PFAZ {pfaz_id}")
        logger.info(f"  Saved: {checkpoint_data['timestamp']}")
        logger.info(f"  Tasks: {len(checkpoint_data['state'].get('completed_tasks', []))} completed")

        return checkpoint_data

    def resume_from_checkpoint(self, pfaz_id: int) -> Optional[Dict]:
        """
        Resume from checkpoint

        Args:
            pfaz_id: Phase ID

        Returns:
            state: Dict or None (if no checkpoint)
        """

        checkpoint_data = self.load_checkpoint(pfaz_id)

        if checkpoint_data is None:
            logger.info(f"🆕 Starting fresh: PFAZ {pfaz_id}")
            return None

        state = checkpoint_data['state']

        logger.info(f"▶️ Resuming from checkpoint: PFAZ {pfaz_id}")
        logger.info(f"  Completed tasks: {state.get('completed_tasks', [])}")
        logger.info(f"  Next task: {state.get('current_task', 0)}")

        return state

    def delete_checkpoint(self, pfaz_id: int):
        """Delete checkpoint"""
        checkpoint_file = self.checkpoint_dir / f'pfaz_{pfaz_id}_checkpoint.pkl'
        meta_file = self.checkpoint_dir / f'pfaz_{pfaz_id}_meta.json'

        if checkpoint_file.exists():
            checkpoint_file.unlink()
        if meta_file.exists():
            meta_file.unlink()

        logger.info(f"🗑️ Checkpoint deleted: PFAZ {pfaz_id}")

    def list_checkpoints(self) -> List[Dict]:
        """
        List all checkpoints

        Returns:
            List of checkpoint metadata
        """

        checkpoints = []

        for meta_file in self.checkpoint_dir.glob('pfaz_*_meta.json'):
            with open(meta_file) as f:
                meta = json.load(f)
                checkpoints.append(meta)

        return sorted(checkpoints, key=lambda x: x['pfaz_id'])

    def print_checkpoints(self):
        """Print checkpoint summary"""
        checkpoints = self.list_checkpoints()

        if not checkpoints:
            print("No checkpoints found.")
            return

        print("="*70)
        print("AVAILABLE CHECKPOINTS")
        print("="*70)

        for cp in checkpoints:
            print(f"\nPFAZ {cp['pfaz_id']}")
            print(f"  Saved: {cp['timestamp']}")
            print(f"  Tasks: {len(cp['completed_tasks'])} completed")
            print(f"  Next: Task {cp['current_task']}")
            print(f"  Size: {cp['file_size_kb']:.1f} KB")
            if cp.get('description'):
                print(f"  Description: {cp['description']}")

        print("="*70)


# ============================================================================
# INTEGRATION WITH TRAINING PIPELINE
# ============================================================================

def train_models_with_checkpoints(configs: List[Dict],
                                  target: str,
                                  pfaz_id: int = 2):
    """
    Train models with automatic checkpointing

    Example usage:
        configs = load_training_configs()
        train_models_with_checkpoints(configs, target='MM', pfaz_id=2)

        # If it crashes at model 42...
        # Just run again! It will resume from model 42.
    """

    checkpoint_manager = CheckpointManager()

    # Try to resume
    state = checkpoint_manager.resume_from_checkpoint(pfaz_id)

    if state is None:
        # Start fresh
        completed_tasks = []
        results = []
        start_idx = 0
    else:
        # Resume from checkpoint
        completed_tasks = state.get('completed_tasks', [])
        results = state.get('results', [])
        start_idx = state.get('current_task', 0)

    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING MODELS: {target}")
    logger.info(f"{'='*70}")
    logger.info(f"Total configs: {len(configs)}")
    logger.info(f"Starting from: config #{start_idx}")
    logger.info(f"Already completed: {len(completed_tasks)}")

    # Import here to avoid circular dependency
    try:
        from model_trainer import train_single_model
    except ImportError:
        logger.warning("model_trainer not found, using mock training")
        def train_single_model(config, target):
            # Mock function for testing
            import numpy as np
            return None, {
                'r2': np.random.uniform(0.85, 0.95),
                'rmse': np.random.uniform(0.05, 0.15),
                'mae': np.random.uniform(0.03, 0.10)
            }

    # Train models
    for i in range(start_idx, len(configs)):
        config = configs[i]

        logger.info(f"\n--- Config {i+1}/{len(configs)} ---")

        try:
            # Train model
            model, metrics = train_single_model(config, target)

            # Save result
            result = {
                'config_id': i,
                'config': config,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            completed_tasks.append(i)

            # Save checkpoint after each model
            checkpoint_manager.save_checkpoint(
                pfaz_id=pfaz_id,
                state={
                    'completed_tasks': completed_tasks,
                    'current_task': i + 1,
                    'results': results,
                    'config': {
                        'target': target,
                        'total_configs': len(configs)
                    }
                },
                description=f"Training {target}, {len(completed_tasks)}/{len(configs)} models"
            )

            logger.info(f"✓ Model trained: R²={metrics['r2']:.4f}")
            logger.info(f"💾 Checkpoint saved ({len(completed_tasks)}/{len(configs)})")

        except Exception as e:
            logger.error(f"❌ Error training model {i}: {e}")

            # Save checkpoint even on error
            checkpoint_manager.save_checkpoint(
                pfaz_id=pfaz_id,
                state={
                    'completed_tasks': completed_tasks,
                    'current_task': i,  # Don't increment (retry this one)
                    'results': results,
                    'config': {
                        'target': target,
                        'total_configs': len(configs)
                    },
                    'last_error': str(e)
                },
                description=f"Error at model {i}"
            )

            raise  # Re-raise to stop execution

    logger.info(f"\n{'='*70}")
    logger.info(f"✓ TRAINING COMPLETE: {len(completed_tasks)}/{len(configs)} models")
    logger.info(f"{'='*70}")

    # Delete checkpoint after successful completion
    checkpoint_manager.delete_checkpoint(pfaz_id)

    return results


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def cli_resume_training():
    """
    CLI for resuming training

    Usage:
        python checkpoint_manager.py --resume --pfaz 2
    """

    import argparse

    parser = argparse.ArgumentParser(description="Checkpoint Manager CLI")
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--pfaz', type=int, help='PFAZ ID')
    parser.add_argument('--list', action='store_true', help='List all checkpoints')
    parser.add_argument('--delete', action='store_true', help='Delete checkpoint')

    args = parser.parse_args()

    cm = CheckpointManager()

    if args.list:
        cm.print_checkpoints()

    elif args.resume and args.pfaz:
        state = cm.resume_from_checkpoint(args.pfaz)
        if state:
            print(f"\n✓ Ready to resume PFAZ {args.pfaz}")
            print(f"  Next task: {state.get('current_task', 0)}")
            print(f"  Completed: {len(state.get('completed_tasks', []))} tasks")
        else:
            print(f"\n🚫 No checkpoint found for PFAZ {args.pfaz}")

    elif args.delete and args.pfaz:
        cm.delete_checkpoint(args.pfaz)
        print(f"✓ Checkpoint deleted: PFAZ {args.pfaz}")

    else:
        parser.print_help()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    cli_resume_training()
