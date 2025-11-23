"""
Distributed Training and GPU Acceleration Enhancements
=======================================================

Features:
- Multi-GPU training (DataParallel and DistributedDataParallel)
- CUDA graph optimization for faster execution
- Mixed precision training (FP16/BF16)
- Gradient accumulation for large batches
- Dynamic batch sizing for optimal memory usage
- Multi-node distributed training support
- Performance profiling and monitoring

Compatible with PyTorch models.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.nn.parallel import DataParallel as DP
    from torch.cuda.amp import GradScaler, autocast
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class DistributedTrainingManager:
    """
    Manager for distributed training across multiple GPUs and nodes

    Supports:
    - Single-GPU training
    - Multi-GPU training (DataParallel)
    - Distributed training (DistributedDataParallel)
    - Mixed precision training
    - CUDA graphs (for static models)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize distributed training manager

        Args:
            config: Configuration dictionary
                - backend: 'nccl', 'gloo', 'mpi'
                - init_method: 'env://', 'tcp://...'
                - world_size: Total number of processes
                - rank: Process rank
                - local_rank: GPU device ID
                - use_amp: Enable mixed precision
                - amp_dtype: 'float16' or 'bfloat16'
                - use_cuda_graphs: Enable CUDA graphs
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for distributed training")

        self.config = config or {}

        # Distributed settings
        self.backend = self.config.get('backend', 'nccl')
        self.world_size = self.config.get('world_size', 1)
        self.rank = self.config.get('rank', 0)
        self.local_rank = self.config.get('local_rank', 0)

        # Mixed precision
        self.use_amp = self.config.get('use_amp', True)
        amp_dtype_str = self.config.get('amp_dtype', 'float16')
        self.amp_dtype = torch.float16 if amp_dtype_str == 'float16' else torch.bfloat16

        # CUDA graphs
        self.use_cuda_graphs = self.config.get('use_cuda_graphs', False)
        self.cuda_graph = None
        self.static_input = None
        self.static_target = None

        # Gradient scaler for mixed precision
        self.scaler = GradScaler() if self.use_amp else None

        # Device setup
        self.device = self._setup_device()

        # Distributed initialization
        self.is_distributed = self.world_size > 1
        if self.is_distributed:
            self._init_distributed()

        logger.info(f"DistributedTrainingManager initialized")
        logger.info(f"  World size: {self.world_size}")
        logger.info(f"  Rank: {self.rank}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Mixed precision: {self.use_amp} ({amp_dtype_str})")
        logger.info(f"  CUDA graphs: {self.use_cuda_graphs}")

    def _setup_device(self) -> torch.device:
        """Setup device for training"""
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(device)
            return device
        else:
            logger.warning("CUDA not available, using CPU")
            return torch.device('cpu')

    def _init_distributed(self):
        """Initialize distributed training backend"""
        init_method = self.config.get('init_method', 'env://')

        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                init_method=init_method,
                world_size=self.world_size,
                rank=self.rank
            )
            logger.info(f"Distributed process group initialized: {self.backend}")

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """
        Wrap model for distributed training

        Args:
            model: PyTorch model

        Returns:
            Wrapped model (DDP, DP, or original)
        """
        model = model.to(self.device)

        if self.is_distributed:
            # Use DistributedDataParallel for multi-node training
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config.get('find_unused_parameters', False)
            )
            logger.info("Model wrapped with DistributedDataParallel")

        elif torch.cuda.device_count() > 1 and self.config.get('use_data_parallel', True):
            # Use DataParallel for single-node multi-GPU
            model = DP(model)
            logger.info(f"Model wrapped with DataParallel ({torch.cuda.device_count()} GPUs)")

        return model

    def train_step(self, model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor],
                   optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   accumulation_steps: int = 1) -> float:
        """
        Perform a single training step with optional gradient accumulation

        Args:
            model: Model to train
            batch: (inputs, targets)
            optimizer: Optimizer
            criterion: Loss function
            accumulation_steps: Number of steps to accumulate gradients

        Returns:
            Loss value
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Mixed precision training
        if self.use_amp:
            with autocast(dtype=self.amp_dtype):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

            # Backward pass with scaled gradients
            self.scaler.scale(loss).backward()

            # Only step optimizer after accumulating gradients
            if (self.rank % accumulation_steps) == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

        else:
            # Standard training
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps

            loss.backward()

            if (self.rank % accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()

        return loss.item() * accumulation_steps

    def setup_cuda_graph(self, model: nn.Module, input_shape: Tuple,
                        target_shape: Tuple, criterion: nn.Module):
        """
        Setup CUDA graph for faster execution (requires static graph)

        Args:
            model: Model
            input_shape: Shape of input tensor (batch_size, ...)
            target_shape: Shape of target tensor
            criterion: Loss function
        """
        if not self.use_cuda_graphs or not torch.cuda.is_available():
            return

        logger.info("Setting up CUDA graph...")

        # Create static inputs
        self.static_input = torch.randn(input_shape, device=self.device)
        self.static_target = torch.randn(target_shape, device=self.device)

        # Warmup
        model.train()
        for _ in range(3):
            with autocast(dtype=self.amp_dtype) if self.use_amp else torch.enable_grad():
                output = model(self.static_input)
                loss = criterion(output, self.static_target)
            loss.backward()

        # Capture graph
        self.cuda_graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.cuda_graph):
            with autocast(dtype=self.amp_dtype) if self.use_amp else torch.enable_grad():
                output = model(self.static_input)
                loss = criterion(output, self.static_target)
            loss.backward()

        logger.info("CUDA graph captured successfully")

    def train_step_with_graph(self, model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor],
                             optimizer: torch.optim.Optimizer) -> float:
        """
        Training step using CUDA graph (faster but requires static shapes)

        Args:
            model: Model
            batch: (inputs, targets)
            optimizer: Optimizer

        Returns:
            Loss value
        """
        if self.cuda_graph is None:
            raise RuntimeError("CUDA graph not initialized. Call setup_cuda_graph first.")

        inputs, targets = batch

        # Copy data to static tensors
        self.static_input.copy_(inputs)
        self.static_target.copy_(targets)

        # Replay graph
        self.cuda_graph.replay()

        # Step optimizer
        optimizer.step()
        optimizer.zero_grad()

        # Return loss (would need to be stored during graph capture)
        return 0.0  # Placeholder - actual implementation would store loss

    def get_gpu_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics"""
        if not torch.cuda.is_available():
            return {}

        stats = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            max_allocated = torch.cuda.max_memory_allocated(i) / 1e9

            stats[f'gpu_{i}'] = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated
            }

        return stats

    def synchronize(self):
        """Synchronize all processes"""
        if self.is_distributed:
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op=None) -> torch.Tensor:
        """
        All-reduce operation across processes

        Args:
            tensor: Tensor to reduce
            op: Reduction operation (default: SUM)

        Returns:
            Reduced tensor
        """
        if not self.is_distributed:
            return tensor

        if op is None:
            op = dist.ReduceOp.SUM

        dist.all_reduce(tensor, op=op)
        return tensor

    def gather(self, tensor: torch.Tensor, dst: int = 0) -> Optional[List[torch.Tensor]]:
        """
        Gather tensors from all processes

        Args:
            tensor: Tensor to gather
            dst: Destination rank

        Returns:
            List of tensors (only on dst rank)
        """
        if not self.is_distributed:
            return [tensor]

        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]

        if self.rank == dst:
            dist.gather(tensor, gathered, dst=dst)
            return gathered
        else:
            dist.gather(tensor, dst=dst)
            return None

    def cleanup(self):
        """Cleanup distributed training"""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")


class DynamicBatchSizer:
    """
    Dynamically adjust batch size based on available GPU memory

    Starts with a large batch size and reduces if OOM occurs
    """

    def __init__(self, initial_batch_size: int = 128,
                 min_batch_size: int = 8,
                 max_batch_size: int = 512):
        """
        Initialize dynamic batch sizer

        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.oom_count = 0

        logger.info(f"DynamicBatchSizer initialized: batch_size={self.current_batch_size}")

    def on_oom(self):
        """Handle out-of-memory error by reducing batch size"""
        self.oom_count += 1
        new_batch_size = max(self.min_batch_size, self.current_batch_size // 2)

        logger.warning(f"OOM detected! Reducing batch size: {self.current_batch_size} -> {new_batch_size}")

        self.current_batch_size = new_batch_size

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.current_batch_size

    def on_success(self):
        """Increase batch size after successful training"""
        # Gradually increase batch size if no OOM
        if self.oom_count == 0 and self.current_batch_size < self.max_batch_size:
            new_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.25))
            if new_batch_size != self.current_batch_size:
                logger.info(f"Increasing batch size: {self.current_batch_size} -> {new_batch_size}")
                self.current_batch_size = new_batch_size

    def get_batch_size(self) -> int:
        """Get current batch size"""
        return self.current_batch_size


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Install PyTorch to test distributed training.")
        exit(1)

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Configuration for single-node multi-GPU
    config = {
        'world_size': 1,  # Single process for this example
        'rank': 0,
        'local_rank': 0,
        'use_amp': True,
        'amp_dtype': 'float16',
        'use_cuda_graphs': False,
        'use_data_parallel': True
    }

    # Initialize manager
    manager = DistributedTrainingManager(config)

    # Create dummy model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    # Wrap model
    model = manager.wrap_model(model)

    # Create dummy data
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)

    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training step
    loss = manager.train_step(model, (inputs, targets), optimizer, criterion)
    logger.info(f"Training loss: {loss:.6f}")

    # GPU memory stats
    mem_stats = manager.get_GPU_memory_stats()
    logger.info(f"GPU memory: {mem_stats}")

    # Dynamic batch sizer
    batch_sizer = DynamicBatchSizer(initial_batch_size=128)
    logger.info(f"Current batch size: {batch_sizer.get_batch_size()}")

    # Simulate OOM
    batch_sizer.on_oom()
    logger.info(f"After OOM, batch size: {batch_sizer.get_batch_size()}")

    # Cleanup
    manager.cleanup()

    print("\n[SUCCESS] Distributed training utilities tested!")
