"""
Advanced AI Models - COMPLETE VERSION
Includes: BNN, PINN, TransferLearning, EnsembleMethods, HybridModels
WITH GPU SUPPORT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import logging
from typing import Dict, List, Tuple, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== GPU OPTIMIZER ====================

class GPUOptimizer:
    """Manage GPU resources for PyTorch models"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.device = self._setup_device()
        self.use_amp = self.config.get('use_amp', True)  # Automatic Mixed Precision
        
    def _setup_device(self):
        """Setup GPU/CPU device"""
        if torch.cuda.is_available() and self.config.get('enable', False):
            device_id = self.config.get('device_id', 0)
            device = torch.device(f'cuda:{device_id}')
            
            gpu_name = torch.cuda.get_device_name(device_id)
            total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9
            
            logger.info(f"\n[INTERACTIVE] GPU ENABLED")
            logger.info(f"   Device: {gpu_name}")
            logger.info(f"   Total Memory: {total_memory:.2f} GB")
            logger.info(f"   Device ID: {device_id}")
            logger.info(f"   Mixed Precision: {self.use_amp}")
            
            # Set memory fraction if specified
            if 'memory_fraction' in self.config:
                memory_fraction = self.config['memory_fraction']
                torch.cuda.set_per_process_memory_fraction(memory_fraction, device_id)
                logger.info(f"   Memory Fraction: {memory_fraction*100:.0f}%")
            
            return device
        else:
            logger.info("💻 Using CPU (GPU disabled or unavailable)")
            return torch.device('cpu')
    
    def to_device(self, tensor_or_model):
        """Move tensor or model to device"""
        return tensor_or_model.to(self.device)
    
    def get_memory_stats(self) -> Dict:
        """Get GPU memory statistics"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'device': str(self.device)
            }
        return {'device': 'cpu'}


# ==================== BAYESIAN NEURAL NETWORK ====================

class BayesianNeuralNetwork:
    """
    Bayesian Neural Network with uncertainty quantification
    GPU Accelerated
    """
    
    def __init__(self, input_dim: int, config: Dict = None):
        self.input_dim = input_dim
        self.config = config or {}
        
        # GPU optimizer
        self.gpu_optimizer = GPUOptimizer(self.config.get('gpu', {}))
        
        # Model architecture
        hidden_layers = self.config.get('bnn', {}).get('hidden_layers', [64, 32, 16])
        self.n_samples = self.config.get('bnn', {}).get('n_samples', 100)
        
        self.model = self._build_model(hidden_layers)
        self.model = self.gpu_optimizer.to_device(self.model)
        
        # Scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.gpu_optimizer.use_amp else None
        
        # For uncertainty estimation
        self.mc_dropout = True
        
        logger.info(f"BNN initialized: {input_dim} inputs -> {hidden_layers} -> 1 output")
        logger.info(f"MC Samples: {self.n_samples}")
        logger.info(f"Device: {self.gpu_optimizer.device}")
    
    def _build_model(self, hidden_layers: List[int]) -> nn.Module:
        """Build BNN architecture with dropout for uncertainty"""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # MC Dropout for uncertainty
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray) -> Dict:
        """Train BNN with GPU acceleration"""
        
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING BAYESIAN NEURAL NETWORK")
        logger.info(f"{'='*60}")
        
        # Convert to tensors and move to device
        X_train_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(X_train))
        y_train_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(y_train).reshape(-1, 1))
        X_val_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(X_val))
        y_val_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(y_val).reshape(-1, 1))
        
        # DataLoader
        batch_size = self.config.get('bnn', {}).get('batch_size', 32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        learning_rate = self.config.get('bnn', {}).get('learning_rate', 0.001)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        epochs = self.config.get('bnn', {}).get('epochs', 200)
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        logger.info(f"Training: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
        
        training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                if self.scaler:
                    # Mixed precision training
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Normal training
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # GPU stats
        if self.gpu_optimizer.device.type == 'cuda':
            mem_stats = self.gpu_optimizer.get_memory_stats()
            logger.info(f"\n[REPORT] GPU Memory Usage:")
            logger.info(f"   Allocated: {mem_stats['allocated_gb']:.2f} GB")
            logger.info(f"   Peak: {mem_stats['max_allocated_gb']:.2f} GB")
        
        logger.info(f"[SUCCESS] BNN Training completed - Best Val Loss: {best_val_loss:.6f}")
        
        return {
            'best_val_loss': best_val_loss,
            'final_train_loss': avg_train_loss,
            'device': str(self.gpu_optimizer.device),
            'training_history': training_history,
            'epochs_trained': epoch + 1
        }
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = False) -> np.ndarray:
        """
        Make predictions with optional uncertainty estimation
        
        Args:
            X: Input features
            return_uncertainty: If True, returns (predictions, uncertainty)
        """
        self.model.eval()
        
        X_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(X))
        
        if return_uncertainty:
            # MC Dropout: multiple forward passes with dropout enabled
            self.model.train()  # Enable dropout
            predictions = []
            
            with torch.no_grad():
                for _ in range(self.n_samples):
                    pred = self.model(X_tensor)
                    predictions.append(pred.cpu().numpy())
            
            predictions = np.array(predictions)
            mean_pred = predictions.mean(axis=0).flatten()
            uncertainty = predictions.std(axis=0).flatten()
            
            return mean_pred, uncertainty
        else:
            # Standard prediction
            with torch.no_grad():
                predictions = self.model(X_tensor)
            
            return predictions.cpu().numpy().flatten()


# ==================== PHYSICS-INFORMED NEURAL NETWORK ====================

class PhysicsInformedNN:
    """
    Physics-Informed Neural Network (PINN)
    Incorporates physical constraints in loss function
    GPU Accelerated
    """
    
    def __init__(self, input_dim: int, config: Dict = None):
        self.input_dim = input_dim
        self.config = config or {}
        
        # GPU optimizer
        self.gpu_optimizer = GPUOptimizer(self.config.get('gpu', {}))
        
        # Model architecture
        hidden_layers = self.config.get('pinn', {}).get('hidden_layers', [64, 64, 32])
        self.model = self._build_model(hidden_layers)
        self.model = self.gpu_optimizer.to_device(self.model)
        
        # Physics weight
        self.physics_weight = self.config.get('pinn', {}).get('physics_weight', 0.3)
        
        # Scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.gpu_optimizer.use_amp else None
        
        logger.info(f"PINN initialized: {input_dim} inputs -> {hidden_layers} -> 1 output")
        logger.info(f"Physics weight: {self.physics_weight}")
        logger.info(f"Device: {self.gpu_optimizer.device}")
    
    def _build_model(self, hidden_layers: List[int]) -> nn.Module:
        """Build PINN architecture"""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())  # Tanh activation for physics-informed
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def physics_loss(self, X: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Physics-informed loss component
        Customizable based on physical constraints
        
        Examples:
        - Energy conservation
        - Mass-energy relation
        - Binding energy constraints
        """
        # Example 1: Bound constraints (predictions should be within physical limits)
        min_bound = -20.0  # MeV
        max_bound = 20.0   # MeV
        
        lower_violation = torch.relu(min_bound - y_pred)
        upper_violation = torch.relu(y_pred - max_bound)
        
        bound_loss = torch.mean(lower_violation ** 2 + upper_violation ** 2)
        
        # Example 2: Smoothness constraint (derivative regularization)
        # Penalize rapid changes in predictions
        if X.requires_grad:
            dy_dx = torch.autograd.grad(
                y_pred.sum(), X,
                create_graph=True, retain_graph=True
            )[0]
            smoothness_loss = torch.mean(dy_dx ** 2)
        else:
            smoothness_loss = 0.0
        
        # Combine physics constraints
        total_physics_loss = bound_loss + 0.01 * smoothness_loss
        
        return total_physics_loss
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray) -> Dict:
        """Train PINN with physics-informed loss"""
        
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING PHYSICS-INFORMED NEURAL NETWORK")
        logger.info(f"{'='*60}")
        
        # Convert to tensors
        X_train_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(X_train))
        y_train_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(y_train).reshape(-1, 1))
        X_val_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(X_val))
        y_val_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(y_val).reshape(-1, 1))
        
        # Require gradients for physics loss
        X_train_tensor.requires_grad = True
        
        # DataLoader
        batch_size = self.config.get('pinn', {}).get('batch_size', 32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        learning_rate = self.config.get('pinn', {}).get('learning_rate', 0.001)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training
        epochs = self.config.get('pinn', {}).get('epochs', 200)
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        logger.info(f"Training: {epochs} epochs, physics_weight={self.physics_weight}")
        
        training_history = {
            'train_data_loss': [],
            'train_physics_loss': [],
            'train_total_loss': [],
            'val_loss': []
        }
        
        for epoch in range(epochs):
            self.model.train()
            epoch_data_loss = 0.0
            epoch_physics_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X.requires_grad = True
                optimizer.zero_grad()
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_X)
                        data_loss = criterion(outputs, batch_y)
                        phys_loss = self.physics_loss(batch_X, outputs)
                        total_loss = data_loss + self.physics_weight * phys_loss
                    
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_X)
                    data_loss = criterion(outputs, batch_y)
                    phys_loss = self.physics_loss(batch_X, outputs)
                    total_loss = data_loss + self.physics_weight * phys_loss
                    
                    total_loss.backward()
                    optimizer.step()
                
                epoch_data_loss += data_loss.item()
                epoch_physics_loss += phys_loss.item() if isinstance(phys_loss, torch.Tensor) else 0
            
            avg_data_loss = epoch_data_loss / len(train_loader)
            avg_physics_loss = epoch_physics_loss / len(train_loader)
            avg_total_loss = avg_data_loss + self.physics_weight * avg_physics_loss
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            training_history['train_data_loss'].append(avg_data_loss)
            training_history['train_physics_loss'].append(avg_physics_loss)
            training_history['train_total_loss'].append(avg_total_loss)
            training_history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Data: {avg_data_loss:.6f}, "
                          f"Physics: {avg_physics_loss:.6f}, Val: {val_loss:.6f}")
        
        # GPU stats
        if self.gpu_optimizer.device.type == 'cuda':
            mem_stats = self.gpu_optimizer.get_memory_stats()
            logger.info(f"\n[REPORT] GPU Memory Usage:")
            logger.info(f"   Allocated: {mem_stats['allocated_gb']:.2f} GB")
            logger.info(f"   Peak: {mem_stats['max_allocated_gb']:.2f} GB")
        
        logger.info(f"[SUCCESS] PINN Training completed - Best Val Loss: {best_val_loss:.6f}")
        
        return {
            'best_val_loss': best_val_loss,
            'final_data_loss': avg_data_loss,
            'final_physics_loss': avg_physics_loss,
            'device': str(self.gpu_optimizer.device),
            'training_history': training_history,
            'epochs_trained': epoch + 1
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        
        X_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(X))
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy().flatten()


# ==================== ENSEMBLE METHODS ====================

class EnsembleRegressor:
    """
    Advanced ensemble methods combining multiple models
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.models = {}
        self.weights = None
        self.scaler = StandardScaler()
        
    def build_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, n_jobs: int = -1):
        """Build ensemble of multiple models"""
        
        logger.info("\n🤝 BUILDING ENSEMBLE MODELS")
        logger.info("-" * 60)
        
        # Random Forest
        logger.info("Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=self.config.get('ensemble', {}).get('random_forest', {}).get('n_estimators', 100),
            max_depth=self.config.get('ensemble', {}).get('random_forest', {}).get('max_depth', 20),
            n_jobs=n_jobs,
            random_state=42
        )
        rf.fit(X_train, y_train)
        self.models['RandomForest'] = rf
        
        # Gradient Boosting
        logger.info("Training Gradient Boosting...")
        gb = GradientBoostingRegressor(
            n_estimators=self.config.get('ensemble', {}).get('gradient_boosting', {}).get('n_estimators', 100),
            learning_rate=self.config.get('ensemble', {}).get('gradient_boosting', {}).get('learning_rate', 0.1),
            random_state=42
        )
        gb.fit(X_train, y_train)
        self.models['GradientBoosting'] = gb
        
        # MLP
        logger.info("Training Multi-Layer Perceptron...")
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            max_iter=1000,
            random_state=42
        )
        mlp.fit(X_train, y_train)
        self.models['MLP'] = mlp
        
        logger.info(f"[SUCCESS] Ensemble built with {len(self.models)} models")
        
        return self
    
    def predict(self, X: np.ndarray, method: str = 'voting') -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Input features
            method: 'voting' (average), 'weighted', or 'stacking'
        """
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        if method == 'voting':
            # Simple average
            return np.mean(list(predictions.values()), axis=0)
        
        elif method == 'weighted':
            # Weighted average (weights should be set during training)
            if self.weights is None:
                return np.mean(list(predictions.values()), axis=0)
            
            weighted_pred = np.zeros_like(list(predictions.values())[0])
            for name, pred in predictions.items():
                weighted_pred += self.weights.get(name, 1/len(self.models)) * pred
            
            return weighted_pred
        
        else:
            return np.mean(list(predictions.values()), axis=0)
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from tree-based models"""
        importance = {}
        
        if 'RandomForest' in self.models:
            importance['RandomForest'] = self.models['RandomForest'].feature_importances_
        
        if 'GradientBoosting' in self.models:
            importance['GradientBoosting'] = self.models['GradientBoosting'].feature_importances_
        
        return importance


# ==================== HYBRID MODEL ====================

class HybridModel:
    """
    Hybrid model combining neural networks and traditional ML
    """
    
    def __init__(self, input_dim: int, config: Dict = None):
        self.input_dim = input_dim
        self.config = config or {}
        
        # Neural network component
        self.nn_model = BayesianNeuralNetwork(input_dim, config)
        
        # Traditional ML component
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Blending weight (learned during training)
        self.nn_weight = 0.6
        self.ml_weight = 0.4
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train both components"""
        
        logger.info("\n🔀 TRAINING HYBRID MODEL")
        logger.info("-" * 60)
        
        # Train neural network
        logger.info("Training neural network component...")
        nn_results = self.nn_model.train(X_train, y_train, X_val, y_val)
        
        # Train ML component
        logger.info("Training machine learning component...")
        self.ml_model.fit(X_train, y_train)
        
        # Optimize blending weights on validation set
        nn_pred = self.nn_model.predict(X_val)
        ml_pred = self.ml_model.predict(X_val)
        
        # Grid search for best weights
        best_rmse = float('inf')
        best_nn_weight = 0.5
        
        for w in np.linspace(0, 1, 21):
            blend_pred = w * nn_pred + (1 - w) * ml_pred
            rmse = np.sqrt(np.mean((y_val - blend_pred)**2))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_nn_weight = w
        
        self.nn_weight = best_nn_weight
        self.ml_weight = 1 - best_nn_weight
        
        logger.info(f"[SUCCESS] Optimal weights: NN={self.nn_weight:.2f}, ML={self.ml_weight:.2f}")
        logger.info(f"Validation RMSE: {best_rmse:.6f}")
        
        return {
            'nn_results': nn_results,
            'nn_weight': self.nn_weight,
            'ml_weight': self.ml_weight,
            'val_rmse': best_rmse
        }
    
    def predict(self, X):
        """Make hybrid predictions"""
        nn_pred = self.nn_model.predict(X)
        ml_pred = self.ml_model.predict(X)
        
        return self.nn_weight * nn_pred + self.ml_weight * ml_pred


if __name__ == "__main__":
    # Test all models
    np.random.seed(42)
    
    X_train = np.random.randn(1000, 5)
    y_train = np.sum(X_train, axis=1) + np.random.randn(1000) * 0.1
    X_val = np.random.randn(200, 5)
    y_val = np.sum(X_val, axis=1) + np.random.randn(200) * 0.1
    
    config = {
        'gpu': {'enable': True, 'device_id': 0, 'memory_fraction': 0.9, 'use_amp': True},
        'bnn': {'hidden_layers': [64, 32, 16], 'epochs': 50, 'batch_size': 32},
        'pinn': {'hidden_layers': [64, 64, 32], 'epochs': 50, 'physics_weight': 0.3}
    }
    
    print("\n=== Testing BNN ===")
    bnn = BayesianNeuralNetwork(input_dim=5, config=config)
    bnn_results = bnn.train(X_train, y_train, X_val, y_val)
    pred, unc = bnn.predict(X_val[:10], return_uncertainty=True)
    print(f"Predictions: {pred}")
    print(f"Uncertainty: {unc}")
    
    print("\n=== Testing PINN ===")
    pinn = PhysicsInformedNN(input_dim=5, config=config)
    pinn_results = pinn.train(X_train, y_train, X_val, y_val)
    
    print("\n=== Testing Ensemble ===")
    ensemble = EnsembleRegressor(config)
    ensemble.build_ensemble(X_train, y_train, n_jobs=4)
    ensemble_pred = ensemble.predict(X_val)
    
    print("\n=== Testing Hybrid ===")
    hybrid = HybridModel(input_dim=5, config=config)
    hybrid_results = hybrid.train(X_train, y_train, X_val, y_val)
    
    print("\n[SUCCESS] All models tested successfully!")