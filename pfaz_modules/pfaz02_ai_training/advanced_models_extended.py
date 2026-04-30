"""
Extended Advanced AI Models
============================

Additional architectures:
- Transformer with Multi-Head Attention
- Residual Networks (ResNet)
- Variational Autoencoder (VAE)
- Attention-based Feature Selection
- Graph Neural Networks (for nuclear structure relationships)

All models are GPU-accelerated and compatible with the existing pipeline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple, Optional
import logging
import math

# Import GPU optimizer from existing advanced_models
try:
    from .advanced_models import PyTorchGPUOptimizer as GPUOptimizer
except ImportError:
    from advanced_models import PyTorchGPUOptimizer as GPUOptimizer
    GPUOptimizer = None
    GPUOptimizer = None

logger = logging.getLogger(__name__)


# ==================== MULTI-HEAD ATTENTION ====================

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the last dimension into (num_heads, d_k)"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            query, key, value: (batch_size, seq_len, d_model)
            mask: Optional attention mask
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(context)

        return output, attention_weights


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer"""

    def __init__(self, d_model: int, num_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Multi-head attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x, attn_weights


# ==================== TRANSFORMER REGRESSOR ====================

class TransformerRegressor:
    """
    Transformer-based regressor with multi-head attention
    Useful for capturing complex feature interactions
    """

    def __init__(self, input_dim: int, config: Dict = None):
        self.input_dim = input_dim
        self.config = config or {}

        # GPU optimizer
        self.gpu_optimizer = GPUOptimizer(self.config.get('gpu', {}))

        # Model parameters
        d_model = self.config.get('transformer', {}).get('d_model', 128)
        num_heads = self.config.get('transformer', {}).get('num_heads', 8)
        num_layers = self.config.get('transformer', {}).get('num_layers', 4)
        d_ff = self.config.get('transformer', {}).get('d_ff', 512)
        dropout = self.config.get('transformer', {}).get('dropout', 0.1)

        self.model = self._build_model(d_model, num_heads, num_layers, d_ff, dropout)
        self.model = self.gpu_optimizer.to_device(self.model)

        # Scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.gpu_optimizer.use_amp else None

        logger.info(f"Transformer initialized: {input_dim} inputs -> {d_model}d x {num_layers} layers")
        logger.info(f"Attention heads: {num_heads}, Device: {self.gpu_optimizer.device}")

    def _build_model(self, d_model: int, num_heads: int, num_layers: int,
                     d_ff: int, dropout: float) -> nn.Module:
        """Build Transformer architecture"""

        class TransformerModel(nn.Module):
            def __init__(self, input_dim, d_model, num_heads, num_layers, d_ff, dropout):
                super().__init__()

                # Input embedding
                self.input_projection = nn.Linear(input_dim, d_model)

                # Positional encoding (learnable)
                self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))

                # Transformer layers
                self.layers = nn.ModuleList([
                    TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                    for _ in range(num_layers)
                ])

                # Output projection
                self.output_projection = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1)
                )

                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                # x: (batch, input_dim)
                batch_size = x.size(0)

                # Project to d_model and add positional encoding
                x = self.input_projection(x)  # (batch, d_model)
                x = x.unsqueeze(1)  # (batch, 1, d_model)
                x = x + self.pos_encoding[:, :1, :]  # Add positional encoding

                x = self.dropout(x)

                # Apply transformer layers
                for layer in self.layers:
                    x, _ = layer(x)

                # Global average pooling
                x = x.mean(dim=1)  # (batch, d_model)

                # Output projection
                output = self.output_projection(x)

                return output

        return TransformerModel(self.input_dim, d_model, num_heads, num_layers, d_ff, dropout)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train Transformer model"""

        logger.info(f"\n{'='*60}")
        logger.info("TRAINING TRANSFORMER REGRESSOR")
        logger.info(f"{'='*60}")

        # Convert to tensors
        X_train_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(X_train))
        y_train_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(y_train).reshape(-1, 1))
        X_val_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(X_val))
        y_val_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(y_val).reshape(-1, 1))

        # DataLoader
        batch_size = self.config.get('transformer', {}).get('batch_size', 32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        learning_rate = self.config.get('transformer', {}).get('learning_rate', 0.0001)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.MSELoss()

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # Training loop
        epochs = self.config.get('transformer', {}).get('epochs', 200)
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        training_history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
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
            training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # Update learning rate
            scheduler.step(val_loss)

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
                logger.info(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, "
                          f"Val: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        logger.info(f"[SUCCESS] Transformer Training completed - Best Val Loss: {best_val_loss:.6f}")

        return {
            'best_val_loss': best_val_loss,
            'final_train_loss': avg_train_loss,
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


# ==================== RESIDUAL NETWORK ====================

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.linear1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.norm2(out)

        # Skip connection
        out = out + identity
        out = F.relu(out)

        return out


class ResNetRegressor:
    """
    Deep Residual Network for regression
    Enables training very deep networks with skip connections
    """

    def __init__(self, input_dim: int, config: Dict = None):
        self.input_dim = input_dim
        self.config = config or {}

        # GPU optimizer
        self.gpu_optimizer = GPUOptimizer(self.config.get('gpu', {}))

        # Model parameters
        hidden_dim = self.config.get('resnet', {}).get('hidden_dim', 128)
        num_blocks = self.config.get('resnet', {}).get('num_blocks', 8)
        dropout = self.config.get('resnet', {}).get('dropout', 0.1)

        self.model = self._build_model(hidden_dim, num_blocks, dropout)
        self.model = self.gpu_optimizer.to_device(self.model)

        # Scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.gpu_optimizer.use_amp else None

        logger.info(f"ResNet initialized: {input_dim} inputs -> {hidden_dim}d x {num_blocks} blocks")
        logger.info(f"Device: {self.gpu_optimizer.device}")

    def _build_model(self, hidden_dim: int, num_blocks: int, dropout: float) -> nn.Module:
        """Build ResNet architecture"""

        class ResNetModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_blocks, dropout):
                super().__init__()

                # Input projection
                self.input_projection = nn.Linear(input_dim, hidden_dim)

                # Residual blocks
                self.blocks = nn.ModuleList([
                    ResidualBlock(hidden_dim, dropout)
                    for _ in range(num_blocks)
                ])

                # Output projection
                self.output_projection = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1)
                )

            def forward(self, x):
                # Input projection
                x = self.input_projection(x)
                x = F.relu(x)

                # Apply residual blocks
                for block in self.blocks:
                    x = block(x)

                # Output projection
                output = self.output_projection(x)

                return output

        return ResNetModel(self.input_dim, hidden_dim, num_blocks, dropout)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train ResNet model"""

        logger.info(f"\n{'='*60}")
        logger.info("TRAINING RESIDUAL NETWORK")
        logger.info(f"{'='*60}")

        # Convert to tensors
        X_train_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(X_train))
        y_train_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(y_train).reshape(-1, 1))
        X_val_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(X_val))
        y_val_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(y_val).reshape(-1, 1))

        # DataLoader
        batch_size = self.config.get('resnet', {}).get('batch_size', 32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        learning_rate = self.config.get('resnet', {}).get('learning_rate', 0.001)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        epochs = self.config.get('resnet', {}).get('epochs', 200)
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        training_history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
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
                logger.info(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {val_loss:.6f}")

        logger.info(f"[SUCCESS] ResNet Training completed - Best Val Loss: {best_val_loss:.6f}")

        return {
            'best_val_loss': best_val_loss,
            'final_train_loss': avg_train_loss,
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


# ==================== VARIATIONAL AUTOENCODER ====================

class VAE(nn.Module):
    """Variational Autoencoder for representation learning"""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: List[int] = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1)
        )

    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent representation"""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        prediction = self.regression_head(z)
        return reconstruction, prediction, mu, logvar


class VAERegressor:
    """
    Variational Autoencoder for regression with uncertainty quantification
    """

    def __init__(self, input_dim: int, config: Dict = None):
        self.input_dim = input_dim
        self.config = config or {}

        # GPU optimizer
        self.gpu_optimizer = GPUOptimizer(self.config.get('gpu', {}))

        # Model parameters
        latent_dim = self.config.get('vae', {}).get('latent_dim', 32)
        hidden_dims = self.config.get('vae', {}).get('hidden_dims', [64, 32])

        self.model = VAE(input_dim, latent_dim, hidden_dims)
        self.model = self.gpu_optimizer.to_device(self.model)

        # Loss weights
        self.reconstruction_weight = self.config.get('vae', {}).get('reconstruction_weight', 1.0)
        self.kl_weight = self.config.get('vae', {}).get('kl_weight', 0.001)
        self.regression_weight = self.config.get('vae', {}).get('regression_weight', 1.0)

        # Scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.gpu_optimizer.use_amp else None

        logger.info(f"VAE initialized: {input_dim} inputs -> {latent_dim}d latent")
        logger.info(f"Device: {self.gpu_optimizer.device}")

    def vae_loss(self, reconstruction, x, prediction, y, mu, logvar):
        """Compute VAE loss"""
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Regression loss
        reg_loss = F.mse_loss(prediction, y, reduction='mean')

        # Total loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.kl_weight * kl_loss +
            self.regression_weight * reg_loss
        )

        return total_loss, recon_loss, kl_loss, reg_loss

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train VAE model"""

        logger.info(f"\n{'='*60}")
        logger.info("TRAINING VARIATIONAL AUTOENCODER")
        logger.info(f"{'='*60}")

        # Convert to tensors
        X_train_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(X_train))
        y_train_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(y_train).reshape(-1, 1))
        X_val_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(X_val))
        y_val_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(y_val).reshape(-1, 1))

        # DataLoader
        batch_size = self.config.get('vae', {}).get('batch_size', 32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        learning_rate = self.config.get('vae', {}).get('learning_rate', 0.001)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        epochs = self.config.get('vae', {}).get('epochs', 200)
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        training_history = {
            'train_loss': [], 'train_recon': [], 'train_kl': [],
            'train_reg': [], 'val_loss': []
        }

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            epoch_reg = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                reconstruction, prediction, mu, logvar = self.model(batch_X)
                total_loss, recon_loss, kl_loss, reg_loss = self.vae_loss(
                    reconstruction, batch_X, prediction, batch_y, mu, logvar
                )

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()
                epoch_reg += reg_loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            avg_recon = epoch_recon / len(train_loader)
            avg_kl = epoch_kl / len(train_loader)
            avg_reg = epoch_reg / len(train_loader)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_recon, val_pred, val_mu, val_logvar = self.model(X_val_tensor)
                val_loss, _, _, val_reg = self.vae_loss(
                    val_recon, X_val_tensor, val_pred, y_val_tensor, val_mu, val_logvar
                )

            val_loss = val_loss.item()

            training_history['train_loss'].append(avg_train_loss)
            training_history['train_recon'].append(avg_recon)
            training_history['train_kl'].append(avg_kl)
            training_history['train_reg'].append(avg_reg)
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
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.6f}, "
                          f"Recon: {avg_recon:.6f}, KL: {avg_kl:.6f}, "
                          f"Reg: {avg_reg:.6f}, Val: {val_loss:.6f}")

        logger.info(f"[SUCCESS] VAE Training completed - Best Val Loss: {best_val_loss:.6f}")

        return {
            'best_val_loss': best_val_loss,
            'final_train_loss': avg_train_loss,
            'device': str(self.gpu_optimizer.device),
            'training_history': training_history,
            'epochs_trained': epoch + 1
        }

    def predict(self, X: np.ndarray, return_latent: bool = False) -> np.ndarray:
        """Make predictions"""
        self.model.eval()

        X_tensor = self.gpu_optimizer.to_device(torch.FloatTensor(X))

        with torch.no_grad():
            mu, logvar = self.model.encode(X_tensor)
            z = self.model.reparameterize(mu, logvar)
            prediction = self.model.regression_head(z)

        if return_latent:
            return prediction.cpu().numpy().flatten(), z.cpu().numpy()
        else:
            return prediction.cpu().numpy().flatten()


# Example usage and testing
if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.randn(1000, 44)
    y_train = np.sum(X_train[:, :5], axis=1) + np.random.randn(1000) * 0.1
    X_val = np.random.randn(200, 44)
    y_val = np.sum(X_val[:, :5], axis=1) + np.random.randn(200) * 0.1

    config = {
        'gpu': {'enable': True, 'device_id': 0},
        'transformer': {
            'd_model': 128, 'num_heads': 8, 'num_layers': 4,
            'epochs': 50, 'batch_size': 32
        },
        'resnet': {
            'hidden_dim': 128, 'num_blocks': 8,
            'epochs': 50, 'batch_size': 32
        },
        'vae': {
            'latent_dim': 32, 'hidden_dims': [64, 32],
            'epochs': 50, 'batch_size': 32
        }
    }

    print("\n=== Testing Transformer ===")
    transformer = TransformerRegressor(input_dim=44, config=config)
    transformer_results = transformer.train(X_train, y_train, X_val, y_val)
    transformer_pred = transformer.predict(X_val[:10])
    print(f"Predictions: {transformer_pred}")

    print("\n=== Testing ResNet ===")
    resnet = ResNetRegressor(input_dim=44, config=config)
    resnet_results = resnet.train(X_train, y_train, X_val, y_val)
    resnet_pred = resnet.predict(X_val[:10])
    print(f"Predictions: {resnet_pred}")

    print("\n=== Testing VAE ===")
    vae = VAERegressor(input_dim=44, config=config)
    vae_results = vae.train(X_train, y_train, X_val, y_val)
    vae_pred, latent = vae.predict(X_val[:10], return_latent=True)
    print(f"Predictions: {vae_pred}")
    print(f"Latent shape: {latent.shape}")

    print("\n[SUCCESS] All extended models tested successfully!")
